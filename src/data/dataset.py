"""PyTorch Dataset, collate function, and reproducible dataset splitting."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Sampler

from src.data.audio import load_audio
from src.tokens.tokenizer import CharTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ManifestRow:
    id: int
    audio_path: str
    text: str
    section: Optional[str] = None


def probe_num_samples(
    path: str, base_dir: Optional[Path] = None, sample_rate: int = 16000
) -> int:
    """Sample count of an audio file without decoding its samples.

    Reads the container header only (soundfile), so it is cheap enough to run
    over a whole manifest. Returns 0 when the file cannot be inspected, which
    callers treat as "unknown length".
    """
    import soundfile as sf

    candidates = [Path(path)]
    if base_dir is not None and not Path(path).is_absolute():
        candidates.append(base_dir / path)
    for candidate in candidates:
        try:
            info = sf.info(str(candidate))
        except Exception:  # unreadable header, missing file, no libsndfile
            continue
        if info.samplerate:
            return int(round(info.frames * sample_rate / info.samplerate))
    return 0


class LengthGroupedBatchSampler(Sampler):
    """Yield batches of similar-length clips so padding - and VRAM - stays small.

    The classic length-grouped scheme: shuffle the indices, cut them into
    megabatches, sort each megabatch by length, split into fixed-size batches,
    then shuffle the *batch order* so epochs are still random. Only the
    ordering changes; every sample is still seen exactly once per epoch.

    ``max_frames`` additionally caps ``len(batch) * longest_clip_in_batch``,
    which is the padded sample count the encoder actually sees and therefore
    the direct bound on activation memory. With it, short clips still batch
    together while one 40s outlier lands in a batch of its own instead of
    inflating all of its neighbours.
    """

    def __init__(
        self,
        num_samples: List[int],
        batch_size: int,
        seed: int = 42,
        megabatch_mult: int = 50,
        max_frames: int = 0,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.num_samples = list(num_samples)
        self.batch_size = batch_size
        self.seed = seed
        self.megabatch_mult = megabatch_mult
        self.max_frames = max_frames
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.epoch = 0
        # Unknown lengths must not sort as "shortest" and get over-batched.
        known = [n for n in self.num_samples if n > 0]
        self._fallback = sorted(known)[len(known) // 2] if known else 0

    def _key(self, idx: int) -> int:
        n = self.num_samples[idx]
        return n if n > 0 else self._fallback

    def _batches(self, epoch: int) -> List[List[int]]:
        indices = list(range(len(self.num_samples)))
        if self.shuffle:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(indices)

        mega = self.batch_size * self.megabatch_mult
        batches: List[List[int]] = []
        for start in range(0, len(indices), mega):
            chunk = sorted(indices[start : start + mega], key=self._key)
            current: List[int] = []
            for idx in chunk:
                candidate = current + [idx]
                longest = max(self._key(i) for i in candidate)
                if current and (
                    len(candidate) > self.batch_size
                    or (
                        self.max_frames
                        and len(candidate) * longest > self.max_frames
                    )
                ):
                    batches.append(current)
                    current = [idx]
                else:
                    current = candidate
            if current and not (self.drop_last and len(current) < self.batch_size):
                batches.append(current)

        if self.shuffle:
            random.Random(self.seed + epoch).shuffle(batches)
        return batches

    def __iter__(self):
        # Advance per pass so a shuffled sampler really reshuffles every epoch
        # (a fixed order would hand the model the same batch sequence 30 times).
        # shuffle=False keeps epoch 0, so eval loaders stay deterministic.
        if self.shuffle:
            self.epoch += 1
        return iter(self._batches(self.epoch))

    def __len__(self) -> int:
        # Same count every epoch: ordering varies, batch sizes do not.
        return len(self._batches(0))


def read_manifest(manifest_path: Path) -> List[ManifestRow]:
    rows = []
    with manifest_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for line in reader:
            rows.append(
                ManifestRow(
                    id=int(line["id"]),
                    audio_path=line["audio"],
                    text=line["text"],
                    section=line.get("section") or None,
                )
            )
    return rows


class SpeechDataset(Dataset):
    """Load audio-transcription pairs from a manifest.

    Audio is decoded lazily in ``__getitem__`` and never cached, so only the
    batch being trained on is resident. ``num_samples()`` reads headers only
    (no decoding) to let a sampler group clips of similar duration.
    """

    def __init__(
        self,
        manifest_path: Path,
        tokenizer: CharTokenizer,
        sample_rate: int = 16000,
        load_audio_samples: bool = True,
    ):
        self.manifest_path = manifest_path
        self.rows = read_manifest(manifest_path)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.load_audio_samples = load_audio_samples
        self._num_samples: Optional[List[int]] = None

    def __len__(self) -> int:
        return len(self.rows)

    def num_samples(self) -> List[int]:
        """Per-row sample count at ``self.sample_rate``, 0 when unreadable."""
        if self._num_samples is None:
            base = getattr(self, "manifest_path", None)
            base = base.parent if base is not None else None
            counts = []
            for row in self.rows:
                counts.append(probe_num_samples(row.audio_path, base, self.sample_rate))
            self._num_samples = counts
        return self._num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]

        if self.load_audio_samples:
            result = load_audio(row.audio_path, target_sr=self.sample_rate)
            waveform = result.waveform
            sr = result.sample_rate
            if sr != self.sample_rate:
                raise RuntimeError(f"Unexpected sample rate {sr} for {row.audio_path}")
        else:
            waveform = torch.tensor([0.0])

        tokens = torch.tensor(self.tokenizer.encode(row.text), dtype=torch.long)

        return {
            "audio": waveform,  # (1, T)
            "text": row.text,
            "tokens": tokens,  # (L,)
            "id": torch.tensor(row.id, dtype=torch.long),
        }


@dataclass
class Batch:
    audio: torch.Tensor  # (B, 1, T_masked)
    audio_lengths: torch.Tensor  # (B,)
    tokens: torch.Tensor  # (B, L_masked)
    token_lengths: torch.Tensor  # (B,)
    ids: torch.Tensor  # (B,)
    audio_padding_mask: torch.Tensor  # (B, T)
    token_padding_mask: torch.Tensor  # (B, L)
    texts: List[str]


def collate_speech(samples: List[Dict[str, torch.Tensor]]) -> Batch:
    audio_max = max(s["audio"].shape[1] for s in samples)
    token_max = max(s["tokens"].shape[0] for s in samples)
    b = len(samples)

    audio_batch = torch.zeros(b, 1, audio_max)
    audio_lengths = torch.zeros(b, dtype=torch.long)
    tokens_batch = torch.zeros(b, token_max, dtype=torch.long)
    token_lengths = torch.zeros(b, dtype=torch.long)
    ids_batch = torch.zeros(b, dtype=torch.long)
    texts = []

    for i, s in enumerate(samples):
        a = s["audio"]
        t = s["tokens"]
        audio_batch[i, :, : a.shape[1]] = a
        audio_lengths[i] = a.shape[1]
        tokens_batch[i, : t.shape[0]] = t
        token_lengths[i] = t.shape[0]
        ids_batch[i] = s["id"]
        texts.append(s["text"])

    audio_padding_mask = torch.arange(audio_max).unsqueeze(0) >= audio_lengths.unsqueeze(1)
    token_padding_mask = torch.arange(token_max).unsqueeze(0) >= token_lengths.unsqueeze(1)

    return Batch(
        audio=audio_batch,
        audio_lengths=audio_lengths,
        tokens=tokens_batch,
        token_lengths=token_lengths,
        ids=ids_batch,
        audio_padding_mask=audio_padding_mask,
        token_padding_mask=token_padding_mask,
        texts=texts,
    )


def split_dataset(
    rows: List[ManifestRow],
    seed: int = 42,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> Tuple[List[ManifestRow], List[ManifestRow], List[ManifestRow]]:
    """Split a manifest into train/valid/test without overlapping sample IDs."""
    rng = random.Random(seed)

    ids = list({r.id for r in rows})
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_ids = set(ids[:n_train])
    valid_ids = set(ids[n_train : n_train + n_valid])
    test_ids = set(ids[n_train + n_valid :])

    train = [r for r in rows if r.id in train_ids]
    valid = [r for r in rows if r.id in valid_ids]
    test = [r for r in rows if r.id in test_ids]

    logger.info(
        "Split %d rows -> train=%d valid=%d test=%d",
        len(rows),
        len(train),
        len(valid),
        len(test),
    )
    return train, valid, test


def manifest_fingerprint(manifest_path: Path) -> str:
    """Content hash of a manifest file, used to detect stale persisted splits."""
    with open(manifest_path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def split_fingerprint(path: Path) -> Optional[str]:
    """Return the manifest fingerprint a saved split was built from, if any."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return data.get("manifest_fingerprint")


def split_row_counts(path: Path) -> List[int]:
    """Per-group row counts of a saved split file ([] if unreadable).

    Used as a legacy staleness heuristic for splits saved before the
    manifest fingerprint field was introduced.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    return [len(data.get(k, [])) for k in ("train", "valid", "test")]


def save_split(path: Path, train: List[ManifestRow], valid: List[ManifestRow], test: List[ManifestRow], seed: int, fingerprint: Optional[str] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "seed": seed,
        "train": [r.__dict__ for r in train],
        "valid": [r.__dict__ for r in valid],
        "test": [r.__dict__ for r in test],
    }
    if fingerprint is not None:
        data["manifest_fingerprint"] = fingerprint
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Split saved to %s", path)


def _rows_to_dataset(
    rows: List[Dict],
    tokenizer: CharTokenizer,
    sample_rate: int,
    load_audio_samples: bool = True,
) -> SpeechDataset:
    """Build a SpeechDataset from row dicts (used when loading saved splits)."""
    ds = SpeechDataset.__new__(SpeechDataset)
    ds.rows = [ManifestRow(id=int(r["id"]), audio_path=r["audio_path"], text=r["text"], section=r.get("section")) for r in rows]
    ds.manifest_path = None
    ds._num_samples = None
    ds.tokenizer = tokenizer
    ds.sample_rate = sample_rate
    ds.load_audio_samples = load_audio_samples
    return ds


def load_split(path: Path, tokenizer: CharTokenizer, sample_rate: int = 16000, load_audio_samples: bool = True):
    """Load saved splits and return (train_ds, valid_ds, test_ds)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    train_ds = _rows_to_dataset(data["train"], tokenizer, sample_rate, load_audio_samples)
    valid_ds = _rows_to_dataset(data["valid"], tokenizer, sample_rate, load_audio_samples)
    test_ds = _rows_to_dataset(data["test"], tokenizer, sample_rate, load_audio_samples)
    return train_ds, valid_ds, test_ds