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
from torch.utils.data import Dataset

from src.data.audio import load_audio
from src.tokens.tokenizer import CharTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ManifestRow:
    id: int
    audio_path: str
    text: str
    section: Optional[str] = None


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
    """Load audio-transcription pairs from a manifest."""

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

    def __len__(self) -> int:
        return len(self.rows)

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