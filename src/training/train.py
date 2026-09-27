"""Training entry point with resume, incremental, and replay support."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import (
    Batch,
    LengthGroupedBatchSampler,
    ManifestRow,
    SpeechDataset,
    collate_speech,
    load_split,
    manifest_fingerprint,
    read_manifest,
    save_split,
    split_dataset,
    split_fingerprint,
    split_row_counts,
    _rows_to_dataset,
)
from src.data.prepare import write_manifest
from src.data.registry import resolve_dataset
from src.device_info import auto_device, resolve_device, summarize
from src.models.stt import STTConfig, STTModel, build_stt_model
from src.tokens.tokenizer import BLANK, CharTokenizer
from src.training.checkpoint import CheckpointManager, TrainingState
from src.training.metrics import cer, compute_metrics, wer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_BACKBONE_ALIASES = {
    "fb": "facebook/mms-300m",
    "mms": "facebook/mms-300m",
    "mms-300m": "facebook/mms-300m",
}

# torch >=2.14 reports device OOM as AcceleratorError; OutOfMemoryError is a
# sibling, not a subclass, and older builds only raise the plain RuntimeError.
OOM_ERRORS: tuple = tuple(
    {
        exc
        for exc in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch, "AcceleratorError", None),
        )
        if isinstance(exc, type) and issubclass(exc, BaseException)
    }
) or (RuntimeError,)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    batch_size: int = 8
    length_group: bool = True
    max_batch_frames: int = 0
    num_workers: int = 0
    learning_rate: float = 1e-3
    epochs: int = 30
    optimizer: str = "AdamW"
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    dataset: str = "data/processed/manifest.csv"
    data: Optional[str] = None
    data_name: Optional[str] = None
    text_field: str = "transcript"
    preset_splits: Optional[Dict[str, List[ManifestRow]]] = None
    validation_frequency: int = 1
    replay_ratio: float = 0.0
    replay_manifest: Optional[str] = None
    resume: Optional[str] = None
    from_checkpoint: Optional[str] = None
    mixed_precision: bool = True
    split_dir: str = "data/processed"
    eval_manifest: Optional[str] = None
    max_audio_len: int = 5
    max_text_len: int = 64
    beam_size: int = 5
    retain_every: int = 0
    retain_protect: List[str] = field(default_factory=list)
    lr_patience: int = 5
    lr_factor: float = 0.1
    lr_threshold: float = 1e-4
    lr_cooldown: int = 0
    new_run: bool = False
    continue_mode: Optional[str] = None
    model: STTConfig = field(default_factory=STTConfig)

    sample_rate: int = 16000


def _is_true_env_flag(name: str) -> bool:
    return os.environ.get(name, "1").lower() not in {"0", "false", "no", "off"}


def _build_plateau(
    optimizer: torch.optim.Optimizer,
    lr_patience: int = 5,
    lr_factor: float = 0.1,
    lr_threshold: float = 1e-4,
    lr_cooldown: int = 0,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """ReduceLROnPlateau with the project's configurable hyperparameters.

    ``mode="min"`` (lower validation_loss is better) with relative threshold,
    matching the historical hardcoded scheduler until a flag overrides it.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=lr_patience,
        factor=lr_factor,
        threshold=lr_threshold,
        cooldown=lr_cooldown,
    )


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.manager = CheckpointManager(config.checkpoint_dir)
        self.tokenizer: Optional[CharTokenizer] = None
        self.model: Optional[STTModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.state = TrainingState()
        self.global_step = 0
        self.last_saved = False
        self.parent_model_sd: Optional[Dict[str, torch.Tensor]] = None

    def _validate_training_args(self):
        if self.config.resume and self.config.from_checkpoint:
            raise ValueError("Use either --resume or --from-checkpoint, not both.")
        modes = (
            self.config.new_run,
            self.config.continue_mode is not None,
            bool(self.config.resume),
            bool(self.config.from_checkpoint),
        )
        if sum(modes) > 1:
            raise ValueError(
                "Use only one of -new, -continue, --resume, or --from-checkpoint."
            )

    def _prepare_epoch_from_checkpoint(self):
        """Incremental run: inherit the parent's weights, arch, and vocabulary.

        The parent's ``model_sd`` is stashed and applied in
        :meth:`_build_model_and_optimizer` — after the tokenizer has been extended
        with the new data's characters — so the vocabulary-sized tensors can be
        resized instead of silently resetting to a fresh random model.
        """
        data = self.manager.load(self.config.from_checkpoint)
        self.tokenizer = data["tokenizer"]
        self.config.model = data["config"]
        self.state.parent_checkpoint = self.config.from_checkpoint
        self.parent_model_sd = data["model_sd"]

        parent_state = data["state"]
        self.state.dataset_version = parent_state.dataset_version

    def _load_parent_weights(self):
        """Restore the parent's tensors onto the freshly built model.

        Layers whose shape changed with a grown vocabulary (token embedding,
        output projection) are dropped and keep their fresh initialization;
        everything else - including the fine-tuned backbone - is restored.
        """
        own = self.model.state_dict()
        usable = {}
        skipped = []
        for key, value in self.parent_model_sd.items():
            current = own.get(key)
            if current is None or current.shape != value.shape:
                skipped.append(key)
            else:
                usable[key] = value
        self.model.load_state_dict(usable, strict=False)
        logger.info(
            "Continued from %s: restored %d parent tensors",
            self.config.from_checkpoint,
            len(usable),
        )
        if skipped:
            logger.info(
                "Kept fresh init for %d resized tensor(s) (vocabulary grew): %s",
                len(skipped),
                ", ".join(skipped),
            )

    def _partition_rows(
        self, rows: List[ManifestRow]
    ) -> tuple[List[ManifestRow], List[ManifestRow], List[ManifestRow]]:
        """Use the dataset's official partition when it ships one, else split randomly.

        A preset split (e.g. FLEURS train/validation/test) is reproduced as-is so
        numbers stay comparable with the corpus' published results; everything
        else keeps the historical seeded 80/10/10 cut.
        """
        preset = self.config.preset_splits
        if preset:
            logger.info(
                "Using preset splits from dataset %r: %s",
                self.config.data_name,
                ", ".join(f"{k}={len(v)}" for k, v in preset.items()),
            )
            return (
                list(preset.get("train", [])),
                list(preset.get("valid", [])),
                list(preset.get("test", [])),
            )
        return split_dataset(rows, seed=self.config.seed)

    def _datasets_from_rows(
        self,
        train_rows: List[ManifestRow],
        valid_rows: List[ManifestRow],
        test_rows: List[ManifestRow],
    ) -> tuple[SpeechDataset, SpeechDataset, SpeechDataset]:
        def to_dataset(rows: List[ManifestRow]) -> SpeechDataset:
            return _rows_to_dataset(
                [r.__dict__ for r in rows], self.tokenizer, self.config.sample_rate
            )

        return to_dataset(train_rows), to_dataset(valid_rows), to_dataset(test_rows)

    def _load_datasets(
        self,
    ) -> tuple[SpeechDataset, SpeechDataset, SpeechDataset, SpeechDataset]:
        manifest = Path(self.config.dataset)
        train_rows, valid_rows, test_rows = None, None, None

        split_file = (
            Path(self.config.split_dir)
            / f"split_{Path(manifest).stem}.json"
        )

        new_run = not self.config.resume and not self.config.from_checkpoint

        rows = read_manifest(manifest)
        current_fp = manifest_fingerprint(manifest)

        if new_run:
            self.tokenizer = CharTokenizer()
            self.tokenizer.build_from_texts([r.text for r in rows])
            if self.config.model.head == "ctc":
                self.tokenizer.add_blank()
            train_rows, valid_rows, test_rows = self._partition_rows(rows)
            save_split(
                split_file, train_rows, valid_rows, test_rows,
                self.config.seed, fingerprint=current_fp,
            )
            train_ds, valid_ds, test_ds = self._datasets_from_rows(
                train_rows, valid_rows, test_rows
            )
        else:
            if self.tokenizer is None:
                ckpt = self.config.resume or self.config.from_checkpoint
                self.tokenizer = self.manager.load(ckpt)["tokenizer"]

            if self.config.from_checkpoint:
                # Incremental: keep every parent token id (they carry trained
                # embeddings) and append the characters the new data introduces.
                # --resume deliberately skips this - its data is unchanged and
                # its model was already restored at the parent's vocabulary.
                before = self.tokenizer.vocab_size()
                self.tokenizer.build_from_texts([r.text for r in rows])
                added = self.tokenizer.vocab_size() - before
                logger.info(
                    "Incremental vocabulary: %d -> %d token(s) (%d added for the new data)",
                    before,
                    self.tokenizer.vocab_size(),
                    added,
                )

            reused = False
            if split_file.exists():
                saved_fp = split_fingerprint(split_file)
                if saved_fp is not None:
                    stale = saved_fp != current_fp
                else:
                    stale = sum(split_row_counts(split_file)) != len(rows)
                if not stale:
                    train_ds, valid_ds, test_ds = load_split(
                        split_file, self.tokenizer, self.config.sample_rate
                    )
                    reused = True
                else:
                    logger.warning(
                        "Split %s was built from different data (manifest changed); "
                        "re-splitting from the current manifest.",
                        split_file,
                    )

            if not reused:
                train_rows, valid_rows, test_rows = self._partition_rows(rows)
                save_split(
                    split_file, train_rows, valid_rows, test_rows,
                    self.config.seed, fingerprint=current_fp,
                )
                train_ds, valid_ds, test_ds = self._datasets_from_rows(
                    train_rows, valid_rows, test_rows
                )

        replay_ds = None
        if self.config.replay_manifest and self.config.replay_ratio > 0:
            replay_rows = read_manifest(Path(self.config.replay_manifest))
            replay_ds = _rows_to_dataset(
                [r.__dict__ for r in replay_rows],
                self.tokenizer,
                self.config.sample_rate,
            )

        return train_ds, valid_ds, test_ds, replay_ds

    def _warn_decode_budget(self, dataset: SpeechDataset, sample: int = 500):
        """Warn when transcripts are longer than the decoding cap.

        Training pads to the batch maximum, but eval decoding stops at
        ``--max-text-len``; on long-sentence corpora (FLEURS medians sit well
        above 100 characters) the default 64 would silently truncate every
        hypothesis and inflate CER.
        """
        rows = dataset.rows[:sample]
        if not rows:
            return
        lengths = sorted(len(self.tokenizer.encode(r.text)) for r in rows)
        p90 = lengths[int(len(lengths) * 0.9) - 1]
        over = sum(1 for length in lengths if length > self.config.max_text_len)
        if over:
            logger.warning(
                "%d/%d sampled train transcripts exceed --max-text-len %d "
                "(p90=%d chars incl. BOS/EOS): eval decoding will truncate them "
                "and inflate CER - pass -mtl %d",
                over,
                len(lengths),
                self.config.max_text_len,
                p90,
                max(p90, self.config.max_text_len),
            )

    def _build_model_and_optimizer(self):
        if self.model is None:
            if self.config.model.head == "ctc" and self.config.model.blank_token_id is None:
                blank_id = self.tokenizer.blank_id
                if blank_id is None:
                    raise RuntimeError("CTC training requires a blank token in the tokenizer")
                self.config.model.blank_token_id = blank_id
            self.model = build_stt_model(self.config.model, self.tokenizer.vocab_size())
            if self.parent_model_sd is not None:
                self._load_parent_weights()
            self.model.to(self.device)

        # AMP lives with the model, so a caller that drives _train_epoch
        # directly (tests, tooling) gets the same numerics as train().
        self.use_amp = self.config.mixed_precision and self.device.type == "cuda"
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.use_amp)
            if self.use_amp
            else None
        )

        if self.optimizer is None:
            opt_name = self.config.optimizer.lower()
            trainable = list(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
            logger.info(
                "Trainable params: %d / %d",
                sum(p.numel() for p in trainable),
                sum(p.numel() for p in self.model.parameters()),
            )
            if opt_name in ("adamw", "adam"):
                self.optimizer = torch.optim.AdamW(
                    trainable, lr=self.config.learning_rate
                )
            elif opt_name == "sgd":
                self.optimizer = torch.optim.SGD(
                    trainable, lr=self.config.learning_rate
                )
            else:
                raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        if self.scheduler is None:
            self.scheduler = self._build_scheduler()

    def _build_scheduler(self) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """Build the LR scheduler from configurable plateau hyperparameters.

        Used by both the fresh-path and the resume path so a --lr-patience /
        --lr-factor / --lr-threshold / --lr-cooldown choice is honored
        identically everywhere. ``--resume`` then overlays the saved
        scheduler state (best / num_bad_epochs / last_lr) on top.
        """
        return _build_plateau(
            self.optimizer,
            lr_patience=self.config.lr_patience,
            lr_factor=self.config.lr_factor,
            lr_threshold=self.config.lr_threshold,
            lr_cooldown=self.config.lr_cooldown,
        )

    def _restore_checkpoint(self, path: str):
        data = self.manager.load(path)
        self.tokenizer = data["tokenizer"]
        self.config.model = data["config"]
        self.model = STTModel(data["config"], vocab_size=self.tokenizer.vocab_size())
        self.model.load_state_dict(data["model_sd"])
        self.model.to(self.device)

        opt_name = self.config.optimizer.lower()
        trainable = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )
        if opt_name in ("adamw", "adam"):
            self.optimizer = torch.optim.AdamW(trainable, lr=self.config.learning_rate)
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(trainable, lr=self.config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        if data["optimizer_sd"] is not None:
            self.optimizer.load_state_dict(data["optimizer_sd"])

        self.scheduler = self._build_scheduler()
        if data["scheduler_sd"] is not None:
            self.scheduler.load_state_dict(data["scheduler_sd"])

        self.state = data["state"]
        self.global_step = data["state"].global_step
        self.state.global_step = self.global_step

        rng_path = Path(data["path"]) / "rng.pt"
        if rng_path.exists():
            rng = torch.load(rng_path, map_location="cpu", weights_only=False)
            if "torch" in rng:
                torch.set_rng_state(rng["torch"])
            if "python" in rng:
                random.setstate(rng["python"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])

    def _build_loader(
        self,
        dataset: SpeechDataset,
        shuffle: bool,
        num_workers: Optional[int] = None,
    ) -> DataLoader:
        """One loader, streaming: audio is decoded per item and dropped again.

        With ``--length-group`` (default) the batches are cut by duration
        instead of at random, so a batch is padded to roughly its own longest
        clip rather than to the longest clip in the epoch. On a 6 GB card that
        is the difference between training on 13s FLEURS sentences and
        immediately running out of memory on a 40s one.

        The frame cap applies to *every* loader, validation included: an
        uncapped eval batch padded to the longest clip in a window is the same
        memory spike as an uncapped training batch, and the peak lands after
        the training allocator has already cached a full epoch's blocks.
        """
        workers = self.config.num_workers if num_workers is None else num_workers
        if not self.config.length_group or not isinstance(dataset, SpeechDataset):
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                collate_fn=collate_speech,
                num_workers=workers,
            )

        sampler = LengthGroupedBatchSampler(
            dataset.num_samples(),
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            max_frames=self.config.max_batch_frames,
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_speech,
            num_workers=workers,
        )

    def train(self):
        self._validate_training_args()

        if self.config.resume:
            self._restore_checkpoint(self.config.resume)
            train_ds, valid_ds, test_ds, replay_ds = self._load_datasets()
        elif self.config.from_checkpoint:
            self._prepare_epoch_from_checkpoint()
            train_ds, valid_ds, test_ds, replay_ds = self._load_datasets()
        else:
            train_ds, valid_ds, test_ds, replay_ds = self._load_datasets()

        logger.info("Train samples: %d", len(train_ds))
        logger.info("Validation samples: %d", len(valid_ds))
        logger.info("Test samples: %d", len(test_ds))
        self._warn_decode_budget(train_ds)

        self._build_model_and_optimizer()

        train_loader = self._build_loader(train_ds, shuffle=True)
        valid_loader = self._build_loader(valid_ds, shuffle=False)
        typed_valid_loaders = self._build_typed_validation_loaders(valid_ds)
        test_loader = self._build_loader(test_ds, shuffle=False)

        if replay_ds is not None:
            logger.info("Replay samples: %d", len(replay_ds))
            replay_loader = self._build_loader(replay_ds, shuffle=True)
        else:
            replay_loader = None

        logger.info(
            "Using device: %s (AMP=%s); auto would select %s",
            self.device,
            self.use_amp,
            auto_device(),
        )
        logger.info("\n".join(summarize()))

        start_epoch = self.state.epoch
        for epoch in range(start_epoch, self.config.epochs):
            train_loss, train_cer, train_wer = self._train_epoch(
                train_loader, replay_loader, epoch
            )
            self.state.epoch = epoch + 1

            should_val = (epoch + 1) % self.config.validation_frequency == 0
            val_loss = 0.0
            val_cer = 0.0
            val_wer = 0.0
            if should_val:
                # Hand the training pass's cached blocks back before the eval
                # peak, otherwise the allocator holds a full epoch's worth and
                # the first validation batch has nothing left to grow into.
                self._free_memory()
                val_loss, val_cer, val_wer = self._evaluate(valid_loader)
                validation_by_type = self._evaluate_by_type(typed_valid_loaders)
                self.scheduler.step(val_loss)
            else:
                validation_by_type = {}

            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"epoch={epoch + 1}/{self.config.epochs} "
                f"step={self.global_step} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"CER={val_cer:.4f} WER={val_wer:.4f} "
                f"lr={lr:.2e}"
            )
            for sample_type, metrics in validation_by_type.items():
                print(
                    f"  {sample_type}: val_loss={metrics['loss']:.4f} "
                    f"CER={metrics['cer']:.4f} WER={metrics['wer']:.4f}"
                )

            self.state.train_loss = train_loss
            self.state.validation_loss = val_loss
            self.state.cer = val_cer
            self.state.wer = val_wer
            self.state.validation_by_type = validation_by_type

            if should_val:
                self._save_checkpoint(
                    manifest={
                        "dataset": self.config.dataset,
                        "replay_manifest": self.config.replay_manifest,
                        "replay_ratio": self.config.replay_ratio,
                    }
                )
                self.manager.update_best(
                    self.state,
                    self.config.dataset,
                    manifest_fingerprint(Path(self.config.dataset)),
                )
                self.last_saved = True
                self._prune_checkpoints()

        # Regression eval when incremental
        if self.config.from_checkpoint and self.config.eval_manifest:
            self._regression_eval()

        # Final test-set evaluation
        print("\n=== Final evaluation ===")
        _, test_cer, test_wer = self._evaluate(test_loader)
        print(f"Test CER={test_cer:.4f} WER={test_wer:.4f}")
        self.state.cer = test_cer
        self.state.wer = test_wer
        if not self.last_saved:
            self._save_checkpoint(
                manifest={
                    "dataset": self.config.dataset,
                    "replay_manifest": self.config.replay_manifest,
                    "replay_ratio": self.config.replay_ratio,
                }
            )
        else:
            self.manager.update_latest_state(self.state)

        self._prune_checkpoints()

    def _train_epoch(
        self,
        train_loader: DataLoader,
        replay_loader: Optional[DataLoader],
        epoch: int,
    ) -> tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        skipped = 0
        all_preds: List[str] = []
        all_refs: List[str] = []

        replay_iter = iter(replay_loader) if replay_loader is not None else None
        replay_ratio = self.config.replay_ratio

        for batch_num, batch in enumerate(train_loader):
            self.model.train()

            # Mix in a replay batch with probability replay_ratio
            use_replay = replay_iter is not None and random.random() < replay_ratio
            if use_replay:
                try:
                    batch = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    batch = next(replay_iter)

            audio = batch.audio.to(self.device)
            audio_lengths = batch.audio_lengths.to(self.device)
            tokens = batch.tokens.to(self.device)
            token_lengths = batch.token_lengths.to(self.device)

            self.optimizer.zero_grad()

            try:
                with torch.autocast(
                    device_type=self.device.type, enabled=self.use_amp
                ):
                    loss = self._batch_loss(
                        audio, audio_lengths, tokens, token_lengths
                    )

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            except OOM_ERRORS as exc:
                # A long clip can still spike past what the card can hold;
                # drop this batch rather than losing a multi-hour run.
                self.optimizer.zero_grad(set_to_none=True)
                del audio, audio_lengths, tokens, token_lengths
                self._free_memory()
                skipped += 1
                logger.warning(
                    "Out of memory on batch %d (epoch %d, %d clips, longest %d samples); "
                    "skipping it - lower -b or --max-batch-frames (%s)",
                    batch_num + 1,
                    epoch + 1,
                    len(batch.audio_lengths),
                    int(batch.audio_lengths.max()),
                    type(exc).__name__,
                )
                continue

            if self.config.grad_clip > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    self.config.grad_clip,
                )

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.global_step += 1
            total_loss += loss.item()
            n_batches += 1

            if (batch_num + 1) % 10 == 0:
                logger.info(
                    "epoch %d batch %d loss %.4f step %d",
                    epoch + 1,
                    batch_num + 1,
                    loss.item(),
                    self.global_step,
                )

        # Compute train metrics on a sample for reporting
        self.model.eval()
        sample_batches = 0
        with torch.no_grad():
            for batch in train_loader:
                if sample_batches >= max(2, 64 // self.config.batch_size):
                    break
                sample_batches += 1
                hyp_ids = self.model.decode(
                    batch.audio.to(self.device),
                    batch.audio_lengths.to(self.device),
                    max_len=self.config.max_text_len,
                    beam_size=self.config.beam_size,
                    bos_token_id=self.tokenizer.bos_id,
                    eos_token_id=self.tokenizer.eos_id,
                )
                for h, ref in zip(hyp_ids, batch.texts):
                    all_preds.append(self.tokenizer.decode(h))
                    all_refs.append(ref)

        avg_loss = total_loss / max(n_batches, 1)
        metrics = compute_metrics(all_refs, all_preds)
        if skipped:
            logger.warning(
                "epoch %d finished with %d batch(es) skipped after CUDA OOM",
                epoch + 1,
                skipped,
            )
        return avg_loss, metrics.cer, metrics.wer

    def _free_memory(self):
        """Release cached CUDA blocks so the next batch starts from clean VRAM."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _batch_loss(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the training loss for one batch (CTC or seq2seq)."""
        if self.config.model.head == "ctc":
            log_probs = self.model.forward_ctc(audio, audio_lengths)
            return self._compute_ctc_loss(log_probs, audio_lengths, tokens, token_lengths)

        in_tokens = tokens[:, :-1]
        in_tok_lens = torch.clamp(token_lengths - 1, min=1)
        target = tokens[:, 1:]
        logits = self.model(audio, audio_lengths, in_tokens, in_tok_lens)
        return self._compute_loss(logits, target, token_lengths)

    def _compute_ctc_loss(
        self,
        log_probs: torch.Tensor,
        audio_lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """CTC loss over per-timestep log-probs.

        The dataset encodes labels as [BOS, chars..., EOS] (with PAD after);
        CTC targets are the raw character sequence only - BOS/EOS are
        stripped. CTCLoss needs time-major log-probs (T, N, C), input
        lengths (encoder frames, pre-padding), and target lengths.
        """
        b, t, c = log_probs.shape
        lp = log_probs.transpose(0, 1).contiguous()  # (T, N, C)

        input_lengths = self.model.ctc_input_lengths(audio_lengths).clamp(max=t)

        target_parts = []
        target_lengths = torch.zeros(b, dtype=torch.long, device=log_probs.device)
        for i in range(b):
            tl = int(token_lengths[i].item())
            target_parts.append(tokens[i, 1 : tl - 1])  # skip BOS and EOS
            target_lengths[i] = max(tl - 2, 0)
        targets = (
            torch.cat(target_parts)
            if target_parts
            and any(p.numel() for p in target_parts)
            else torch.zeros(0, dtype=torch.long, device=log_probs.device)
        )

        blank_id = (
            self.config.model.blank_token_id
            if self.config.model.blank_token_id is not None
            else 0
        )
        return nn.functional.ctc_loss(
            lp,
            targets,
            input_lengths,
            target_lengths,
            blank=blank_id,
            zero_infinity=True,
        )

    def _compute_loss(
        self, logits: torch.Tensor, target: torch.Tensor, token_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy loss with padding positions masked out.

        target rows are the teacher-forced shift (tokens[:, 1:]).
        Positions at/after token_lengths - 1 correspond to padding and are ignored.
        """
        b, l, vocab = logits.shape

        # Consider only valid text positions: up to token_lengths-1 (the EOS is at
        # token_lengths-1, everything after is padding).
        lengths = token_lengths.clamp(max=l)
        arange = torch.arange(l, device=logits.device).unsqueeze(0)  # (1, L)
        valid = arange < (lengths - 1).unsqueeze(1)  # (B, L)

        masked_target = target.masked_fill(~valid, -100)

        logits = logits.reshape(-1, vocab)
        masked_target = masked_target.reshape(-1)
        return nn.functional.cross_entropy(logits, masked_target, ignore_index=-100)

    def _evaluate(
        self, loader: DataLoader, label: str = "validation"
    ) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        skipped = 0
        all_preds: List[str] = []
        all_refs: List[str] = []

        with torch.no_grad():
            for batch_num, batch in enumerate(loader):
                audio = batch.audio.to(self.device)
                audio_lengths = batch.audio_lengths.to(self.device)
                tokens = batch.tokens.to(self.device)
                token_lengths = batch.token_lengths.to(self.device)

                try:
                    with torch.autocast(
                        device_type=self.device.type, enabled=self.use_amp
                    ):
                        loss = self._batch_loss(
                            audio, audio_lengths, tokens, token_lengths
                        )
                    total_loss += loss.item()
                    n_batches += 1
                except OOM_ERRORS as exc:
                    # Never let one oversized eval batch discard a completed
                    # epoch: the weights are already trained, only the metrics
                    # for this window are lost.
                    del audio, audio_lengths, tokens, token_lengths
                    self._free_memory()
                    skipped += 1
                    logger.warning(
                        "Out of memory during %s batch %d (%d clips, longest %d "
                        "samples); skipping it - lower -b or "
                        "--max-batch-frames (%s)",
                        label,
                        batch_num + 1,
                        len(batch.audio_lengths),
                        int(batch.audio_lengths.max()),
                        type(exc).__name__,
                    )
                    continue

                hyp_ids = self.model.decode(
                    audio,
                    audio_lengths,
                    max_len=self.config.max_text_len,
                    beam_size=self.config.beam_size,
                    bos_token_id=self.tokenizer.bos_id,
                    eos_token_id=self.tokenizer.eos_id,
                )
                for h, ref in zip(hyp_ids, batch.texts):
                    all_preds.append(self.tokenizer.decode(h))
                    all_refs.append(ref)

        if skipped:
            logger.warning(
                "%s: skipped %d of %d batch(es) after OOM - CER/WER below cover "
                "the remaining %d sample(s) only",
                label,
                skipped,
                skipped + n_batches,
                len(all_refs),
            )
        metrics = compute_metrics(all_refs, all_preds)
        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss, metrics.cer, metrics.wer

    def _build_typed_validation_loaders(
        self, valid_ds: SpeechDataset
    ) -> Dict[str, DataLoader]:
        """Build automatic word/sentence validation views from transcript text.

        These go through ``_build_loader`` so they inherit the same frame cap
        as training. Built by hand they were the worst batch in the run: FLEURS
        validation has no single-word clips at all, so the whole split lands in
        the "sentence" view, and an uncapped ``batch_size`` x longest-clip
        window reached 5.3M padded samples against a 800k training cap.
        """
        typed: Dict[str, DataLoader] = {}
        for sample_type in ("word", "sentence"):
            rows = [
                row for row in valid_ds.rows
                if (len(row.text.split()) == 1) == (sample_type == "word")
            ]
            if not rows:
                continue
            dataset = _rows_to_dataset(
                [row.__dict__ for row in rows],
                self.tokenizer,
                self.config.sample_rate,
            )
            typed[sample_type] = self._build_loader(
                dataset, shuffle=False, num_workers=0
            )
            logger.info("%s validation samples: %d", sample_type, len(dataset))
        return typed

    def _evaluate_by_type(
        self, loaders: Dict[str, DataLoader]
    ) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        for sample_type, loader in loaders.items():
            loss, sample_cer, sample_wer = self._evaluate(
                loader, label=f"{sample_type} validation"
            )
            results[sample_type] = {
                "loss": loss,
                "cer": sample_cer,
                "wer": sample_wer,
            }
        return results

    def _save_checkpoint(self, manifest: Dict):
        self.state.global_step = self.global_step
        version_dir = self.manager.save(
            self.model,
            self.optimizer,
            self.scheduler,
            self.tokenizer,
            self.config.model,
            self.state,
            extra=manifest,
        )

        rng_path = Path(version_dir) / "rng.pt"
        torch.save(
            {
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            },
            str(rng_path),
        )

    def _prune_checkpoints(self):
        if self.config.retain_every < 0:
            return
        self.manager.prune(
            retain_every=self.config.retain_every,
            keep_best=True,
            protect=self.config.retain_protect,
        )

    def _regression_eval(self):
        """Evaluate new model on old/reference data to detect forgetting."""
        eval_manifest = Path(self.config.eval_manifest)
        if not eval_manifest.exists():
            logger.warning("Eval manifest not found: %s", eval_manifest)
            return

        rows = read_manifest(eval_manifest)
        ds = _rows_to_dataset(
            [r.__dict__ for r in rows], self.tokenizer, self.config.sample_rate
        )
        loader = DataLoader(ds, batch_size=self.config.batch_size, collate_fn=collate_speech)

        if self.config.from_checkpoint:
            logger.info("=== Regression evaluation on reference dataset ===")
            _, cer_val, wer_val = self._evaluate(loader)
            print(f"Reference dataset: CER={cer_val:.4f} WER={wer_val:.4f}")
            self.state.regression = {
                "reference_manifest": str(eval_manifest),
                "cer": cer_val,
                "wer": wer_val,
            }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or continue training the STT model"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "-new",
        "--new",
        action="store_true",
        help="Start a fresh run from scratch on the current dataset "
        "(rebuilds tokenizer and splits).",
    )
    modes.add_argument(
        "-continue",
        "--continue",
        nargs="?",
        const="best",
        choices=["best", "latest"],
        metavar="SOURCE",
        help="Continue from an existing checkpoint line onto the current dataset. "
        "SOURCE selects the parent version: 'best' (lowest validation_loss, "
        "default) or 'latest'. Fresh optimizer; auto-wires replay + regression "
        "eval from the parent's dataset manifest.",
    )
    modes.add_argument(
        "--resume",
        "-resume",
        nargs="?",
        const="_AUTO_",
        default=None,
        metavar="CHECKPOINT",
        help="Continue the same run/state (optimizer+scheduler+epoch restored). "
        "With no value, auto-resolves checkpoints/<newest-line>/latest; otherwise "
        "a version dir or path like checkpoints/mms/latest.",
    )
    modes.add_argument(
        "--from-checkpoint",
        default=None,
        metavar="CHECKPOINT",
        help="Incremental: load weights/vocab/arch from an existing checkpoint, "
        "start a fresh optimizer and record parent_checkpoint.",
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data",
        default=None,
        metavar="NAME_OR_PATH",
        help="Named dataset or path. 'default' = data/processed/manifest.csv "
        "(random 80/10/10 split); 'fleurs_ceb_ph' = data/fleurs_ceb_ph with its "
        "official train/validation/test splits; a dataset directory or manifest "
        "CSV/TSV path also works. Run `ns.py datasets` for the registry. "
        "Mutually exclusive with --dataset.",
    )
    data_group.add_argument(
        "--dataset",
        default="data/processed/manifest.csv",
        help="Manifest CSV path (legacy; use --data for named datasets)",
    )
    parser.add_argument(
        "--text-field",
        default="transcript",
        choices=["transcript", "raw_transcript"],
        help="Which column of a multi-column export becomes the target text. "
        "Default 'transcript' (the normalized column).",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument(
        "--no-length-group",
        dest="length_group",
        action="store_false",
        help="Disable duration-bucketed batches (default: group clips of similar "
             "length together so padding, and VRAM, stay small).",
    )
    parser.add_argument(
        "--max-batch-frames",
        type=int,
        default=0,
        help="Cap padded samples per batch (len(batch) * longest clip), the "
             "direct bound on activation memory. 0 = only -b applies. Units are "
             "16 kHz samples: 400000 = 25s of audio per batch, e.g. two 13s "
             "FLEURS sentences or eight 3s clips. Measured on a 6GB RTX 3050 "
             "(MMS-300m, last 4 layers unfrozen): 166k -> 2.4GB peak, 400k -> "
             "4.1GB, 800k -> 5.2GB, 1.3M -> 7.2GB (oversubscribed).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes for audio decoding. 0 keeps only the "
             "current batch in memory; >0 prefetches that many times per worker.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("--optimizer", default="AdamW", choices=["AdamW", "SGD"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("-d", "--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "-c",
        "--checkpoint-dir",
        default=None,
        help="Checkpoint line to read from/write to. Continue/resume auto-detect the "
        "newest line under checkpoints/ when omitted.",
    )
    parser.add_argument("--validation-frequency", type=int, default=1)
    parser.add_argument("--replay-ratio", type=float, default=0.0)
    parser.add_argument("--replay-manifest", default=None)
    parser.add_argument("--eval-manifest", default=None)
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision")
    parser.add_argument("--split-dir", default="data/processed")
    parser.add_argument("-mtl", "--max-text-len", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=5, help="Beam width for eval decoding (1 = greedy)")
    parser.add_argument("--model-d-model", type=int, default=128)
    parser.add_argument("--model-heads", type=int, default=4)
    parser.add_argument("--model-encoder-layers", type=int, default=3)
    parser.add_argument("--model-decoder-layers", type=int, default=3)
    parser.add_argument(
        "--model-head",
        default="decoder",
        choices=["decoder", "ctc"],
        help="Head: 'decoder' (seq2seq, teacher-forced CE) or 'ctc' (CTC head, "
        "diagnostic). CTC disables teacher forcing entirely.",
    )
    parser.add_argument(
        "-m",
        "--model-backbone",
        default="fb",
        help="Pretrained acoustic backbone. 'none' = from-scratch encoder. "
        "E.g. 'facebook/mms-300m' (hidden 1024, 24 layers, ~315M params, "
        "self-supervised on 1,406 languages incl. ceb). Aliases: 'fb'/'mms' "
        "= facebook/mms-300m. Default 'fb' (the previously used model).",
    )
    parser.add_argument(
        "--backbone-unfreeze-layers",
        type=int,
        default=4,
        help="Last K transformer layers of the backbone left trainable "
        "(standard low-resource fine-tuning; everything else frozen).",
    )
    parser.add_argument(
        "--retain-every",
        type=int,
        default=0,
        help="Checkpoint retention: keep the final, metric-best, and protected "
        "versions. Positive N also keeps every Nth version; 0 disables milestone "
        "retention; negative values disable pruning. Default 0.",
    )
    parser.add_argument(
        "--retain-protect",
        action="append",
        default=[],
        metavar="VERSION",
        help="Never delete this version dir (e.g. --retain-protect v026). "
        "Repeatable. Independent of the milestone/best/final keep-set.",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience (epochs without improvement before the "
        "LR drops). Default 5 (matches the old hardcoded value).",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.1,
        help="ReduceLROnPlateau reduction factor (new_lr = lr * factor). Default 0.1.",
    )
    parser.add_argument(
        "--lr-threshold",
        type=float,
        default=1e-4,
        help="ReduceLROnPlateau threshold for 'no significant improvement' "
        "(relative to the current best; threshold_mode='rel'). Default 1e-4.",
    )
    parser.add_argument(
        "--lr-cooldown",
        type=int,
        default=0,
        help="ReduceLROnPlateau cooldown: epochs to wait after a drop before "
        "the patience counter resumes. Default 0.",
    )
    return parser


def _detect_checkpoint_line(checkpoint_dir: Optional[str]) -> Path:
    """Resolve the checkpoint line to operate on.

    An explicit ``--checkpoint-dir`` is used as-is (must exist). Otherwise the
    newest line under ``checkpoints/`` (a subdir holding a ``latest.json``) is
    auto-detected.
    """
    if checkpoint_dir:
        line = Path(checkpoint_dir)
        if not line.is_dir():
            raise FileNotFoundError(f"Checkpoint line not found: {line}")
        return line
    base = Path("checkpoints")
    if not base.is_dir():
        raise FileNotFoundError(
            "No checkpoints/ directory found; pass --checkpoint-dir or run -new first."
        )
    lines = [
        p for p in base.iterdir() if p.is_dir() and (p / "latest.json").exists()
    ]
    if not lines:
        raise FileNotFoundError(
            "No checkpoint line found under checkpoints/ (a subdir with latest.json). "
            "Run with -new once first, or pass --checkpoint-dir explicitly."
        )
    newest = max(lines, key=lambda p: (p / "latest.json").stat().st_mtime)
    logger.info("Auto-detected newest checkpoint line: %s", newest)
    return newest


def _best_version_dir(manager: CheckpointManager) -> Path:
    """Pick the parent version for -continue: lowest validation_loss, latest as fallback."""
    versions = manager.versions()
    if not versions:
        raise FileNotFoundError(f"No checkpoint versions under {manager.root}")
    measured = [
        v
        for v in versions
        if manager.version_metrics(v).get("validation_loss", 0.0) > 0.0
    ]
    best = (
        min(measured, key=lambda v: manager.version_metrics(v)["validation_loss"])
        if measured
        else max(versions)
    )
    logger.info("Parent for -continue: v%03d (best by validation_loss)", best)
    return manager.version_dir(best)


def _apply_data_defaults(config: TrainConfig) -> TrainConfig:
    """Resolve ``--data`` into a manifest path (and preset splits, if any).

    Runs before ``_apply_mode_defaults`` so ``-continue`` compares the parent's
    manifest against the *resolved* dataset and can still wire the parent data
    in as replay/regression reference.
    """
    if not config.data:
        return config

    spec = resolve_dataset(config.data)
    missing = spec.missing_sources()
    if missing:
        hint = spec.prepare_hint or "export the dataset first"
        raise FileNotFoundError(
            f"Dataset {spec.name!r} is incomplete; missing: {', '.join(missing)}. "
            f"Hint: {hint}"
        )

    config.dataset = str(spec.manifest)
    config.data_name = spec.name
    if spec.has_preset_splits:
        config.preset_splits = spec.read_splits(text_field=config.text_field)

    logger.info(
        "--data %s -> %s (manifest: %s)",
        config.data,
        spec.name,
        config.dataset,
    )
    return config


def _apply_mode_defaults(config: TrainConfig) -> TrainConfig:
    """Resolve the abbreviated mode flags into concrete targets.

    -new is a plain fresh run (no resolution needed). -continue resolves a
    parent checkpoint (best/latest) and auto-wires replay + regression eval
    from the parent's dataset manifest. A bare -resume resolves to the
    newest line's latest version.
    """
    if config.continue_mode:
        line = _detect_checkpoint_line(config.checkpoint_dir)
        if not config.checkpoint_dir:
            config.checkpoint_dir = str(line)
        manager = CheckpointManager(config.checkpoint_dir)
        parent = (
            _best_version_dir(manager)
            if config.continue_mode == "best"
            else manager.resolve("latest")
        )
        config.from_checkpoint = str(parent)
        config.continue_mode = None

        parent_dataset = manager.load(str(parent))["manifest"].get("dataset")
        if parent_dataset:
            p = Path(parent_dataset)
            if p.exists() and p.resolve() != Path(config.dataset).resolve():
                if not config.replay_manifest:
                    config.replay_manifest = str(p)
                    if config.replay_ratio == 0.0:
                        config.replay_ratio = 0.3
                if not config.eval_manifest:
                    config.eval_manifest = str(p)
                logger.info(
                    "-continue: parent data %s reused for replay (ratio %.2f) and "
                    "regression eval",
                    p,
                    config.replay_ratio,
                )
            elif not p.exists():
                logger.info(
                    "-continue: parent dataset %s not found; skipping replay/regression",
                    p,
                )
            else:
                logger.info(
                    "-continue: parent dataset is the current dataset; no replay needed"
                )
    elif config.resume == "_AUTO_":
        line = _detect_checkpoint_line(config.checkpoint_dir)
        if not config.checkpoint_dir:
            config.checkpoint_dir = str(line)
        config.resume = str(CheckpointManager(config.checkpoint_dir).resolve("latest"))

    return config


def main():
    args = build_arg_parser().parse_args()

    set_seed(args.seed)

    model_config = STTConfig(
        d_model=args.model_d_model,
        nhead=args.model_heads,
        encoder_layers=args.model_encoder_layers,
        decoder_layers=args.model_decoder_layers,
        head=args.model_head,
        backbone=_BACKBONE_ALIASES.get(args.model_backbone, args.model_backbone),
        backbone_unfreeze_layers=args.backbone_unfreeze_layers,
    )
    config = TrainConfig(
        batch_size=args.batch_size,
    length_group=args.length_group,
    max_batch_frames=args.max_batch_frames,
    num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        optimizer=args.optimizer,
        grad_clip=args.grad_clip,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        dataset=args.dataset,
        data=args.data,
        text_field=args.text_field,
        validation_frequency=args.validation_frequency,
        replay_ratio=args.replay_ratio,
        replay_manifest=args.replay_manifest,
        resume=args.resume,
        from_checkpoint=args.from_checkpoint,
        mixed_precision=not args.no_fp16,
        split_dir=args.split_dir,
        eval_manifest=args.eval_manifest,
        max_text_len=args.max_text_len,
        beam_size=args.beam_size,
        retain_every=args.retain_every,
        retain_protect=args.retain_protect,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_threshold=args.lr_threshold,
        lr_cooldown=args.lr_cooldown,
        new_run=args.new,
        continue_mode=getattr(args, "continue"),
        model=model_config,
    )

    config = _apply_data_defaults(config)
    config = _apply_mode_defaults(config)
    if config.checkpoint_dir is None:
        config.checkpoint_dir = "checkpoints"

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()