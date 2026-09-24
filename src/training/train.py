"""Training entry point with resume, incremental, and replay support."""

from __future__ import annotations

import argparse
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
    SpeechDataset,
    collate_speech,
    load_split,
    read_manifest,
    save_split,
    split_dataset,
    _rows_to_dataset,
)
from src.data.prepare import write_manifest
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 1e-3
    epochs: int = 30
    optimizer: str = "AdamW"
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    dataset: str = "data/processed/manifest.csv"
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
    retain_every: int = 10
    retain_protect: List[str] = field(default_factory=list)
    model: STTConfig = field(default_factory=STTConfig)

    sample_rate: int = 16000


def _is_true_env_flag(name: str) -> bool:
    return os.environ.get(name, "1").lower() not in {"0", "false", "no", "off"}


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

    def _validate_training_args(self):
        if self.config.resume and self.config.from_checkpoint:
            raise ValueError("Use either --resume or --from-checkpoint, not both.")

    def _prepare_epoch_from_checkpoint(self):
        """For incremental training, build tokenizer/vocab from the parent vocab plus new data."""
        self.manager.load(self.config.from_checkpoint)
        data = self.manager.load(self.config.from_checkpoint)
        parent_tokenizer = data["tokenizer"]
        self.tokenizer = parent_tokenizer
        parent_config = data["config"]
        self.config.model = parent_config
        self.state.parent_checkpoint = self.config.from_checkpoint

        parent_state = data["state"]
        self.state.dataset_version = parent_state.dataset_version

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

        if new_run:
            rows = read_manifest(manifest)
            self.tokenizer = CharTokenizer()
            self.tokenizer.build_from_texts([r.text for r in rows])
            if self.config.model.head == "ctc":
                self.tokenizer.add_blank()
            train_rows, valid_rows, test_rows = split_dataset(
                rows, seed=self.config.seed
            )
            save_split(split_file, train_rows, valid_rows, test_rows, self.config.seed)
            train_ds = _rows_to_dataset([r.__dict__ for r in train_rows], self.tokenizer, self.config.sample_rate)
            valid_ds = _rows_to_dataset([r.__dict__ for r in valid_rows], self.tokenizer, self.config.sample_rate)
            test_ds = _rows_to_dataset([r.__dict__ for r in test_rows], self.tokenizer, self.config.sample_rate)
        else:
            if self.tokenizer is None:
                ckpt = self.config.resume or self.config.from_checkpoint
                self.tokenizer = self.manager.load(ckpt)["tokenizer"]

            if split_file.exists():
                train_ds, valid_ds, test_ds = load_split(
                    split_file, self.tokenizer, self.config.sample_rate
                )
            else:
                rows = read_manifest(manifest)
                train_rows, valid_rows, test_rows = split_dataset(rows, seed=self.config.seed)
                train_ds = _rows_to_dataset([r.__dict__ for r in train_rows], self.tokenizer, self.config.sample_rate)
                valid_ds = _rows_to_dataset([r.__dict__ for r in valid_rows], self.tokenizer, self.config.sample_rate)
                test_ds = _rows_to_dataset([r.__dict__ for r in test_rows], self.tokenizer, self.config.sample_rate)

        replay_ds = None
        if self.config.replay_manifest and self.config.replay_ratio > 0:
            replay_rows = read_manifest(Path(self.config.replay_manifest))
            replay_ds = _rows_to_dataset(
                [r.__dict__ for r in replay_rows],
                self.tokenizer,
                self.config.sample_rate,
            )

        return train_ds, valid_ds, test_ds, replay_ds

    def _build_model_and_optimizer(self):
        if self.model is None:
            if self.config.model.head == "ctc" and self.config.model.blank_token_id is None:
                blank_id = self.tokenizer.blank_id
                if blank_id is None:
                    raise RuntimeError("CTC training requires a blank token in the tokenizer")
                self.config.model.blank_token_id = blank_id
            self.model = build_stt_model(self.config.model, self.tokenizer.vocab_size())
            self.model.to(self.device)

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
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5
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

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5
        )
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

        self._build_model_and_optimizer()

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_speech,
            num_workers=0,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_speech,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_speech,
            num_workers=0,
        )

        if replay_ds is not None:
            logger.info("Replay samples: %d", len(replay_ds))
            replay_loader = DataLoader(
                replay_ds,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=collate_speech,
                num_workers=0,
            )
        else:
            replay_loader = None

        self.use_amp = self.config.mixed_precision and self.device.type == "cuda"
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.use_amp)
            if self.use_amp
            else None
        )
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
                val_loss, val_cer, val_wer = self._evaluate(valid_loader)
                self.scheduler.step(val_loss)

            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"epoch={epoch + 1}/{self.config.epochs} "
                f"step={self.global_step} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"CER={val_cer:.4f} WER={val_wer:.4f} "
                f"lr={lr:.2e}"
            )

            self.state.train_loss = train_loss
            self.state.validation_loss = val_loss
            self.state.cer = val_cer
            self.state.wer = val_wer

            if should_val:
                self._save_checkpoint(
                    manifest={
                        "dataset": self.config.dataset,
                        "replay_manifest": self.config.replay_manifest,
                        "replay_ratio": self.config.replay_ratio,
                    }
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
        return avg_loss, metrics.cer, metrics.wer

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
        self, loader: DataLoader
    ) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds: List[str] = []
        all_refs: List[str] = []

        with torch.no_grad():
            for batch in loader:
                audio = batch.audio.to(self.device)
                audio_lengths = batch.audio_lengths.to(self.device)
                tokens = batch.tokens.to(self.device)
                token_lengths = batch.token_lengths.to(self.device)

                with torch.autocast(
                    device_type=self.device.type, enabled=self.use_amp
                ):
                    loss = self._batch_loss(
                        audio, audio_lengths, tokens, token_lengths
                    )
                total_loss += loss.item()
                n_batches += 1

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

        metrics = compute_metrics(all_refs, all_preds)
        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss, metrics.cer, metrics.wer

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
        if self.config.retain_every == 0 and not self.config.retain_protect:
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
    parser.add_argument("--dataset", default="data/processed/manifest.csv")
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("--optimizer", default="AdamW", choices=["AdamW", "SGD"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("-d", "--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-c", "--checkpoint-dir", default="checkpoints")
    parser.add_argument("--validation-frequency", type=int, default=1)
    parser.add_argument("--replay-ratio", type=float, default=0.0)
    parser.add_argument("--replay-manifest", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--from-checkpoint", default=None)
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
        default="none",
        help="Pretrained acoustic backbone. 'none' = from-scratch encoder. "
        "E.g. 'facebook/mms-300m' (hidden 1024, 24 layers, ~315M params, "
        "self-supervised on 1,406 languages incl. ceb). Aliases: 'fb'/'mms' "
        "= facebook/mms-300m.",
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
        default=10,
        help="Checkpoint retention: keep every Nth version plus the best-val-loss "
        "and final versions, deleting the rest after each validated save. "
        "0 = keep every version (old behaviour). Default 10.",
    )
    parser.add_argument(
        "--retain-protect",
        action="append",
        default=[],
        metavar="VERSION",
        help="Never delete this version dir (e.g. --retain-protect v026). "
        "Repeatable. Independent of the milestone/best/final keep-set.",
    )
    return parser


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
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        optimizer=args.optimizer,
        grad_clip=args.grad_clip,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        dataset=args.dataset,
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
        model=model_config,
    )

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()