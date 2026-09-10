"""Versioned checkpoint management."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.models.stt import STTConfig, STTModel
from src.tokens.tokenizer import CharTokenizer

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"^v(\d{3})$")


@dataclass
class TrainingState:
    version: int = 1
    epoch: int = 0
    global_step: int = 0
    dataset_version: str = "v001"
    parent_checkpoint: Optional[str] = None
    train_loss: float = 0.0
    validation_loss: float = 0.0
    wer: float = 0.0
    cer: float = 0.0
    regression: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingState":
        known = {k: v for k, v in cls.__dataclass_fields__.items()}
        return cls(**{k: v for k, v in data.items() if k in known})


class CheckpointManager:
    """Manages immutable versioned checkpoints under checkpoints/vNNN/."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root / "latest.json"

        for p in self.root.iterdir():
            m = _VERSION_RE.match(p.name)
            if m:
                pass  # keep existing versions

    def _next_version(self) -> int:
        existing = [m.group(1) for p in self.root.iterdir() if (m := _VERSION_RE.match(p.name))]
        if not existing:
            return 1
        return max(int(v) for v in existing) + 1

    def version_dir(self, version: int) -> Path:
        return self.root / f"v{version:03d}"

    def save(
        self,
        model: STTModel,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        tokenizer: CharTokenizer,
        config: STTConfig,
        state: TrainingState,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a checkpoint as a new immutable version."""
        version = self._next_version()

        target = self.version_dir(version)
        if target.exists():
            raise RuntimeError(f"Checkpoint directory already exists: {target}. Never overwrite versions.")

        tmp = Path(tempfile.mkdtemp(prefix=f"v{version:03d}-", dir=str(self.root)))
        try:
            torch.save(model.state_dict(), tmp / "model.pt")
            if optimizer is not None:
                torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
            if scheduler is not None:
                torch.save(scheduler.state_dict(), tmp / "scheduler.pt")
            tokenizer.save(tmp / "vocabulary.json")
            (tmp / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
            state.version = version
            (tmp / "training_state.json").write_text(
                json.dumps(state.to_dict(), indent=2), encoding="utf-8"
            )
            manifest = extra if extra else {}
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            tmp.replace(target)
            logger.info("Checkpoint saved: %s", target)
        finally:
            pass

        self._update_latest(version)
        return target

    def _update_latest(self, version: int):
        data = {
            "version": f"v{version:03d}",
            "path": str(self.version_dir(version)),
            "latest": True,
        }
        self.latest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def update_latest_state(self, state: TrainingState):
        """Rewrite the training_state.json of the current latest version.

        Used to persist final evaluation metrics into the version that the
        active run just produced. Model weights are not touched, so an older
        trained version is never overwritten.
        """
        if not self.latest_path.exists():
            raise FileNotFoundError(f"No latest checkpoint: {self.latest_path}")
        data = json.loads(self.latest_path.read_text(encoding="utf-8"))
        version_dir = Path(data["path"])
        state.version = int(data["version"][1:])
        (version_dir / "training_state.json").write_text(
            json.dumps(state.to_dict(), indent=2), encoding="utf-8"
        )
        logger.info("Updated latest checkpoint state: %s", version_dir)

    def resolve(self, checkpoint: str) -> Path:
        """Resolve a checkpoint alias ('v001', 'latest', 'checkpoints/latest', or a path) to a directory."""
        ckpt = str(checkpoint)
        if ckpt == "latest" or ckpt.endswith("latest") or ckpt.endswith(os.sep + "latest"):
            if not self.latest_path.exists():
                raise FileNotFoundError(f"No latest checkpoint: {self.latest_path}")
            data = json.loads(self.latest_path.read_text(encoding="utf-8"))
            return Path(data["path"])

        path = Path(checkpoint)
        if not path.is_absolute():
            candidate = self.root / path if (self.root / path).exists() else path
            path = candidate
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return path

    def load(self, checkpoint: str) -> Dict[str, Any]:
        """Load a checkpoint directory into a dict of states.

        Returns:
            Dict with 'model_sd', 'optimizer_sd', 'scheduler_sd', 'tokenizer',
            'config', 'state', 'manifest'.
        """
        path = self.resolve(checkpoint)
        result: Dict[str, Any] = {}

        result["model_sd"] = torch.load(path / "model.pt", map_location="cpu")
        opt_path = path / "optimizer.pt"
        result["optimizer_sd"] = (
            torch.load(opt_path, map_location="cpu", weights_only=False)
            if opt_path.exists()
            else None
        )
        sched_path = path / "scheduler.pt"
        result["scheduler_sd"] = (
            torch.load(sched_path, map_location="cpu", weights_only=False)
            if sched_path.exists()
            else None
        )
        result["tokenizer"] = CharTokenizer.load(path / "vocabulary.json")
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        result["config"] = STTConfig.from_dict(config)
        state_file = path / "training_state.json"
        result["state"] = TrainingState.from_dict(json.loads(state_file.read_text(encoding="utf-8"))) if state_file.exists() else TrainingState()
        manifest_file = path / "manifest.json"
        result["manifest"] = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
        result["path"] = path
        return result


def build_and_load(
    manager: CheckpointManager,
    checkpoint: str,
    device: torch.device,
) -> tuple[STTModel, CharTokenizer, STTConfig, TrainingState, Dict]:
    """Build a model from a checkpoint and load weights."""
    data = manager.load(checkpoint)
    model = STTModel(data["config"], vocab_size=data["tokenizer"].vocab_size())
    model.load_state_dict(data["model_sd"])
    model.to(device)
    model.eval()
    return model, data["tokenizer"], data["config"], data["state"], data["manifest"]