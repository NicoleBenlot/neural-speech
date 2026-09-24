"""Versioned checkpoint management."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
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
    validation_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
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
        self.best_path = self.root / "best.json"
        self.protect_path = self.root / "protect.json"

        for p in self.root.iterdir():
            m = _VERSION_RE.match(p.name)
            if m:
                pass  # keep existing versions

    def _read_protect(self) -> List[str]:
        """Persistent per-line protection list (``protect.json``), e.g. ["v030"].

        Entries are merged into every ``prune`` call regardless of interval
        rules or CLI flags, so a historical/reference version can be pinned
        against auto-pruning forever.
        """
        try:
            data = json.loads(self.protect_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except (ValueError, OSError):
            pass
        return []

    def _read_best(self) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(self.best_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def _write_best(self, entry: Dict[str, Any]) -> None:
        self.best_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")

    def update_best(
        self,
        state: TrainingState,
        dataset: str,
        dataset_fingerprint: str,
    ) -> None:
        """Record the version just saved as line-best when it qualifies.

                Minima are tracked per dataset regime (stored in ``best.json``):

          * ``val_loss`` — the canonical metric (computed every validation pass
            and fed to the LR scheduler). ``prune``'s on-disk ``keep_best``
            recomputation is a belt-and-suspenders copy of this.
          * ``cer`` — a separate CER-specific best, kept because CER and
            validation_loss don't always peak at the same epoch (e.g. the
            previous combined run: val_loss best ~epoch 20, CER best ~epoch 28).
                    * ``word_cer`` / ``sentence_cer`` and their validation losses —
                        optional subset-specific minima populated when the trainer has
                        word/sentence validation rows.

        A new version updates an entry on the SAME dataset (fingerprint match)
        only when it beats the stored value. When the dataset changed (fingerprint
        mismatch) both entries are replaced immediately: an older dataset's lower
        val_loss must never shadow a newer, harder dataset's best and let the
        interval pruner delete it — that cross-dataset comparison is what deleted
        the first combined word+sentence run's best model.

        Both tracked versions are protected by ``prune`` regardless of their
        epoch or retain interval, and persist across runs until a checkpoint on
        the same dataset beats them.
        """
        loss = float(state.validation_loss)
        if not loss > 0.0:
            return
        vdir = self.version_dir(state.version)
        if not vdir.exists():
            return
        ver = f"v{state.version:03d}"

        def entry(
            validation_loss: float, cer: float, wer: float, sample_type: Optional[str] = None
        ) -> Dict[str, Any]:
            return {
                "version": ver,
                "path": str(vdir),
                "epoch": state.epoch,
                "validation_loss": validation_loss,
                "cer": cer,
                "wer": wer,
                **({"sample_type": sample_type} if sample_type else {}),
            }

        candidates: Dict[str, tuple[float, str, Dict[str, Any]]] = {
            "val_loss": (loss, "validation_loss", entry(loss, float(state.cer), float(state.wer))),
            "cer": (float(state.cer), "cer", entry(loss, float(state.cer), float(state.wer))),
        }
        for sample_type, metrics in state.validation_by_type.items():
            if not metrics:
                continue
            sample_loss = float(metrics.get("loss", 0.0))
            sample_cer = float(metrics.get("cer", 0.0))
            sample_wer = float(metrics.get("wer", 0.0))
            if sample_loss > 0.0:
                candidates[f"{sample_type}_val_loss"] = (
                    sample_loss,
                    "validation_loss",
                    entry(sample_loss, sample_cer, sample_wer, sample_type),
                )
            if sample_cer >= 0.0:
                candidates[f"{sample_type}_cer"] = (
                    sample_cer,
                    "cer",
                    entry(sample_loss, sample_cer, sample_wer, sample_type),
                )

        stored = self._read_best()
        if stored is None or stored.get("dataset_fingerprint") != dataset_fingerprint:
            fresh = {name: candidate[2] for name, candidate in candidates.items()}
            fresh.update(
                {
                    "dataset": dataset,
                    "dataset_fingerprint": dataset_fingerprint,
                    "best": True,
                }
            )
            self._write_best(fresh)
            logger.info(
                "New best checkpoint (new dataset regime): %s val_loss=%.4f cer=%.4f",
                ver,
                loss,
                state.cer,
            )
            return

        changed = False
        for name, (value, field_name, candidate) in candidates.items():
            current = stored.get(name)
            current_value = float(current[field_name]) if current else float("inf")
            if value < current_value:
                stored[name] = candidate
                changed = True
        if changed:
            self._write_best(stored)
            logger.info(
                "New best checkpoint (same dataset): val_loss %s=%.4f | cer %s=%.4f",
                stored["val_loss"]["version"],
                stored["val_loss"]["validation_loss"],
                stored["cer"]["version"],
                stored["cer"]["cer"],
            )

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

    def versions(self) -> List[int]:
        """Sorted ordinal list of every version dir under this root."""
        return sorted(
            int(m.group(1))
            for p in self.root.iterdir()
            if (m := _VERSION_RE.match(p.name))
        )

    def version_metrics(self, version: int) -> Dict[str, Any]:
        """Best-effort summary of a version from its training_state.json."""
        info: Dict[str, Any] = {
            "version": version,
            "path": str(self.version_dir(version)),
        }
        state_file = self.version_dir(version) / "training_state.json"
        if state_file.exists():
            try:
                info.update(json.loads(state_file.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                pass
        return info

    def prune(
        self,
        retain_every: int = 0,
        keep_best: bool = True,
        protect: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> List[Path]:
        """Delete checkpoint versions outside the retention policy.

        Retention policy (applied going forward and safe to reuse retroactively):
          * the final (newest) version — always kept (``latest.json`` stays valid)
          * the best version by ``validation_loss`` (if any real measurement exists)
                    * every version whose ordinal is a multiple of ``retain_every``
                        (diagnostic trend visibility when positive; ``retain_every=0``
                        keeps only the final + best)
          * anything in ``protect`` (exact names like ``"v026"`` or paths) —
            never deleted; the line's persistent ``protect.json`` list is
            merged in automatically on every call, so a historical reference
            version can be pinned against auto-pruning forever (no CLI flag
            needed on future runs)
          * the recorded line-best (``best.json``, written by ``update_best`` at
            save time) — protected unconditionally, independent of its epoch,
            the retain interval, and older-dataset overshadowing. ``latest``
            (most recent epoch) and ``best`` (best validation performance) are
            separate pointers and are both kept.

        The final version is always retained, so ``--resume .../latest`` keeps
        working; resuming from an explicitly-pruned intermediate version does not.

        Returns the deleted version dirs. With ``dry_run=True`` nothing is
        deleted and the would-be deletions are returned.
        """
        vers = self.versions()
        if not vers:
            return []

        protect_nums = set()
        for entry in ([*(protect or []), *self._read_protect()]):
            entry = str(entry).strip()
            m = _VERSION_RE.match(entry)
            if m:
                protect_nums.add(int(m.group(1)))
            else:
                try:
                    protect_nums.add(int(entry))
                except ValueError:
                    logger.warning("Ignoring unknown --retain-protect value: %r", entry)

        keep: set = {vers[-1]}
        if keep_best:
            measured = [
                self.version_metrics(v)
                for v in vers
                if self.version_metrics(v).get("validation_loss", 0.0) > 0.0
            ]
            if measured:
                keep.add(min(measured, key=lambda info: info["validation_loss"])["version"])
        if retain_every > 0:
            keep.update(v for v in vers if v % retain_every == 0)
        best_entry = self._read_best()
        if best_entry:
            best_vers = []
            for v in best_entry.values():
                if isinstance(v, dict) and v.get("version"):
                    best_vers.append(str(v["version"]))
            if best_entry.get("version"):  # legacy single-entry format
                best_vers.append(str(best_entry["version"]))
            for bversion in best_vers:
                bm = _VERSION_RE.match(bversion)
                if bm and int(bm.group(1)) in vers:
                    keep.add(int(bm.group(1)))
                    logger.info(
                        "Retention: protecting recorded best %s (best.json)",
                        bversion,
                    )
        keep.update(protect_nums)

        to_delete = [self.version_dir(v) for v in vers if v not in keep]
        for path in to_delete:
            if dry_run:
                logger.info("[dry-run] retention would delete %s", path)
            else:
                logger.warning("Retention policy: deleting %s", path)
                shutil.rmtree(path, ignore_errors=True)
        return to_delete

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
        """Resolve a checkpoint alias ('v001', 'latest', 'checkpoints/latest',
        'best', 'checkpoints/mms/best', a path, or a line root) to a directory."""
        ckpt = str(checkpoint)
        if ckpt == "latest" or ckpt.endswith("latest") or ckpt.endswith(os.sep + "latest"):
            if not self.latest_path.exists():
                raise FileNotFoundError(f"No latest checkpoint: {self.latest_path}")
            data = json.loads(self.latest_path.read_text(encoding="utf-8"))
            return Path(data["path"])
        if ckpt == "best" or Path(ckpt).name == "best":
            if not self.best_path.exists():
                raise FileNotFoundError(f"No best checkpoint recorded: {self.best_path}")
            data = json.loads(self.best_path.read_text(encoding="utf-8"))
            block = data.get("val_loss") or data  # new dual entry or legacy single
            path = Path(block["path"])
            if not (path / "model.pt").exists():
                raise FileNotFoundError(
                    f"Recorded best checkpoint {path} is missing; it may have been "
                    "deleted before best.json tracking was introduced."
                )
            return path

        path = Path(checkpoint)
        if not path.is_absolute():
            candidate = self.root / path if (self.root / path).exists() else path
            path = candidate
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if (path / "latest.json").exists():
            return self.resolve(str(path / "latest"))
        if not (path / "model.pt").exists():
            raise FileNotFoundError(
                f"No checkpoint inside {path} (expected a version dir or a line with latest.json)"
            )
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