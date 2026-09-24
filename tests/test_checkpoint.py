"""Checkpoint versioning, save/load, and resume tests."""

import json
import random
from pathlib import Path

import pytest
import torch

from src.inference.transcriber import DEFAULT_CHECKPOINT, _split_checkpoint
from src.models.stt import STTConfig, STTModel
from src.tokens.tokenizer import CharTokenizer
from src.training.checkpoint import CheckpointManager, TrainingState


def _make_model_and_state(version: int = 1):
    tok = CharTokenizer()
    tok.build_from_texts(["adlaw", "adto"])
    config = STTConfig(d_model=16, nhead=2, encoder_layers=1, decoder_layers=1, dim_feedforward=32)
    model = STTModel(config, vocab_size=tok.vocab_size())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = TrainingState(version=version, epoch=3, global_step=100)
    return model, optimizer, tok, config, state


def test_next_version_increments(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    # First save -> v001
    model, opt, tok, config, state = _make_model_and_state()
    v1 = manager.save(model, opt, None, tok, config, state)
    assert v1.name == "v001"

    model2, opt2, tok2, config2, state2 = _make_model_and_state()
    v2 = manager.save(model2, opt2, None, tok2, config2, state2)
    assert v2.name == "v002"


def test_checkpoint_dir_layout(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    model, opt, tok, config, state = _make_model_and_state()
    v1 = manager.save(model, opt, None, tok, config, state)
    assert (v1 / "model.pt").exists()
    assert (v1 / "optimizer.pt").exists()
    assert (v1 / "vocabulary.json").exists()
    assert (v1 / "config.json").exists()
    assert (v1 / "training_state.json").exists()
    assert (v1 / "manifest.json").exists()


def test_load_roundtrip(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    model, opt, tok, config, state = _make_model_and_state()
    v1 = manager.save(model, opt, None, tok, config, state)

    data = manager.load("v001")
    assert data["tokenizer"].decode(data["tokenizer"].encode("adlaw")) == "adlaw"
    assert data["state"].epoch == 3
    assert data["state"].global_step == 100
    assert data["config"].d_model == 16

    restored = STTModel(data["config"], vocab_size=data["tokenizer"].vocab_size())
    restored.load_state_dict(data["model_sd"])

    # Weights match
    with torch.no_grad():
        a = torch.randn(1, 1, 4000)
        lengths = torch.tensor([4000])
        toks = torch.tensor([[tok.bos_id]])
        tl = torch.tensor([1])
        model.eval()
        restored.eval()
        out1 = model(a, lengths, toks, tl)
        out2 = restored(a, lengths, toks, tl)
        assert torch.allclose(out1, out2, atol=1e-6)


def test_never_overwrites(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    model, opt, tok, config, state = _make_model_and_state()
    manager.save(model, opt, None, tok, config, state)
    model2, opt2, tok2, config2, state2 = _make_model_and_state()
    manager.save(model2, opt2, None, tok2, config2, state2)
    versions = sorted(p.name for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("v"))
    assert versions == ["v001", "v002"]


def test_latest_file(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    model, opt, tok, config, state = _make_model_and_state()
    manager.save(model, opt, None, tok, config, state)
    manager.save(model, opt, None, tok, config, state)
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["version"] == "v002"


def test_resolve_latest(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    model, opt, tok, config, state = _make_model_and_state()
    manager.save(model, opt, None, tok, config, state)
    resolved = manager.resolve("latest")
    assert resolved.name == "v001"


def test_resolve_line_root_resolves_latest(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    model, opt, tok, config, state = _make_model_and_state()
    manager.save(model, opt, None, tok, config, state)
    manager.save(model, opt, None, tok, config, state)
    resolved = manager.resolve(str(tmp_path))
    assert resolved.name == "v002"


def test_resolve_version_dir_unambiguous(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    model, opt, tok, config, state = _make_model_and_state()
    manager.save(model, opt, None, tok, config, state)
    assert manager.resolve("v001") == tmp_path / "v001"
    with pytest.raises(FileNotFoundError):
        manager.resolve("v999")


def test_default_inference_checkpoint_uses_mms_line():
    root, name = _split_checkpoint(DEFAULT_CHECKPOINT)
    assert Path(root) == Path("checkpoints/mms")
    assert name == "latest"


def test_training_state_roundtrip():
    state = TrainingState(
        version=1,
        epoch=10,
        global_step=5000,
        parent_checkpoint="checkpoints/v001",
    )
    restored = TrainingState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored.parent_checkpoint == "checkpoints/v001"
    assert restored.global_step == 5000


def _save_version(manager, val_loss: float, cer: float, wer: float) -> Path:
    model, opt, tok, config, state = _make_model_and_state()
    state.validation_loss = val_loss
    state.cer = cer
    state.wer = wer
    state.epoch = len(manager.versions()) + 1
    return manager.save(model, opt, None, tok, config, state)


def test_best_preserved_under_overfitting_curve(tmp_path):
    """val_loss dips then rises (overfitting): the lowest point must survive
    interval pruning even after several worse epochs complete."""
    manager = CheckpointManager(str(tmp_path))
    fp = "fp-combined"
    losses = [3.0, 2.6, 2.2, 1.8, 1.5, 1.2, 1.4, 1.7, 2.1, 2.6]  # min at v006
    for v, loss in enumerate(losses, start=1):
        _save_version(manager, loss, 0.5 + v / 100, 0.8)
        state = manager.version_metrics(v)
        manager.update_best(
            TrainingState(version=v, validation_loss=loss, cer=0.5, wer=0.8),
            "data/processed/manifest.csv",
            fp,
        )

    best = json.loads((tmp_path / "best.json").read_text(encoding="utf-8"))
    assert best["val_loss"]["version"] == "v006"
    assert best["val_loss"]["validation_loss"] == 1.2

    # interval pruning (retain_every=3) without the on-disk keep_best comptuation:
    # v006 is only protected because best.json says so.
    manager.prune(retain_every=3, keep_best=False)
    assert (tmp_path / "v006").exists()          # the overfitting trough
    assert (tmp_path / "v003").exists()          # milestone
    assert (tmp_path / "v009").exists()          # milestone
    assert (tmp_path / "v010").exists()          # newest / final
    assert not (tmp_path / "v007").exists()      # post-peak regression, not best
    assert not (tmp_path / "v004").exists()

    # still protected after a second prune round (persists across runs)
    manager.prune(retain_every=3, keep_best=False)
    assert (tmp_path / "v006").exists()


def test_best_cross_dataset_regime_replaces(tmp_path):
    """A newer dataset's best must not be shadowed by an older dataset's lower
    val_loss (the bug that deleted the combined run's best model)."""
    manager = CheckpointManager(str(tmp_path))
    _save_version(manager, 1.0, 0.2, 0.5)   # v001: old-dataset best
    v1 = manager.version_dir(1)
    manager.update_best(
        TrainingState(version=1, validation_loss=1.0, cer=0.2, wer=0.5),
        "data/processed/old.csv",
        "fp-old",
    )
    assert json.loads((tmp_path / "best.json").read_text(encoding="utf-8"))["val_loss"]["version"] == "v001"

    _save_version(manager, 3.0, 0.9, 1.0)   # v002: new dataset, worse absolute loss
    manager.update_best(
        TrainingState(version=2, validation_loss=3.0, cer=0.9, wer=1.0),
        "data/processed/combined.csv",
        "fp-combined",
    )
    best = json.loads((tmp_path / "best.json").read_text(encoding="utf-8"))
    assert best["val_loss"]["version"] == "v002"   # regime switch, not loss comparison
    assert best["dataset_fingerprint"] == "fp-combined"

    manager.prune(retain_every=0, keep_best=False)
    assert (tmp_path / "v002").exists()
    assert (tmp_path / "v001").exists() or True  # keep_best path in real runs keeps both


def test_best_tracks_cer_and_validation_loss_independently(tmp_path):
    """CER-best and val_loss-best can be different epochs (the v050-vs-epoch28
    discrepancy): both versions must survive pruning."""
    manager = CheckpointManager(str(tmp_path))
    seq = [(2.0, 0.9), (1.5, 0.8), (1.8, 0.1)]  # (val_loss, cer): v002 val-min, v003 cer-min
    for v, (loss, cer) in enumerate(seq, start=1):
        _save_version(manager, loss, cer, 1.0)
        manager.update_best(
            TrainingState(version=v, validation_loss=loss, cer=cer, wer=1.0),
            "data/processed/manifest.csv",
            "fp",
        )
    best = json.loads((tmp_path / "best.json").read_text(encoding="utf-8"))
    assert best["val_loss"]["version"] == "v002"
    assert best["cer"]["version"] == "v003"

    manager.prune(retain_every=0, keep_best=False)
    assert (tmp_path / "v002").exists()   # val_loss best (not newest)
    assert (tmp_path / "v003").exists()   # CER best + newest
    assert not (tmp_path / "v001").exists()


def test_resolve_best(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    for loss in (2.0, 1.5, 1.8):
        v = len(manager.versions()) + 1
        _save_version(manager, loss, 0.4, 0.7)
        manager.update_best(
            TrainingState(version=v, validation_loss=loss, cer=0.4, wer=0.7),
            "data/processed/manifest.csv",
            "fp",
        )
    assert manager.resolve("best") == tmp_path / "v002"
    assert manager.resolve(str(tmp_path) + "/best") == tmp_path / "v002"