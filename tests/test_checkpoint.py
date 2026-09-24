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