"""End-to-end training, resume, and incremental training tests."""

from pathlib import Path

import pytest
import torch

from src.training.train import _BACKBONE_ALIASES, build_arg_parser, TrainConfig, Trainer, set_seed
from src.training.checkpoint import CheckpointManager
from src.models.stt import STTConfig


def _tiny_model_config():
    return STTConfig(
        d_model=16,
        nhead=2,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=32,
        feat_channels=4,
        feat_layers=2,
    )


@pytest.fixture
def base_config(tmp_path: Path, tiny_manifest: Path):
    return {
        "dataset": str(tiny_manifest),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "split_dir": str(tmp_path / "processed"),
        "epochs": 1,
        "validation_frequency": 1,
        "batch_size": 2,
        "device": "cpu",
        "mixed_precision": False,
        "eval_manifest": None,
        # These tests exercise version immutability/resume, not retention;
        # opt out of auto-pruning so every version stays on disk.
        "retain_every": 0,
        "model": _tiny_model_config(),
    }


def train_once(config: dict):
    trainer = Trainer(TrainConfig(**config))
    return trainer


def test_train_creates_version(base_config, tmp_path):
    set_seed(0)
    trainer = train_once(base_config)
    trainer.train()

    ckpt_root = tmp_path / "checkpoints"
    assert (ckpt_root / "v001").exists()
    assert (ckpt_root / "latest.json").exists()

    # second training run from scratch -> v002, never overwrites v001
    set_seed(0)
    trainer2 = train_once(base_config)
    trainer2.train()
    assert (ckpt_root / "v002").exists()
    assert (ckpt_root / "v001" / "model.pt").exists()


def test_resume_continues_epoch(base_config, tmp_path):
    config = dict(base_config)
    config["epochs"] = 1
    set_seed(1)
    Trainer(TrainConfig(**config)).train()

    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    data = manager.load("v001")
    assert data["state"].epoch == 1

    # resume for 1 more epoch -> should start at epoch 1 and produce v002
    resume_config = dict(base_config)
    resume_config["epochs"] = 2
    resume_config["resume"] = "v001"
    set_seed(1)
    Trainer(TrainConfig(**resume_config)).train()

    assert (tmp_path / "checkpoints" / "v002").exists()
    data2 = manager.load("v002")
    assert data2["state"].epoch == 2


def test_incremental_from_checkpoint(base_config, tmp_path):
    config = dict(base_config)
    config["epochs"] = 1
    set_seed(2)
    Trainer(TrainConfig(**config)).train()

    incr = dict(base_config)
    incr["epochs"] = 1
    incr["from_checkpoint"] = "v001"
    incr["replay_manifest"] = base_config["dataset"]
    incr["replay_ratio"] = 0.3
    set_seed(2)
    Trainer(TrainConfig(**incr)).train()

    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    assert (tmp_path / "checkpoints" / "v002").exists()
    data = manager.load("v002")
    assert data["state"].parent_checkpoint == "v001"
    assert data["manifest"]["replay_ratio"] == 0.3


def test_short_flag_aliases():
    args = build_arg_parser().parse_args(
        ["-m", "fb", "-c", "checkpoints/mms-sent", "-e", "30", "-b", "10", "-mtl", "123", "-d", "auto"]
    )
    assert args.model_backbone == "fb"
    assert _BACKBONE_ALIASES[args.model_backbone] == "facebook/mms-300m"
    assert args.checkpoint_dir == "checkpoints/mms-sent"
    assert args.epochs == 30
    assert args.batch_size == 10
    assert args.max_text_len == 123
    assert args.device == "auto"


def test_backbone_aliases_all_map_to_mms():
    assert _BACKBONE_ALIASES["fb"] == "facebook/mms-300m"
    assert _BACKBONE_ALIASES["mms"] == "facebook/mms-300m"
    assert _BACKBONE_ALIASES["mms-300m"] == "facebook/mms-300m"
    assert "none" not in _BACKBONE_ALIASES