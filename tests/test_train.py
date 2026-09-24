"""End-to-end training, resume, and incremental training tests."""

from pathlib import Path

import pytest
import torch

from src.training.train import (
    _BACKBONE_ALIASES,
    _apply_mode_defaults,
    _build_plateau,
    build_arg_parser,
    TrainConfig,
    Trainer,
    set_seed,
)
from src.training.checkpoint import CheckpointManager
from src.models.stt import STTConfig
from src.data.dataset import manifest_fingerprint, split_fingerprint


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
        "retain_every": -1,
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


def test_mode_flag_parsing():
    parser = build_arg_parser()

    a = parser.parse_args(["-new", "-d", "auto"])
    assert a.new is True
    assert a.resume is None
    assert getattr(a, "continue") is None

    c = parser.parse_args(["-continue", "-d", "auto", "-e", "30", "-b", "8"])
    assert getattr(c, "continue") == "best"

    c2 = parser.parse_args(["-continue", "latest"])
    assert getattr(c2, "continue") == "latest"

    r = parser.parse_args(["-resume"])
    assert r.resume == "_AUTO_"
    r2 = parser.parse_args(["-resume", "v001"])
    assert r2.resume == "v001"

    assert parser.parse_args(["-new"]).model_backbone == "fb"

    with pytest.raises(SystemExit):
        parser.parse_args(["-new", "-continue", "best"])
    with pytest.raises(SystemExit):
        parser.parse_args(["-resume", "v001", "--from-checkpoint", "v002"])


def test_apply_continue_defaults_wires_replay_and_regression(base_config, tmp_path):
    config = dict(base_config)
    config["epochs"] = 1
    set_seed(2)
    Trainer(TrainConfig(**config)).train()

    new_data = tmp_path / "other" / "manifest.csv"
    cfg = TrainConfig(
        **dict(
            config,
            continue_mode="best",
            checkpoint_dir=str(tmp_path / "checkpoints"),
            dataset=str(new_data),
        )
    )
    out = _apply_mode_defaults(cfg)

    assert Path(out.from_checkpoint).name == "v001"
    assert out.continue_mode is None
    assert out.replay_manifest == config["dataset"]
    assert out.eval_manifest == config["dataset"]
    assert out.replay_ratio == 0.3


def test_changed_manifest_resplits_on_continue(base_config, tmp_path):
    config = dict(base_config)
    config["epochs"] = 1
    set_seed(1)
    Trainer(TrainConfig(**config)).train()

    split_file = Path(config["split_dir"]) / "split_manifest.json"
    assert split_file.exists()

    # A new manifest sharing the same stem (new data, more samples reusing the
    # same audio assets) -> different fingerprint -> the old split must be
    # discarded and rebuilt from the current manifest.
    old_manifest = Path(config["dataset"])
    assets = old_manifest.parent / "assets"
    other = tmp_path / "other"
    other.mkdir(exist_ok=True)
    words = ["adlaw", "adto", "ako", "akong", "amo", "amo", "akong"]
    rows = []
    for i, word in enumerate(words):
        wid = 300 + i
        rows.append(f"{wid},{assets / f'{100 + i % 5}.wav'},{word},ad")
    new_manifest = other / "manifest.csv"
    new_manifest.write_text(
        "id,audio,text,section\n" + "\n".join(rows), encoding="utf-8"
    )
    assert manifest_fingerprint(new_manifest) != manifest_fingerprint(old_manifest)

    resume_config = dict(config, dataset=str(new_manifest), resume="v001")
    trainer = Trainer(TrainConfig(**resume_config))
    train_ds, valid_ds, test_ds, _ = trainer._load_datasets()

    # 7 rows -> 5 train (80%); the stale 4-row train split must NOT be reused,
    # and the split file must be rewritten against the new manifest.
    assert len(train_ds) == 5
    assert split_fingerprint(split_file) == manifest_fingerprint(new_manifest)


def test_lr_patience_flag_changes_plateau_trigger():
    """ReduceLROnPlateau with patience=3 must fire on a sequence where the
    default patience=5 does not -- proves the configurable flag actually
    changes behavior rather than being accepted and ignored."""
    seq = [1.0, 0.8, 0.9, 0.95, 1.05, 1.1]  # best=0.8, then 4 consecutive bad

    def drops(patience):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        sch = _build_plateau(opt, lr_patience=patience, lr_factor=0.1)
        events = []
        for v in seq:
            lr_before = opt.param_groups[0]["lr"]
            sch.step(v)
            lr_after = opt.param_groups[0]["lr"]
            if lr_after != lr_before:
                events.append((v, lr_after))
        return events

    p3 = drops(3)
    p5 = drops(5)
    assert p3                  # 4 consecutive bad epochs > patience 3 -> fires
    assert not p5              # 4 bad epochs <= patience 5 -> default never fires
    assert len(p3) == 1
    assert abs(p3[0][1] - 1e-4) < 1e-12  # factor 0.1: 1e-3 -> 1e-4


def test_lr_flags_wire_into_config_and_defaults():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--lr-patience", "3",
            "--lr-factor", "0.2",
            "--lr-threshold", "0.01",
            "--lr-cooldown", "2",
        ]
    )
    assert args.lr_patience == 3
    assert args.lr_factor == 0.2
    assert args.lr_threshold == 0.01
    assert args.lr_cooldown == 2

    # defaults match the historical hardcoded scheduler exactly
    defaults = parser.parse_args([])
    assert defaults.lr_patience == 5
    assert defaults.lr_factor == 0.1
    assert defaults.lr_threshold == 1e-4
    assert defaults.lr_cooldown == 0
    assert TrainConfig().lr_patience == 5
    assert defaults.retain_every == 0
    assert TrainConfig().retain_every == 0