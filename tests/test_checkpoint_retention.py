"""Checkpoint retention policy: prune() keep-set and dry-run behaviour."""

import json
import logging
import shutil

import pytest

from src.training.checkpoint import CheckpointManager


def _make_versions(root, n, val_losses):
    """Create n dummy version dirs with fixed validation_losses (1-indexed)."""
    root.mkdir(parents=True, exist_ok=True)
    for version in range(1, n + 1):
        d = root / f"v{version:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.pt").write_bytes(b"dummy")
        vl = val_losses[version - 1]
        state = {
            "version": version,
            "epoch": version,
            "validation_loss": vl,
        }
        (d / "training_state.json").write_text(json.dumps(state), encoding="utf-8")
    (root / "latest.json").write_text(
        json.dumps({"version": f"v{n:03d}", "path": str(root / f"v{n:03d}")}),
        encoding="utf-8",
    )


@pytest.fixture
def manager(tmp_path):
    root = tmp_path / "line"
    # 8 versions: losses dip at v4 (best), v8 newest.
    _make_versions(root, 8, [9.0, 8.0, 7.0, 1.0, 6.0, 5.0, 4.0, 3.0])
    return CheckpointManager(str(root))


def test_versions_sorted(manager):
    assert manager.versions() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_prune_keeps_final_best_and_multiples(manager):
    deleted = manager.prune(retain_every=5, keep_best=True)
    kept = manager.versions()
    # best = v4 (val_loss 1.0), final = v8, milestones 5,10 -> 5 (within range)
    assert kept == [4, 5, 8]
    assert set(deleted) == {manager.version_dir(v) for v in [1, 2, 3, 6, 7]}
    assert not (manager.root / "v001").exists()


def test_prune_protect_overrides_deletion(manager):
    deleted = manager.prune(retain_every=5, keep_best=True, protect=["v001", "3"])
    kept = manager.versions()
    assert kept == [1, 3, 4, 5, 8]
    # v1 and v3 were not in the default keep-set, but are protected -> survive
    assert (manager.root / "v001").exists()
    assert (manager.root / "v003").exists()
    assert manager.version_dir(1) not in deleted
    assert manager.version_dir(3) not in deleted
    assert set(deleted) == {manager.version_dir(v) for v in [2, 6, 7]}


def test_prune_retain_every_zero_keeps_only_final_and_best(manager):
    deleted = manager.prune(retain_every=0, keep_best=True)
    assert manager.versions() == [4, 8]
    assert len(deleted) == 6


def test_prune_dry_run_deletes_nothing(manager):
    before = manager.versions()
    deleted = manager.prune(retain_every=5, keep_best=True, dry_run=True)
    assert len(deleted) > 0
    assert manager.versions() == before


def test_prune_no_valid_validation_loss(manager):
    # All zero validation_loss -> no "best", final + milestones still kept.
    _make_versions(manager.root, 8, [0.0] * 8)
    manager.prune(retain_every=5, keep_best=True)
    assert manager.versions() == [5, 8]


def test_prune_empty_line(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    manager = CheckpointManager(str(root))
    assert manager.prune(retain_every=10) == []


def test_prune_unknown_protect_warns(tmp_path, caplog):
    root = tmp_path / "line"
    _make_versions(root, 3, [1.0, 2.0, 3.0])
    manager = CheckpointManager(str(root))
    with caplog.at_level(logging.WARNING, logger="src.training.checkpoint"):
        manager.prune(retain_every=10, protect=["bogus"])
    assert "Ignoring unknown --retain-protect" in caplog.text


def test_prune_keeps_latest_for_resume(manager):
    manager.prune(retain_every=5, keep_best=True)
    # latest.json still resolves to the final (kept) version
    data = json.loads(manager.latest_path.read_text(encoding="utf-8"))
    assert data["version"] == "v008"
    assert (manager.root / "v008").exists()