"""Named-dataset registry, preset-split honoring, and ``--data`` wiring."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

import pytest

from src.data.dataset import manifest_fingerprint, read_manifest, split_fingerprint
from src.data.registry import (
    FLEURS_SPLIT_DIRS,
    DatasetSpec,
    read_split_tsv,
    resolve_dataset,
)
from src.models.stt import STTConfig
from src.training.train import TrainConfig, Trainer, _apply_data_defaults, build_arg_parser
from tests.conftest import _write_wav

HEADER = "filename\ttranscript\traw_transcript\tspeaker_id\tgender"

# (filename, transcript, raw_transcript, speaker_id, gender), keyed by pipeline
# split (the export directory is FLEURS_SPLIT_DIRS[split]).
SPLIT_ROWS = {
    "train": [
        ("00000.wav", "adlaw adto ako", "Adlaw. Adto ako.", "0", "f"),
        ("00001.wav", 'adlaw  "adto"  ako', 'Adlaw "adto" ako.', "0", "f"),
        # FLEURS writes text unescaped: a trailing tab shifts the tail columns for
        # the csv module but must not truncate the text here.
        ("00002.wav", "akong amtan\t", "Akong amtan.", "", "1"),
    ],
    "valid": [("00000.wav", "baligato ug bisan", "Baligato ug bisan.", "2", "m")],
    "test": [
        ("00000.wav", "kaayo kani adlaw", "Kaayo kani adlaw.", "3", "f"),
        ("00001.wav", "duha ka tawo", "Duha ka tawo.", "3", "m"),
    ],
}


def write_export(root: Path, with_audio: bool = True) -> Path:
    """Write a FLEURS-shaped export (train/validation/test + manifest.tsv each)."""
    for split, dirname in FLEURS_SPLIT_DIRS.items():
        split_dir = root / dirname
        split_dir.mkdir(parents=True, exist_ok=True)
        lines = [HEADER]
        for row in SPLIT_ROWS[split]:
            lines.append("\t".join(row))
            if with_audio:
                _write_wav(split_dir / row[0], duration=0.1)
        (split_dir / "manifest.tsv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return root


def spec_for(root: Path, processed: Path) -> DatasetSpec:
    return DatasetSpec(
        name=root.name,
        description="test export",
        manifest=processed / f"manifest_{root.name}.csv",
        split_dirs={s: root / d for s, d in FLEURS_SPLIT_DIRS.items()},
    )


def tiny_model_config() -> STTConfig:
    return STTConfig(
        d_model=16,
        nhead=2,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=32,
        feat_channels=4,
        feat_layers=2,
    )


def test_read_split_tsv_handles_unescaped_quotes_and_tabs(tmp_path):
    root = write_export(tmp_path / "fleurs_test", with_audio=False)
    split_dir = root / "train"

    rows = read_split_tsv(split_dir / "manifest.tsv", split_dir, "train", id_offset=7)

    assert [r.id for r in rows] == [7, 8, 9]
    assert all(r.section == "train" for r in rows)
    # internal whitespace collapsed, stray quote kept, trailing tab dropped
    assert rows[1].text == 'adlaw "adto" ako'
    assert rows[2].text == "akong amtan"
    assert rows[0].audio_path == os.path.join(str(split_dir), "00000.wav")


def test_read_split_tsv_ids_continue_across_splits(tmp_path):
    root = write_export(tmp_path / "fleurs_test", with_audio=False)
    train = read_split_tsv(root / "train" / "manifest.tsv", root / "train", "train")
    valid = read_split_tsv(
        root / "validation" / "manifest.tsv", root / "validation", "valid", id_offset=3
    )
    assert [r.id for r in valid] == [3]
    assert not ({r.id for r in train} & {r.id for r in valid})


def test_read_split_tsv_rejects_bad_text_field(tmp_path):
    root = write_export(tmp_path / "fleurs_test", with_audio=False)
    with pytest.raises(ValueError, match="text_field"):
        read_split_tsv(
            root / "train" / "manifest.tsv", root / "train", "train", text_field="nope"
        )


def test_read_split_tsv_requires_columns(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    (split_dir / "manifest.tsv").write_text("id\ttext\n1\thi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="filename"):
        read_split_tsv(split_dir / "manifest.tsv", split_dir, "train")


def test_read_split_tsv_skips_rows_without_text(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    (split_dir / "manifest.tsv").write_text(
        HEADER + "\n00000.wav\thi\tHi\t0\tf\n00001.wav\t \t \t0\tf\n", encoding="utf-8"
    )
    rows = read_split_tsv(split_dir / "manifest.tsv", split_dir, "train")
    assert len(rows) == 1
    assert rows[0].id == 0  # ids stay contiguous after the skip


def test_combined_manifest_marks_sections_and_unique_ids(tmp_path):
    root = write_export(tmp_path / "fleurs_test", with_audio=False)
    spec = spec_for(root, tmp_path / "processed")
    splits = spec.read_splits()

    assert {k: len(v) for k, v in splits.items()} == {
        "train": 3,
        "valid": 1,
        "test": 2,
    }
    rows = read_manifest(spec.manifest)
    assert len(rows) == 6
    assert len({r.id for r in rows}) == 6
    assert [r.section for r in rows] == [
        "train",
        "train",
        "train",
        "valid",
        "test",
        "test",
    ]
    for row in rows:
        assert row.audio_path.endswith(".wav")


def test_raw_transcript_field_is_selectable(tmp_path):
    root = write_export(tmp_path / "fleurs_test", with_audio=False)
    spec = spec_for(root, tmp_path / "processed")
    splits = spec.read_splits(text_field="raw_transcript")
    assert splits["valid"][0].text == "Baligato ug bisan."


def test_manifest_only_rebuilt_when_source_changes(tmp_path):
    root = write_export(tmp_path / "fleurs_test", with_audio=False)
    spec = spec_for(root, tmp_path / "processed")
    spec.read_splits()
    first = manifest_fingerprint(spec.manifest)
    stamp = spec.manifest.stat().st_mtime

    spec.read_splits()
    assert manifest_fingerprint(spec.manifest) == first
    assert spec.manifest.stat().st_mtime == stamp

    tsv = root / "test" / "manifest.tsv"
    os.utime(tsv, (tsv.stat().st_atime + 10, tsv.stat().st_mtime + 10))
    spec.read_splits()
    assert manifest_fingerprint(spec.manifest) == first  # identical content -> same hash
    assert spec.manifest.stat().st_mtime > stamp          # but it was rewritten


def test_missing_split_dirs_are_reported(tmp_path):
    root = write_export(tmp_path / "fleurs_test", with_audio=False)
    (root / "test" / "manifest.tsv").unlink()
    spec = spec_for(root, tmp_path / "processed")
    assert len(spec.missing_sources()) == 1
    with pytest.raises(FileNotFoundError, match="incomplete"):
        spec.read_splits()


def test_resolve_dataset_by_name_and_path(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    write_export(data_root / "fleurs_ceb_ph", with_audio=False)
    monkeypatch.chdir(tmp_path)

    named = resolve_dataset("fleurs_ceb_ph")
    assert named.manifest == Path("data/processed/manifest_fleurs_ceb_ph.csv")
    assert named.split_names == ("train", "valid", "test")
    assert named.missing_sources() == []

    default = resolve_dataset("default")
    assert default.manifest == Path("data/processed/manifest.csv")
    assert not default.has_preset_splits
    assert default.prepare_hint

    adhoc = resolve_dataset("data/fleurs_ceb_ph")
    assert adhoc.name == "fleurs_ceb_ph"
    assert adhoc.has_preset_splits

    manifest_file = tmp_path / "data" / "processed" / "manifest.csv"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text("id,audio,text,section\n1,a.wav,hi,x\n", encoding="utf-8")
    single = resolve_dataset(str(manifest_file))
    assert single.manifest == manifest_file
    assert not single.has_preset_splits

    with pytest.raises(FileNotFoundError, match="Unknown dataset"):
        resolve_dataset("does_not_exist")

    bare = tmp_path / "empty_dir"
    bare.mkdir()
    with pytest.raises(FileNotFoundError, match="no train/validation/test split dirs"):
        resolve_dataset(str(bare))


def test_apply_data_defaults_wires_manifest_and_preset_splits(tmp_path, monkeypatch):
    write_export(tmp_path / "data" / "fleurs_ceb_ph", with_audio=False)
    monkeypatch.chdir(tmp_path)

    config = _apply_data_defaults(TrainConfig(data="fleurs_ceb_ph"))

    assert config.dataset == str(Path("data/processed/manifest_fleurs_ceb_ph.csv"))
    assert config.data_name == "fleurs_ceb_ph"
    assert {k: len(v) for k, v in config.preset_splits.items()} == {
        "train": 3,
        "valid": 1,
        "test": 2,
    }


def test_apply_data_defaults_without_data_is_noop():
    config = TrainConfig()
    out = _apply_data_defaults(config)
    assert out.dataset == "data/processed/manifest.csv"
    assert out.preset_splits is None


def test_apply_data_defaults_reports_incomplete_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Hint"):
        _apply_data_defaults(TrainConfig(data="fleurs_ceb_ph"))


def test_data_and_dataset_are_mutually_exclusive():
    parser = build_arg_parser()
    assert parser.parse_args(["--data", "fleurs_ceb_ph"]).data == "fleurs_ceb_ph"
    assert parser.parse_args([]).data is None
    assert parser.parse_args([]).dataset == "data/processed/manifest.csv"
    with pytest.raises(SystemExit):
        parser.parse_args(["--data", "default", "--dataset", "x.csv"])


def test_trainer_uses_official_split_instead_of_random(tmp_path):
    root = write_export(tmp_path / "fleurs_test", with_audio=True)
    spec = spec_for(root, tmp_path / "processed")
    splits = spec.read_splits()

    trainer = Trainer(
        TrainConfig(
            dataset=str(spec.manifest),
            data_name=spec.name,
            preset_splits=splits,
            split_dir=str(tmp_path / "processed"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            epochs=1,
            batch_size=2,
            device="cpu",
            mixed_precision=False,
            retain_every=-1,
            model=tiny_model_config(),
        )
    )
    train_ds, valid_ds, test_ds, _ = trainer._load_datasets()

    # 6 rows would be a 4/0/2 random cut; the official partition must survive.
    assert (len(train_ds), len(valid_ds), len(test_ds)) == (3, 1, 2)
    split_file = tmp_path / "processed" / f"split_{spec.manifest.stem}.json"
    assert split_file.exists()
    assert split_fingerprint(split_file) == manifest_fingerprint(spec.manifest)


def test_trainer_falls_back_to_random_split(tmp_path, tiny_manifest):
    trainer = Trainer(
        TrainConfig(
            dataset=str(tiny_manifest),
            split_dir=str(tmp_path / "processed"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            device="cpu",
            retain_every=-1,
            model=tiny_model_config(),
        )
    )
    train_ds, valid_ds, test_ds, _ = trainer._load_datasets()
    assert (len(train_ds), len(valid_ds), len(test_ds)) == (4, 0, 1)


def test_decode_budget_warning_fires_for_long_transcripts(tmp_path, caplog):
    root = write_export(tmp_path / "fleurs_test", with_audio=True)
    spec = spec_for(root, tmp_path / "processed")
    splits = spec.read_splits()

    trainer = Trainer(
        TrainConfig(
            dataset=str(spec.manifest),
            data_name=spec.name,
            preset_splits=splits,
            max_text_len=4,
            split_dir=str(tmp_path / "processed"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            device="cpu",
            retain_every=-1,
            model=tiny_model_config(),
        )
    )
    train_ds, _, _, _ = trainer._load_datasets()
    with caplog.at_level(logging.WARNING, logger="src.training.train"):
        trainer._warn_decode_budget(train_ds)
    assert any("max-text-len" in r.getMessage() for r in caplog.records)


def test_decode_budget_quiet_when_transcripts_fit(tmp_path, caplog):
    root = write_export(tmp_path / "fleurs_test", with_audio=True)
    spec = spec_for(root, tmp_path / "processed")
    splits = spec.read_splits()

    trainer = Trainer(
        TrainConfig(
            dataset=str(spec.manifest),
            data_name=spec.name,
            preset_splits=splits,
            max_text_len=256,
            split_dir=str(tmp_path / "processed"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            device="cpu",
            retain_every=-1,
            model=tiny_model_config(),
        )
    )
    train_ds, _, _, _ = trainer._load_datasets()
    with caplog.at_level(logging.WARNING, logger="src.training.train"):
        trainer._warn_decode_budget(train_ds)
    assert not [r for r in caplog.records if "max-text-len" in r.getMessage()]


def test_manifest_csv_column_order_stays_canonical(tmp_path):
    root = write_export(tmp_path / "fleurs_test", with_audio=False)
    spec = spec_for(root, tmp_path / "processed")
    spec.read_splits()
    with spec.manifest.open(encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    assert header == ["id", "audio", "text", "section"]


def test_datasets_command_is_registered():
    import ns

    assert "datasets" in ns.COMMANDS
    assert "--data" in ns.TRAIN_FLAGS
