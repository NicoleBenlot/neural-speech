"""Tests for the Kaggle upload bundler (path rewriting only).

These lock in the one property the port depends on: a Windows-written path must
come out Linux-readable, while ids, text, metrics and the manifest fingerprint
stay byte-identical. No zip/audio needed.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_MODULE = Path(__file__).resolve().parent.parent / "kaggle" / "make_upload.py"
_spec = importlib.util.spec_from_file_location("kaggle_make_upload", _MODULE)
make_upload = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_upload)


def test_posix_flips_every_separator():
    assert make_upload._posix(r"data\raw\assets\7.opus") == "data/raw/assets/7.opus"
    assert make_upload._posix("data/raw/7.opus") == "data/raw/7.opus"
    assert make_upload._posix(r"C:\Users\x\a.wav") == "C:/Users/x/a.wav"


def test_rewrite_manifest_csv_normalizes_audio_only(tmp_path):
    src = tmp_path / "manifest.csv"
    src.write_text(
        "id,audio,text,section\n"
        "0,data\\fleurs\\train\\00000.wav,adlaw train,1\n"
        "1,data\\fleurs\\test\\00001.wav,adlaw test,1\n",
        encoding="utf-8",
    )
    dst = tmp_path / "out.csv"
    rows = make_upload.rewrite_manifest_csv(src, dst)

    assert rows == 2
    lines = dst.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "id,audio,text,section"  # header/order preserved
    assert "data/fleurs/train/00000.wav" in lines[1]
    assert "data/fleurs/test/00001.wav" in lines[2]
    assert "adlaw train" in lines[1]  # text untouched
    assert "\\" not in dst.read_text(encoding="utf-8")


def test_rewrite_split_json_normalizes_every_row_and_keeps_fingerprint(tmp_path):
    src = tmp_path / "split_manifest.json"
    payload = {
        "seed": 42,
        "train": [
            {"id": 0, "audio_path": r"data\raw\a.wav", "text": "adlaw", "section": "train"}
        ],
        "valid": [
            {"id": 1, "audio_path": r"data\raw\b.wav", "text": "bat", "section": "valid"}
        ],
        "test": [],
        "manifest_fingerprint": "deadbeef",
    }
    src.write_text(json.dumps(payload), encoding="utf-8")

    dst = tmp_path / "out.json"
    assert make_upload.rewrite_split_json(src, dst) == 2

    data = json.loads(dst.read_text(encoding="utf-8"))
    assert data["train"][0]["audio_path"] == "data/raw/a.wav"
    assert data["valid"][0]["audio_path"] == "data/raw/b.wav"
    assert data["train"][0]["text"] == "adlaw"
    assert data["seed"] == 42
    # a fingerprint change only re-warns + re-splits; it must not silently
    # redefine the rows
    assert data["manifest_fingerprint"] == "deadbeef"


def test_rewrite_ckpt_index_covers_dual_best_entries(tmp_path):
    src = tmp_path / "best.json"
    src.write_text(
        json.dumps(
            {
                "val_loss": {"version": "v004", "path": r"checkpoints\mms\v004", "value": 1.2},
                "cer": {"version": "v003", "path": r"checkpoints\mms\v003", "value": 0.4},
                "dataset": r"data\processed\manifest_fleurs_ceb_ph.csv",
                "replay_manifest": r"data\processed\manifest.csv",
            }
        ),
        encoding="utf-8",
    )
    dst = tmp_path / "out.json"
    touched = make_upload.rewrite_ckpt_index(src, dst)

    data = json.loads(dst.read_text(encoding="utf-8"))
    assert data["val_loss"]["path"] == "checkpoints/mms/v004"
    assert data["cer"]["path"] == "checkpoints/mms/v003"
    assert data["dataset"] == "data/processed/manifest_fleurs_ceb_ph.csv"
    assert data["replay_manifest"] == "data/processed/manifest.csv"
    assert data["val_loss"]["value"] == 1.2  # metrics untouched
    assert set(touched) == {"dataset", "replay_manifest", "val_loss.path", "cer.path"}


def test_rewrite_ckpt_index_tolerates_legacy_latest_json(tmp_path):
    src = tmp_path / "latest.json"
    src.write_text(json.dumps({"version": "v004", "path": r"checkpoints\mms\v004", "latest": True}), encoding="utf-8")
    dst = tmp_path / "out.json"
    make_upload.rewrite_ckpt_index(src, dst)
    data = json.loads(dst.read_text(encoding="utf-8"))
    assert data["path"] == "checkpoints/mms/v004"
    assert data["latest"] is True


def test_check_bundle_accepts_a_posix_archive(tmp_path):
    import zipfile

    good = tmp_path / "code.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("ns.py", "print('hi')")
        zf.writestr("src/data/dataset.py", "# ok")
    assert make_upload.check_bundle(good, "checkpoints/mms") == []


def test_check_bundle_flags_a_bundle_missing_its_payload(tmp_path):
    import zipfile

    # every arc name we write is built with as_posix()/relative_to(), so a
    # backslash in a zip *entry* is unreachable here; check the payload instead
    empty = tmp_path / "data.zip"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("README", "no data here")
    problems = make_upload.check_bundle(empty, "checkpoints/mms")
    assert any("no data/ entries" in p for p in problems)

    bare = tmp_path / "ckpt.zip"
    with zipfile.ZipFile(bare, "w") as zf:
        zf.writestr("checkpoints/mms/latest.json", "{}")
    problems = make_upload.check_bundle(bare, "checkpoints/mms")
    assert any("no checkpoints/mms/vNNN/ version dir" in p for p in problems)
    assert any("no model.pt" in p for p in problems)


def test_check_bundle_flags_windows_paths_inside_a_manifest(tmp_path):
    import zipfile

    bundle = tmp_path / "data.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("data/raw/a.wav", "x")
        zf.writestr(
            "data/processed/manifest.csv",
            "id,audio,text,section\n0,data\\raw\\a.wav,adlaw,train\n",
        )
    problems = make_upload.check_bundle(bundle, "checkpoints/mms")
    assert any("backslash in audio" in p for p in problems)


@pytest.mark.parametrize("name", ["latest.json", "best.json", "protect.json"])
def test_ckpt_index_names_are_the_ones_the_line_uses(name, tmp_path):
    assert name in make_upload.CKPT_INDEXES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
