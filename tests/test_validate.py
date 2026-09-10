"""Dataset validation + manifest preparation tests."""

import pytest

from src.data.validate import validate
from src.data.prepare import prepare, write_manifest
from src.data.parser import IndexEntry, parse_index
from tests.conftest import _write_wav


def test_validate_clean_dataset(tiny_dataset, tmp_path):
    raw, assets, _ = tiny_dataset
    # assets are .wav; write a manifest-independent check via parser + validate
    # validation uses index + assets dir
    result = validate(str(raw / "index.txt"), assets_dir=str(raw / "assets"))
    assert result.invalid_entries == 0
    assert result.duplicate_ids == 0
    assert result.missing_audio == 0


def test_validate_missing_audio(tiny_dataset, tmp_path):
    raw, assets, _ = tiny_dataset
    (assets / "44.opus").unlink()
    # Should report exactly one missing file
    result = validate(str(raw / "index.txt"), assets_dir=str(raw / "assets"))
    assert result.missing_audio == 1


def test_validate_duplicate_id(tiny_dataset, tmp_path):
    raw, assets, _ = tiny_dataset
    idx = raw / "index.txt"
    idx.write_text(
        "[ad]\nadlaw = 104\nadto = 83\nakong = 104\n",
        encoding="utf-8",
    )
    result = validate(str(idx), assets_dir=str(raw / "assets"))
    assert result.duplicate_ids >= 1


def test_prepare_writes_manifest(tiny_dataset, tmp_path):
    raw, assets, _ = tiny_dataset
    out = tmp_path / "manifest.csv"
    write_manifest(parse_index(raw / "index.txt").entries, out)
    content = out.read_text(encoding="utf-8")
    lines = content.strip().splitlines()
    assert lines[0] == "id,audio,text,section"
    assert len(lines) == 6  # header + 5 entries
    assert "adlaw" in content


def test_fail_on_error_exit(tiny_dataset, tmp_path):
    raw, assets, _ = tiny_dataset
    (assets / "44.opus").unlink()
    with pytest.raises(SystemExit) as exc:
        validate(str(raw / "index.txt"), assets_dir=str(raw / "assets"), fail_on_error=True)
    assert exc.value.code == 1