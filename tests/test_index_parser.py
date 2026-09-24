"""Tests for the index.txt parser."""

import os
from pathlib import Path

import pytest

from src.data.parser import parse_index


def test_parse_valid_entries(tiny_dataset):
    raw, _, _ = tiny_dataset
    result = parse_index(raw / "index.txt", assets_dir=Path(raw / "assets"))

    assert result.error_count == 0
    assert result.entry_count == 5

    by_id = {e.id: e for e in result.entries}
    assert by_id[104].text == "adlaw"
    assert by_id[104].audio_path.endswith(os.path.join("assets", "104.opus"))
    assert by_id[35].text == "akong"


def test_section_parsing(tiny_dataset):
    raw, _, _ = tiny_dataset
    result = parse_index(raw / "index.txt", assets_dir=Path(raw / "assets"))

    sections = {e.id: e.section for e in result.entries}
    assert sections[104] == "ad"
    assert sections[24] == "ak"
    assert sections[44] == "am"


def test_section_not_required():
    # No sections at all -> still parses fine
    path = Path("n/a")
    # build in-memory by monkeypatching read; simpler: write temp file
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        idx = Path(td) / "index.txt"
        idx.write_text("hello = 1\nworld = 2\n", encoding="utf-8")
        result = parse_index(idx, assets_dir=Path(td) / "assets")
        assert result.error_count == 0
        assert result.entry_count == 2
        assert result.entries[0].section is None


def test_blank_lines_ignored(tiny_dataset):
    raw, _, _ = tiny_dataset
    raw_index = raw / "index.txt"
    raw_index.write_text("\n\nhello = 1\n\n\n\n", encoding="utf-8")
    result = parse_index(raw_index, assets_dir=Path(raw / "assets"))
    assert result.error_count == 0
    assert result.entry_count == 1


def test_invalid_line_detected(tiny_dataset):
    raw, _, _ = tiny_dataset
    idx = raw / "index.txt"
    idx.write_text("not_an_entry\nabc = 12\n", encoding="utf-8")
    result = parse_index(idx, assets_dir=Path(raw / "assets"))
    assert result.error_count == 1
    assert result.entry_count == 1


def test_missing_index_file(tmp_path):
    result = parse_index(tmp_path / "missing.txt")
    assert result.error_count == 1
    assert result.entry_count == 0


def test_sentence_entry_with_spaces():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        idx = Path(td) / "index.txt"
        idx.write_text(
            "[sentences]\n"
            "ang bata nagdula sa bola = 196\n"
            "maayo nga adlaw kaninyo! = 197\n"
            "hello = 1\n",
            encoding="utf-8",
        )
        result = parse_index(idx, assets_dir=Path(td) / "assets")
        assert result.error_count == 0
        assert result.entry_count == 3
        by_id = {e.id: e for e in result.entries}
        assert by_id[196].text == "ang bata nagdula sa bola"
        assert by_id[196].section == "sentences"
        assert by_id[197].text == "maayo nga adlaw kaninyo!"
        assert by_id[197].audio_path.endswith(os.path.join("assets", "197.opus"))
        assert by_id[1].text == "hello"


def test_sentence_entry_extra_whitespace_around_equals():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        idx = Path(td) / "index.txt"
        idx.write_text("nindot  kaayo   =   198   \n", encoding="utf-8")
        result = parse_index(idx, assets_dir=Path(td) / "assets")
        assert result.error_count == 0
        assert result.entries[0].text == "nindot  kaayo"
        assert result.entries[0].id == 198