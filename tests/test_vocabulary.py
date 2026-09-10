"""Vocabulary / tokenizer tests."""

import json
from pathlib import Path

import pytest

from src.tokens.tokenizer import (
    BOS,
    CharTokenizer,
    EOS,
    PAD,
    UNK,
    SPECIAL_TOKENS,
)


def test_special_tokens_present():
    tok = CharTokenizer()
    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.bos_id == 2
    assert tok.eos_id == 3


def test_encode_decode_roundtrip():
    tok = CharTokenizer()
    tok.build_from_texts(["adlaw", "adto"])
    ids = tok.encode("adlaw")
    assert tok.decode(ids) == "adlaw"


def test_unk_characters():
    tok = CharTokenizer()
    tok.build_from_texts(["adlaw"])
    ids = tok.encode("xyz")
    assert tok.unk_id in ids


def test_save_load(tmp_path):
    tok = CharTokenizer()
    tok.build_from_texts(["hello", "world"])
    path = tmp_path / "vocab.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.char_to_id == tok.char_to_id
    assert loaded.decode(loaded.encode("hello")) == "hello"


def test_bos_eos_behavior():
    tok = CharTokenizer()
    tok.build_from_texts(["hi"])
    ids = tok.encode("hi")
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == "hi"
    ids_no_special = tok.encode("hi", add_bos=False, add_eos=False)
    assert tok.bos_id not in ids_no_special