"""Dataset loading, collation, and splitting tests."""

import torch
import pytest

from src.data.dataset import (
    SpeechDataset,
    collate_speech,
    read_manifest,
    split_dataset,
    save_split,
    load_split,
)
from src.tokens.tokenizer import CharTokenizer


def test_read_manifest(tiny_manifest):
    rows = read_manifest(tiny_manifest)
    assert len(rows) == 5
    assert rows[0].text == "adlaw"
    assert rows[0].id == 100


def test_dataset_len_getitem(tiny_manifest):
    tok = CharTokenizer()
    tok.build_from_texts(["adlaw", "adto"])
    ds = SpeechDataset(tiny_manifest, tokenizer=tok)
    assert len(ds) == 5
    item = ds[0]
    assert item["audio"].shape[0] == 1
    assert item["text"]
    assert item["tokens"].shape[0] > 0
    assert item["id"].item() >= 100


def test_collate_padding(tiny_manifest):
    tok = CharTokenizer()
    tok.build_from_texts(["adlaw", "adto", "ako"])
    ds = SpeechDataset(tiny_manifest, tokenizer=tok)
    samples = [ds[i] for i in range(5)]

    batch = collate_speech(samples)
    assert batch.audio.shape[0] == 5
    assert batch.audio.shape[1] == 1
    assert batch.tokens.shape[0] == 5
    assert batch.tokens.shape[1] == max(s["tokens"].shape[0] for s in samples)
    assert batch.audio_padding_mask.shape == (5, batch.audio.shape[2])
    assert batch.token_padding_mask.shape == (5, batch.tokens.shape[1])
    assert len(batch.texts) == 5


def test_split_no_overlap(tiny_manifest):
    rows = read_manifest(tiny_manifest)
    train, valid, test = split_dataset(rows, seed=7)
    train_ids = {r.id for r in train}
    valid_ids = {r.id for r in valid}
    test_ids = {r.id for r in test}
    assert not (train_ids & valid_ids)
    assert not (train_ids & test_ids)
    assert not (valid_ids & test_ids)
    assert len(train) + len(valid) + len(test) == len(rows)


def test_split_reproducible(tiny_manifest):
    rows = read_manifest(tiny_manifest)
    a = split_dataset(rows, seed=42)
    b = split_dataset(rows, seed=42)
    assert [r.id for r in a[0]] == [r.id for r in b[0]]


def test_save_load_split(tmp_path, tiny_manifest):
    rows = read_manifest(tiny_manifest)
    train, valid, test = split_dataset(rows, seed=3)
    path = tmp_path / "split.json"
    save_split(path, train, valid, test, seed=3)

    tok = CharTokenizer()
    tok.build_from_texts([r.text for r in rows])
    train_ds, valid_ds, test_ds = load_split(path, tok)
    assert len(train_ds) == len(train)
    assert len(valid_ds) == len(valid)
    assert len(test_ds) == len(test)
    assert train_ds[0]["id"].item() == train[0].id