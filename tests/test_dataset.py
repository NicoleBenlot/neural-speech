"""Dataset loading, collation, and splitting tests."""

import torch
import pytest

from src.data.dataset import (
    LengthGroupedBatchSampler,
    SpeechDataset,
    collate_speech,
    read_manifest,
    split_dataset,
    save_split,
    load_split,
    manifest_fingerprint,
    split_fingerprint,
    split_row_counts,
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


def test_manifest_fingerprint_tracks_content(tmp_path, tiny_manifest):
    fp1 = manifest_fingerprint(tiny_manifest)
    tiny_manifest.write_text(
        tiny_manifest.read_text(encoding="utf-8") + "105,noop.wav,extra,ad\n",
        encoding="utf-8",
    )
    assert manifest_fingerprint(tiny_manifest) != fp1


def test_save_split_stores_fingerprint(tmp_path, tiny_manifest):
    rows = read_manifest(tiny_manifest)
    train, valid, test = split_dataset(rows, seed=3)
    path = tmp_path / "split.json"
    save_split(path, train, valid, test, seed=3, fingerprint=manifest_fingerprint(tiny_manifest))
    assert split_fingerprint(path) == manifest_fingerprint(tiny_manifest)
    assert sum(split_row_counts(path)) == len(rows)


def test_legacy_split_has_no_fingerprint(tmp_path, tiny_manifest):
    rows = read_manifest(tiny_manifest)
    train, valid, test = split_dataset(rows, seed=3)
    path = tmp_path / "split.json"
    save_split(path, train, valid, test, seed=3)
    assert split_fingerprint(path) is None
    assert sum(split_row_counts(path)) == len(rows)

def test_length_grouped_sampler_covers_every_row_once():
    lengths = [1000, 5000, 100, 4000, 2000, 50, 3000, 1500]
    sampler = LengthGroupedBatchSampler(lengths, batch_size=2, seed=7, shuffle=True)
    batches = list(sampler)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    assert all(len(b) <= 2 for b in batches)
    assert len(sampler) == len(batches)


def test_length_grouped_sampler_batches_similar_lengths():
    lengths = [1000, 5000, 100, 4000, 2000, 50, 3000, 1500]
    sampler = LengthGroupedBatchSampler(
        lengths, batch_size=2, seed=7, megabatch_mult=100, shuffle=False
    )
    spreads = [max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in sampler]
    # a random order would put 50-sample clips in the same batch as 5000s ones
    assert max(spreads) < 2000


def test_length_grouped_sampler_respects_max_frames():
    lengths = [16000] * 10 + [640000]  # ten 1s clips plus one 40s clip
    sampler = LengthGroupedBatchSampler(
        lengths, batch_size=8, seed=7, max_frames=320000, shuffle=False
    )
    batches = list(sampler)
    for b in batches:
        padded = len(b) * max(lengths[i] for i in b)
        # a single clip longer than the whole budget still runs, alone
        assert padded <= 320000 or len(b) == 1
    # the outlier gets its own batch rather than padding its neighbours
    assert [len(b) for b in batches] == [8, 2, 1]



def test_length_grouped_sampler_treats_unknown_length_as_median():
    sampler = LengthGroupedBatchSampler(
        [0, 100, 200, 0], batch_size=2, seed=1, shuffle=False
    )
    batches = list(sampler)
    assert sorted(i for b in batches for i in b) == [0, 1, 2, 3]
    assert all(0 < len(b) <= 2 for b in batches)


def test_length_grouped_sampler_reshuffles_every_epoch():
    lengths = [1000 * (i % 17 + 1) for i in range(200)]
    sampler = LengthGroupedBatchSampler(lengths, batch_size=4, seed=3, shuffle=True)
    first = [tuple(b) for b in sampler]
    second = [tuple(b) for b in sampler]
    third = [tuple(b) for b in sampler]

    assert first != second != third
    # every epoch still covers each row exactly once, and the count is stable
    for order in (first, second, third):
        assert sorted(i for b in order for i in b) == list(range(200))
    assert len(sampler) == len(first) == len(second) == len(third)


def test_length_grouped_sampler_without_shuffle_is_deterministic():
    lengths = [1000 * (i % 17 + 1) for i in range(50)]
    sampler = LengthGroupedBatchSampler(lengths, batch_size=4, shuffle=False)
    assert [tuple(b) for b in sampler] == [tuple(b) for b in sampler]


def test_probe_num_samples_reads_header(tiny_manifest, tmp_path):
    from src.data.dataset import probe_num_samples

    row = read_manifest(tiny_manifest)[0]
    assert probe_num_samples(row.audio_path) == pytest.approx(4800, rel=0.01)  # 0.3s
    # unreadable paths degrade to "unknown" instead of raising
    assert probe_num_samples("nope/missing.opus") == 0

