"""WER / CER metric tests."""

import pytest

from src.training.metrics import cer, wer, compute_metrics, _levenshtein


def test_levenshtein():
    assert _levenshtein(list("kitten"), list("sitting")) == 3
    assert _levenshtein([], []) == 0
    assert _levenshtein(list("abc"), list("abc")) == 0


def test_cer_exact():
    assert cer("adlaw", "adlaw") == 0.0


def test_cer_substitution():
    # "adlaw" -> "adlax": 1 char change over 5 chars
    assert cer("adlaw", "adlax") == pytest.approx(1 / 5)


def test_wer_exact():
    assert wer("ako si juan", "ako si juan") == 0.0


def test_wer_one_word_error():
    assert wer("ako si juan", "ako juan") == pytest.approx(1 / 3)


def test_wer_empty_reference():
    assert wer("", "hello") == 1.0
    assert wer("", "") == 0.0


def test_compute_metrics():
    result = compute_metrics(["adlaw adlaw"], ["adlaw"])
    assert result.cer > 0
    assert result.wer > 0


def test_compute_metrics_empty():
    result = compute_metrics([], [])
    assert result.cer == 0.0
    assert result.wer == 0.0