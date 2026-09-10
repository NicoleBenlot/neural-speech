"""WER and CER metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Compute Levenshtein distance between two token sequences (in characters or words)."""
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate for a single pair."""
    if not reference:
        return 1.0 if hypothesis else 0.0
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    return _levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate for a single pair."""
    if not reference:
        return 1.0 if hypothesis else 0.0
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def to_words(text: str) -> List[str]:
    return text.split()


@dataclass
class MetricsSummary:
    cer: float = 0.0
    wer: float = 0.0


def compute_metrics(references: List[str], hypotheses: List[str]) -> MetricsSummary:
    """Compute average CER/WER over a batch/list of sentence pairs."""
    if not references:
        return MetricsSummary()

    total_cer = sum(cer(r, h) for r, h in zip(references, hypotheses))
    total_wer = sum(wer(r, h) for r, h in zip(references, hypotheses))
    n = len(references)
    return MetricsSummary(cer=total_cer / n, wer=total_wer / n)