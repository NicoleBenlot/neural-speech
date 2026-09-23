"""CTC projection head replacing the causal seq2seq decoder.

CTC has no decoder input other than the encoder output: a per-timestep
linear projection to the vocabulary (including a blank token), followed by a
log-softmax. No autoregressive feedback, no positional decoding, no
teacher forcing. Targets are plain character sequences; decoding is a
collapse-repeats / remove-blank greedy search (see STTModel.ctc_greedy_decode).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CTCHead(nn.Module):
    """Log-softmax linear head applied per timestep over encoder output.

    Args:
        d_model: Encoder feature dimension (128).
        vocab_size: Vocabulary size (includes the CTC blank token).
        blank_id: Index of the CTC blank token in the vocabulary.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        blank_id: int = 0,
    ):
        super().__init__()
        self.blank_id = blank_id
        self.proj = nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Project encoder output to per-timestep log-probabilities.

        Args:
            encoder_out: (B, T, D)

        Returns:
            (B, T, V) log-probabilities (log-softmax over the vocabulary).
        """
        return self.log_softmax(self.proj(encoder_out))


def build_ctc_head(config: dict, vocab_size: int, blank_id: int) -> CTCHead:
    return CTCHead(
        d_model=config.get("d_model", 128),
        vocab_size=vocab_size,
        blank_id=blank_id,
    )


__all__ = ["CTCHead", "build_ctc_head"]