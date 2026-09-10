"""Autoregressive text decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class TextDecoder(nn.Module):
    """Transformer decoder generating token probabilities from encoder output."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        bos_token_id: int = 1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(512, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.bos_token_id = bos_token_id

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((size, size), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        tokens: torch.Tensor,
        token_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            encoder_out: (B, T_e, D)
            encoder_padding_mask: (B, T_e)
            tokens: (B, L)
            token_padding_mask: (B, L)

        Returns:
            (B, L, vocab_size)
        """
        b, l = tokens.shape
        emb = self.embed(tokens)
        positions = torch.arange(l, device=tokens.device).unsqueeze(0)
        pos = self.pos(positions[:, :l])
        tgt = emb + pos

        tgt_mask = self._causal_mask(l, tokens.device)
        tgt_key_padding_mask = token_padding_mask if token_padding_mask is not None else None

        memory_key_padding_mask = encoder_padding_mask if encoder_padding_mask is not None else None

        out = self.decoder(
            tgt,
            encoder_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out(out), out

    def step(
        self,
        encoder_out: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        token_embed: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a single step's logits for inference.

        Args:
            encoder_out: (T_e, D)
            encoder_padding_mask: (T_e,)
            token_embed: (1, L, D)
            memory: (1, T_e, D)

        Returns:
            (vocab,) logits for the last position.
        """
        out, _ = self.forward_step(encoder_out[None], encoder_padding_mask[None], token_embed)
        return out[0, -1]

    def forward_step(
        self,
        encoder_out: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        token_embed: torch.Tensor,
    ) -> torch.Tensor:
        tgt = token_embed
        l = tgt.shape[1]
        tgt_mask = self._causal_mask(l, tgt.device)
        out = self.decoder(
            tgt,
            encoder_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=encoder_padding_mask,
        )
        logits = self.out(out)
        return logits, out


def build_decoder(config: dict, vocab_size: int) -> TextDecoder:
    return TextDecoder(
        vocab_size=vocab_size,
        d_model=config.get("d_model", 128),
        nhead=config.get("nhead", 4),
        num_layers=config.get("decoder_layers", 3),
        dim_feedforward=config.get("dim_feedforward", 512),
        dropout=config.get("dropout", 0.1),
        bos_token_id=config.get("bos_token_id", 1),
    )