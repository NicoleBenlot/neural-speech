"""Audio feature extraction + sequence encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioFeatureExtractor(nn.Module):
    """Convert raw waveform into a compact frame feature sequence.

    Stacks Conv1d layers that downsample the raw samples into a sequence
    of frame vectors with fixed context.
    """

    def __init__(
        self,
        input_channels: int = 1,
        channels: int = 64,
        n_layers: int = 3,
        kernel_size: int = 3,
        stride: int = 2,
        output_dim: int = 128,
    ):
        super().__init__()
        layers = []
        in_ch = input_channels

        for _ in range(n_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, channels, kernel_size, stride=stride, padding=kernel_size // 2),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(),
                )
            )
            in_ch = channels

        self.features = nn.Sequential(*layers)
        self.proj = nn.Linear(channels, output_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            waveform: (B, 1, T) float waveform.

        Returns:
            (B, T', output_dim) frame features.
        """
        x = self.features(waveform)  # (B, C, T')
        x = x.transpose(1, 2)  # (B, T', C)
        return self.proj(x)


class AudioEncoder(nn.Module):
    """Transformer encoder over audio frame features.

    Produces a fixed-dim sequence representation with a learned length
    embedding used to keep the model robust to variable-length inputs.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_length: int = 2048,
    ):
        super().__init__()
        self.length_embed = nn.Embedding(max_length, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, t, d = features.shape
        positions = torch.arange(t, device=features.device).unsqueeze(0)
        length = min(t, self.length_embed.num_embeddings)
        pos = self.length_embed(positions[:, :length])
        x = features[:, :length] + pos

        enc = self.encoder(x, src_key_padding_mask=padding_mask[:, :length])

        # Replace masked positions with zeros
        if padding_mask is not None:
            enc = enc.masked_fill(padding_mask[:, :length].unsqueeze(-1), 0.0)
        return enc


def build_encoder(config: dict) -> AudioEncoder:
    return AudioEncoder(
        d_model=config.get("d_model", 128),
        nhead=config.get("nhead", 4),
        num_layers=config.get("encoder_layers", 3),
        dim_feedforward=config.get("dim_feedforward", 512),
        dropout=config.get("dropout", 0.1),
        max_length=config.get("max_frames", 2048),
    )