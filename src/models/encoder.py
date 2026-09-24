"""Audio feature extraction + sequence encoder.

Two encoders live here:

- ``AudioFeatureExtractor`` + ``AudioEncoder``: the original from-scratch
  pipeline (conv feature extractor + random-init Transformer encoder). Kept
  for backward compatibility and tests.
- ``MMSAudioEncoder``: a pretrained MMS/wav2vec2 acoustic backbone
  (feature extractor conv stack + transformer encoder stack) loaded from a
  HuggingFace checkpoint. This is the default for real training: the
  from-scratch encoder collapsed at this data scale (encoder outputs were
  bit-identical across all probe clips, pairwise cosine = 1.0000) because a
  random-init transformer gets insufficient gradient signal on ~150 clips.
  A pretrained multilingual backbone already has general acoustic
  representations, so fine-tuning only adapts to the language-specific head.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


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
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

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


class MMSAudioEncoder(nn.Module):
    """Pretrained MMS/wav2vec2 acoustic backbone.

    Wraps a HuggingFace ``Wav2Vec2Model`` (conv feature extractor +
    transformer encoder) loaded from a pretrained checkpoint. The feature
    extractor normalizes the raw waveform (zero-mean, unit-variance per
    utterance, matching the backbone's own processing) and downsamples it to
    ~20 ms frames; the transformer stack produces one 1024-dim frame vector
    per 20 ms.

    Low-resource fine-tuning recipe: all backbone weights are frozen except
    the last ``unfreeze_layers`` transformer encoder layers (the CTC head is
    trained separately anyway).
    """

    def __init__(
        self,
        model_id: str = "facebook/mms-300m",
        unfreeze_layers: int = 4,
        device: Optional[torch.device] = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        if unfreeze_layers < 0:
            raise ValueError("unfreeze_layers must be >= 0")

        from transformers import Wav2Vec2Config, Wav2Vec2Model

        # Official MMS fine-tuning recipe: zero out all backbone dropout so the
        # frozen layers produce identical outputs in train and eval mode.
        mms_config = Wav2Vec2Config.from_pretrained(
            model_id,
            use_safetensors=True,
            local_files_only=local_files_only,
        )
        mms_config.attention_dropout = 0.0
        mms_config.activation_dropout = 0.0
        mms_config.hidden_dropout = 0.0
        mms_config.feat_proj_dropout = 0.0
        mms_config.feat_extract_dropout = 0.0
        mms_config.layerdrop = 0.0
        mms_config.mask_time_prob = 0.0

        self.model_id = model_id
        self.backbone = Wav2Vec2Model.from_pretrained(
            model_id,
            config=mms_config,
            use_safetensors=True,
            local_files_only=local_files_only,
        )
        if device is not None:
            self.backbone.to(device)

        # Freeze everything, then unfreeze the last K transformer layers.
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        n_layers = self.backbone.config.num_hidden_layers
        self.unfreeze_layers = min(unfreeze_layers, n_layers)
        for layer in self.backbone.encoder.layers[-self.unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)

        self.config = self.backbone.config
        self.hidden_dim = self.backbone.config.hidden_size

    def normalize_waveform(
        self, waveform: torch.Tensor, audio_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Zero-mean, unit-variance normalize each utterance over its valid samples."""
        b, c, t = waveform.shape
        lengths = audio_lengths.clamp(max=t).float()

        mask = torch.arange(t, device=waveform.device).unsqueeze(0) < lengths.unsqueeze(1)
        w = waveform[:, 0, :]  # (B, T)

        mean = w.masked_fill(~mask, 0.0).sum(dim=-1, keepdim=True) / lengths.clamp(min=1).unsqueeze(-1)
        w = w - mean
        var = (w * w).masked_fill(~mask, 0.0).sum(dim=-1, keepdim=True) / lengths.clamp(min=1).unsqueeze(-1)
        w = w / (var + 1e-7).sqrt()
        return w * mask

    def forward(
        self,
        waveform: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode raw waveform to backbone hidden states.

        Args:
            waveform: (B, 1, T) float waveform (peak-normalized by the loader).
            audio_lengths: (B,) number of valid samples per utterance.

        Returns:
            (B, T', hidden_dim) backbone encoder output (last hidden state).
            Frames at/after each utterance's downsampled length are from
            padding; downstream code masks by ``feat_out_lengths``.
        """
        b, t = waveform.shape[0], waveform.shape[2]
        attention_mask = (
            torch.arange(t, device=waveform.device).unsqueeze(0)
            < audio_lengths.clamp(max=t).unsqueeze(1)
        ).long()

        input_values = self.normalize_waveform(waveform, audio_lengths)
        out = self.backbone(input_values=input_values, attention_mask=attention_mask)
        return out.last_hidden_state

    def feat_out_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        """Per-utterance frame counts after the conv feature-extractor downsampling."""
        return self.backbone._get_feat_extract_output_lengths(audio_lengths).long()


def build_encoder(config: dict) -> AudioEncoder:
    return AudioEncoder(
        d_model=config.get("d_model", 128),
        nhead=config.get("nhead", 4),
        num_layers=config.get("encoder_layers", 3),
        dim_feedforward=config.get("dim_feedforward", 512),
        dropout=config.get("dropout", 0.1),
        max_length=config.get("max_frames", 2048),
    )


__all__ = [
    "AudioFeatureExtractor",
    "AudioEncoder",
    "MMSAudioEncoder",
    "build_encoder",
]