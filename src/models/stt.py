"""Full STT model combining encoder + decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn

from src.models.encoder import AudioEncoder, build_encoder
from src.models.decoder import TextDecoder, build_decoder


@dataclass
class STTConfig:
    d_model: int = 128
    nhead: int = 4
    encoder_layers: int = 3
    decoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_frames: int = 2048
    feat_channels: int = 64
    feat_layers: int = 3
    feat_stride: int = 2
    feat_kernel: int = 3
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "STTConfig":
        known = {k: v for k, v in cls.__dataclass_fields__.items()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class STTModel(nn.Module):
    """waveform -> features -> encoder -> decoder -> token probs."""

    def __init__(self, config: STTConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        from src.models.encoder import AudioFeatureExtractor

        self.feature_extractor = AudioFeatureExtractor(
            input_channels=1,
            channels=config.feat_channels,
            n_layers=config.feat_layers,
            kernel_size=config.feat_kernel,
            stride=config.feat_stride,
            output_dim=config.d_model,
        )
        self.encoder = build_encoder(config.to_dict())
        self.decoder = build_decoder(config.to_dict(), vocab_size)

    def encode(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder output for audio.

        Args:
            audio: (B, 1, T)
            audio_lengths: (B,)

        Returns:
            (encoder_out (B, T_e, D), encoder_lengths (B,))
        """
        features = self.feature_extractor(audio)

        # Downsample lengths proportionally to the 2^n stride rule.
        # The Conv1d uses padding = kernel//2 and stride=2; effective output
        # length is ceil(T / stride^n).
        b, t, _ = features.shape
        lengths = audio_lengths
        n_layers = self.config.feat_layers
        for _ in range(n_layers):
            lengths = torch.ceil(lengths / self.config.feat_stride).long()

        lengths = torch.clamp(lengths, max=t)

        t_frames = features.shape[1]
        arange = torch.arange(t_frames, device=audio.device)
        padding_mask = arange.unsqueeze(0) >= lengths.unsqueeze(1)

        encoder_out = self.encoder(features, padding_mask)
        return encoder_out, lengths

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training.

        Args:
            audio: (B, 1, T)
            audio_lengths: (B,)
            tokens: (B, L)
            token_lengths: (B,)

        Returns:
            (B, L, vocab_size)
        """
        encoder_out, encoder_lengths = self.encode(audio, audio_lengths)
        encoder_lengths = encoder_lengths.clamp(max=encoder_out.shape[1])

        b, t_e = encoder_out.shape[:2]
        arange = torch.arange(t_e, device=audio.device)
        encoder_mask = arange.unsqueeze(0) >= encoder_lengths.unsqueeze(1)

        b, l = tokens.shape
        arange = torch.arange(l, device=tokens.device)
        token_mask = arange.unsqueeze(0) >= token_lengths.unsqueeze(1)

        logits, _ = self.decoder(encoder_out, encoder_mask, tokens, token_mask)
        return logits

    def greedy_decode(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        max_len: int = 128,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> List[List[int]]:
        """Greedy autoregressive decoding."""
        encoder_out, encoder_lengths = self.encode(audio, audio_lengths)
        b = encoder_out.shape[0]

        device = next(self.parameters()).device
        results: List[List[int]] = []

        for i in range(b):
            memory = encoder_out[i : i + 1]
            mem_mask = (
                torch.arange(encoder_out.shape[1], device=device).unsqueeze(0)
                >= encoder_lengths[i].cpu().item()
            )
            mem_mask = mem_mask.to(device)

            out_ids: List[int] = []
            cur_embeds = self.decoder.embed(
                torch.tensor([[bos_token_id]], device=device)
            )
            for _ in range(max_len):
                logits, _ = self.decoder.forward_step(
                    memory, mem_mask, cur_embeds
                )
                next_id = logits[0, -1].argmax(dim=-1).item()
                out_ids.append(next_id)
                if next_id == eos_token_id:
                    break
                next_emb = self.decoder.embed(
                    torch.tensor([[next_id]], device=device)
                )
                cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)
            else:
                out_ids.append(eos_token_id)
            results.append(out_ids)

        return results


def build_stt_model(config: STTConfig, vocab_size: int) -> STTModel:
    return STTModel(config, vocab_size)