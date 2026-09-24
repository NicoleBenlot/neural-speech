"""Full STT model combining encoder + decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.models.ctc_head import CTCHead, build_ctc_head
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
    bos_token_id: int = 2
    eos_token_id: int = 3
    pad_token_id: int = 0
    head: str = "decoder"  # "decoder" (seq2seq) or "ctc" (CTC head)
    blank_token_id: Optional[int] = None  # set to the CTC blank index when head == "ctc"
    backbone: str = "none"  # "none" (from-scratch encoder) or a HF model id like "facebook/mms-300m"
    backbone_unfreeze_layers: int = 4  # last K transformer layers left trainable when backbone is used

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
        self.mms_encoder = None

        uses_backbone = config.backbone is not None and config.backbone != "none"
        if uses_backbone:
            from src.models.encoder import MMSAudioEncoder

            self.mms_encoder = MMSAudioEncoder(
                model_id=config.backbone,
                unfreeze_layers=config.backbone_unfreeze_layers,
            )
            self.feature_extractor = None
            self.encoder = None
            self.config.d_model = self.mms_encoder.hidden_dim
        else:
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

        if config.head == "ctc":
            self.ctc_head = build_ctc_head(
                config.to_dict(), vocab_size, config.blank_token_id or 0
            )
            self.decoder = None
            if config.blank_token_id is None:
                config.blank_token_id = self.ctc_head.blank_id
        else:
            self.decoder = build_decoder(config.to_dict(), vocab_size)
            self.ctc_head = None

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
        if self.mms_encoder is not None:
            encoder_out = self.mms_encoder(audio, audio_lengths)
            lengths = self.mms_encoder.feat_out_lengths(audio_lengths)
            lengths = torch.clamp(lengths, max=encoder_out.shape[1])
            return encoder_out, lengths

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
        if self.decoder is None:
            raise RuntimeError("Model built with head='ctc'; forward() is the seq2seq path, use forward_ctc()")
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

    def forward_ctc(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """CTC forward pass: per-timestep log-probs over encoder output.

        Args:
            audio: (B, 1, T)
            audio_lengths: (B,)

        Returns:
            (B, T_e, vocab_size) log-probabilities (log-softmax).
        """
        if self.ctc_head is None:
            raise RuntimeError("Model built with head='decoder'; forward_ctc() needs head='ctc'")
        encoder_out, _ = self.encode(audio, audio_lengths)
        return self.ctc_head(encoder_out)

    def ctc_input_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        """Encoder output length per sample after the frame downsampling.

        With a pretrained backbone this is the conv feature-extractor
        downsampling of the MMS encoder; otherwise it mirrors encode()'s
        repeated ``ceil(T / feat_stride)`` length rule so CTCLoss gets input
        lengths that match the actual per-frame counts.
        """
        if self.mms_encoder is not None:
            return self.mms_encoder.feat_out_lengths(audio_lengths).clamp(min=1)
        lengths = audio_lengths
        for _ in range(self.config.feat_layers):
            lengths = torch.ceil(lengths / self.config.feat_stride).long()
        return lengths

    def ctc_greedy_decode(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        blank_id: Optional[int] = None,
    ) -> List[List[int]]:
        """CTC greedy decode: argmax per frame, collapse repeats, drop blanks.

        Returns raw character token ids (no BOS/EOS), ready for
        ``tokenizer.decode``. CTC beam search is a later optimization.
        """
        if self.ctc_head is None:
            raise RuntimeError("Model built with head='decoder'; ctc_greedy_decode() needs head='ctc'")
        if blank_id is None:
            blank_id = (
                self.config.blank_token_id
                if self.config.blank_token_id is not None
                else self.ctc_head.blank_id
            )

        encoder_out, lengths = self.encode(audio, audio_lengths)
        log_probs = self.ctc_head(encoder_out)  # (B, T, V)
        argmax = log_probs.argmax(dim=-1)  # (B, T)

        results: List[List[int]] = []
        for row, L in zip(argmax, lengths):
            ids: List[int] = []
            prev: Optional[int] = None
            for tok in row[: L.item()].tolist():
                if tok == blank_id:
                    prev = None
                else:
                    if tok != prev:
                        ids.append(tok)
                    prev = tok
            results.append(ids)
        return results

    def greedy_decode(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        max_len: int = 128,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
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

    def decode(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        max_len: int = 128,
        beam_size: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        length_penalty: float = 1.0,
    ) -> List[List[int]]:
        """Autoregressive decoding; beam search if ``beam_size`` > 1.

        With ``head="ctc"`` this dispatches to CTC greedy decoding and the
        autoregressive arguments are ignored. TODO: CTC beam search (with or
        without an LM) is a later optimization.
        """
        if self.ctc_head is not None:
            return self.ctc_greedy_decode(audio, audio_lengths)
        if beam_size and beam_size > 1:
            return self.beam_decode(
                audio,
                audio_lengths,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
            )
        return self.greedy_decode(
            audio,
            audio_lengths,
            max_len=max_len,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

    def beam_decode(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        max_len: int = 128,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
    ) -> List[List[int]]:
        """Beam-search autoregressive decoding."""
        encoder_out, encoder_lengths = self.encode(audio, audio_lengths)
        b = encoder_out.shape[0]

        device = next(self.parameters()).device
        results: List[List[int]] = []

        for i in range(b):
            mem_mask = (
                torch.arange(encoder_out.shape[1], device=device).unsqueeze(0)
                >= encoder_lengths[i].cpu().item()
            ).to(device)
            results.append(
                self._beam_search(
                    encoder_out[i : i + 1],
                    mem_mask,
                    max_len=max_len,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                )
            )
        return results

    def _beam_search(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        max_len: int,
        beam_size: int,
        length_penalty: float,
        bos_token_id: int,
        eos_token_id: int,
    ) -> List[int]:
        """Beam search over a single utterance's encoder output.

        All active hypotheses share the same length at each step, so the
        decoder runs once per step over the whole beam as a batch.

        Returns:
            Generated token ids (starting after BOS, ending with EOS if found).
        """
        vocab = self.vocab_size
        beam_size = min(beam_size, vocab)

        active: List[tuple[List[int], float]] = [([bos_token_id], 0.0)]
        finished: List[tuple[List[int], float]] = []

        for _ in range(max_len):
            if not active:
                break

            tokens = torch.tensor(
                [hyp for hyp, _ in active], dtype=torch.long, device=memory.device
            )
            k = tokens.shape[0]
            tok = self.decoder.embed(tokens)
            mem = memory.expand(k, -1, -1).contiguous()
            mask = memory_mask.expand(k, -1).contiguous()

            logits, _ = self.decoder.forward_step(mem, mask, tok)
            log_probs = logits[:, -1].log_softmax(dim=-1)

            candidates: List[tuple[List[int], float]] = []
            for idx, (hyp, score) in enumerate(active):
                top = log_probs[idx].topk(min(beam_size, vocab))
                for value, token_id in zip(top.values, top.indices):
                    token_id = token_id.item()
                    candidates.append((hyp + [token_id], score + value.item()))

            candidates.sort(key=lambda c: c[1], reverse=True)

            active = []
            for hyp, score in candidates[:beam_size]:
                if hyp[-1] == eos_token_id:
                    finished.append((hyp, score))
                else:
                    active.append((hyp, score))

        scored = [
            (hyp, score / (len(hyp) ** length_penalty))
            for hyp, score in finished + active
        ]
        best = max(scored, key=lambda c: c[1])[0]
        if best[-1] != eos_token_id:
            best = best + [eos_token_id]
        return best[1:]


def build_stt_model(config: STTConfig, vocab_size: int) -> STTModel:
    return STTModel(config, vocab_size)