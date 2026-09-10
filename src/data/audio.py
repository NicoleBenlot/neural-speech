"""Audio loading and preprocessing utilities.

Loads audio through torchaudio (best for .opus when FFmpeg is available) and
falls back to soundfile/librosa-free decoding so the pipeline works even
without FFmpeg/torchcodec installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000


@dataclass
class AudioResult:
    waveform: torch.Tensor  # (1, samples) float32
    sample_rate: int


def _torchaudio_load(path: str) -> tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(path)
    return waveform, sr


def _soundfile_load(path: str) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))
    return waveform, sr


def _decode_audio(path: str) -> tuple[torch.Tensor, int]:
    """Decode an audio file to (waveform, sample_rate).

    Tries torchaudio first (needs FFmpeg for Opus), then falls back to
    soundfile which supports WAV/FLAC/OGG/Opus via libsndfile.
    """
    try:
        return _torchaudio_load(path)
    except Exception as torchaudio_exc:  # noqa: BLE001 - fallback expected
        try:
            return _soundfile_load(path)
        except Exception as sf_exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not decode {path} "
                f"(torchaudio: {torchaudio_exc}; soundfile: {sf_exc})"
            ) from sf_exc


def load_audio(
    path: str,
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> AudioResult:
    """Load an audio file, convert to mono, resample, and normalize.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate.

    Returns:
        AudioResult with waveform tensor and sample rate.
    """
    waveform, sr = _decode_audio(path)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    if not waveform.isfinite().all():
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)

    return AudioResult(waveform=waveform.float(), sample_rate=sr)


def load_audio_raw(
    path: str,
    target_sr: int = DEFAULT_SAMPLE_RATE,
) -> tuple[torch.Tensor, int]:
    """Load audio and return (waveform, sample_rate) tuple."""
    result = load_audio(path, target_sr=target_sr)
    return result.waveform, result.sample_rate