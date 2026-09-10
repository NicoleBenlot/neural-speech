"""Shared test fixtures: tiny synthetic dataset using generated audio."""

from __future__ import annotations

import torch
import torchaudio
import pytest
import tempfile
import wave
import math
from pathlib import Path
import struct


def _write_wav(path: Path, duration: float = 0.3, sr: int = 16000, freq: float = 440.0):
    """Write a tiny synthetic WAV tone (tests avoid requiring Opus decoders)."""
    n = int(duration * sr)
    samples = []
    for i in range(n):
        v = 0.3 * math.sin(2 * math.pi * freq * i / sr)
        samples.append(0.5 + v / 2)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(s * 32767 * 0.5)) for s in samples
        )
        w.writeframes(frames)


@pytest.fixture
def tiny_dataset(tmp_path: Path):
    """Create a small raw dataset: index.txt + synthetic assets.

    The parser always resolves `<id>.opus` paths (matching the real data
    layout), so the fixtures write WAV bytes under .opus filenames. Decoding
    is content-based, so this exercises the full pipeline without FFmpeg.
    """
    raw = tmp_path / "data" / "raw"
    assets = raw / "assets"
    assets.mkdir(parents=True)

    words = ["adlaw", "adto", "ako", "akong", "amo"]
    index_lines = ["[ad]", "adlaw = 104", "adto = 83", "", "[ak]", "ako = 24", "akong = 35", "[am]", "amo = 44"]
    (raw / "index.txt").write_text("\n".join(index_lines), encoding="utf-8")

    # Write synthetic 16k mono wavs named 104.opus, 83.opus ...
    for wid in (104, 83, 24, 35, 44):
        _write_wav(assets / f"{wid}.opus")

    return raw, assets, words


@pytest.fixture
def tiny_manifest(tmp_path: Path) -> Path:
    """A manifest.csv pointing at synthetic wav files."""
    assets = tmp_path / "assets"
    assets.mkdir(parents=True)
    rows = []
    for i, word in enumerate(["adlaw", "adto", "ako", "akong", "amo"]):
        wid = 100 + i
        _write_wav(assets / f"{wid}.wav")
        rows.append(f"{wid},{assets / f'{wid}.wav'},{word},ad")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,audio,text,section\n" + "\n".join(rows), encoding="utf-8")
    return manifest


@pytest.fixture
def sample_waveform():
    sr = 16000
    t = torch.linspace(0, 0.2, int(0.2 * sr))
    wav = 0.3 * torch.sin(2 * math.pi * 440 * t)
    return wav.unsqueeze(0), sr