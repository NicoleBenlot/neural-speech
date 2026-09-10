"""Audio preprocessing tests."""

import pytest

from src.data.audio import load_audio, load_audio_raw
from tests.conftest import _write_wav


def test_load_audio_mono_float(tmp_path):
    p = tmp_path / "tone.wav"
    _write_wav(p, duration=0.2)
    wave, sr = load_audio_raw(str(p), target_sr=16000)
    assert sr == 16000
    assert wave.dtype == pytest.importorskip("torch").float32
    assert wave.shape[0] == 1  # mono
    assert wave.abs().max().item() <= 1.0


def test_load_audio_resample(tmp_path):
    p = tmp_path / "tone.wav"
    _write_wav(p, duration=0.2)
    wave, sr = load_audio_raw(str(p), target_sr=8000)
    assert sr == 8000


def test_load_audio_missing_file():
    import torchaudio
    with pytest.raises(Exception):
        load_audio("does-not-exist.wav")