"""Tests for the pluggable TTS speaker (offline piper / system / online stub)."""

import io
import types
import wave

import pytest

from src.inference import speaker as speaker_mod
from src.inference.speaker import Speaker, create_speaker


def _decode_wav(wav_bytes):
    """Return (frame_rate, channels, sample_width, pcm) from a WAV blob."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())


class _Chunk:
    def __init__(self, data: bytes):
        self.audio_int16_bytes = data


class _Voice:
    config = types.SimpleNamespace(sample_rate=22050)

    def synthesize(self, text, syn_config=None):
        return [_Chunk(b"AAA"), _Chunk(b"BBB")]


class _FakePiper:
    class PiperVoice:
        @staticmethod
        def load(*args, **kwargs):
            return _Voice()

    class SynthesisConfig:
        def __init__(self, *args, **kwargs):
            pass


class _FakeEngine:
    def __init__(self):
        self.texts = []

    def say(self, text):
        self.texts.append(text)

    def runAndWait(self):
        pass


class _FakePyttsx3:
    @staticmethod
    def init():
        return _FakeEngine()


def _make_voice(speaker_mod, tmp_path):
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    voice = voice_dir / "en_US-lessac-medium.onnx"
    voice.write_bytes(b"fake-onnx")
    return voice_dir


def test_create_none_backend_returns_none(monkeypatch):
    monkeypatch.setattr(speaker_mod, "piper", _FakePiper)
    assert create_speaker(backend="none") is None


def test_auto_picks_piper_when_voice_present(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker_mod, "piper", _FakePiper)
    voice_dir = _make_voice(speaker_mod, tmp_path)
    speaker = create_speaker(backend="auto", voice_dir=str(voice_dir))
    assert speaker is not None
    assert speaker.resolved_backend == "piper"


def test_auto_falls_back_to_system(monkeypatch):
    monkeypatch.setattr(speaker_mod, "piper", None)
    monkeypatch.setattr(speaker_mod, "pyttsx3", _FakePyttsx3)
    speaker = create_speaker(backend="auto")
    assert speaker.resolved_backend == "system"


def test_auto_returns_none_when_nothing_available(monkeypatch):
    monkeypatch.setattr(speaker_mod, "piper", None)
    monkeypatch.setattr(speaker_mod, "pyttsx3", None)
    assert create_speaker(backend="auto") is None


def test_explicit_piper_without_voice_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker_mod, "piper", _FakePiper)
    with pytest.raises(RuntimeError, match="unavailable"):
        create_speaker(backend="piper", voice_dir=str(tmp_path))


def test_explicit_system_without_pyttsx3_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker_mod, "piper", None)
    monkeypatch.setattr(speaker_mod, "pyttsx3", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        create_speaker(backend="system")


def test_online_registers_but_not_implemented(tmp_path, monkeypatch):
    speaker = create_speaker(backend="online")
    assert speaker is not None
    assert speaker.resolved_backend == "online"
    with pytest.raises(NotImplementedError, match="not implemented"):
        speaker.synthesize("hello")


def test_piper_synthesize_wraps_raw_pcm_into_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker_mod, "piper", _FakePiper)
    voice_dir = _make_voice(speaker_mod, tmp_path)
    speaker = Speaker(backend="piper", voice_dir=str(voice_dir))
    result = speaker.synthesize("adlaw")
    assert result.backend == "piper"
    assert result.text == "adlaw"
    assert result.wav_bytes[:4] == b"RIFF"
    sr, channels, width, pcm = _decode_wav(result.wav_bytes)
    assert (sr, channels, width, pcm) == (22050, 1, 2, b"AAABBB")


def test_system_synthesize_uses_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker_mod, "piper", None)
    monkeypatch.setattr(speaker_mod, "pyttsx3", _FakePyttsx3)
    speaker = Speaker(backend="system")
    result = speaker.synthesize("ado")
    assert result.backend == "system"
    assert result.wav_bytes is None
    assert speaker._engine.texts == ["ado"]


def test_speak_plays_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker_mod, "piper", _FakePiper)
    played = {}

    def fake_play(wav_bytes, sample_rate=None):
        played["wav"] = wav_bytes

    monkeypatch.setattr(speaker_mod, "_play_wav", fake_play)
    voice_dir = _make_voice(speaker_mod, tmp_path)
    speaker = Speaker(backend="piper", voice_dir=str(voice_dir))
    result = speaker.speak("ako")
    assert result.wav_bytes[:4] == b"RIFF"
    _, _, _, pcm = _decode_wav(result.wav_bytes)
    assert pcm == b"AAABBB"
    assert played["wav"] == result.wav_bytes