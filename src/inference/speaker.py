"""Pluggable text-to-speech output for the inference pipeline.

Backends
--------
- piper  : offline neural TTS (piper-tts + an .onnx voice in ``voice_dir``).
           The default offline backend; best on-device quality.
- system : OS voices via pyttsx3 (SAPI on Windows). Used automatically when
           piper (or its voice) is unavailable. Zero downloads.
- online : reserved for a hosted neural backend (edge-tts / Azure Speech).
           Kept as a registered name so the API/CLI plumbing is ready; swap
           the implementation when it lands.
- none   : disable TTS entirely.

``Speaker.synthesize`` returns WAV bytes (safe to stream to Android / other
clients); ``speak`` additionally plays them on the local default device.
"""

from __future__ import annotations

import io
import logging
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import piper
except ImportError:  # pragma: no cover - optional dependency
    piper = None

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None

DEFAULT_VOICE_DIR = "voices"
DEFAULT_VOICE = "en_US-lessac-medium"

BACKENDS = ("auto", "piper", "system", "online", "none")


@dataclass
class SpeechResult:
    """Outcome of one synthesis call."""

    backend: str
    text: str
    wav_bytes: Optional[bytes]  # 16-bit PCM mono WAV; None if the backend plays natively
    sample_rate: Optional[int] = None


def _chunk_audio(chunk) -> bytes:
    """Best-effort PCM bytes for a piper AudioChunk across piper versions."""
    if hasattr(chunk, "audio_int16_bytes"):
        return chunk.audio_int16_bytes
    if hasattr(chunk, "audio") and isinstance(chunk.audio, bytes):
        return chunk.audio
    raise RuntimeError("Unsupported piper AudioChunk shape")


def _merge_piper_chunks(chunks) -> tuple[bytes, int]:
    """Combine piper sentence chunks into one WAV and return (wav, sample_rate).

    piper 1.8 yields raw int16 PCM per sentence (no WAV header); older
    releases emitted full RIFF blobs. Both are handled here.
    """
    sample_rate = 22050
    sample_width = 2
    channels = 1
    pieces: list[bytes] = []
    for chunk in chunks:
        raw = _chunk_audio(chunk)
        if raw[:4] == b"RIFF":
            return raw, getattr(chunk, "sample_rate", sample_rate)
        pieces.append(raw)
        sample_rate = getattr(chunk, "sample_rate", sample_rate)
        sample_width = getattr(chunk, "sample_width", sample_width)
        channels = getattr(chunk, "sample_channels", channels)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(b"".join(pieces))
    return buf.getvalue(), sample_rate


def _play_wav(wav_bytes: bytes, sample_rate: Optional[int] = None) -> None:
    """Play WAV bytes on the default audio output device."""
    import sounddevice as sd
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if sample_rate is not None and sr == 0:
        sr = sample_rate
    sd.play(data, sr)
    sd.wait()


class Speaker:
    """Synthesize speech from text using a pluggable backend.

    Args:
        backend: One of ``auto`` (default), ``piper``, ``system``, ``online``,
            ``none``.
        voice_dir: Directory holding downloaded ``<voice>.onnx`` voices.
        voice: Piper voice name (without the ``.onnx`` suffix).
    """

    def __init__(
        self,
        backend: str = "auto",
        voice_dir: str = DEFAULT_VOICE_DIR,
        voice: str = DEFAULT_VOICE,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"Unknown TTS backend: {backend!r} (choose from {BACKENDS})")
        self.backend = backend
        self.voice_dir = Path(voice_dir)
        self.voice = voice
        self.voice_path = self.voice_dir / f"{voice}.onnx"
        self._piper_voice = None
        self._engine = None

    @property
    def resolved_backend(self) -> str:
        """The backend that will actually run, or '' when nothing is available."""
        if self.backend == "auto":
            if piper is not None and self.voice_path.is_file():
                return "piper"
            if pyttsx3 is not None:
                return "system"
            return ""
        if self.backend == "none":
            return ""
        if self.backend == "piper":
            return "piper" if piper is not None and self.voice_path.is_file() else ""
        if self.backend == "system":
            return "system" if pyttsx3 is not None else ""
        if self.backend == "online":
            return "online"
        return ""

    def synthesize(self, text: str) -> SpeechResult:
        """Generate speech for ``text``.

        Returns a :class:`SpeechResult`. ``wav_bytes`` is populated for the
        piper backend; the system backend plays natively and returns None.
        """
        backend = self.resolved_backend
        if backend == "piper":
            return self._synthesize_piper(text)
        if backend == "system":
            return self._synthesize_system(text)
        if backend == "online":
            raise NotImplementedError(
                "The 'online' TTS backend is reserved for a hosted service "
                "(edge-tts / Azure Speech) and is not implemented yet."
            )
        raise RuntimeError(
            f"No TTS backend available. Install piper-tts and place '{self.voice}.onnx' "
            f"in {self.voice_dir}, or install pyttsx3 for OS voices."
        )

    def speak(self, text: str) -> SpeechResult:
        """Synthesize ``text`` and play it on the local audio device."""
        result = self.synthesize(text)
        if result.wav_bytes is not None:
            _play_wav(result.wav_bytes, result.sample_rate)
        return result

    def _synthesize_piper(self, text: str) -> SpeechResult:
        if self._piper_voice is None:
            self._piper_voice = piper.PiperVoice.load(str(self.voice_path))
            logger.info("Piper voice loaded: %s", self.voice_path)
        chunks = list(
            self._piper_voice.synthesize(text, syn_config=piper.SynthesisConfig())
        )
        wav, sample_rate = _merge_piper_chunks(chunks)
        return SpeechResult(backend="piper", text=text, wav_bytes=wav, sample_rate=sample_rate)

    def _synthesize_system(self, text: str) -> SpeechResult:
        if self._engine is None:
            self._engine = pyttsx3.init()
            logger.info("Initialized OS TTS engine (pyttsx3)")
        if not getattr(self._engine, "say", None):
            raise RuntimeError("OS TTS engine has no 'say' API")
        self._engine.say(text)
        self._engine.runAndWait()
        return SpeechResult(backend="system", text=text, wav_bytes=None)


def create_speaker(
    backend: str = "auto",
    voice_dir: str = DEFAULT_VOICE_DIR,
    voice: str = DEFAULT_VOICE,
) -> Optional[Speaker]:
    """Build a configured :class:`Speaker`.

    Returns ``None`` when ``backend="none"`` or nothing is available (auto).
    Explicit requests (piper/system/online) that cannot be satisfied raise
    ``RuntimeError`` so a requested backend is never silently replaced.
    """
    if backend == "none":
        return None
    speaker = Speaker(backend=backend, voice_dir=voice_dir, voice=voice)
    if speaker.resolved_backend:
        return speaker
    if backend == "auto":
        return None
    if backend == "online":
        return speaker
    raise RuntimeError(
        f"TTS backend '{backend}' is unavailable "
        f"(voice: {speaker.voice_path if backend == 'piper' else 'pyttsx3'}). "
        "Install the backend or provide a valid voice."
    )


__all__ = [
    "BACKENDS",
    "DEFAULT_VOICE",
    "DEFAULT_VOICE_DIR",
    "SpeechResult",
    "Speaker",
    "create_speaker",
]