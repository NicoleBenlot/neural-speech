"""FastAPI application entry point.

uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from src.api.routes import health, stt, tts
from src.inference.speaker import create_speaker
from src.inference.transcriber import DEFAULT_CHECKPOINT, Transcriber

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="neural-speech", version="0.1.0")

app.include_router(health.router)
app.include_router(stt.router)
app.include_router(tts.router)


def load_model_sync():
    """Load the transcriber once at startup.

    FastAPI is an inference layer only. The checkpoint path is resolved via
    STT_CHECKPOINT; the app fails clearly if it does not exist.
    """
    checkpoint = os.environ.get("STT_CHECKPOINT", DEFAULT_CHECKPOINT)
    device = os.environ.get("STT_DEVICE", "auto")

    try:
        transcriber = Transcriber(checkpoint=checkpoint, device=device)
    except FileNotFoundError as exc:
        logger.error("Checkpoint not found: %s (%s)", checkpoint, exc)
        raise SystemExit(f"Configured checkpoint does not exist: {checkpoint}")
    except Exception as exc:
        logger.error("Failed to load model: %s", exc, exc_info=True)
        raise SystemExit(f"Failed to load model from checkpoint {checkpoint}: {exc}")

    stt.set_transcriber(transcriber)
    logger.info("Transcriber loaded from %s on device %s", checkpoint, device)


def load_speaker_sync():
    """Load the TTS speaker once at startup.

    TTS backend comes from TTS_BACKEND (auto/piper/system/online/none). The
    server still starts if TTS is unavailable - the /tts route then returns
    503 so clients can fall back to their own speech output.
    """
    backend = os.environ.get("TTS_BACKEND", "auto")
    voice_dir = os.environ.get("TTS_VOICE_DIR", "voices")

    try:
        instance = create_speaker(backend=backend, voice_dir=voice_dir)
    except RuntimeError as exc:
        logger.warning("TTS disabled: %s", exc)
        instance = None

    tts.set_speaker(instance)
    if instance is not None:
        logger.info("TTS enabled (%s backend)", instance.resolved_backend)


@app.on_event("startup")
async def load_model():
    load_model_sync()
    load_speaker_sync()