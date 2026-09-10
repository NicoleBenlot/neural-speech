"""STT inference route."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.inference.transcriber import Transcriber

logger = logging.getLogger(__name__)

router = APIRouter()
transcriber: Optional[Transcriber] = None  # set at startup in main.py


def set_transcriber(instance: Transcriber):
    """Inject the shared transcriber instance (loaded once at startup)."""
    global transcriber
    transcriber = instance


ALLOWED_EXT = {".opus", ".ogg", ".wav", ".flac", ".mp3"}


@router.post("/stt")
async def stt(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file."""
    global transcriber

    if transcriber is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(status_code=415, detail=f"Unsupported audio format: {suffix}")

    try:
        data = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            text = transcriber.transcribe(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return {"text": text}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Inference error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc