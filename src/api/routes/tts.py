"""TTS route - returns synthesized speech audio for clients.

This is the integration point that stays stable whether TTS runs offline
(piper / system) or through a hosted backend (``online``) later: the route
always returns audio bytes, and the backend swap happens in startup config.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from src.inference.speaker import Speaker

logger = logging.getLogger(__name__)

router = APIRouter()
speaker: Optional[Speaker] = None  # set at startup in main.py


def set_speaker(instance: Optional[Speaker]):
    """Inject the shared speaker instance (loaded once at startup)."""
    global speaker
    speaker = instance


class TTSRequest(BaseModel):
    text: str


@router.post("/tts")
async def tts(req: TTSRequest):
    """Synthesize speech for ``text`` and return WAV audio (16-bit PCM mono)."""
    global speaker

    if speaker is None:
        raise HTTPException(status_code=503, detail="TTS not configured")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    try:
        result = speaker.synthesize(text)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

    if not result.wav_bytes:
        raise HTTPException(
            status_code=501,
            detail="Current backend produces no streamable audio "
            "(use a piper voice or an online backend)",
        )

    return Response(content=result.wav_bytes, media_type="audio/wav")