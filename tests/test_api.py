"""API route tests using a stubbed transcriber + synthetic audio."""

import io
import math

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import health, stt, tts
from tests.conftest import _write_wav


class FakeTranscriber:
    def __init__(self, text="adlaw"):
        self.text = text

    def transcribe(self, path):
        return self.text


@pytest.fixture
def client(tmp_path, monkeypatch):
    app = FastAPI()
    app.include_router(health.router)
    app.include_router(stt.router)
    monkeypatch.setattr(stt, "transcriber", FakeTranscriber("adlaw"))
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_stt_with_audio(client, tmp_path):
    wav_path = tmp_path / "sample.wav"
    _write_wav(wav_path)
    with open(wav_path, "rb") as fh:
        resp = client.post(
            "/stt",
            files={"file": ("sample.wav", fh, "audio/wav")},
        )
    assert resp.status_code == 200
    assert resp.json() == {"text": "adlaw"}


def test_stt_rejects_unknown_format(client):
    resp = client.post(
        "/stt",
        files={"file": ("notes.txt", io.BytesIO(b"hi"), "text/plain")},
    )
    assert resp.status_code == 415


def test_api_fails_clearly_on_missing_checkpoint(monkeypatch):
    from src.api import main as api_main

    monkeypatch.setenv("STT_CHECKPOINT", "checkpoints/does_not_exist")
    with pytest.raises(SystemExit):
        api_main.load_model_sync()


class FakeSpeaker:
    def __init__(self, wav_bytes=b"RIFFfakewav"):
        self.wav_bytes = wav_bytes

    def synthesize(self, text):
        return type(
            "Result",
            (),
            {"backend": "piper", "text": text, "wav_bytes": self.wav_bytes},
        )


@pytest.fixture
def tts_client(monkeypatch):
    app = FastAPI()
    app.include_router(tts.router)
    monkeypatch.setattr(tts, "speaker", FakeSpeaker())
    return TestClient(app)


def test_tts_returns_wav(tts_client):
    resp = tts_client.post("/tts", json={"text": "adlaw"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content == b"RIFFfakewav"


def test_tts_rejects_empty_text(tts_client):
    resp = tts_client.post("/tts", json={"text": "   "})
    assert resp.status_code == 400


def test_tts_503_when_unconfigured(tts_client, monkeypatch):
    monkeypatch.setattr(tts, "speaker", None)
    resp = tts_client.post("/tts", json={"text": "hi"})
    assert resp.status_code == 503


def test_tts_501_when_backend_has_no_audio(tts_client, monkeypatch):
    monkeypatch.setattr(tts, "speaker", FakeSpeaker(wav_bytes=None))
    resp = tts_client.post("/tts", json={"text": "hi"})
    assert resp.status_code == 501