"""API route tests using a stubbed transcriber + synthetic audio."""

import io
import math

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import health, stt
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