"""API tests with fake models — no TensorFlow, Hugging Face or internet needed (FFmpeg is)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.app import Models, TranscriptionUnavailable, app
from backend.features import EMOTIONS


class FakeModels:
    def __init__(self):
        self.transcript = "great job everyone"
        self.sentiment_result = ("POSITIVE", 0.98)
        self.voice = "angry"
        self.asr_down = False
        self.acoustic_calls = []

    def acoustic(self, windows):
        self.acoustic_calls.append(windows.shape)
        probs = np.full((len(windows), len(EMOTIONS)), 0.02)
        probs[:, EMOTIONS.index(self.voice)] = 0.86
        return probs

    def sentiment(self, text):
        return self.sentiment_result

    def transcribe(self, path):
        if self.asr_down:
            raise TranscriptionUnavailable("network down")
        return self.transcript


@pytest.fixture
def fake():
    return FakeModels()


@pytest.fixture
def client(fake):
    app.state.models = Models(acoustic=fake.acoustic, sentiment=fake.sentiment, transcribe=fake.transcribe)
    with TestClient(app) as test_client:
        yield test_client
    app.state.models = None


def post(client, data, name="clip.wav"):
    return client.post("/analyze", files={"file": (name, data, "audio/wav")})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "models_loaded": True}


def test_sarcasm_end_to_end(client, fake, wav_bytes):
    res = post(client, wav_bytes)
    assert res.status_code == 200
    body = res.json()
    assert body["fusion_decision"] == "Sarcasm or Passive-Aggressive Tone Detected"
    assert body["fusion_mode"] == "multimodal"
    assert body["text_intent"] == {"label": "POSITIVE", "confidence": 0.98, "raw_label": "POSITIVE"}
    assert body["voice_emotion"]["label"] == "angry"
    assert set(body["voice_emotion"]["probabilities"]) == set(EMOTIONS)
    assert body["asr_status"] == "ok"
    assert fake.acoustic_calls == [(1, 150, 40)]


def test_no_speech_still_returns_voice_emotion(client, fake, wav_bytes):
    fake.transcript = None
    body = post(client, wav_bytes).json()
    assert body["asr_status"] == "no_speech"
    assert body["text_intent"]["label"] == "N/A"
    assert body["voice_emotion"]["label"] == "angry"
    assert body["fusion_mode"] == "voice_only"


def test_asr_outage_degrades_gracefully(client, fake, wav_bytes):
    fake.asr_down = True
    res = post(client, wav_bytes)
    assert res.status_code == 200
    assert res.json()["asr_status"] == "unavailable"


def test_rejects_unsupported_extension(client, wav_bytes):
    assert post(client, wav_bytes, name="notes.txt").status_code == 415


def test_rejects_undecodable_audio(client):
    assert post(client, b"definitely not audio" * 100).status_code == 422


def test_rejects_empty_upload(client):
    assert post(client, b"").status_code == 422


def test_rejects_too_short_audio(client, make_wav):
    assert post(client, make_wav(seconds=0.1)).status_code == 422


def test_rejects_oversized_upload(client, monkeypatch, wav_bytes):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1000)
    assert post(client, wav_bytes).status_code == 413


def test_long_clip_uses_multiple_windows(client, fake, make_wav):
    post(client, make_wav(seconds=10))
    assert fake.acoustic_calls[0][0] > 1
