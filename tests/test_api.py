"""API tests with fake models — no TensorFlow, Hugging Face or internet needed (FFmpeg is)."""

import subprocess

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
        self.acoustic_fails = False
        self.acoustic_calls = []

    def acoustic(self, windows):
        if self.acoustic_fails:
            raise RuntimeError("model crashed")
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


ORIGIN = "http://127.0.0.1:5500"  # where the frontend is served from


def post(client, data, name="clip.wav"):
    return client.post("/analyze", files={"file": (name, data, "audio/wav")}, headers={"Origin": ORIGIN})


def assert_browser_readable(res):
    """The frontend is cross-origin: without CORS headers the browser hides the error."""
    assert res.headers.get("access-control-allow-origin") in ("*", ORIGIN)
    assert res.json()["detail"]


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
    assert body["transcript"] is None
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


def test_rejects_too_long_audio(client, monkeypatch, make_wav):
    monkeypatch.setattr(app_module, "MAX_AUDIO_SECONDS", 2)
    res = post(client, make_wav(seconds=5))
    assert res.status_code == 422
    assert "too long" in res.json()["detail"]


def test_rejects_silence(client, fake, make_wav):
    res = post(client, make_wav(amplitude=0.0))
    assert res.status_code == 422
    assert "No sound" in res.json()["detail"]
    assert fake.acoustic_calls == []


def test_oversized_upload_rejected_from_content_length(client, monkeypatch, wav_bytes):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1000)
    res = post(client, wav_bytes)
    assert res.status_code == 413
    assert_browser_readable(res)


def test_oversized_upload_rejected_while_streaming(client, monkeypatch, wav_bytes):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(app_module, "MULTIPART_OVERHEAD_BYTES", 10**9)  # get past the header check
    res = post(client, wav_bytes)
    assert res.status_code == 413
    assert_browser_readable(res)


def test_internal_error_is_readable_by_the_browser(client, fake, wav_bytes):
    fake.acoustic_fails = True
    res = post(client, wav_bytes)
    assert res.status_code == 500
    assert_browser_readable(res)


def test_long_clip_uses_multiple_windows(client, fake, make_wav):
    post(client, make_wav(seconds=10))
    assert fake.acoustic_calls[0][0] > 1


def test_browser_webm_recording_is_decoded(client, fake, tmp_path, wav_bytes):
    """Browsers record WebM/Opus, which goes through ffprobe + ffmpeg rather than pydub's WAV reader."""
    wav, webm = tmp_path / "clip.wav", tmp_path / "clip.webm"
    wav.write_bytes(wav_bytes)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(wav), "-c:a", "libopus", str(webm)], check=True)
    res = post(client, webm.read_bytes(), name="live_recording.webm")
    assert res.status_code == 200
    assert res.json()["duration_seconds"] == pytest.approx(2.0, abs=0.1)
