"""EmoVoice AI — FastAPI backend for multimodal speech emotion recognition.

Pipeline for every request:
    upload -> FFmpeg decode -> two normalized WAV copies
        acoustic copy (mono, native rate)  -> MFCC windows -> LSTM -> voice emotion
        ASR copy (mono, 16 kHz, normalized) -> Google ASR -> DistilBERT -> text intent
    -> rule-based fusion -> JSON response

The two paths are independent: if no speech is recognized (or the ASR service
is unreachable) the voice path still produces a result.

Run from the repository root:
    uvicorn backend.app:app --reload
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence TensorFlow C++ startup noise

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydub import AudioSegment

try:  # package import: `uvicorn backend.app:app` from the repo root
    from .features import (EMOTIONS, MAX_LEN, N_MFCC, RES_TYPE, SAMPLE_RATE,
                           load_audio, mfcc_frames, sliding_windows)
    from .fusion import (DEFAULT_TEXT_NEUTRAL_THRESHOLD, DEFAULT_VOICE_THRESHOLD,
                         classify_text_intent, fuse)
except ImportError:  # script import: `cd backend && uvicorn app:app`
    from features import (EMOTIONS, MAX_LEN, N_MFCC, RES_TYPE, SAMPLE_RATE,
                          load_audio, mfcc_frames, sliding_windows)
    from fusion import (DEFAULT_TEXT_NEUTRAL_THRESHOLD, DEFAULT_VOICE_THRESHOLD,
                        classify_text_intent, fuse)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("emovoice")

# ----------------------------------------------------------------------------
# Configuration (override any value with an environment variable)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("EMOVOICE_MODEL_PATH", str(ROOT / "models" / "ser_lstm_model.keras")))
SENTIMENT_MODEL = os.getenv("EMOVOICE_SENTIMENT_MODEL",
                            "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("EMOVOICE_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
MAX_UPLOAD_BYTES = int(float(os.getenv("EMOVOICE_MAX_UPLOAD_MB", "10")) * 1024 * 1024)
MAX_AUDIO_SECONDS = float(os.getenv("EMOVOICE_MAX_AUDIO_SECONDS", "30"))
MIN_AUDIO_SECONDS = float(os.getenv("EMOVOICE_MIN_AUDIO_SECONDS", "0.3"))
VOICE_THRESHOLD = float(os.getenv("EMOVOICE_VOICE_THRESHOLD", str(DEFAULT_VOICE_THRESHOLD)))
TEXT_NEUTRAL_THRESHOLD = float(os.getenv("EMOVOICE_TEXT_NEUTRAL_THRESHOLD", str(DEFAULT_TEXT_NEUTRAL_THRESHOLD)))
ASR_LANGUAGE = os.getenv("EMOVOICE_ASR_LANGUAGE", "en-US")
ASR_TIMEOUT_SECONDS = float(os.getenv("EMOVOICE_ASR_TIMEOUT", "10"))
ALLOWED_EXTENSIONS = {".webm", ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4"}


# ----------------------------------------------------------------------------
# Model container — real models in production, fakes in tests
# ----------------------------------------------------------------------------
class TranscriptionUnavailable(Exception):
    """The speech-recognition service could not be reached or failed."""


@dataclass
class Models:
    acoustic: Callable[[np.ndarray], np.ndarray]          # (n, 150, 40) -> (n, 8) probabilities
    sentiment: Callable[[str], Tuple[str, float]]         # text -> (label, score)
    transcribe: Callable[[str], Optional[str]]            # wav path -> text, or None if no speech


def load_acoustic_model() -> Callable[[np.ndarray], np.ndarray]:
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Acoustic model not found at {MODEL_PATH}. "
                           "Train it with `python train_model.py` or set EMOVOICE_MODEL_PATH.")
    import keras
    import tensorflow as tf
    model = keras.models.load_model(MODEL_PATH, compile=False)
    logger.info("Loaded acoustic model from %s", MODEL_PATH)

    # Compile inference into a TF graph once. Calling the Keras model eagerly
    # steps through the LSTM in Python (~1 s per request); the traced graph runs
    # in ~20 ms with identical outputs, and is safe to call from many threads.
    @tf.function(input_signature=[tf.TensorSpec([None, MAX_LEN, N_MFCC], tf.float32)])
    def infer(windows):
        return model(windows, training=False)

    return lambda windows: infer(tf.constant(windows, dtype=tf.float32)).numpy()


def warm_up(models: Models) -> None:
    """Trigger one-off compilation (TF graph tracing, numba JIT in the resampler)
    at startup so the first real request is as fast as every later one."""
    started = time.perf_counter()
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    librosa.resample(silence, orig_sr=48000, target_sr=SAMPLE_RATE, res_type=RES_TYPE)
    models.acoustic(sliding_windows(mfcc_frames(silence)).astype(np.float32))
    logger.info("Warm-up finished in %d ms", (time.perf_counter() - started) * 1000)


def load_sentiment_model() -> Callable[[str], Tuple[str, float]]:
    from transformers import pipeline
    classifier = pipeline("text-classification", model=SENTIMENT_MODEL)
    logger.info("Loaded sentiment model %s", SENTIMENT_MODEL)

    def analyze(text: str) -> Tuple[str, float]:
        result = classifier(text, truncation=True)[0]
        return result["label"], float(result["score"])
    return analyze


def make_google_transcriber() -> Callable[[str], Optional[str]]:
    import speech_recognition as sr

    def transcribe(wav_path: str) -> Optional[str]:
        recognizer = sr.Recognizer()
        recognizer.operation_timeout = ASR_TIMEOUT_SECONDS
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language=ASR_LANGUAGE) or None
        except sr.UnknownValueError:
            return None  # audio decoded fine but contained no recognizable speech
        except sr.RequestError as exc:
            raise TranscriptionUnavailable(str(exc)) from exc
    return transcribe


@asynccontextmanager
async def lifespan(app: FastAPI):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg was not found on PATH. Install it and restart the server.")
    if getattr(app.state, "models", None) is None:  # tests inject fakes beforehand
        app.state.models = Models(acoustic=load_acoustic_model(),
                                  sentiment=load_sentiment_model(),
                                  transcribe=make_google_transcriber())
        warm_up(app.state.models)
    yield


app = FastAPI(title="EmoVoice AI — Multimodal Emotion Fusion API", version="2.0.0", lifespan=lifespan)
# No cookies or auth are used, so credentials stay disabled; a wildcard origin
# combined with credentials would let any website make authenticated calls.
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


# ----------------------------------------------------------------------------
# Response schema (also documents the API at /docs)
# ----------------------------------------------------------------------------
class TextIntentOut(BaseModel):
    label: str              # POSITIVE | NEGATIVE | NEUTRAL | N/A
    confidence: float
    raw_label: Optional[str] = None


class VoiceEmotionOut(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]


class AnalysisOut(BaseModel):
    transcript: str
    text_intent: TextIntentOut
    voice_emotion: VoiceEmotionOut
    fusion_decision: str
    fusion_mode: str        # multimodal | text_only | voice_only | none
    asr_status: str         # ok | no_speech | unavailable
    duration_seconds: float
    processing_ms: int


# ----------------------------------------------------------------------------
# Pipeline steps
# ----------------------------------------------------------------------------
def save_upload(upload: UploadFile, destination: Path) -> None:
    """Copy the upload to disk, rejecting anything over MAX_UPLOAD_BYTES."""
    if upload.size is not None and upload.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    written = 0
    with destination.open("wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
            out.write(chunk)
    if written == 0:
        raise HTTPException(422, "Uploaded file is empty.")


def decode_audio(path: Path) -> AudioSegment:
    try:
        return AudioSegment.from_file(str(path))
    except Exception as exc:  # pydub raises several exception types for corrupt input
        logger.info("Could not decode %s: %s", path.name, exc)
        raise HTTPException(422, "Could not decode the audio file. Is it a valid audio recording?") from exc


def predict_voice_emotion(models: Models, wav_path: Path) -> Tuple[str, float, Dict[str, float]]:
    windows = sliding_windows(mfcc_frames(load_audio(wav_path))).astype(np.float32)
    probabilities = models.acoustic(windows).mean(axis=0)  # average over windows of long clips
    best = int(np.argmax(probabilities))
    distribution = {emotion: round(float(p), 4) for emotion, p in zip(EMOTIONS, probabilities)}
    return EMOTIONS[best], float(probabilities[best]), distribution


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "models_loaded": getattr(request.app.state, "models", None) is not None}


# Declared as a plain `def` on purpose: model inference and the ASR call are
# blocking, so FastAPI runs this handler in its thread pool and the event loop
# stays free to accept concurrent requests.
@app.post("/analyze", response_model=AnalysisOut)
def analyze_audio(request: Request, file: UploadFile = File(...)) -> AnalysisOut:
    started = time.perf_counter()
    models: Models = request.app.state.models

    extension = Path(file.filename or "recording.webm").suffix.lower() or ".webm"
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{extension}'. "
                                 f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.")

    # TemporaryDirectory removes every intermediate file, even on errors.
    with tempfile.TemporaryDirectory(prefix="emovoice_", ignore_cleanup_errors=True) as tmp:
        tmp_dir = Path(tmp)
        raw_path = tmp_dir / f"input{extension}"
        save_upload(file, raw_path)

        segment = decode_audio(raw_path)
        duration = len(segment) / 1000.0
        if duration < MIN_AUDIO_SECONDS:
            raise HTTPException(422, f"Recording too short ({duration:.2f}s). Please speak for at least a second.")
        if duration > MAX_AUDIO_SECONDS:
            raise HTTPException(422, f"Recording too long ({duration:.0f}s, max {MAX_AUDIO_SECONDS:.0f}s).")

        mono = segment.set_channels(1)
        # Acoustic copy keeps the native sample rate: the model was trained on
        # 48 kHz RAVDESS audio resampled to 22.05 kHz, so downsampling to 16 kHz
        # first would strip the 8-11 kHz band the model learned from.
        acoustic_path = tmp_dir / "acoustic.wav"
        mono.export(str(acoustic_path), format="wav")
        # ASR copy: 16 kHz, peak-normalized PCM — what speech recognizers expect.
        asr_path = tmp_dir / "asr.wav"
        mono.set_frame_rate(16000).normalize(headroom=0.1).export(str(asr_path), format="wav", codec="pcm_s16le")

        voice_label, voice_score, voice_distribution = predict_voice_emotion(models, acoustic_path)

        try:
            transcript = models.transcribe(str(asr_path))
            asr_status = "ok" if transcript else "no_speech"
        except TranscriptionUnavailable as exc:
            logger.warning("Speech recognition unavailable: %s", exc)
            transcript, asr_status = None, "unavailable"

    raw_label, text_score = (models.sentiment(transcript) if transcript else (None, 0.0))
    text_intent = classify_text_intent(raw_label, text_score, TEXT_NEUTRAL_THRESHOLD)
    fusion = fuse(text_intent, voice_label, voice_score, VOICE_THRESHOLD)

    if transcript is None:
        transcript = ("Speech recognition service unavailable — check your internet connection."
                      if asr_status == "unavailable" else "Could not understand the speech in this audio.")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info("analyze: %.1fs audio | text=%s(%.2f) voice=%s(%.2f) -> %s [%s] in %d ms",
                duration, text_intent, text_score, voice_label, voice_score,
                fusion.decision, fusion.mode, elapsed_ms)

    return AnalysisOut(
        transcript=transcript,
        text_intent=TextIntentOut(label=text_intent.upper() if text_intent else "N/A",
                                  confidence=round(text_score, 2), raw_label=raw_label),
        voice_emotion=VoiceEmotionOut(label=voice_label, confidence=round(voice_score, 4),
                                      probabilities=voice_distribution),
        fusion_decision=fusion.decision,
        fusion_mode=fusion.mode,
        asr_status=asr_status,
        duration_seconds=round(duration, 2),
        processing_ms=elapsed_ms,
    )


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Check the server logs."})


if __name__ == "__main__":
    import uvicorn
    # Bind to localhost by default; set EMOVOICE_HOST=0.0.0.0 to expose on the network.
    uvicorn.run(app, host=os.getenv("EMOVOICE_HOST", "127.0.0.1"), port=int(os.getenv("EMOVOICE_PORT", "8000")))
