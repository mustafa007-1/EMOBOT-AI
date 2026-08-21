from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import uvicorn
import speech_recognition as sr
from pydub import AudioSegment
import os
import tempfile
import librosa
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI(title="Multimodal Emotion Fusion API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

sentiment_analyzer = pipeline("sentiment-analysis")

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ser_lstm_model.keras')
acoustic_model = load_model(model_path)

ACOUSTIC_EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def extract_mfcc_from_file(file_path, max_len=150):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if mfccs.shape[1] < max_len:
            pad_width = max_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_len]
        return mfccs.T.reshape(1, max_len, 40)
    except Exception as e:
        print(f"MFCC error: {e}")
        return None


def perform_fusion(text_intent, text_score, acoustic_emotion, voice_score=0.0):
    """Improved Fusion Logic"""
    if text_intent in ["N/A", "UNKNOWN"] or acoustic_emotion == "unknown":
        return "Insufficient data for fusion analysis."

    text_intent = text_intent.lower()
    reliable_voice = voice_score >= 0.40

    # ==================== NEGATIVE TEXT ====================
    if text_intent == "negative":
        if reliable_voice and acoustic_emotion in ["angry", "disgust"]:
            return "High Hostility • Anger Detected"
        elif reliable_voice and acoustic_emotion in ["sad", "fearful"]:
            return "Deep Emotional Distress"
        elif reliable_voice and acoustic_emotion in ["calm", "neutral"]:
            return "Calm Negative / Disappointed Tone"          # Improved case
        else:
            return "Negative Intent"

    # ==================== POSITIVE TEXT ====================
    elif text_intent == "positive":
        if reliable_voice and acoustic_emotion in ["angry", "disgust", "fearful"]:
            return "Sarcasm or Passive-Aggressive Tone Detected"
        elif reliable_voice and acoustic_emotion == "sad":
            return "Bittersweet / Melancholic Positivity"
        elif reliable_voice and acoustic_emotion in ["calm", "neutral"]:
            return "Calm Positive / Content"
        else:
            return "Genuine Positive Emotion"

    # ==================== NEUTRAL TEXT ====================
    else:
        if reliable_voice and acoustic_emotion in ["angry", "disgust"]:
            return "Hidden Anger / Frustration Detected"
        elif reliable_voice and acoustic_emotion in ["sad", "fearful"]:
            return "Underlying Sadness or Anxiety"
        elif reliable_voice and acoustic_emotion in ["happy", "surprised"]:
            return "Subtle Positive Emotion"
        else:
            return "Neutral Emotional State"

    return "Mixed Emotional Signals"


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    raw_audio_bytes = await file.read()
    
    ext = os.path.splitext(file.filename or ".webm")[1].lower()
    temp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_raw.close()
    temp_wav.close()

    try:
        with open(temp_raw.name, "wb") as f:
            f.write(raw_audio_bytes)

        audio = AudioSegment.from_file(temp_raw.name)
        audio = audio.set_channels(1).set_frame_rate(16000).normalize(headroom=0.1)
        audio.export(temp_wav.name, format="wav", codec="pcm_s16le")

        # Transcription
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav.name) as source:
            audio_data = recognizer.record(source)
            actual_transcript = recognizer.recognize_google(audio_data, language="en-US")

        # Text Sentiment
        if actual_transcript and not actual_transcript.startswith("Error"):
            text_result = sentiment_analyzer(actual_transcript)[0]
            text_intent = text_result['label']
            text_score = round(text_result['score'], 2)
        else:
            text_intent, text_score = "UNKNOWN", 0.0

        # Voice Emotion
        real_voice_emotion, voice_score = "unknown", 0.0
        features = extract_mfcc_from_file(temp_wav.name)
        if features is not None:
            predictions = acoustic_model.predict(features, verbose=0)
            predicted_index = np.argmax(predictions[0])
            real_voice_emotion = ACOUSTIC_EMOTIONS[predicted_index]
            voice_score = round(float(predictions[0][predicted_index]), 4)
            print(f"[VOICE MODEL] → {real_voice_emotion} ({voice_score})")

        final_action = perform_fusion(text_intent, text_score, real_voice_emotion, voice_score)

    except sr.UnknownValueError:
        actual_transcript = "Error: Could not understand the audio."
        text_intent, text_score, real_voice_emotion, voice_score, final_action = "N/A", 0, "N/A", 0, "Audio unclear."
    except Exception as e:
        print(f"[ERROR] {e}")
        actual_transcript = f"System Error"
        text_intent, text_score, real_voice_emotion, voice_score, final_action = "N/A", 0, "N/A", 0, "Error"
    finally:
        for temp in [temp_raw.name, temp_wav.name]:
            if os.path.exists(temp):
                try: os.remove(temp)
                except: pass

    return {
        "transcript": actual_transcript,
        "text_intent": {"label": text_intent, "confidence": text_score},
        "voice_emotion": {"label": real_voice_emotion, "confidence": voice_score},
        "fusion_decision": final_action
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)