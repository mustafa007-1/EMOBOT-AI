"""Acoustic feature extraction shared by training (train_model.py) and serving (app.py).

Keeping a single implementation guarantees the model sees identically computed
features at training time and at inference time (no train/serve skew).
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

# --- Feature contract: must match how ser_lstm_model.keras was trained -------
SAMPLE_RATE = 22050        # librosa's default rate, used during training
RES_TYPE = "kaiser_fast"   # resampler used during training (needs `resampy`)
N_MFCC = 40                # coefficients per frame
MAX_LEN = 150              # frames per window (~3.5 s at 22.05 kHz, hop 512)

# RAVDESS emotion codes 01..08, in model output order.
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def load_audio(path: str | Path) -> np.ndarray:
    """Load any WAV file as mono float32 at SAMPLE_RATE."""
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, res_type=RES_TYPE)
    return audio


def mfcc_frames(audio: np.ndarray) -> np.ndarray:
    """Return MFCCs with shape (n_frames, N_MFCC)."""
    return librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T


def fit_to_window(frames: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    """Zero-pad or truncate (n_frames, N_MFCC) to exactly (max_len, N_MFCC)."""
    if frames.shape[0] >= max_len:
        return frames[:max_len]
    return np.pad(frames, ((0, max_len - frames.shape[0]), (0, 0)), mode="constant")


def extract_mfcc(path: str | Path) -> np.ndarray:
    """Training features: first MAX_LEN frames of the clip, shape (MAX_LEN, N_MFCC)."""
    return fit_to_window(mfcc_frames(load_audio(path)))


def sliding_windows(frames: np.ndarray, max_len: int = MAX_LEN, hop: int | None = None) -> np.ndarray:
    """Split a clip of any length into model-sized windows, shape (n_windows, max_len, N_MFCC).

    Clips no longer than one window produce exactly the training-time input.
    Longer clips are covered by 50%-overlapping windows so the model hears the
    whole utterance instead of only its first ~3.5 seconds.
    """
    n_frames = frames.shape[0]
    if n_frames <= max_len:
        return fit_to_window(frames, max_len)[np.newaxis]
    hop = hop or max_len // 2
    # Stop as soon as a window reaches the end of the clip, so the final window
    # always holds more than (max_len - hop) real, non-padded frames.
    starts = range(0, n_frames - max_len + hop, hop)
    return np.stack([fit_to_window(frames[s:s + max_len], max_len) for s in starts])
