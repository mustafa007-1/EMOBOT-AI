import io
import wave

import numpy as np
import pytest


def make_wav_bytes(seconds: float = 2.0, rate: int = 48000, freq: float = 220.0,
                   amplitude: float = 0.3) -> bytes:
    """A mono 16-bit WAV containing a modulated tone (stand-in for a voice clip)."""
    t = np.linspace(0, seconds, int(seconds * rate), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * freq * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    pcm = (signal * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav_bytes()


@pytest.fixture
def make_wav():
    return make_wav_bytes
