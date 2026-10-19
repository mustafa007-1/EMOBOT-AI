"""Smoke test against the real trained model (skipped if TensorFlow/Keras or the model is absent)."""

import pytest

keras = pytest.importorskip("keras")

from backend import app as app_module  # noqa: E402
from backend.app import Models, predict_voice_emotion  # noqa: E402
from backend.features import EMOTIONS  # noqa: E402

pytestmark = pytest.mark.skipif(not app_module.MODEL_PATH.is_file(), reason="trained model not present")


def test_real_model_produces_a_distribution(tmp_path, wav_bytes):
    path = tmp_path / "clip.wav"
    path.write_bytes(wav_bytes)
    models = Models(acoustic=app_module.load_acoustic_model(), sentiment=None, transcribe=None)
    label, score, distribution = predict_voice_emotion(models, path)
    assert label in EMOTIONS
    assert 0.0 < score <= 1.0
    assert abs(sum(distribution.values()) - 1.0) < 1e-2
