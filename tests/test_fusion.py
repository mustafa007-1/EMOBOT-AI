import pytest

from backend.fusion import INSUFFICIENT, classify_text_intent, fuse


@pytest.mark.parametrize("text, voice, expected", [
    # Negative text
    ("negative", "angry", "High Hostility • Anger Detected"),
    ("negative", "disgust", "High Hostility • Anger Detected"),
    ("negative", "sad", "Deep Emotional Distress"),
    ("negative", "fearful", "Deep Emotional Distress"),
    ("negative", "calm", "Calm Negative / Disappointed Tone"),
    ("negative", "neutral", "Calm Negative / Disappointed Tone"),
    ("negative", "happy", "Negative Intent"),
    # Positive text
    ("positive", "angry", "Sarcasm or Passive-Aggressive Tone Detected"),
    ("positive", "fearful", "Sarcasm or Passive-Aggressive Tone Detected"),
    ("positive", "sad", "Bittersweet / Melancholic Positivity"),
    ("positive", "calm", "Calm Positive / Content"),
    ("positive", "happy", "Genuine Positive Emotion"),
    # Neutral text
    ("neutral", "angry", "Hidden Anger / Frustration Detected"),
    ("neutral", "sad", "Underlying Sadness or Anxiety"),
    ("neutral", "surprised", "Subtle Positive Emotion"),
    ("neutral", "calm", "Neutral Emotional State"),
])
def test_multimodal_rules(text, voice, expected):
    result = fuse(text, voice, voice_score=0.9)
    assert result.decision == expected
    assert result.mode == "multimodal"


def test_unreliable_voice_falls_back_to_text_only():
    result = fuse("positive", "angry", voice_score=0.2)
    assert result.decision == "Genuine Positive Emotion"
    assert result.mode == "text_only"


def test_voice_only_when_no_speech():
    result = fuse(None, "angry", voice_score=0.8)
    assert result.mode == "voice_only"
    assert "Angry" in result.decision


def test_insufficient_when_nothing_reliable():
    assert fuse(None, "angry", voice_score=0.1).decision == INSUFFICIENT
    assert fuse(None, None, voice_score=0.0).mode == "none"


@pytest.mark.parametrize("label, score, expected", [
    ("POSITIVE", 0.99, "positive"),
    ("NEGATIVE", 0.95, "negative"),
    ("POSITIVE", 0.55, "neutral"),   # low-confidence polarity -> neutral
    ("LABEL_X", 0.99, "neutral"),
    (None, 0.0, None),
])
def test_classify_text_intent(label, score, expected):
    assert classify_text_intent(label, score, neutral_threshold=0.7) == expected
