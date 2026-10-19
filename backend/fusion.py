"""Rule-based multimodal decision fusion.

Pure functions with no I/O, so every rule can be unit-tested in isolation
(see tests/test_fusion.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

TextIntent = Literal["positive", "negative", "neutral"]
FusionMode = Literal["multimodal", "text_only", "voice_only", "none"]

# Voice predictions below this softmax confidence are ignored by the fusion rules.
DEFAULT_VOICE_THRESHOLD = 0.40
# The SST-2 sentiment model only emits POSITIVE/NEGATIVE. Low-confidence calls
# are treated as NEUTRAL so the neutral-text rules can actually fire.
DEFAULT_TEXT_NEUTRAL_THRESHOLD = 0.70

INSUFFICIENT = "Insufficient data for fusion analysis."


@dataclass(frozen=True)
class FusionResult:
    decision: str
    mode: FusionMode


def classify_text_intent(label: Optional[str], score: float,
                         neutral_threshold: float = DEFAULT_TEXT_NEUTRAL_THRESHOLD) -> Optional[TextIntent]:
    """Map a raw sentiment-model output to positive / negative / neutral (None if absent)."""
    if not label:
        return None
    label = label.lower()
    if label not in ("positive", "negative"):
        return "neutral"
    return label if score >= neutral_threshold else "neutral"  # type: ignore[return-value]


def fuse(text_intent: Optional[TextIntent], voice_emotion: Optional[str], voice_score: float,
         voice_threshold: float = DEFAULT_VOICE_THRESHOLD) -> FusionResult:
    """Combine linguistic intent with acoustic emotion into one decision.

    Either input may be missing (None): no intelligible speech means no text
    intent, and an undecodable/too-short clip means no voice emotion. The
    engine degrades to whichever modality is available instead of failing.
    """
    reliable_voice = voice_emotion is not None and voice_score >= voice_threshold

    if text_intent is None:
        if reliable_voice:
            return FusionResult(f"Voice-only: {voice_emotion.capitalize()} tone (speech not transcribed)",
                                "voice_only")
        return FusionResult(INSUFFICIENT, "none")

    mode: FusionMode = "multimodal" if reliable_voice else "text_only"
    emotion = voice_emotion if reliable_voice else None

    if text_intent == "negative":
        if emotion in ("angry", "disgust"):
            return FusionResult("High Hostility • Anger Detected", mode)
        if emotion in ("sad", "fearful"):
            return FusionResult("Deep Emotional Distress", mode)
        if emotion in ("calm", "neutral"):
            return FusionResult("Calm Negative / Disappointed Tone", mode)
        return FusionResult("Negative Intent", mode)

    if text_intent == "positive":
        if emotion in ("angry", "disgust", "fearful"):
            return FusionResult("Sarcasm or Passive-Aggressive Tone Detected", mode)
        if emotion == "sad":
            return FusionResult("Bittersweet / Melancholic Positivity", mode)
        if emotion in ("calm", "neutral"):
            return FusionResult("Calm Positive / Content", mode)
        return FusionResult("Genuine Positive Emotion", mode)

    # Neutral text
    if emotion in ("angry", "disgust"):
        return FusionResult("Hidden Anger / Frustration Detected", mode)
    if emotion in ("sad", "fearful"):
        return FusionResult("Underlying Sadness or Anxiety", mode)
    if emotion in ("happy", "surprised"):
        return FusionResult("Subtle Positive Emotion", mode)
    return FusionResult("Neutral Emotional State", mode)
