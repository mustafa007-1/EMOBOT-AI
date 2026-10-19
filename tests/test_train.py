import pytest

pytest.importorskip("sklearn")

from train_model import parse_ravdess_name  # noqa: E402


@pytest.mark.parametrize("name, expected", [
    ("03-01-05-02-01-01-12.wav", (4, 12)),   # speech, angry, actor 12
    ("03-01-01-01-01-01-01.wav", (0, 1)),    # speech, neutral
    ("03-02-05-02-01-01-12.wav", None),      # song clip, not speech
    ("03-01-09-01-01-01-01.wav", None),      # emotion code out of range
    ("03-01-05-02-01-01.wav", None),         # too few fields
    ("recording.wav", None),
])
def test_parse_ravdess_name(name, expected):
    assert parse_ravdess_name(name) == expected
