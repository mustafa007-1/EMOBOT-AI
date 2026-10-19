import librosa
import numpy as np

from backend.features import MAX_LEN, N_MFCC, extract_mfcc, fit_to_window, sliding_windows


def original_extract_mfcc(file_path, max_len=150):
    """Verbatim copy of the feature code the shipped model was trained with."""
    audio, sr = librosa.load(file_path, res_type="kaiser_fast")
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfccs.shape[1] < max_len:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])), mode="constant")
    else:
        mfccs = mfccs[:, :max_len]
    return mfccs.T


def test_features_match_original_training_code(tmp_path, wav_bytes):
    path = tmp_path / "clip.wav"
    path.write_bytes(wav_bytes)
    np.testing.assert_allclose(extract_mfcc(path), original_extract_mfcc(str(path)), rtol=1e-5, atol=1e-4)


def test_fit_to_window_pads_and_truncates():
    assert fit_to_window(np.ones((10, N_MFCC))).shape == (MAX_LEN, N_MFCC)
    assert fit_to_window(np.ones((400, N_MFCC))).shape == (MAX_LEN, N_MFCC)


def test_short_clip_gives_single_training_window():
    frames = np.random.rand(120, N_MFCC)
    windows = sliding_windows(frames)
    assert windows.shape == (1, MAX_LEN, N_MFCC)
    np.testing.assert_array_equal(windows[0], fit_to_window(frames))


def test_long_clip_is_fully_covered():
    frames = np.random.rand(600, N_MFCC)  # ~14 s
    windows = sliding_windows(frames)
    assert windows.shape[1:] == (MAX_LEN, N_MFCC)
    assert windows.shape[0] == 7           # starts 0, 75, ..., 450
    np.testing.assert_array_equal(windows[-1], frames[450:600])
