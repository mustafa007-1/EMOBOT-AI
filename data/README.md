# Dataset

The acoustic model is trained on the speech subset of **RAVDESS**
(24 actors, 1,440 clips, 8 emotions). It is not committed to this repository.

1. Download it from Kaggle: <https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio>
2. Extract it into `data/ravdess/` (any nesting works; duplicate copies are skipped).
3. Run `python train_model.py` from the repository root.

RAVDESS is licensed CC BY-NC-SA 4.0 (Livingstone & Russo, 2018).
