"""Train the EmoVoice acoustic emotion model (LSTM over MFCCs) on RAVDESS.

Examples
--------
    # Default: stratified random split (same protocol as the original 92.88% run)
    python train_model.py

    # Speaker-independent evaluation: test actors never appear in training
    python train_model.py --split actor

Outputs (next to --output):
    ser_lstm_model.keras        trained model (best epoch by validation loss)
    ser_lstm_model.metrics.json test accuracy, per-class report, confusion matrix, run config
    ser_lstm_model.history.csv  per-epoch loss/accuracy
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from backend.features import EMOTIONS, MAX_LEN, N_MFCC, extract_mfcc

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("train")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def parse_ravdess_name(filename: str) -> tuple[int, int] | None:
    """Return (emotion_index 0..7, actor_id) from a RAVDESS speech filename, or None otherwise.

    Format: modality-channel-emotion-intensity-statement-repetition-actor.wav
    e.g. 03-01-05-02-01-01-12.wav -> emotion 05 (angry), actor 12.
    Song clips (vocal channel 02) are rejected: the full RAVDESS release ships
    them with the same naming scheme, and the model is trained on speech only.
    """
    parts = Path(filename).stem.split("-")
    if len(parts) != 7 or not all(p.isdigit() for p in parts):
        return None
    if int(parts[1]) != 1:  # vocal channel: 01 = speech, 02 = song
        return None
    emotion, actor = int(parts[2]), int(parts[6])
    if not 1 <= emotion <= len(EMOTIONS):
        return None
    return emotion - 1, actor


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for every unique RAVDESS clip under data_dir.

    The Kaggle download can contain the same clips twice (e.g. `Actor_01/` and
    `audio_speech_actors_01-24/Actor_01/`). Duplicates would land on both sides
    of the train/test split and inflate accuracy, so clips are de-duplicated by
    filename, which is unique per recording in RAVDESS.
    """
    if not data_dir.is_dir():
        sys.exit(f"Dataset folder not found: {data_dir.resolve()}\n"
                 "Download RAVDESS (see data/README.md) or pass --data-dir.")

    unique: dict[str, Path] = {}
    duplicates = skipped = 0
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() != ".wav":
            continue
        if path.name in unique:
            duplicates += 1
        elif parse_ravdess_name(path.name) is None:
            skipped += 1
        else:
            unique[path.name] = path

    log.info("Found %d unique RAVDESS clips (%d duplicates ignored, %d song/non-RAVDESS files skipped)",
             len(unique), duplicates, skipped)
    if not unique:
        sys.exit("No RAVDESS .wav files found. Check --data-dir.")

    features, labels, actors, failed = [], [], [], 0
    for i, (name, path) in enumerate(unique.items(), start=1):
        emotion, actor = parse_ravdess_name(name)  # type: ignore[misc]
        try:
            features.append(extract_mfcc(path))
        except Exception as exc:  # corrupt file: report it, keep going
            failed += 1
            log.warning("Skipping %s: %s", path, exc)
            continue
        labels.append(emotion)
        actors.append(actor)
        if i % 200 == 0:
            log.info("  extracted %d / %d", i, len(unique))

    if failed:
        log.warning("%d files could not be read", failed)
    return np.asarray(features, dtype=np.float32), np.asarray(labels), np.asarray(actors)


def split_dataset(X, y, actors, mode: str, test_size: float, val_size: float, seed: int):
    """Return (train, val, test) index arrays. The test set is never used for training decisions."""
    indices = np.arange(len(y))
    if mode == "actor":
        unique_actors = np.unique(actors)
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(unique_actors)
        n_test = max(1, round(len(unique_actors) * test_size))
        n_val = max(1, round(len(unique_actors) * val_size))
        test_actors = {int(a) for a in shuffled[:n_test]}
        val_actors = {int(a) for a in shuffled[n_test:n_test + n_val]}
        test = indices[np.isin(actors, list(test_actors))]
        val = indices[np.isin(actors, list(val_actors))]
        train = indices[~np.isin(actors, list(test_actors | val_actors))]
        log.info("Actor split — test actors %s, val actors %s", sorted(test_actors), sorted(val_actors))
    else:
        train_val, test = train_test_split(indices, test_size=test_size, random_state=seed, stratify=y)
        relative_val = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=relative_val, random_state=seed,
                                      stratify=y[train_val])
    return train, val, test


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model():
    from keras import layers, models
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN, N_MFCC)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(len(EMOTIONS), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ravdess"))
    parser.add_argument("--output", type=Path, default=Path("models/ser_lstm_model.keras"))
    parser.add_argument("--split", choices=["random", "actor"], default="random",
                        help="random: stratified clip split; actor: speaker-independent split")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50, help="maximum epochs")
    parser.add_argument("--patience", type=int, default=10, help="early-stopping patience (0 disables)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import keras
    keras.utils.set_random_seed(args.seed)  # seeds Python, NumPy and the backend for reproducibility

    X, y, actors = load_dataset(args.data_dir)
    log.info("Dataset: %d samples, feature shape %s, class counts %s",
             len(y), X.shape[1:], {EMOTIONS[k]: v for k, v in sorted(Counter(y.tolist()).items())})

    train, val, test = split_dataset(X, y, actors, args.split, args.test_size, args.val_size, args.seed)
    log.info("Split (%s): train=%d val=%d test=%d", args.split, len(train), len(val), len(test))

    weights = compute_class_weight("balanced", classes=np.unique(y[train]), y=y[train])
    class_weight = {int(c): float(w) for c, w in zip(np.unique(y[train]), weights)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [keras.callbacks.ModelCheckpoint(str(args.output), monitor="val_loss", save_best_only=True)]
    if args.patience > 0:
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                                       restore_best_weights=True))

    model = build_model()
    model.summary(print_fn=log.info)
    history = model.fit(X[train], y[train], validation_data=(X[val], y[val]),
                        epochs=args.epochs, batch_size=args.batch_size,
                        class_weight=class_weight, callbacks=callbacks, verbose=2)

    model = keras.models.load_model(str(args.output))  # best checkpoint by validation loss
    test_loss, test_acc = model.evaluate(X[test], y[test], verbose=0)
    predictions = model.predict(X[test], verbose=0).argmax(axis=1)
    labels = list(range(len(EMOTIONS)))
    report = classification_report(y[test], predictions, labels=labels, target_names=EMOTIONS,
                                   output_dict=True, zero_division=0)

    print("\n" + classification_report(y[test], predictions, labels=labels, target_names=EMOTIONS,
                                       zero_division=0))
    log.info("Final test accuracy: %.4f (%.2f%%) on %d held-out samples [%s split]",
             test_acc, test_acc * 100, len(test), args.split)

    stem = args.output.with_suffix("")
    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y[test], predictions, labels=labels).tolist(),
        "labels": EMOTIONS,
        "samples": {"total": int(len(y)), "train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "epochs_trained": len(history.history["loss"]),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(f"{stem}.metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(f"{stem}.history.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        keys = list(history.history)
        writer.writerow(["epoch", *keys])
        for epoch, row in enumerate(zip(*history.history.values()), start=1):
            writer.writerow([epoch, *row])

    log.info("Model saved to %s (metrics + history alongside)", args.output)


if __name__ == "__main__":
    main()
