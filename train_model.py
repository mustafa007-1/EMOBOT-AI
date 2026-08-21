import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

# ====================== CONFIG ======================
DATA_DIR = "./data/RAVDEES DATASET"   # ← Corrected spelling
MODEL_SAVE_PATH = "./models/ser_lstm_model.keras"
MAX_LEN = 150

# ====================== FEATURE EXTRACTION ======================
def extract_mfcc(file_path, max_len=MAX_LEN):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfccs.shape[1] < max_len:
            pad_width = max_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_len]
        return mfccs.T
    except Exception as e:
        return None

# ====================== LOAD DATA ======================
def load_data(data_dir):
    X, y = [], []
    print(f"Scanning: {os.path.abspath(data_dir)}")
    
    file_count = 0
    loaded_count = 0

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                file_count += 1
                try:
                    if file_count <= 5:
                        print(f"   Found: {file}")
                    
                    emotion_code = int(file.split('-')[2]) - 1
                    file_path = os.path.join(root, file)
                    
                    features = extract_mfcc(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(emotion_code)
                        loaded_count += 1
                except:
                    continue

    print(f"\n✅ Total .wav files found: {file_count}")
    print(f"✅ Successfully loaded: {loaded_count} samples")
    return np.array(X), np.array(y)

# ====================== BUILD MODEL ======================
def build_model():
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN, 40)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(8, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ====================== MAIN ======================
if __name__ == "__main__":
    X, y = load_data(DATA_DIR)

    if len(X) == 0:
        print("❌ No data loaded. Check folder name and contents.")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    model = build_model()
    model.summary()

    print("\n Training started... This may take some time.")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n Final Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    model.save(MODEL_SAVE_PATH)
    print(f" Model saved at: {MODEL_SAVE_PATH}")