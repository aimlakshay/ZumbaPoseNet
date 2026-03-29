import numpy as np
import os
from extract_pose import extract_keypoints
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.utils import shuffle

WINDOW = 30

def load_folder(path, label):
    X, y = [], []

    for file in os.listdir(path):
        if not file.endswith(".mp4"):
            continue

        print("Processing:", file)
        data = extract_keypoints(os.path.join(path, file))

        for i in range(len(data) - WINDOW):
            X.append(data[i:i+WINDOW])
            y.append(label)

    return X, y


print("Loading videos...")

X1, y1 = load_folder("videos/correct", 1)
X0, y0 = load_folder("videos/incorrect", 0)

X = np.array(X1 + X0)
y = np.array(y1 + y0)

X, y = shuffle(X, y)

print("Samples:", len(X))

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(WINDOW, X.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Training...")

model.fit(
    X,
    y,
    epochs=40,
    batch_size=32,
    validation_split=0.2
)

model.save("model/zumba_lstm_model.h5")

print("Model saved.")

