"""Visión computacional: CNN sobre imágenes sintéticas de las interacciones."""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from config import RANDOM_STATE

def interactions_to_images(x_scaled):
    """Convierte las 8 variables estandarizadas en imágenes 16x16 monocromáticas."""
    x = np.asarray(x_scaled, dtype=np.float32)
    x = np.clip((x + 3.0) / 6.0, 0.0, 1.0)
    images = np.zeros((len(x), 16, 16, 1), dtype=np.float32)
    for i, row in enumerate(x):
        for feature in range(8):
            r0 = feature * 2
            images[i, r0:r0+2, :, 0] = row[feature]
        images[i, :, :, 0] += np.eye(16, dtype=np.float32) * np.mean(row[:4])
        images[i, :, :, 0] += np.fliplr(np.eye(16, dtype=np.float32)) * np.mean(row[4:])
    return np.clip(images, 0, 1)

def build_cnn(input_shape=(16,16,1)):
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name="CONAUTI_CNN")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_and_evaluate(images_train, y_train, images_test, y_test, epochs=50):
    model = build_cnn(images_train.shape[1:])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
    model.fit(images_train, np.asarray(y_train), validation_split=0.2, epochs=epochs,
              batch_size=16, callbacks=[stop], verbose=0)
    probs = model.predict(images_test, verbose=0).ravel()
    pred = (probs >= 0.5).astype(int)
    y = np.asarray(y_test).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    return model, pred, probs, metrics
