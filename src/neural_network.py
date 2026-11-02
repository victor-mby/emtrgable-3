"""Red neuronal MLP supervisada implementada con TensorFlow/Keras."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from config import RANDOM_STATE


def build_mlp(input_dim: int):
    """Construye una MLP binaria para clasificar interacciones."""
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="CONAUTI_MLP",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(x_train, y_train, x_test, y_test, epochs: int = 60):
    """Entrena la red y devuelve modelo, historial, predicciones y métricas."""
    model = build_mlp(input_dim=x_train.shape[1])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        np.asarray(y_train),
        validation_split=0.20,
        epochs=epochs,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0,
    )

    probabilities = model.predict(x_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    y_true = np.asarray(y_test).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0
        ),
    }
    return model, history, predictions, probabilities, metrics
