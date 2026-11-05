"""Deep Learning: Autoencoder para detección de anomalías."""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from config import RANDOM_STATE

def build_autoencoder(input_dim: int):
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
    ])
    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(input_dim, activation="linear"),
    ])
    inputs = tf.keras.Input(shape=(input_dim,))
    outputs = decoder(encoder(inputs))
    model = tf.keras.Model(inputs, outputs, name="CONAUTI_Autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model

def train_autoencoder(x_train, y_train, x_all, epochs=60):
    normal_train = x_train[np.asarray(y_train) == 0]
    model = build_autoencoder(x_train.shape[1])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(normal_train, normal_train, validation_split=0.2, epochs=epochs,
              batch_size=16, callbacks=[stop], verbose=0)
    normal_recon = model.predict(normal_train, verbose=0)
    threshold = float(np.percentile(np.mean(np.square(normal_train-normal_recon), axis=1), 95))
    recon = model.predict(x_all, verbose=0)
    errors = np.mean(np.square(x_all-recon), axis=1)
    anomaly = (errors > threshold).astype(int)
    return model, errors, threshold, anomaly
