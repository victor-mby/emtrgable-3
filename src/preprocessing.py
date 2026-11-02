"""Preprocesamiento compartido para los modelos."""

from __future__ import annotations

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import FEATURES, RANDOM_STATE, TARGET, TEST_SIZE


def split_and_scale(df):
    """Divide datos y ajusta el escalador solo con el conjunto de entrenamiento."""
    x = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler
