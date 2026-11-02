"""Orquestación del flujo de generación, entrenamiento y evaluación."""

from __future__ import annotations

import json
import joblib
import pandas as pd

from config import DATA_PATH, DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FEATURES
from src.data_generator import generate_dataset, validate_dataset
from src.preprocessing import split_and_scale
from src.clustering import fit_kmeans
from src.anomaly_detection import fit_isolation_forest
from src.neural_network import train_and_evaluate


def run_pipeline():
    for directory in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Datos sintéticos
    df = generate_dataset()
    validate_dataset(df)
    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

    # 2. División y escalado supervisado sin fuga de datos
    x_train, x_test, y_train, y_test, scaler = split_and_scale(df)

    # 3. K-Means: agrupamiento exploratorio de todo el conjunto
    x_all_scaled = scaler.transform(df[FEATURES].astype(float))
    kmeans, cluster_labels = fit_kmeans(x_all_scaled)
    df["Grupo_KMeans"] = cluster_labels

    # 4. Isolation Forest: se ajusta con entrenamiento; predice todo el conjunto
    isolation_model, isolation_predictions, isolation_scores = fit_isolation_forest(
        x_train, x_all_scaled
    )
    df["IsolationForest"] = isolation_predictions
    df["Puntuacion_IsolationForest"] = isolation_scores

    # 5. MLP TensorFlow: clasificación supervisada
    mlp, history, test_predictions, probabilities, metrics = train_and_evaluate(
        x_train, y_train, x_test, y_test
    )

    # Guardar resultados y artefactos
    df.to_csv(OUTPUTS_DIR / "resultados_interacciones.csv", index=False, encoding="utf-8-sig")
    test_results = pd.DataFrame(
        {
            "Indice_Registro": y_test.index,
            "Anomalia_Real": y_test.to_numpy(),
            "Anomalia_Predicha": test_predictions,
            "Probabilidad_Anomalia": probabilities,
        }
    )
    test_results.to_csv(
        OUTPUTS_DIR / "predicciones_mlp.csv", index=False, encoding="utf-8-sig"
    )

    mlp.save(MODELS_DIR / "mlp_conauti.keras")
    joblib.dump(kmeans, MODELS_DIR / "kmeans.joblib")
    joblib.dump(isolation_model, MODELS_DIR / "isolation_forest.joblib")
    joblib.dump(scaler, MODELS_DIR / "standard_scaler.joblib")

    report = {
        "registros": int(len(df)),
        "normales": int((df["Anomalia"] == 0).sum()),
        "anomalias": int((df["Anomalia"] == 1).sum()),
        "kmeans_grupos": int(df["Grupo_KMeans"].nunique()),
        "isolation_forest_atipicos": int((df["IsolationForest"] == -1).sum()),
        "mlp": metrics,
    }
    with (OUTPUTS_DIR / "metricas.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    return report
