"""Orquestación de los 6 modelos del entregable final."""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_PATH, DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FEATURES, TARGET, RANDOM_STATE, TEST_SIZE
from src.data_generator import generate_dataset, validate_dataset
from src.preprocessing import split_and_scale
from src.clustering import fit_kmeans
from src.decision_tree import train_and_evaluate as train_tree
from src.dbscan import fit_dbscan
from src.neural_network import train_and_evaluate as train_mlp
from src.autoencoder import train_autoencoder
from src.cnn import interactions_to_images, train_and_evaluate as train_cnn

def run_pipeline():
    for directory in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    validate_dataset(df)
    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

    x_train, x_test, y_train, y_test, scaler = split_and_scale(df)
    x_all = scaler.transform(df[FEATURES].astype(float))

    tree, tree_pred, tree_metrics = train_tree(x_train, y_train, x_test, y_test)

    kmeans, kmeans_labels = fit_kmeans(x_all)
    df["Grupo_KMeans"] = kmeans_labels

    dbscan, dbscan_labels, dbscan_info = fit_dbscan(x_all)
    df["Grupo_DBSCAN"] = dbscan_labels

    mlp, mlp_history, mlp_pred, mlp_prob, mlp_metrics = train_mlp(
        x_train, y_train, x_test, y_test
    )

    autoencoder, ae_errors, ae_threshold, ae_anomaly = train_autoencoder(
        x_train, y_train, x_all
    )
    df["Autoencoder_Error"] = ae_errors
    df["Autoencoder_Anomalia"] = ae_anomaly

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    images = interactions_to_images(x_all)
    cnn, cnn_pred, cnn_prob, cnn_metrics = train_cnn(
        images[train_idx], df[TARGET].to_numpy()[train_idx],
        images[test_idx], df[TARGET].to_numpy()[test_idx]
    )

    df["CNN_Probabilidad_Anomalia"] = np.nan
    df.loc[test_idx, "CNN_Probabilidad_Anomalia"] = cnn_prob
    df.to_csv(OUTPUTS_DIR / "resultados_interacciones.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "Indice_Registro": y_test.index,
        "Anomalia_Real": y_test.to_numpy(),
        "Arbol_Decision": tree_pred,
        "MLP": mlp_pred,
        "MLP_Probabilidad": mlp_prob,
    }).to_csv(OUTPUTS_DIR / "predicciones_supervisadas.csv", index=False, encoding="utf-8-sig")

    joblib.dump(tree, MODELS_DIR / "arbol_decision.joblib")
    joblib.dump(kmeans, MODELS_DIR / "kmeans.joblib")
    joblib.dump(dbscan, MODELS_DIR / "dbscan.joblib")
    joblib.dump(scaler, MODELS_DIR / "standard_scaler.joblib")
    mlp.save(MODELS_DIR / "mlp_conauti.keras")
    autoencoder.save(MODELS_DIR / "autoencoder_conauti.keras")
    cnn.save(MODELS_DIR / "cnn_conauti.keras")

    report = {
        "registros": int(len(df)),
        "normales": int((df[TARGET] == 0).sum()),
        "anomalias": int((df[TARGET] == 1).sum()),
        "modelos": {
            "arbol_decision": tree_metrics,
            "kmeans": {"grupos": int(df["Grupo_KMeans"].nunique())},
            "dbscan": dbscan_info,
            "mlp": mlp_metrics,
            "autoencoder": {
                "threshold": ae_threshold,
                "anomalias_detectadas": int(ae_anomaly.sum())
            },
            "cnn": cnn_metrics,
        },
    }
    with (OUTPUTS_DIR / "metricas.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    return report
