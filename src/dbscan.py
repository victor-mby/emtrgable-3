"""Modelo no supervisado: DBSCAN para detectar grupos y ruido."""
from __future__ import annotations
import numpy as np
from sklearn.cluster import DBSCAN

def fit_dbscan(x_scaled, eps=1.35, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(x_scaled)
    noise = int(np.sum(labels == -1))
    groups = int(len(set(labels)) - (1 if -1 in labels else 0))
    return model, labels, {"grupos": groups, "ruido": noise}
