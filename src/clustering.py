"""Modelo K-Means para explorar grupos de interacciones."""

from sklearn.cluster import KMeans

from config import RANDOM_STATE


def fit_kmeans(x_scaled, n_clusters: int = 4):
    """Ajusta K-Means y devuelve el modelo y las etiquetas de grupo."""
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(x_scaled)
    return model, labels
