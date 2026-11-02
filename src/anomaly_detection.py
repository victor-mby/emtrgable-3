"""Modelo Isolation Forest para detectar registros atípicos."""

from sklearn.ensemble import IsolationForest

from config import RANDOM_STATE


def fit_isolation_forest(x_train_scaled, x_all_scaled=None):
    """Entrena Isolation Forest sin consultar la etiqueta objetivo.

    Devuelve predicciones: 1 = comportamiento habitual, -1 = atípico.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=0.20,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train_scaled)
    data = x_train_scaled if x_all_scaled is None else x_all_scaled
    predictions = model.predict(data)
    scores = model.decision_function(data)
    return model, predictions, scores
