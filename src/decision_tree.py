"""Modelo supervisado: Árbol de Decisión para clasificar anomalías."""
from __future__ import annotations
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from config import RANDOM_STATE

def train_and_evaluate(x_train, y_train, x_test, y_test):
    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    return model, pred, metrics
