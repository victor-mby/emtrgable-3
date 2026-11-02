"""Panel web Flask para consultar resultados y clasificar interacciones."""
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

from config import FEATURES, MODELS_DIR, OUTPUTS_DIR

app = Flask(__name__)
app.secret_key = "conauti-academic-local-key"

DEFAULTS = {
    "Longitud_Solicitud": 350,
    "Tiempo_Respuesta_ms": 700,
    "Herramientas_Utilizadas": 2,
    "Numero_Intentos": 1,
    "Tokens_Consumidos": 180,
    "Errores_Ejecucion": 0,
    "Reintentos_API": 0,
    "Uso_CPU_Porcentaje": 25,
}


def load_results():
    metrics_path = OUTPUTS_DIR / "metricas.json"
    results_path = OUTPUTS_DIR / "resultados_interacciones.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    df = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    return metrics, df


@app.route("/", methods=["GET"])
def index():
    metrics, df = load_results()
    recent = df.tail(12).to_dict(orient="records") if not df.empty else []
    return render_template("index.html", metrics=metrics, recent=recent, defaults=DEFAULTS, features=FEATURES)


@app.route("/predecir", methods=["POST"])
def predict():
    try:
        values = {name: float(request.form[name]) for name in FEATURES}
        if values["Longitud_Solicitud"] < 0 or values["Tiempo_Respuesta_ms"] < 0:
            raise ValueError("La longitud y el tiempo no pueden ser negativos.")
        if not 0 <= values["Uso_CPU_Porcentaje"] <= 100:
            raise ValueError("El uso de CPU debe estar entre 0 y 100.")
        for name in ("Herramientas_Utilizadas", "Numero_Intentos", "Tokens_Consumidos", "Errores_Ejecucion", "Reintentos_API"):
            if values[name] < 0 or not values[name].is_integer():
                raise ValueError(f"{name} debe ser un entero no negativo.")

        model = __import__("tensorflow").keras.models.load_model(MODELS_DIR / "mlp_conauti.keras")
        scaler = joblib.load(MODELS_DIR / "standard_scaler.joblib")
        row = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
        scaled = scaler.transform(row)
        probability = float(model.predict(scaled, verbose=0)[0][0])
        prediction = int(probability >= 0.5)
        return render_template("prediction.html", prediction=prediction, probability=probability, values=values)
    except Exception as exc:
        flash(f"No se pudo realizar la predicción: {exc}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
