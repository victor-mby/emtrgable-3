# CONAUTI S.A.C. — Entregable final

Sistema académico para analizar anomalías en interacciones de agentes de inteligencia artificial.

## Modelos implementados

| Tipo | Modelo | Tecnología |
|---|---|---|
| Supervisado | Árbol de Decisión | Scikit-learn |
| No supervisado | K-Means | Scikit-learn |
| No supervisado | DBSCAN | Scikit-learn |
| Deep Learning | MLP | TensorFlow/Keras |
| Deep Learning | Autoencoder | TensorFlow/Keras |
| Visión Computacional | CNN | TensorFlow/Keras |

## Datos

Se generan 250 registros sintéticos: 200 normales y 50 anómalos, con 8 variables predictoras y 1 variable objetivo (`Anomalia`).

## Estructura

- `main.py`: punto de entrada.
- `app.py`: interfaz web local con Flask.
- `config.py`: configuración.
- `src/data_generator.py`: generación y validación de datos.
- `src/preprocessing.py`: división y escalado.
- `src/decision_tree.py`: modelo supervisado.
- `src/clustering.py`: K-Means.
- `src/dbscan.py`: DBSCAN.
- `src/neural_network.py`: MLP.
- `src/autoencoder.py`: Autoencoder.
- `src/cnn.py`: CNN.
- `src/pipeline.py`: ejecución integrada de los seis modelos.
- `data/`: datos generados.
- `models/`: modelos entrenados.
- `outputs/`: métricas y predicciones.
- `templates/`: interfaz HTML.

## Ejecución

Desde la carpeta del proyecto:

```bash
py -3.13 -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Para entrenar sin abrir el panel:

```bash
python main.py --solo-entrenar
```

El panel queda disponible en `http://127.0.0.1:5000`.

## Visión computacional

La CNN trabaja con una representación visual 16x16 generada a partir de las ocho variables numéricas estandarizadas de cada interacción. Así se incorpora una arquitectura CNN al caso sin inventar fotografías reales que no forman parte del conjunto de datos.

## Resultados

El pipeline genera `outputs/metricas.json`, `outputs/resultados_interacciones.csv` y `outputs/predicciones_supervisadas.csv`. Los modelos entrenados se guardan en `models/`.

Los resultados son académicos y se basan en datos sintéticos.
