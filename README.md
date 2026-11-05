# CONAUTI S.A.C. — Entregable final

Sistema académico para analizar anomalías en interacciones de agentes de inteligencia artificial.

## Modelos implementados

1. Supervisado: Árbol de Decisión.
2. No supervisado: K-Means.
3. No supervisado: DBSCAN.
4. Deep Learning: MLP con TensorFlow/Keras.
5. Deep Learning: Autoencoder con TensorFlow/Keras.
6. Visión Computacional: CNN con TensorFlow/Keras.

## Datos

Se generan 250 registros sintéticos: 200 normales y 50 anómalos.

## Ejecución

Desde la carpeta extraída del ZIP:

py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py

Para entrenar sin abrir Flask:
python main.py --solo-entrenar

El panel queda disponible en http://127.0.0.1:5000.

## Visión computacional

La CNN recibe una imagen sintética de 16x16 generada a partir de las ocho variables normalizadas. Esta representación permite aplicar una arquitectura de convolución al comportamiento de cada interacción sin inventar fotografías reales de una operación que no existe en los datos disponibles.

Los resultados son académicos y se basan en datos sintéticos.
