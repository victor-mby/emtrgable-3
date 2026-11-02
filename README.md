# CONAUTI S.A.C. — Detección de anomalías en agentes de IA

Proyecto académico para analizar interacciones de agentes de inteligencia artificial mediante tres modelos: K-Means, Isolation Forest y una red neuronal MLP con TensorFlow.

## 1. Requisitos

- Python 3.10 a 3.12 recomendado.
- pip.
- No necesita servicios externos ni credenciales.

## 2. Instalación

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 3. Ejecutar el proyecto

Desde la carpeta raíz:

```bash
python main.py
```

El programa:
1. Genera 250 interacciones simuladas (200 normales y 50 anómalas).
2. Guarda el conjunto de datos en `data/interacciones_agentes.csv`.
3. Preprocesa y escala las variables predictoras.
4. Entrena K-Means e Isolation Forest.
5. Entrena la red neuronal MLP con TensorFlow.
6. Evalúa la clasificación supervisada y guarda métricas y predicciones.

Los resultados se guardan en `outputs/` y los modelos entrenados en `models/`.

## 4. Variables

Predictoras:
- `Longitud_Solicitud`
- `Tiempo_Respuesta_ms`
- `Herramientas_Utilizadas`
- `Numero_Intentos`
- `Tokens_Consumidos`
- `Errores_Ejecucion`
- `Reintentos_API`
- `Uso_CPU_Porcentaje`

Objetivo:
- `Anomalia`: 0 = normal, 1 = anómala.

Metadatos:
- `ID_Interaccion`
- `ID_Agente`
- `Fecha_Hora`

Los metadatos no se usan como entradas de los modelos.

## 5. Modelos

### K-Means
Agrupa interacciones con características similares. Se usa para explorar patrones de comportamiento; los números de grupo no representan por sí solos una clase normal o anómala.

### Isolation Forest
Identifica registros atípicos sin utilizar la etiqueta `Anomalia` durante el ajuste. Su resultado es una señal complementaria, no un diagnóstico definitivo.

### MLP con TensorFlow
Aprende con las ocho variables predictoras y la etiqueta `Anomalia`. Clasifica nuevas interacciones como normales o anómalas.

## 6. Estructura

```text
emtrgable-3/
├── config.py
├── main.py
├── requirements.txt
├── data/
├── models/
├── outputs/
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── anomaly_detection.py
│   ├── neural_network.py
│   └── pipeline.py
└── tests/
    └── test_data_generator.py
```

## 7. Alcance

Los registros son simulados para fines académicos. Los resultados sirven para demostrar el flujo de análisis y no representan mediciones operativas reales de CONAUTI S.A.C.
