# CONAUTI S.A.C. — Detección de anomalías en agentes de IA

Aplicación académica que genera 250 interacciones sintéticas y aplica K-Means, Isolation Forest y una red neuronal MLP con TensorFlow. Incluye un panel web para consultar resultados y evaluar nuevas interacciones.

## Requisitos
Python 3.10–3.12 recomendado. TensorFlow debe instalarse en una versión compatible con tu Python.

## Instalación (Windows)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecución
```powershell
python main.py
```
Abre http://127.0.0.1:5000. El primer inicio genera los datos, entrena los modelos y luego levanta el panel.

Para entrenar sin abrir el panel:
```powershell
python main.py --solo-entrenar
```

## Variables predictoras
Longitud_Solicitud, Tiempo_Respuesta_ms, Herramientas_Utilizadas, Numero_Intentos, Tokens_Consumidos, Errores_Ejecucion, Reintentos_API y Uso_CPU_Porcentaje.

Objetivo: Anomalia (0 = normal, 1 = anómala). Metadatos: ID_Interaccion, ID_Agente y Fecha_Hora; no se usan como entradas predictoras.

## Modelos
- K-Means: agrupa interacciones similares para explorar patrones.
- Isolation Forest: detecta registros atípicos sin utilizar la etiqueta objetivo durante el ajuste.
- MLP TensorFlow: aprende con las ocho variables y la etiqueta Anomalia para clasificar nuevas interacciones.

## Estructura
- app.py y templates/: panel web.
- main.py: ejecución del flujo.
- config.py: variables y rutas.
- src/: generación, preprocesamiento y modelos.
- tests/: pruebas del generador.
- data/: conjunto sintético.
- models/: modelos entrenados.
- outputs/: predicciones y métricas.

Los datos son simulados para fines académicos; no son mediciones reales de CONAUTI S.A.C.
