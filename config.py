"""Configuración central del proyecto CONAUTI."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

RANDOM_STATE = 42
N_RECORDS = 250
N_NORMAL = 200
N_ANOMALOUS = 50
TEST_SIZE = 0.20

FEATURES = [
    "Longitud_Solicitud",
    "Tiempo_Respuesta_ms",
    "Herramientas_Utilizadas",
    "Numero_Intentos",
    "Tokens_Consumidos",
    "Errores_Ejecucion",
    "Reintentos_API",
    "Uso_CPU_Porcentaje",
]
TARGET = "Anomalia"
METADATA = ["ID_Interaccion", "ID_Agente", "Fecha_Hora"]

DATA_PATH = DATA_DIR / "interacciones_agentes.csv"
