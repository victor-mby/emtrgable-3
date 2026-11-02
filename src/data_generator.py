"""Generación de datos sintéticos coherentes para las interacciones de agentes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    FEATURES,
    METADATA,
    N_ANOMALOUS,
    N_NORMAL,
    N_RECORDS,
    RANDOM_STATE,
    TARGET,
)


def generate_dataset(random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Crea 250 registros: 200 normales y 50 anómalos.

    Las etiquetas se generan a partir de combinaciones de señales.
    Una solicitud extensa o un consumo alto aislado no implica anomalía.
    """
    if N_NORMAL + N_ANOMALOUS != N_RECORDS:
        raise ValueError("La suma de registros normales y anómalos debe ser 250.")

    rng = np.random.default_rng(random_state)
    n = N_RECORDS

    # Variables de base con relaciones naturales entre longitud y tokens.
    longitud = np.clip(rng.lognormal(mean=5.0, sigma=0.48, size=n), 40, 2500).astype(int)
    tokens = np.clip(
        longitud * rng.uniform(0.22, 0.62, size=n) + rng.normal(35, 45, size=n),
        20,
        2200,
    ).astype(int)

    herramientas = rng.poisson(lam=1.4, size=n)
    intentos = np.ones(n, dtype=int)
    errores = np.zeros(n, dtype=int)
    reintentos = np.zeros(n, dtype=int)

    # La mayoría opera en rangos regulares, con variabilidad entre agentes.
    tiempo = (
        420
        + tokens * rng.uniform(0.25, 0.55, size=n)
        + herramientas * rng.normal(95, 25, size=n)
        + rng.normal(0, 150, size=n)
    )
    tiempo = np.clip(tiempo, 120, 2400)

    cpu = (
        12
        + tokens * rng.uniform(0.006, 0.014, size=n)
        + herramientas * rng.uniform(1.0, 3.5, size=n)
        + rng.normal(0, 5, size=n)
    )
    cpu = np.clip(cpu, 3, 78)

    # Se eligen exactamente 50 interacciones anómalas.
    anomalous_idx = rng.choice(n, size=N_ANOMALOUS, replace=False)
    is_anomalous = np.zeros(n, dtype=int)
    is_anomalous[anomalous_idx] = 1

    # Las anomalías son variadas y dependen de combinaciones, no de un único umbral.
    groups = np.array_split(anomalous_idx, 5)
    for group_no, idx in enumerate(groups):
        if group_no == 0:  # latencia alta + uso CPU alto
            tiempo[idx] *= rng.uniform(2.1, 3.2, size=len(idx))
            cpu[idx] += rng.uniform(18, 35, size=len(idx))
        elif group_no == 1:  # fallos y reintentos reiterados
            errores[idx] = rng.integers(2, 6, size=len(idx))
            reintentos[idx] = rng.integers(3, 8, size=len(idx))
            intentos[idx] = rng.integers(3, 6, size=len(idx))
            tiempo[idx] *= rng.uniform(1.3, 2.0, size=len(idx))
        elif group_no == 2:  # consumo e intentos desproporcionados
            tokens[idx] = (tokens[idx] * rng.uniform(2.0, 3.1, size=len(idx))).astype(int)
            intentos[idx] = rng.integers(2, 5, size=len(idx))
            cpu[idx] += rng.uniform(8, 20, size=len(idx))
        elif group_no == 3:  # varias señales moderadas combinadas
            herramientas[idx] += rng.integers(2, 5, size=len(idx))
            errores[idx] = rng.integers(1, 3, size=len(idx))
            reintentos[idx] = rng.integers(2, 5, size=len(idx))
            tiempo[idx] *= rng.uniform(1.4, 2.0, size=len(idx))
        else:  # concurrencia de varias señales de recursos y ejecución
            cpu[idx] += rng.uniform(15, 30, size=len(idx))
            errores[idx] = rng.integers(1, 4, size=len(idx))
            intentos[idx] = rng.integers(2, 5, size=len(idx))
            reintentos[idx] = rng.integers(1, 5, size=len(idx))

    # Algunas interacciones normales pueden tener solicitudes extensas,
    # muchos tokens o tiempos relativamente altos sin ser anómalas.
    normal_idx = np.flatnonzero(is_anomalous == 0)
    long_normal = rng.choice(normal_idx, size=18, replace=False)
    longitud[long_normal] = rng.integers(1200, 2400, size=len(long_normal))
    tokens[long_normal] = rng.integers(700, 1500, size=len(long_normal))
    tiempo[long_normal] = rng.uniform(1000, 2100, size=len(long_normal))
    cpu[long_normal] = rng.uniform(25, 58, size=len(long_normal))

    # Agregar ruido y acotar dominios.
    tiempo = np.clip(tiempo + rng.normal(0, 90, size=n), 100, 15000).round(2)
    cpu = np.clip(cpu + rng.normal(0, 3, size=n), 1, 100).round(2)
    herramientas = np.clip(herramientas, 0, 12).astype(int)
    intentos = np.clip(intentos, 1, 10).astype(int)
    errores = np.clip(errores, 0, 10).astype(int)
    reintentos = np.clip(reintentos, 0, 12).astype(int)

    agents = rng.choice(["AG-001", "AG-002", "AG-003", "AG-004", "AG-005"], size=n)
    timestamps = pd.date_range("2026-09-01 08:00:00", periods=n, freq="17min")

    df = pd.DataFrame(
        {
            "ID_Interaccion": [f"INT-{i:04d}" for i in range(1, n + 1)],
            "ID_Agente": agents,
            "Fecha_Hora": timestamps,
            "Longitud_Solicitud": longitud,
            "Tiempo_Respuesta_ms": tiempo,
            "Herramientas_Utilizadas": herramientas,
            "Numero_Intentos": intentos,
            "Tokens_Consumidos": tokens,
            "Errores_Ejecucion": errores,
            "Reintentos_API": reintentos,
            "Uso_CPU_Porcentaje": cpu,
            "Anomalia": is_anomalous,
        }
    )

    # Mantener un orden de columnas estable.
    return df[METADATA + FEATURES + [TARGET]]


def validate_dataset(df: pd.DataFrame) -> None:
    """Valida estructura, tipos básicos y distribución de clases."""
    expected = METADATA + FEATURES + [TARGET]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    if len(df) != N_RECORDS:
        raise ValueError(f"Se esperaban {N_RECORDS} filas; se recibieron {len(df)}.")
    counts = df[TARGET].value_counts().to_dict()
    if counts.get(0, 0) != N_NORMAL or counts.get(1, 0) != N_ANOMALOUS:
        raise ValueError(f"Distribución de clases inesperada: {counts}")
    if df[FEATURES].isna().any().any():
        raise ValueError("El conjunto contiene valores faltantes.")
