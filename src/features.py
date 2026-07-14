"""Construcción de la matriz de características para el modelado.

Este módulo transforma la tabla integrada en una matriz de diseño lista para el entrenamiento de modelos predictivos.

Las variables se construyen respetando el orden temporal de los datos:

- Solo se utilizan variables disponibles antes del instante de predicción.
- Las variables rezagadas (lags) y estadísticas móviles se calculan exclusivamente a partir del historial previo.
- Las variables categóricas se codifican mediante one-hot encoding.
- El conjunto de columnas se mantiene consistente entre entrenamiento, validación y prueba.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_KEYS = ["parada_observada", "tipo_recorrido", "fecha", "turno"]
CATEGORICAL = ["dia_semana", "turno", "tipo_recorrido", "parada_observada", "franja"]


def add_lag_rolling(df: pd.DataFrame, cfg: dict, target: str = "espera_al_cierre") -> pd.DataFrame:
    """Genera variables rezagadas y estadísticas móviles.

    Los cálculos se realizan de forma independiente para cada sesión (parada, tipo de recorrido, fecha y turno), evitando mezclar información entre sesiones.

    Las estadísticas móviles se calculan únicamente con observaciones previas mediante un desplazamiento temporal (`shift(1)`), evitando utilizar información futura.
    """
    
    out = df.sort_values(GROUP_KEYS + ["inicio_seg"]).copy()
    g = out.groupby(GROUP_KEYS, sort=False)

    for k in cfg["features"]["lags"]:
        out[f"{target}_lag_{k}"] = g[target].shift(k)
        
    # Flujos del intervalo anterior.
    for col in ["llegan", "se_retiran"]:
        if col in out.columns:
            out[f"{col}_lag_1"] = g[col].shift(1)

    # Estadísticas móviles calculadas únicamente sobre el historial previo.
    out["_shift"] = g[target].shift(1)
    gs = out.groupby(GROUP_KEYS, sort=False)["_shift"]
    for w in cfg["features"]["rolling_windows"]:
        roll = gs.rolling(w, min_periods=1)
        out[f"{target}_rollmean_{w}"] = roll.mean().reset_index(level=GROUP_KEYS, drop=True)
        out[f"{target}_rollmax_{w}"] = roll.max().reset_index(level=GROUP_KEYS, drop=True)
        out[f"{target}_rollmin_{w}"] = roll.min().reset_index(level=GROUP_KEYS, drop=True)

    out = out.drop(columns=["_shift"])

    return out


def build_design_matrix(df: pd.DataFrame, cfg: dict, target: str = "espera_al_cierre", dummy_columns: list | None = None):
    """Construye la matriz de diseño utilizada por los modelos.

    Parámetros
    ----------
    df
        Tabla integrada con las observaciones.
    cfg
        Configuración del proyecto.
    target
        Variable objetivo.
    dummy_columns
        Columnas one-hot generadas durante el entrenamiento. Cuando se proporcionan, la matriz se reindexa para mantener exactamente el mismo conjunto de variables.

    Retorna
    -------
    X : pandas.DataFrame
        Matriz de características.

    y : numpy.ndarray
        Variable objetivo.

    meta : pandas.DataFrame
        Información auxiliar utilizada durante la validación temporal.

    dummy_columns : list[str]
        Lista de columnas de la matriz de diseño.
    """

    feat = add_lag_rolling(df, cfg, target)

    # Se descarta el primer intervalo de cada sesión, ya que no posee lag.
    feat = feat[feat[f"{target}_lag_1"].notna()].copy()

    lag_roll_cols = [
        c for c in feat.columns
        if c.startswith(f"{target}_lag_")
        or c.startswith(f"{target}_roll")
        or c.endswith("_lag_1")
    ]
    numeric = ["interval_idx", "hora"] + lag_roll_cols
    # Las ventanas iniciales incompletas se rellenan con cero.
    for c in numeric:
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)

    cat = pd.get_dummies(feat[CATEGORICAL].astype(str), prefix=CATEGORICAL)
    X = pd.concat([feat[numeric].reset_index(drop=True), cat.reset_index(drop=True)], axis=1)

    if dummy_columns is not None:
        X = X.reindex(columns=dummy_columns, fill_value=0.0)
    dummy_columns = list(X.columns)

    y = feat[target].to_numpy(dtype="float64")
    meta = feat[GROUP_KEYS + ["inicio_seg", "interval_idx", "franja", "observado"]].reset_index(drop=True)
    meta["order_key"] = (feat["fecha"].astype("int64").values if np.issubdtype(feat["fecha"].dtype, np.datetime64) else feat["fecha"].values)
    
    return X.astype("float64"), y, meta, dummy_columns
