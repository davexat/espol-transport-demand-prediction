"""Métricas de evaluación para modelos predictivos.

Este módulo implementa las métricas utilizadas para evaluar el desempeño de los modelos y resume sus resultados a través de múltiples particiones de validación temporal.

Métricas disponibles:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- WAPE (Weighted Absolute Percentage Error)
- R² (Coefficient of Determination)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y, yhat):
    """Calcula el error absoluto medio (MAE)."""

    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    """Calcula la raíz del error cuadrático medio (RMSE)."""

    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def wape(y, yhat):
    """Calcula el Weighted Absolute Percentage Error (WAPE).

    Retorna NaN cuando la suma de los valores reales es cero.
    """
    
    y = np.asarray(y, dtype="float64")
    yhat = np.asarray(yhat, dtype="float64")
    denom = np.sum(np.abs(y))

    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y - yhat)) / denom)


def r2(y, yhat):
    """Calcula el coeficiente de determinación (R²).

    Retorna NaN cuando la varianza de los valores reales es nula.
    """
    y = np.asarray(y, dtype="float64")
    yhat = np.asarray(yhat, dtype="float64")
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return float("nan")
    
    return float(1 - ss_res / ss_tot)


def all_metrics(y, yhat) -> dict:
    """Calcula todas las métricas de evaluación para un conjunto de predicciones."""

    return {
        "MAE": mae(y, yhat), 
        "RMSE": rmse(y, yhat),
        "WAPE": wape(y, yhat), 
        "R2": r2(y, yhat)
    }


def summarize_folds(per_fold: pd.DataFrame, by: list[str], metrics=("MAE", "RMSE", "WAPE", "R2")) -> pd.DataFrame:
    """Resume las métricas obtenidas en múltiples folds.

    Para cada grupo calcula la media y la desviación estándar de las métricas especificadas.
    """

    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
    out = per_fold.groupby(by).agg(**agg).reset_index()

    return out
