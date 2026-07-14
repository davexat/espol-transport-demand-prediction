"""Utilidades para validación temporal mediante rolling-origin.

Este módulo genera particiones cronológicas para evaluar modelos de series temporales sin alterar el orden de las observaciones.

Los conjuntos de entrenamiento siempre contienen únicamente datos anteriores al conjunto de prueba, evitando el uso de información futura durante la evaluación.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_origin_date_folds(dates: pd.Series, n_folds: int, initial_fraction: float, scheme: str = "expanding") -> list[dict]:
    """Genera particiones cronológicas a nivel de fecha.

    Parámetros
    ----------
    dates
        Serie con las fechas de las observaciones.
    n_folds
        Número máximo de particiones a generar.
    initial_fraction
        Proporción inicial de fechas destinada al entrenamiento.
    scheme
        Esquema de validación:

        - ``expanding``: el entrenamiento crece en cada fold.
        - ``sliding``: el entrenamiento mantiene un tamaño fijo y se desplaza junto con la ventana de prueba.

    Retorna
    -------
    list[dict]
        Lista de folds con las fechas de entrenamiento y prueba.
    """
    
    uniq = np.array(sorted(pd.unique(dates)))
    n = len(uniq)
    start = max(1, int(round(initial_fraction * n)))
    remaining = n - start
    if remaining < n_folds:
        n_folds = max(1, remaining)
    fold_size = max(1, remaining // n_folds)

    folds = []
    for f in range(n_folds):
        test_lo = start + f * fold_size
        test_hi = n if f == n_folds - 1 else start + (f + 1) * fold_size
        if test_lo >= n:
            break
        test_dates = uniq[test_lo:test_hi]
        if scheme == "expanding":
            train_dates = uniq[:test_lo]
        else:  # Ventana deslizante de tamaño fijo.
            train_dates = uniq[max(0, test_lo - start):test_lo]
        folds.append({
            "fold": f, 
            "train_dates": train_dates,
            "test_dates": test_dates
        })

    return folds


def folds_to_row_indices(meta: pd.DataFrame, folds: list[dict], date_col: str = "fecha") -> list[dict]:
    """Convierte particiones por fecha en índices de filas.

    A partir de una tabla de metadatos y una colección de folds temporales, obtiene los índices correspondientes a entrenamiento y prueba para la matriz de diseño.
    """

    out = []
    dvals = meta[date_col].values
    for fd in folds:
        train_mask = np.isin(dvals, fd["train_dates"])
        test_mask = np.isin(dvals, fd["test_dates"])
        out.append({
            "fold": fd["fold"],
            "train_idx": np.where(train_mask)[0],
            "test_idx": np.where(test_mask)[0]
        })

    return out
