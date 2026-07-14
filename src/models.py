"""Modelos predictivos y baselines.

Este módulo centraliza la construcción de los modelos utilizados durante la evaluación experimental.

Se distinguen dos grupos:

- Baselines, que generan predicciones mediante reglas simples y sirven como referencia mínima de desempeño.
- Modelos de aprendizaje automático, que se ajustan sobre la matriz de características generada durante el preprocesamiento.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except Exception:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_LGBM = False

# Modelos evaluados durante los experimentos.
LEARNING_MODELS = ["ridge", "poisson", "random_forest", "lightgbm"]

# Baselines de referencia.
BASELINE_MODELS = ["naive", "seasonal_naive"]


def make_learning_model(name: str, random_state: int = 42):
    """Construye un modelo de aprendizaje a partir de su identificador."""
    
    if name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", Ridge(alpha=1.0, random_state=random_state))
        ])
    
    if name == "poisson":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", PoissonRegressor(alpha=1.0, max_iter=300))
        ])
    
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=60,
            max_depth=14,
            min_samples_leaf=2, 
            n_jobs=-1,
            random_state=random_state
        )
    
    if name == "lightgbm":
        if _HAS_LGBM:
            return LGBMRegressor(
                n_estimators=200, 
                learning_rate=0.05,
                num_leaves=31, 
                subsample=0.9,
                colsample_bytree=0.9, 
                random_state=random_state,
                n_jobs=-1, 
                verbose=-1
            )
        
        return GradientBoostingRegressor(random_state=random_state)
    
    raise ValueError(f"Modelo desconocido: {name}")


def predict_naive(X_test, target="espera_al_cierre"):
    """Genera predicciones utilizando el último valor observado.

    El baseline toma como predicción el valor de la primera variable rezagada disponible en la matriz de diseño.
    """

    col = f"{target}_lag_1"

    return np.clip(X_test[col].to_numpy(dtype="float64"), 0, None)


def predict_seasonal_naive(X_test, meta_test, y_train, meta_train, target="espera_al_cierre"):
    """Genera predicciones utilizando el comportamiento histórico del mismo intervalo.

    Para cada observación se busca el valor histórico correspondiente al mismo paradero, tipo de recorrido e intervalo temporal.

    Si no existe información suficiente, se utilizan sucesivamente:

    1. La media del intervalo en el entrenamiento.
    2. La media global del entrenamiento.
    """

    import pandas as pd
    train = meta_train.copy()
    train["y"] = y_train

    # Estadísticos históricos utilizados por el baseline estacional.
    lookup = (train.sort_values("fecha")
              .groupby(["parada_observada", "tipo_recorrido", "interval_idx"])["y"]
              .mean())
    interval_mean = train.groupby("interval_idx")["y"].mean()
    global_mean = float(train["y"].mean()) if len(train) else 0.0

    preds = []
    for _, r in meta_test.iterrows():
        key = (r["parada_observada"], r["tipo_recorrido"], r["interval_idx"])
        if key in lookup.index and pd.notna(lookup.loc[key]):
            preds.append(lookup.loc[key])
        elif r["interval_idx"] in interval_mean.index:
            preds.append(interval_mean.loc[r["interval_idx"]])
        else:
            preds.append(global_mean)

    return np.clip(np.asarray(preds, dtype="float64"), 0, None)
