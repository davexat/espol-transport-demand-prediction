"""Implementación de las estrategias de imputación y su evaluación.

Este módulo proporciona las herramientas necesarias para:

- Ajustar y aplicar diferentes estrategias de imputación sobre los intervalos con valores faltantes.
- Comparar dichas estrategias mediante enmascaramiento artificial de valores observados.
- Seleccionar automáticamente la estrategia con mejor desempeño.

Estrategias implementadas:

- ConditionalImputer: imputación mediante medianas condicionales con jerarquía de respaldo.
- KNNImputer: imputación basada en vecinos más cercanos.
- IterativeImputer: imputación iterativa utilizando regresión bayesiana.

La imputación se aplica únicamente a las variables reconstruibles del dataset:

- espera_al_cierre
- llegan
- se_retiran

Todas las estrategias utilizan únicamente información disponible durante el entrenamiento. El conjunto de prueba nunca participa en el ajuste del imputador, evitando fuga de información.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge

CONTEXT_CATEGORICAL = ["dia_semana", "turno", "tipo_recorrido",
                       "parada_observada", "franja"]
CONTEXT_NUMERIC = ["interval_idx", "hora"]
TARGET_COLS = ["espera_al_cierre", "llegan", "se_retiran"]


def _encode_context(df: pd.DataFrame, categories: dict | None):
    """Codifica las variables de contexto mediante one-hot encoding.

    Las categorías se fijan durante el entrenamiento y se reutilizan en la transformación para garantizar la consistencia de la matriz de entrada.
    """
    if categories is None:
        categories = {c: sorted(df[c].dropna().astype(str).unique())
                      for c in CONTEXT_CATEGORICAL}
    parts = [df[CONTEXT_NUMERIC].astype("float64").reset_index(drop=True)]

    for c in CONTEXT_CATEGORICAL:
        vals = df[c].astype(str).values
        for cat in categories[c]:
            parts.append(pd.Series((vals == cat).astype("float64"), name=f"{c}={cat}"))
    
    return pd.concat(parts, axis=1), categories


class ConditionalImputer:
    """Imputador basado en medianas condicionales.

    La imputación se realiza utilizando distintos niveles de granularidad:

    1. día de la semana + intervalo + parada;
    2. intervalo + parada;
    3. intervalo;
    4. mediana global.

    Cada nivel actúa como respaldo cuando el anterior no dispone de información suficiente.
    """

    def __init__(self, target_cols=TARGET_COLS):
        self.target_cols = target_cols
        self.tables = {}
        self.global_ = {}

    def fit(self, df_train: pd.DataFrame):
        obs = df_train[df_train["espera_al_cierre"].notna()]

        for col in self.target_cols:
            self.tables[(col, "fine")] = (obs.groupby(["dia_semana", "interval_idx", "parada_observada"])[col].median())
            self.tables[(col, "mid")] = (obs.groupby(["interval_idx", "parada_observada"])[col].median())
            self.tables[(col, "coarse")] = obs.groupby("interval_idx")[col].median()
            self.global_[col] = float(obs[col].median())
        
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        for col in self.target_cols:
            miss = out[col].isna()
            if not miss.any():
                continue
            fine = self.tables[(col, "fine")]
            mid = self.tables[(col, "mid")]
            coarse = self.tables[(col, "coarse")]

            def lookup(row):
                for key, tab in (
                    ((row["dia_semana"], row["interval_idx"], row["parada_observada"]), fine),
                    ((row["interval_idx"], row["parada_observada"]), mid),
                    (row["interval_idx"], coarse)
                ):
                    if key in tab.index and pd.notna(tab.loc[key]): 
                        return tab.loc[key]
                return self.global_[col]

            out.loc[miss, col] = out.loc[miss].apply(lookup, axis=1)
        return out


class SklearnMatrixImputer:
    """Adaptador para imputadores matriciales de scikit-learn.

    Construye una matriz formada por variables de contexto codificadas y las columnas a imputar, permitiendo utilizar KNNImputer o IterativeImputer con la misma interfaz.
    """

    def __init__(self, kind: str = "knn", target_cols=TARGET_COLS, knn_neighbors: int = 5, random_state: int = 42):
        self.kind = kind
        self.target_cols = target_cols
        self.knn_neighbors = knn_neighbors
        self.random_state = random_state
        self.categories = None
        self.imputer = None
        self.columns_ = None

    def _matrix(self, df):
        ctx, self.categories = _encode_context(df, self.categories)
        tgt = df[self.target_cols].astype("float64").reset_index(drop=True)
        mat = pd.concat([ctx, tgt], axis=1)
        self.columns_ = mat.columns
        return mat

    def fit(self, df_train: pd.DataFrame):
        mat = self._matrix(df_train)

        if self.kind == "knn":
            self.imputer = KNNImputer(n_neighbors=self.knn_neighbors)
        else:
            self.imputer = IterativeImputer(
                estimator=BayesianRidge(),
                random_state=self.random_state,
                max_iter=10, sample_posterior=False
            )
        self.imputer.fit(mat.values)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ctx, _ = _encode_context(df, self.categories)
        tgt = df[self.target_cols].astype("float64").reset_index(drop=True)
        mat = pd.concat([ctx, tgt], axis=1)[self.columns_]
        filled = self.imputer.transform(mat.values)
        filled = pd.DataFrame(filled, columns=self.columns_)
        out = df.copy().reset_index(drop=True)

        for col in self.target_cols:
            miss = out[col].isna()
            out.loc[miss, col] = np.clip(filled.loc[miss, col].values, 0, None)

        return out


def build_imputer(name: str, cfg: dict):
    """Construye una estrategia de imputación a partir de su nombre."""

    if name == "conditional":
        return ConditionalImputer()
    if name == "knn":
        return SklearnMatrixImputer("knn", knn_neighbors=cfg["imputation"]["knn_neighbors"])
    if name == "iterative":
        return SklearnMatrixImputer("iterative", random_state=cfg["seed"])
    raise ValueError(f"Imputador desconocido: {name}")


# ----------------------------------------------------------------------------
# Evaluación por enmascaramiento artificial
# ----------------------------------------------------------------------------
def evaluate_masking(df_train_observed: pd.DataFrame, cfg: dict, target_col: str = "espera_al_cierre") -> pd.DataFrame:
    """Evalúa estrategias de imputación mediante enmascaramiento artificial.

    En cada porcentaje de enmascaramiento:

    1. Se oculta aleatoriamente un subconjunto de valores observados.
    2. Se ajusta el imputador con los datos restantes.
    3. Se reconstruyen los valores ocultos.
    4. Se calcula el error de reconstrucción.

    Se reportan MAE y RMSE tanto de forma global como por franja horaria.
    """
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    strategies = ["conditional", "knn"]
    if cfg["imputation"].get("iterative_enabled", True):
        strategies.append("iterative")

    base = df_train_observed[df_train_observed[target_col].notna()].reset_index(drop=True)
    for rate in cfg["missingness"]["masking_rates"]:
        n = len(base)
        mask_idx = rng.choice(n, size=int(round(rate * n)), replace=False)
        true_vals = base.loc[mask_idx, target_col].to_numpy(dtype="float64")

        for strat in strategies:
            masked = base.copy()
            masked.loc[mask_idx, target_col] = np.nan
            imp = build_imputer(strat, cfg)
            # se ajusta con las filas NO enmascaradas
            imp.fit(masked[masked[target_col].notna()])
            filled = imp.transform(masked)
            pred = filled.loc[mask_idx, target_col].to_numpy(dtype="float64")

            err = pred - true_vals
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            rows.append({
                "estrategia": strat, 
                "tasa_masking": rate,
                "franja": "GLOBAL", 
                "MAE": mae, 
                "RMSE": rmse,
                "n": len(mask_idx)
            })

            # Desglose por franja
            fr = base.loc[mask_idx, "franja"].to_numpy()
            for franja in pd.unique(fr):
                sel = fr == franja
                e = pred[sel] - true_vals[sel]
                rows.append({
                    "estrategia": strat, "tasa_masking": rate,
                    "franja": franja,
                    "MAE": float(np.mean(np.abs(e))),
                    "RMSE": float(np.sqrt(np.mean(e ** 2))),
                    "n": int(sel.sum())
                })
    
    return pd.DataFrame(rows)


def select_best_strategy(masking_df: pd.DataFrame) -> str:
    """Selecciona la estrategia con menor MAE promedio global."""

    g = (masking_df[masking_df["franja"] == "GLOBAL"].groupby("estrategia")["MAE"].mean().sort_values())

    return g.index[0]
