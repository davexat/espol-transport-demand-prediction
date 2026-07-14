"""Análisis del mecanismo de datos faltantes.

Calcula la tasa de datos faltantes por grupos, evalúa su asociación con covariables observadas mediante pruebas chi-cuadrado y proporciona una clasificación operativa del mecanismo de datos faltantes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def missing_rate_by(full: pd.DataFrame, col: str) -> pd.DataFrame:
    """Calcula la tasa de datos faltantes para cada categoría de una variable."""

    g = (full.assign(faltante=1 - full["observado"])
         .groupby(col, as_index=False)
         .agg(n=("faltante", "size"), faltantes=("faltante", "sum")))
    g["tasa_faltante"] = g["faltantes"] / g["n"]

    return g.sort_values("tasa_faltante", ascending=False)


def chi_square_dependence(full: pd.DataFrame, col: str) -> dict:
    """Evalúa la asociación entre la presencia de datos faltantes y una covariable mediante una prueba chi-cuadrado."""

    tbl = pd.crosstab(full[col], full["observado"])
    if tbl.shape[0] < 2 or tbl.shape[1] < 2:
        return {
            "col": col, 
            "chi2": np.nan, 
            "p_value": np.nan, 
            "dof": np.nan
        }
    
    chi2, p, dof, _ = chi2_contingency(tbl)

    return {
        "col": col, 
        "chi2": float(chi2), 
        "p_value": float(p), 
        "dof": int(dof)
    }


def classify(full: pd.DataFrame) -> dict:
    """Clasifica el mecanismo de datos faltantes a partir de las covariables observadas."""

    covariables = ["dia_semana", "parada_observada", "turno", "hora"]
    covariables = [c for c in covariables if c in full.columns]

    by_group = {c: missing_rate_by(full, c) for c in covariables}
    tests = [chi_square_dependence(full, c) for c in covariables]
    tests_df = pd.DataFrame(tests)

    # Evidencia de dependencia con covariables observadas → MAR.
    # En ausencia de dicha evidencia se asume MCAR.
    
    depende_observadas = bool((tests_df["p_value"] < 0.05).any())
    veredicto = "MAR" if depende_observadas else "MCAR"

    return {
        "by_group": by_group,
        "tests": tests_df,
        "veredicto": veredicto,
        "global_missing_rate": float(1 - full["observado"].mean()),
    }
