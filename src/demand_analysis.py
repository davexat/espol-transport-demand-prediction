"""Análisis exploratorio de la demanda de pasajeros en espera.

Caracteriza la variación de `espera_al_cierre` (demanda) y de la demanda no
satisfecha en función de la franja horaria y el día de la semana, sobre las
observaciones reales de la tabla integrada (RQ3, Contribución 2).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FRANJA_ORDER = ["manana", "mediodia", "tarde"]
DIA_ORDER = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]


def _stats_by(df: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    """Media, desviación estándar, mediana y conteo de la demanda por categoría."""

    g = (df.groupby(col)["espera_al_cierre"]
         .agg(demanda_media="mean", demanda_std="std",
              demanda_mediana="median", n="size")
         .reindex(order)
         .reset_index())

    return g


def _unmet_by(df: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    """Tasa e intensidad de la demanda no satisfecha por categoría."""

    g = (df.groupby(col)["demanda_no_satisfecha"]
         .agg(tasa_intervalos_con_no_satisfecha=lambda s: float((s > 0).mean()),
              total_no_satisfecha="sum",
              media_no_satisfecha="mean")
         .reindex(order)
         .reset_index())

    return g


def analyze_demand(df: pd.DataFrame) -> dict:
    """Caracteriza la variación de la demanda por franja horaria y día de la semana."""

    por_franja = _stats_by(df, "franja", FRANJA_ORDER)
    por_dia = _stats_by(df, "dia_semana", DIA_ORDER)

    heatmap = (df.pivot_table(index="franja", columns="dia_semana",
                               values="espera_al_cierre", aggfunc="mean")
               .reindex(index=FRANJA_ORDER, columns=DIA_ORDER))

    no_satisfecha_por_franja = _unmet_by(df, "franja", FRANJA_ORDER)
    no_satisfecha_por_dia = _unmet_by(df, "dia_semana", DIA_ORDER)

    resumen = {
        "tasa_intervalos_con_no_satisfecha": float((df["demanda_no_satisfecha"] > 0).mean()),
        "n_intervalos_con_no_satisfecha": int((df["demanda_no_satisfecha"] > 0).sum()),
        "total_no_satisfecha": float(df["demanda_no_satisfecha"].sum()),
        "media_no_satisfecha_cuando_ocurre": float(
            df.loc[df["demanda_no_satisfecha"] > 0, "demanda_no_satisfecha"].mean()
        ),
    }

    return {
        "por_franja": por_franja,
        "por_dia": por_dia,
        "heatmap": heatmap,
        "no_satisfecha_por_franja": no_satisfecha_por_franja,
        "no_satisfecha_por_dia": no_satisfecha_por_dia,
        "resumen": resumen,
    }


def plot_heatmap(heatmap: pd.DataFrame, path: str | Path) -> None:
    """Genera el mapa de calor de la demanda media por franja horaria y día."""

    fig, ax = plt.subplots(figsize=(6, 3.2))
    im = ax.imshow(heatmap.values, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns)
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            val = heatmap.values[i, j]
            color = "white" if val > heatmap.values.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, label="Demanda media (espera_al_cierre)")
    ax.set_title("Demanda promedio por franja horaria y dia de la semana")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
