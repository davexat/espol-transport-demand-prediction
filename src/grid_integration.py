"""Integración temporal de los datasets y construcción de la grilla de análisis.

Este módulo genera la base utilizada durante el resto del pipeline mediante:

- Integración de las observaciones por intervalo con los eventos de bus.
- Agregación de eventos al intervalo de 10 minutos correspondiente.
- Cálculo de variables operativas derivadas.
- Construcción de la grilla temporal teórica para identificar intervalos observados y no observados.

La variable objetivo del estudio es `espera_al_cierre`. Las variables relacionadas con el abordaje (`personas_suben`, `personas_bajan` y `personas_esperando_antes`) se conservan únicamente para análisis descriptivos y no forman parte del conjunto predictor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INTERVAL_SEC = 600

# Límites horarios utilizados para construir la grilla temporal.
TURNO_BOUNDS = {
    "08h00-11h00": (8 * 3600, 11 * 3600),
    "11h00-14h00": (11 * 3600, 14 * 3600),
    "14h00-16h30": (14 * 3600, int(16.5 * 3600)),
}


def interval_start(seg: pd.Series | np.ndarray) -> pd.Series:
    """Devuelve el inicio del intervalo de 10 minutos correspondiente."""

    return (
        np.floor(np.asarray(seg, dtype="float64") / INTERVAL_SEC) * INTERVAL_SEC
    ).astype("int64")


def aggregate_bus_events(eventos: pd.DataFrame, day_start: int, day_end: int) -> pd.DataFrame:
    """Agrega los eventos de bus al nivel de intervalo temporal.

    Los eventos fuera de la ventana de observación se identifican para su posterior reporte y no participan en la agregación.
    """

    ev = eventos.copy()
    ev["fuera_ventana"] = (~ev["hora_real_llegada_seg"].between(day_start, day_end - 1)).astype(int)

    ev_in = ev[ev["fuera_ventana"] == 0].copy()
    ev_in["inicio_seg"] = interval_start(ev_in["hora_real_llegada_seg"])

    ev_in["demanda_no_satisfecha"] = np.maximum(
        0, ev_in["personas_esperando_antes"] - ev_in["personas_suben"])

    agg = (ev_in.groupby(["sesion_id", "inicio_seg"], as_index=False).agg(
        n_buses=("id", "count"),
        suben_sum=("personas_suben", "sum"),
        bajan_sum=("personas_bajan", "sum"),
        esperando_antes_sum=("personas_esperando_antes", "sum"),
        demanda_no_satisfecha=("demanda_no_satisfecha", "sum"),
        hora_programada_min=("hora_programada_seg", "min")
    ))
    
    return agg, ev

def build_observed_table(data: dict, cfg: dict) -> pd.DataFrame:
    """Tabla observada a nivel de intervalo, con contexto de sesión y buses."""

    ses = data["sesiones"]
    inte = data["intervalos"].copy()
    day_start = int(pd.to_timedelta(cfg["data"]["day_start"]).total_seconds())
    day_end = int(pd.to_timedelta(cfg["data"]["day_end"]).total_seconds())

    bus_agg, _ = aggregate_bus_events(data["eventos"], day_start, day_end)

    df = inte.merge(ses, on="sesion_id", how="left", validate="many_to_one")
    df = df.merge(bus_agg, on=["sesion_id", "inicio_seg"], how="left")

    df["bus_presente"] = df["n_buses"].notna().astype(int)
    for c in ["n_buses", "suben_sum", "bajan_sum", "esperando_antes_sum",
              "demanda_no_satisfecha"]:
        df[c] = df[c].fillna(0)

    df["interval_idx"] = ((df["inicio_seg"] - day_start) // INTERVAL_SEC).astype(int)
    df["hora"] = (df["inicio_seg"] // 3600).astype(int)
    df["franja"] = pd.cut(df["hora"], bins=[-1, 10, 13, 24], labels=["manana", "mediodia", "tarde"]).astype(str)
    df["observado"] = 1

    return df


def build_theoretical_grid(data: dict, cfg: dict) -> pd.DataFrame:
    """Construye la grilla temporal del universo observado."""

    ses = data["sesiones"]
    day_start = int(pd.to_timedelta(cfg["data"]["day_start"]).total_seconds())

    fechas = ses[["fecha", "dia_semana"]].drop_duplicates()
    paradas = ses[["parada_observada", "tipo_recorrido"]].drop_duplicates()
    turnos = list(TURNO_BOUNDS.keys())

    rows = []
    for turno in turnos:
        t0, t1 = TURNO_BOUNDS[turno]
        starts = list(range(t0, t1, INTERVAL_SEC))
        for s in starts:
            rows.append({"turno": turno, "inicio_seg": s})
    grid_intervals = pd.DataFrame(rows)

    grid = (fechas.merge(paradas, how="cross").merge(grid_intervals, how="cross"))
    grid["interval_idx"] = ((grid["inicio_seg"] - day_start) // INTERVAL_SEC).astype(int)
    grid["hora"] = (grid["inicio_seg"] // 3600).astype(int)
    grid["franja"] = pd.cut(grid["hora"], bins=[-1, 10, 13, 24], labels=["manana", "mediodia", "tarde"]).astype(str)

    return grid


def integrate(data: dict, cfg: dict) -> dict:
    """Integra las observaciones con la grilla temporal del estudio."""

    observed = build_observed_table(data, cfg)
    grid = build_theoretical_grid(data, cfg)
    target = cfg["target"]["name"]

    keys = ["fecha", "parada_observada", "tipo_recorrido", "turno", "inicio_seg"]
    obs_small = observed[keys + [target, "llegan", "se_retiran", "bus_presente", "hora_programada_min", "demanda_no_satisfecha"]]
    full = grid.merge(obs_small, on=keys, how="left")
    full["observado"] = full[target].notna().astype(int)

    coverage = full["observado"].mean()
    
    return {
        "observed": observed,
        "grid": grid,
        "full": full,
        "coverage": coverage,
    }
