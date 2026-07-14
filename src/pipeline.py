"""Pipeline de la metodología.

Cada etapa ejecuta una fase del flujo metodológico, guarda sus resultados en `outputs/` y devuelve la información necesaria para la siguiente etapa.
"""
from __future__ import annotations

import json

import pandas as pd

from . import missingness as mo
from .config import load_config, output_path
from .data_loading import load_all
from .experiment import run_experiment
from .grid_integration import integrate
from .imputation import evaluate_masking, select_best_strategy


def _save_csv(df: pd.DataFrame, cfg: dict, *parts: str):
    p = output_path(cfg, *parts)
    df.to_csv(p, index=False)
    return p


def stage_characterize(cfg: dict, data: dict) -> dict:
    """Caracteriza los datasets y genera un resumen descriptivo."""

    ses, inte, ev = data["sesiones"], data["intervalos"], data["eventos"]
    
    resumen = {
        "n_sesiones": int(len(ses)),
        "n_intervalos": int(len(inte)),
        "n_eventos_bus": int(len(ev)),
        "fechas": [str(ses["fecha"].min().date()), str(ses["fecha"].max().date())],
        "n_fechas": int(ses["fecha"].nunique()),
        "n_paradas": int(ses["parada_observada"].nunique()),
        "turnos": sorted(ses["turno"].unique().tolist()),
        "target_stats_espera_al_cierre": {
            "min": float(inte["espera_al_cierre"].min()),
            "max": float(inte["espera_al_cierre"].max()),
            "mean": round(float(inte["espera_al_cierre"].mean()), 3),
        },
    }

    dic = []
    for name, df in [("sesion", ses), ("intervalo", inte), ("evento_bus", ev)]:
        for c in df.columns:
            dic.append({
                "dataset": name, 
                "variable": c, 
                "dtype": str(df[c].dtype)
            })
    
    _save_csv(pd.DataFrame(dic), cfg, "01_diccionario_variables.csv")
    with open(output_path(cfg, "01_caracterizacion.json"), "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, ensure_ascii=False, indent=2)
    
    return resumen


def stage_integrate(cfg: dict, data: dict) -> dict:
    """Integra los datasets y calcula la cobertura temporal observada."""
    
    integrated = integrate(data, cfg)
    _save_csv(integrated["observed"], cfg, "02_tabla_integrada_observada.csv")

    cov = {
        "cobertura_observada": round(float(integrated["coverage"]), 4),
        "n_celdas_grilla": int(len(integrated["full"])),
        "n_observadas": int(integrated["full"]["observado"].sum()),
        "n_faltantes": int((integrated["full"]["observado"] == 0).sum())
    }

    with open(output_path(cfg, "02_cobertura.json"), "w", encoding="utf-8") as fh:
        json.dump(cov, fh, ensure_ascii=False, indent=2)
    integrated["_cobertura"] = cov

    return integrated


def stage_missingness(cfg: dict, integrated: dict) -> dict:
    """Analiza el mecanismo de datos faltantes y genera los reportes correspondientes."""
    
    res = mo.classify(integrated["full"])
    for col, tab in res["by_group"].items():
        _save_csv(tab, cfg, f"03_faltante_por_{col}.csv")
    _save_csv(res["tests"], cfg, "03_pruebas_chi2.csv")
    
    veredicto = {
        "veredicto_mecanismo": res["veredicto"],
        "tasa_faltante_global": round(res["global_missing_rate"], 4)
    }
    
    with open(output_path(cfg, "03_missingness_veredicto.json"), "w", encoding="utf-8") as fh:
        json.dump(veredicto, fh, ensure_ascii=False, indent=2)
    
    return res


def stage_masking(cfg: dict, integrated: dict) -> str:
    """Analiza el mecanismo de datos faltantes y genera los reportes correspondientes."""

    masking = evaluate_masking(integrated["observed"], cfg)
    _save_csv(masking, cfg, "04_masking_evaluacion.csv")
    best = select_best_strategy(masking)

    with open(output_path(cfg, "04_imputador_seleccionado.json"), "w", encoding="utf-8") as fh:
        json.dump({"imputador_seleccionado": best}, fh, ensure_ascii=False, indent=2)

    return best


def stage_experiment(cfg: dict, integrated: dict, best_strategy: str) -> dict:
    """Ejecuta el experimento predictivo utilizando la estrategia de imputación seleccionada."""

    cfg["_selected_imputer"] = best_strategy
    res = run_experiment(integrated, cfg)

    _save_csv(res["per_fold"], cfg, "05_metricas_por_fold.csv")
    _save_csv(res["summary"], cfg, "05_resumen_metricas.csv")

    return res


def run_pipeline(config_path: str | None = None) -> dict:
    """Ejecuta el pipeline completo de la metodología."""

    cfg = load_config(config_path)
    data = load_all(cfg)
    resumen = stage_characterize(cfg, data)
    integrated = stage_integrate(cfg, data)
    missing = stage_missingness(cfg, integrated)
    best = stage_masking(cfg, integrated)
    experiment = stage_experiment(cfg, integrated, best)

    return {
        "cfg": cfg, 
        "resumen": resumen, 
        "cobertura": integrated["_cobertura"],
        "missingness": missing, 
        "imputador": best, 
        "experiment": experiment
    }
