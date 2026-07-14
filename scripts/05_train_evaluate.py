#!/usr/bin/env python3
"""Etapa 5: entrenamiento y evaluación — Available vs. Imputed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config       # noqa: E402
from src.data_loading import load_all     # noqa: E402
from src.pipeline import (stage_integrate, stage_masking,  # noqa: E402
                          stage_experiment)

if __name__ == "__main__":
    cfg = load_config()
    data = load_all(cfg)
    integrated = stage_integrate(cfg, data)
    best = stage_masking(cfg, integrated)
    res = stage_experiment(cfg, integrated, best)
    print("Resumen de métricas (media entre folds):")
    cols = ["escenario", "modelo", "MAE_mean", "MAE_std", "WAPE_mean", "R2_mean"]
    summ = res["summary"]
    print(summ[[c for c in cols if c in summ.columns]].to_string(index=False))
