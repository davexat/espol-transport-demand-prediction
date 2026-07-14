#!/usr/bin/env python3
"""Etapa 4: evaluación de imputadores por enmascaramiento artificial."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config       # noqa: E402
from src.data_loading import load_all     # noqa: E402
from src.pipeline import stage_integrate, stage_masking  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    data = load_all(cfg)
    integrated = stage_integrate(cfg, data)
    best = stage_masking(cfg, integrated)
    print(f"Imputador seleccionado (menor MAE global promedio): {best}")
    print("Detalle por tasa y franja en outputs/04_masking_evaluacion.csv")
