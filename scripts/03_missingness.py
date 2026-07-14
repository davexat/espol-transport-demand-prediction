#!/usr/bin/env python3
"""Etapa 3: clasificación del mecanismo de datos faltantes (MCAR/MAR/MNAR)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config          # noqa: E402
from src.data_loading import load_all        # noqa: E402
from src.pipeline import stage_integrate, stage_missingness  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    data = load_all(cfg)
    integrated = stage_integrate(cfg, data)
    res = stage_missingness(cfg, integrated)
    print(f"Veredicto del mecanismo de faltante: {res['veredicto']}")
    print(f"Tasa de faltante global: {res['global_missing_rate']:.1%}")
