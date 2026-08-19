#!/usr/bin/env python3
"""Etapa 6: análisis exploratorio de la demanda por franja horaria y día (RQ3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config              # noqa: E402
from src.data_loading import load_all            # noqa: E402
from src.grid_integration import integrate       # noqa: E402
from src.pipeline import stage_demand_analysis    # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    data = load_all(cfg)
    integrated = integrate(data, cfg)
    res = stage_demand_analysis(cfg, integrated)
    print("Análisis de demanda por franja y día listo. Detalle en outputs/06_*.")
    print(res["resumen"])
