#!/usr/bin/env python3
"""Etapa 2: integración de datasets y construcción de la grilla temporal."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config       # noqa: E402
from src.data_loading import load_all     # noqa: E402
from src.pipeline import stage_integrate  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    data = load_all(cfg)
    integrated = stage_integrate(cfg, data)
    print("Integración lista.")
    print(integrated["_cobertura"])
