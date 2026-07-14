#!/usr/bin/env python3
"""Etapa 1: caracterización inicial de los datos."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config          # noqa: E402
from src.data_loading import load_all        # noqa: E402
from src.pipeline import stage_characterize  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    data = load_all(cfg)
    resumen = stage_characterize(cfg, data)
    print("Caracterización lista. Cobertura y detalle en outputs/01_*.")
    print(resumen)
