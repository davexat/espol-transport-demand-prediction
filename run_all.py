#!/usr/bin/env python3
"""Ejecuta el pipeline completo de la metodología.

Uso:
    python run_all.py [--config config.yaml]

El pipeline realiza, en orden:

1. Caracterización de los datos.
2. Integración de datasets y construcción de la grilla temporal.
3. Análisis del mecanismo de datos faltantes.
4. Evaluación y selección del método de imputación.
5. Entrenamiento y evaluación de los modelos predictivos.

Todos los resultados se almacenan en `outputs/`.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.pipeline import run_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Ruta a config.yaml")
    args = ap.parse_args()

    out = run_pipeline(args.config)

    print("=" * 64)
    print("RESUMEN DE EJECUCIÓN")
    print("=" * 64)
    print(f"Sesiones: {out['resumen']['n_sesiones']} | "
          f"Intervalos: {out['resumen']['n_intervalos']} | "
          f"Eventos bus: {out['resumen']['n_eventos_bus']}")
    print(f"Cobertura observada de la grilla: "
          f"{out['cobertura']['cobertura_observada']:.1%} "
          f"({out['cobertura']['n_observadas']}/{out['cobertura']['n_celdas_grilla']})")
    print(f"Mecanismo de faltante (veredicto): {out['missingness']['veredicto']}")
    print(f"Imputador seleccionado (menor MAE en masking): {out['imputador']}")
    print(f"Folds evaluados: {out['experiment']['n_folds_evaluados']}")
    print("-" * 64)
    print("Resumen de métricas (media entre folds):")
    summ = out["experiment"]["summary"]
    cols = ["escenario", "modelo", "MAE_mean", "MAE_std", "WAPE_mean", "R2_mean"]
    with_cols = [c for c in cols if c in summ.columns]
    print(summ[with_cols].to_string(index=False))
    print("=" * 64)
    print("Resultados escritos en outputs/. Ver docs/metodologia.md.")


if __name__ == "__main__":
    main()
