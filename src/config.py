"""Carga la configuración del proyecto y resuelve las rutas utilizadas por el pipeline."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# Directorio base del proyecto
METHOD_ROOT = Path(__file__).resolve().parents[1]

def load_config(config_path: str | os.PathLike | None = None) -> dict:
    """Carga el archivo de configuración y resuelve las rutas absolutas."""

    if config_path is None:
        config_path = METHOD_ROOT / "config.yaml"

    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    ds = cfg["paths"]["datasets_dir"]
    ds_path = Path(ds)
    if not ds_path.is_absolute():
        candidate = (METHOD_ROOT / ds).resolve()
        if not candidate.exists():
            candidate = (METHOD_ROOT / "datasets").resolve()
        ds_path = candidate
    
    cfg["paths"]["datasets_dir_resolved"] = str(ds_path)
    out = Path(cfg["paths"]["outputs_dir"])
    if not out.is_absolute():
        out = (METHOD_ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["outputs_dir_resolved"] = str(out)

    return cfg


def dataset_path(cfg: dict, key: str) -> Path:
    """Devuelve la ruta del dataset asociado a una clave de configuración."""

    return Path(cfg["paths"]["datasets_dir_resolved"]) / cfg["data"][key]


def output_path(cfg: dict, *parts: str) -> Path:
    """Devuelve la ruta de un archivo de salida, creando los directorios necesarios."""

    p = Path(cfg["paths"]["outputs_dir_resolved"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
