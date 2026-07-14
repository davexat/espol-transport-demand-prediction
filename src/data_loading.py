"""Carga y normalización inicial de los datasets del estudio.

Se cargan los tres archivos originales y se estandarizan los tipos de datos necesarios para el resto del pipeline.

Relaciones entre datasets:
    sesion_observacion_rows.sesion_id
        ├── intervalo_demanda_rows.sesion_id
        └── evento_bus_rows.sesion_id

En esta etapa únicamente se realizan conversiones de formato y validaciones básicas. No se imputan datos ni se modifican los registros originales.
"""
from __future__ import annotations
import pandas as pd
from .config import dataset_path


def _to_seconds(series: pd.Series) -> pd.Series:
    """Convierte valores horarios a segundos desde medianoche."""

    return pd.to_timedelta(series.astype(str)).dt.total_seconds().astype("Int64")

def load_sesiones(cfg: dict) -> pd.DataFrame:
    """Carga la información de las sesiones de observación."""

    df = pd.read_csv(dataset_path(cfg, "sesiones_file"), dtype=str)

    df = df.rename(columns={"id": "sesion_id"})
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d")

    return df

def load_intervalos(cfg: dict) -> pd.DataFrame:
    """Carga los registros agregados por intervalo de 10 minutos."""

    df = pd.read_csv(
        dataset_path(cfg, "intervalos_file"),
        dtype={"id": str, "sesion_id": str},
    )

    df["inicio_seg"] = _to_seconds(df["hora_inicio"])
    df["fin_seg"] = _to_seconds(df["hora_fin"])

    numeric_columns = [
        "llegan",
        "se_retiran",
        "espera_al_cierre",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df

def load_eventos(cfg: dict) -> pd.DataFrame:
    """Carga los eventos asociados al paso de los buses."""

    df = pd.read_csv(
        dataset_path(cfg, "eventos_file"),
        dtype={"id": str, "sesion_id": str},
    )

    time_columns = [
        "hora_programada",
        "hora_real_llegada",
        "hora_real_salida",
    ]

    for column in time_columns:
        df[f"{column}_seg"] = _to_seconds(df[column])

    numeric_columns = [
        "personas_esperando_antes",
        "personas_suben",
        "personas_bajan",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df

def load_all(cfg: dict) -> dict[str, pd.DataFrame]:
    """Carga todos los datasets requeridos por el pipeline."""

    return {
        "sesiones": load_sesiones(cfg),
        "intervalos": load_intervalos(cfg),
        "eventos": load_eventos(cfg),
    }