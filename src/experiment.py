"""Experimento de comparación entre registros observados y dataset imputado.

Implementa el protocolo experimental utilizado para responder la RQ2:

    ¿La imputación de intervalos temporales faltantes mejora el desempeño predictivo respecto al entrenamiento únicamente con los registros observados?

Protocolo experimental
----------------------
- Validación temporal mediante rolling-origin con ventana expansiva.
- El imputador se ajusta exclusivamente con el conjunto de entrenamiento de cada fold para evitar fuga de información.
- Ambos escenarios se evalúan sobre el mismo conjunto de prueba, compuesto únicamente por observaciones reales.
- Se comparan modelos de referencia (baselines) y modelos de aprendizaje automático utilizando MAE, RMSE, WAPE y R².
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluation as ev
from . import models as mdl
from .features import build_design_matrix
from .imputation import build_imputer
from .splits import folds_to_row_indices, rolling_origin_date_folds

TARGET = "espera_al_cierre"


def _fit_predict(model_name, X_train, y_train, X_test, random_state):
    """Entrena un modelo y genera predicciones no negativas."""
    
    model = mdl.make_learning_model(model_name, random_state=random_state)
    model.fit(X_train.values, y_train)
    predictions = model.predict(X_test.values)
    return np.clip(predictions, 0, None)


def run_experiment(integrated: dict, cfg: dict) -> dict:
    """Ejecuta el experimento completo de comparación Available vs. Imputed."""

    seed = cfg["seed"]

    observed = integrated["observed"]
    full_grid = integrated["full"]

    # Matriz de referencia construida únicamente con observaciones reales.
    X_obs, y_obs, meta_obs, dummy_columns = build_design_matrix(
        observed,
        cfg,
        TARGET,
    )

    folds_dates = rolling_origin_date_folds(
        meta_obs["fecha"],
        cfg["split"]["n_folds"],
        cfg["split"]["initial_train_fraction"],
        cfg["split"]["scheme"],
    )

    folds = folds_to_row_indices(meta_obs, folds_dates, "fecha")

    selected_imputer = cfg.get("_selected_imputer", "conditional")

    per_fold_results = []

    for fold, fold_dates in zip(folds, folds_dates):

        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]

        if len(train_idx) < 30 or len(test_idx) < 5:
            continue

        X_test = X_obs.iloc[test_idx]
        y_test = y_obs[test_idx]

        meta_test = meta_obs.iloc[test_idx].reset_index(drop=True)

        # ==========================
        # Escenario: Available
        # ==========================

        X_train_available = X_obs.iloc[train_idx]
        y_train_available = y_obs[train_idx]

        meta_train_available = (
            meta_obs.iloc[train_idx]
            .reset_index(drop=True)
        )

        # ==========================
        # Escenario: Imputed
        # ==========================

        train_dates = set(fold_dates["train_dates"])

        observed_train = observed[
            observed["fecha"].isin(train_dates)
        ]

        full_train = (
            full_grid[full_grid["fecha"].isin(train_dates)]
            .copy()
        )

        imputer = build_imputer(selected_imputer, cfg)
        imputer.fit(observed_train)

        imputed_train = imputer.transform(full_train)

        X_train_imputed, y_train_imputed, _, _ = build_design_matrix(
            imputed_train,
            cfg,
            TARGET,
            dummy_columns=dummy_columns,
        )

        # ==========================
        # Baselines
        # ==========================

        baselines = {
            "naive": mdl.predict_naive(X_test, TARGET),
            "seasonal_naive": mdl.predict_seasonal_naive(
                X_test,
                meta_test,
                y_train_available,
                meta_train_available,
                TARGET,
            ),
        }

        for model_name, prediction in baselines.items():

            metrics = ev.all_metrics(y_test, prediction)

            metrics.update({
                "fold": fold["fold"],
                "escenario": "baseline",
                "modelo": model_name,
                "n_test": len(y_test),
            })

            per_fold_results.append(metrics)

        # ==========================
        # Modelos de aprendizaje
        # ==========================

        scenarios = {
            "available": (
                X_train_available,
                y_train_available,
            ),
            "imputed": (
                X_train_imputed,
                y_train_imputed,
            ),
        }

        for scenario, (X_train, y_train) in scenarios.items():

            for model_name in mdl.LEARNING_MODELS:

                prediction = _fit_predict(
                    model_name,
                    X_train,
                    y_train,
                    X_test,
                    seed,
                )

                metrics = ev.all_metrics(y_test, prediction)

                metrics.update({
                    "fold": fold["fold"],
                    "escenario": scenario,
                    "modelo": model_name,
                    "n_test": len(y_test),
                })

                per_fold_results.append(metrics)

    per_fold = pd.DataFrame(per_fold_results)

    summary = ev.summarize_folds(
        per_fold,
        ["escenario", "modelo"],
    )

    return {
        "per_fold": per_fold,
        "summary": summary,
        "n_folds_evaluados": (
            per_fold["fold"].nunique()
            if not per_fold.empty
            else 0
        ),
    }