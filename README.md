# ESPOL Transport Demand Prediction

Pipeline que estima la demanda de pasajeros en espera del sistema de transporte interno de ESPOL y evalúa si la imputación de intervalos faltantes mejora el pronóstico.

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python run_all.py                  # pipeline completo
python run_all.py --config custom.yaml  # config alternativa
```

O por etapas:

```bash
python scripts/01_characterize.py
python scripts/02_integrate.py
python scripts/03_missingness.py
python scripts/04_impute_masking_eval.py
python scripts/05_train_evaluate.py
```

## Estructura

| Directorio | Contenido |
|---|---|
| `data/` | CSV de origen (sesiones, intervalos, eventos de bus) |
| `src/` | Módulos del pipeline (config, carga, integración, imputación, modelos, evaluación) |
| `scripts/` | Scripts por etapa |
| `outputs/` | Resultados generados (métricas, cobertura, imputador seleccionado) |

## Configuración

Todas las constantes (rutas, lags, ventanas, modelos, splits) se definen en `config.yaml`.

## Variable objetivo

`espera_al_cierre` — número de personas en espera al cierre de cada intervalo de 10 minutos.
