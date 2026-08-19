# ESPOL Transport Demand Prediction

Pipeline que estima la demanda de pasajeros en espera del sistema de transporte interno de ESPOL y evalúa si la imputación de intervalos faltantes mejora el pronóstico.

## Requisitos

- Python ≥ 3.10

## Instalación

```bash
git clone https://github.com/davexat/espol-transport-demand-prediction.git
cd espol-transport-demand-prediction

# Crear entorno virtual
python -m venv .venv

# Activar entorno
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

Pipeline completo:

```bash
python run_all.py
```

Configuración alternativa:

```bash
python run_all.py --config custom.yaml
```

Por etapas:

```bash
python scripts/01_characterize.py
python scripts/02_integrate.py
python scripts/06_demand_analysis.py
python scripts/03_missingness.py
python scripts/04_impute_masking_eval.py
python scripts/05_train_evaluate.py
```

## Configuración

Todas las constantes se definen en `config.yaml`: rutas, lags, ventanas móviles, modelos, esquema de validación y tasas de enmascaramiento.

## Estructura

| Directorio | Contenido |
|---|---|
| `data/` | CSV de origen (sesiones, intervalos, eventos de bus) |
| `src/` | Módulos del pipeline (config, carga, integración, imputación, modelos, evaluación) |
| `scripts/` | Scripts por etapa |
| `outputs/` | Resultados generados (métricas, cobertura, imputador seleccionado) |

## Outputs

| Archivo | Descripción |
|---|---|
| `01_caracterizacion.json` | Resumen de sesiones, intervalos y eventos |
| `02_cobertura.json` | Cobertura temporal observada |
| `03_missingness_veredicto.json` | Clasificación del mecanismo de faltante (MAR/MCAR) |
| `04_imputador_seleccionado.json` | Estrategia de imputación seleccionada |
| `05_resumen_metricas.csv` | Métricas finales (MAE, RMSE, WAPE, R²) por escenario y modelo |
| `06_demanda_por_franja.csv` / `06_demanda_por_dia.csv` | Demanda media, mediana y desviación estándar por franja horaria y por día |
| `06_demanda_heatmap_franja_dia.csv` | Demanda media cruzada por franja horaria y día |
| `06_no_satisfecha_por_franja.csv` / `06_no_satisfecha_por_dia.csv` | Tasa e intensidad de la demanda no satisfecha por franja horaria y por día |
| `06_no_satisfecha_resumen.json` | Resumen global de la demanda no satisfecha |
