# Multi-objective calibration of a kinetic model and oxygen-constrained feed design for fed-batch cultivation of recombinant *Escherichia coli*

Code and data supporting the article:

> Barrera Hernández GI, Gómez Acata RV, Alfaro-Ponce M, Sosa Hernández VA. *Multi-objective calibration of a kinetic model and oxygen-constrained feed design for fed-batch cultivation of recombinant Escherichia coli.* Manuscript under review.

Archived version: [ZENODO DOI]

## Overview

The repository contains the Python scripts and result files used to:

1. Estimate eight kinetic and oxygen-transfer parameters of a dual-substrate Monod model with Luedeking–Piret product formation and a dissolved-oxygen balance, separately for two batch data sets (F1 and F2), with NSGA-III and one objective per state variable (biomass, glucose, OFP, and dissolved oxygen).
2. Determine, with a constrained genetic algorithm, the feed glucose concentration that maximizes final biomass in fed-batch operation, subject to a maximum culture volume and a minimum dissolved-oxygen level.
3. Compare model predictions with the feeding phase of an experimental fed-batch run.
4. Generate the figures and the sensitivity analysis reported in the article.

## Files

| File | Description |
|---|---|
| `NSGA3_calibracion_por_fermentacion.py` | Parameter estimation with NSGA-III for F1 and F2 (10 seeds each). Contains the experimental batch data. Writes `resultados_NSGA3_por_fermentacion.xlsx`. |
| `FedBatch_optimizacion_validacion.py` | Fed-batch optimization with a GA (10 seeds), verification by grid evaluation and bisection, and predictive validation with S_in = 15.43 g/L. Contains the validation data. Writes `resultados_fedbatch.xlsx`. |
| `figuras_articulo.py` | Calibration figures: `ajuste_F1.png`, `ajuste_F2.png`, `hipervolumen_F1.png`, `hipervolumen_F2.png`. |
| `figuras_fedbatch.py` | Fed-batch figures (`fedbatch_optimo.png`, `fedbatch_validacion.png`, `fedbatch_convergencia_AG.png`) and sensitivity analysis of the optimum (`sensibilidad_fedbatch.xlsx`). |
| `resultados_NSGA3_por_fermentacion.xlsx` | Calibration results reported in the article. |
| `resultados_fedbatch.xlsx` | Fed-batch optimization and validation results reported in the article. |
| `sensibilidad_fedbatch.xlsx` | Optimum as a function of the dissolved-oxygen limit and with the Chebyshev parameter vector. |
| `requirements.txt` | Package versions used for the article. |

Variable names, comments, and Excel sheet names are in Spanish. Glossary of sheet names:

| Sheet | Content |
|---|---|
| `configuracion` | Constants, bounds, and algorithm settings |
| `F1_seleccionada`, `F2_seleccionada` | Selected parameter vector (column `valor`) and Chebyshev solution (column `valor_chebyshev`) |
| `F1_ajuste`, `F2_ajuste` | Goodness of fit (R², RMSE, NRMSE, MBE) |
| `F1_frente_final`, `F2_frente_final` | Pooled final Pareto front |
| `F1_por_semilla`, `F2_por_semilla` | Solution selected in each run (seed) |
| `F1_hipervolumen`, `F2_hipervolumen` | Normalized hypervolume per generation and seed |
| `F1_reglas_seleccion`, `F2_reglas_seleccion` | Fit statistics under alternative selection rules |
| `AG`, `biseccion`, `barrido_Sin` | GA results per seed, bisection, and grid evaluation of S_in |
| `prediccion_vs_medido`, `metricas_validacion` | Predicted vs. measured values and validation metrics |
| `trayectoria_optimo`, `convergencia_sN` | Trajectory at the optimum and GA convergence for seed N |

## Requirements

Python 3.14.3 and the packages listed in `requirements.txt`:

```
python -m pip install -r requirements.txt
```

Use the listed pymoo version. With the same random seeds, other pymoo versions can follow a different search path and give different results.

## How to run

Place all files in the same folder; the scripts read the result files and import each other from their own folder.

To regenerate the figures and the sensitivity analysis from the result files provided:

```
python figuras_articulo.py
python figuras_fedbatch.py
```

To repeat the full computation:

```
python NSGA3_calibracion_por_fermentacion.py
python FedBatch_optimizacion_validacion.py
python figuras_articulo.py
python figuras_fedbatch.py
```

The calibration (20 NSGA-III runs) is the slowest step. Both computation scripts accept `--prueba` for a quick test with fewer seeds and generations; test results are not those of the article. Run the figure scripts last: the computation scripts also save diagnostic plots with Spanish labels under the same file names as the article figures.

## Data

The experimental data are included in the scripts:

- F1: batch cultivation, 0–7 h (calibration).
- F2: batch phase of the fed-batch run, 0–5 h (calibration).
- Feeding phase of the fed-batch run, 6–11 h (validation).

Concentrations are in g/L. Glucose readings below the detection limit of the glucometer (0.1 g/L) are recorded as 0. Dissolved oxygen is stored in g/L, converted from percentage of air saturation with a factor of 0.005697485 g/L, and reconverted in the scripts with O₂* = 0.005474 g/L (see Methods in the article).

## License

Code: MIT License (see `LICENSE`). Data and result files: CC BY 4.0.

## Use of generative AI

Parts of the code were written with the assistance of generative AI (Claude, Anthropic); see the acknowledgements of the article.
