# Soil-Remote-Sensing Correlation Screening (Northern Kazakhstan steppe)

Reproducible code and data for the manuscript *"Digital Soil Mapping of the Steppe Zone of
Northern Kazakhstan: Relationships Between Agrochemical Soil Properties and Multimodal Remote
Sensing Data - Analysis of 530 Features."*

This is the **correlation-screening / feature-quality** study. It is the companion of, and
**independent from**, the predictive-modelling paper (Agriculture 2026, 16(11), 1239;
https://www.mdpi.com/2077-0472/16/11/1239), with which it shares only the underlying soil
dataset. The two studies answer different questions - *which* remote-sensing features relate to
*which* soil property and *why* (this paper) versus ML/DL prediction benchmarking (the modelling
paper) - and share no results or manuscript text.

## Contents
- `math_statistics/` - screening pipeline: Spearman screening with Benjamini-Hochberg FDR
  (`correlation_analysis.py`), leakage-controlled screening (temporal alignment + out-of-farm
  Farm-LOFO), corrected manuscript tables (`corrected_tables.py`), nested variance components via
  Searle EMS (`nested_variance.py`), permutation + bootstrap false-positive controls
  (`permutation_bootstrap.py`), variance decomposition, spatial autocorrelation, confounding.
- `ML/` - spatial cross-validation and feature quality: `cv_scheme_comparison.py`
  (Spatial-block + LLTO), `farm_lofo_*.py` (out-of-farm rho/R2/RMSE/RPD), `feature_quality_cv.py`
  (per-feature within/between/cross-year decomposition), `feature_leakage_taxonomy.py` (empirical
  generalizable / zonal_only / unstable / weak classification).
- `data/features/master_dataset_old.csv` - 530-column matrix (11 meta + 7 soil targets + 512 RS
  features; 1085 samples / 20 farms / 81 fields, 2022-2023).
- `math_statistics/output/`, `ML/results/` - committed, reproducible result tables.
- `article/` - the manuscript.

## Reproduce
    python -m venv .venv && . .venv/bin/activate && pip install -e .
    python -m math_statistics.run_all       # screening + corrected tables
    python ML/cv_scheme_comparison.py        # Spatial-block + LLTO metrics
    python ML/feature_quality_cv.py          # per-feature quality
    python ML/feature_leakage_taxonomy.py    # empirical leakage taxonomy
Results are deterministic (fixed seeds).

## Data note
Soil laboratory values come from 2022-2023 field campaigns; remote-sensing covariates are derived
from public Sentinel-2 / Landsat-8 / Sentinel-1 / SRTM / ERA5-Land imagery via Google Earth Engine.

## Reproducibility
All manuscript numbers are byte-reproducible from data/features/master_dataset_old.csv with the committed code and fixed seeds: re-running the screening, the corrected Tables 7/S1/13, the Table 16 variance components, the spatial-CV comparison, the per-feature quality decomposition, the empirical taxonomy and the Farm-LOFO metrics regenerates the committed CSVs identically (verified by an empty git status after a full re-run).

Caveat: farm_lofo_feature_importance.py is a secondary diagnostic (not used in the manuscript tables); its permutation importances are seeded but vary sub-percent run-to-run under parallel execution, while the feature rankings are stable. The ML/results/*.md files are written narratives, not script-generated.
