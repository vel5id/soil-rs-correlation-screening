# Soil-Remote-Sensing Correlation Screening (Northern Kazakhstan steppe)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20820838.svg)](https://doi.org/10.5281/zenodo.20820838)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Reproducible code and data for the manuscript *"Digital Soil Mapping of the Steppe Zone of
Northern Kazakhstan: Relationships Between Agrochemical Soil Properties and Multimodal Remote
Sensing Data - Analysis of 530 Features."*

This is the **correlation-screening / feature-quality** study. It is the companion of, and
**independent from**, the predictive-modelling paper (Agriculture 2026, 16(11), 1239;
https://www.mdpi.com/2077-0472/16/11/1239), with which it shares only the underlying soil
dataset. The two studies answer different questions - *which* remote-sensing features relate to
*which* soil property and *why* (this paper) versus ML/DL prediction benchmarking (the modelling
paper) - and share no results or manuscript text.

## Key result
![Key result: ICC vs out-of-farm predictability](math_statistics/output/plots/00_key_icc_predictability.png)

**Between-field structure (ICC) governs out-of-farm predictability.** For each soil property, the
honest out-of-farm Farm-LOFO ρ with its best remote-sensing feature scales with the property's
between-field variance share (Spearman ρ(ICC, out-of-farm) = **+0.83**): the high-structure
properties (pH, K₂O, SOC, P₂O₅) generalise out of farm, NO₃ is weak, and sulfur — the property with
the *lowest* between-field structure (ICC = 0.17; ≈77 % of its variance is within-field) — does not
generalise (out-of-farm ρ ≈ 0.04). Naïve in-sample screening |ρ| over all 512 features (open circles)
sits uniformly above the honest values (grey arrows), and the inflation is largest exactly where
structure is weakest: sulfur's screening |ρ| ≈ 0.42 collapses to ≈ 0.04 out of farm. A strong
correlation for a property that is mostly within-field noise is a fingerprint of leakage, not signal —
the reason this study screens features under out-of-farm control. (Across the six properties, n = 6:
the out-of-farm relationship is significant — Spearman p = 0.04; Pearson r = 0.87, p = 0.02 — whereas
the naïve in-sample one is not (p = 0.27), so leakage control is what makes the structure–predictability
law detectable.) Reproducible via `python -m math_statistics.key_figure` (deterministic, byte-identical
across runs).

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
    python -m math_statistics.run_all       # screening + corrected tables + all 18 figures
    python -m math_statistics.key_figure     # headline figure: ICC vs out-of-farm predictability
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

### Figures
`python -m math_statistics.run_all` regenerates all 18 figures from `data/features/master_dataset_old.csv` into `math_statistics/output/plots/`. They are **byte-identical across independent runs** — verified by SHA-256 over two full runs (all 18 PNG *and* all 18 TIFF matched). Determinism follows from a fixed seed (`numpy.random.default_rng(42)` in the bootstrap-CI panel) and otherwise data-only plotting. The committed 300-dpi PNGs are shown in [Figures](#figures) below; the 600-dpi LZW-TIFF variants regenerate identically but are git-ignored (~54 MB). Two of the 20 defined panels are intentionally not produced: a variance-inflation-factor panel (slot 13) was dropped because the engineered 512-feature block is rank-deficient — every feature is an exact linear combination of the others, so every VIF diverges to ∞ and the chart renders empty and misleading; multicollinearity is instead reported as effective dimensionality (≈228 independent features of 512). A derived-soil top-correlations panel (slot 19) is skipped when its input table is empty. The standalone `python -m math_statistics.plots` entry point now reads the same `FEATURES_CSV` as the rest of the pipeline (it previously referenced a private `full_dataset.csv` that is not shipped in this repository).

The headline figure (`00_key_icc_predictability`, see [Key result](#key-result)) is produced separately by `python -m math_statistics.key_figure`. It recomputes ICC from the dataset and reads only committed tables (`all_spearman_correlations.csv`, `leakage_controlled_screening.csv`), has no random component, and is byte-identical across runs (SHA-256 verified). Its Y-axis is the corrected, leakage-controlled metric — not the naive full-pool screening |ρ| — so it does not reproduce the spurious negative ICC–correlation slope of the earlier draft figure (which was an artifact of micronutrient contamination in the feature pool).

## Figures
All panels below are regenerated byte-for-byte by `python -m math_statistics.run_all` (300-dpi PNG).

**Fig. 1 — Distribution (histogram + KDE) of the six soil properties; annotated with n, mean, median and CV.**

![Fig. 1](math_statistics/output/plots/01_histograms.png)

**Fig. 2 — Spearman intercorrelation matrix among the six soil properties (significance stars).**

![Fig. 2](math_statistics/output/plots/02_intercorrelation_heatmap.png)

**Fig. 3 — Spearman ρ heatmap: Sentinel-2 spectral indices × season vs each soil property.**

![Fig. 3](math_statistics/output/plots/03_s2_index_season_heatmap.png)

**Fig. 4 — Scatter plots of the strongest screening correlations (ρ and p annotated).**

![Fig. 4](math_statistics/output/plots/04_top_scatter_plots.png)

**Fig. 5 — Seasonal NDVI trajectory (spring → autumn) by SOC class.**

![Fig. 5](math_statistics/output/plots/05_seasonal_ndvi_by_soc.png)

**Fig. 6 — Sentinel-2 summer single-band correlations with the soil properties.**

![Fig. 6](math_statistics/output/plots/06_band_correlations_summer.png)

**Fig. 7 — Topographic and climate covariate correlations with the soil properties.**

![Fig. 7](math_statistics/output/plots/07_topo_climate_correlations.png)

**Fig. 8 — Spatial distribution of the six soil properties across sample centroids.**

![Fig. 8](math_statistics/output/plots/08_spatial_maps.png)

**Fig. 9 — QQ-plots for normality assessment of the six properties.**

![Fig. 9](math_statistics/output/plots/09_qq_plots.png)

**Fig. 10 — Soil properties by sampling year (2022 vs 2023; Kruskal–Wallis check).**

![Fig. 10](math_statistics/output/plots/10_boxplots_by_year.png)

**Fig. 11 — Bootstrap 95 % CIs for the key correlations vs the published article values.**

![Fig. 11](math_statistics/output/plots/11_bootstrap_ci.png)

**Fig. 12 — Article-reported ρ vs recomputed ρ for the screening claims.**

![Fig. 12](math_statistics/output/plots/12_claim_verification.png)

**Fig. 14 — Best single vs best composite spectral feature, per soil target.**

![Fig. 14](math_statistics/output/plots/14_composite_vs_single.png)

**Fig. 15 — Between-field vs within-field variance decomposition.**

![Fig. 15](math_statistics/output/plots/15_variance_decomposition.png)

**Fig. 16 — Raw vs pH-controlled partial SOC–vegetation-index correlations (confounding).**

![Fig. 16](math_statistics/output/plots/16_ph_confounding.png)

**Fig. 17 — NDVI (summer) saturation curve against SOC (plateau ≈ SOC 2.5 %).**

![Fig. 17](math_statistics/output/plots/17_ndvi_saturation.png)

**Fig. 18 — Property coefficient of variation vs maximum |ρ| with RS features.**

![Fig. 18](math_statistics/output/plots/18_cv_vs_rho.png)

**Fig. 20 — Multi-seasonal delta/amplitude features vs peak single-season, per target.**

![Fig. 20](math_statistics/output/plots/20_delta_vs_peak.png)
