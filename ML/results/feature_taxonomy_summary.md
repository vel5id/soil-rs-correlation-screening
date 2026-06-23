# Empirical Feature-Quality Taxonomy

**Source:** `ML/results/feature_quality_tagged.csv` (3,072 feature×target rows; 6 targets × 512 features).
**Derived from:** `ML/results/feature_quality_cv.csv` (block cross-validated Spearman correlations).
**Classifier:** `ML/feature_leakage_taxonomy.py` (deterministic; no randomness).

This taxonomy decides feature quality **empirically from cross-validated correlation behaviour**, not from feature names. The legacy name-based `leakage_suspect` column is retained in the CSV for comparison but is **not** used by the classifier.

---

## 1. Class definitions and final thresholds

Each feature×target row carries four CV statistics: `rho_full` (pooled Spearman over both years), `block_within` (within-block / within-year partial association), `block_between` (between-block / cross-year association), and `year_consistent` (do the 2022 and 2023 sign agree). Classes are assigned by the following precedence: **generalizable > unstable > zonal_only > weak**.

| Class | Rule (in precedence order) | Interpretation |
|---|---|---|
| **generalizable** | `\|block_within\| ≥ TAU_WITHIN` **and** `year_consistent = yes` (no cross-year sign-flip) | A within-year (local) signal that does not collapse and does not reverse between years — a genuinely usable predictor. |
| **unstable** | cross-year **sign-flip** (rho_2022 and rho_2023 have opposite signs) **and** `\|rho_full\| ≥ UNSTABLE_MIN_FULL` | Carries pooled signal but the relationship **reverses** between years → leakage-prone / non-transferable. |
| **zonal_only** | `\|block_between\| ≥ TAU_BETWEEN` **and** `\|rho_full\| ≥ TAU_FULL` **and** `\|block_within\| < TAU_WITHIN` | Signal lives **between** zones/years (spatial stratification) but **collapses within** a block → reflects geography, not soil. |
| **weak** | everything else | No exploitable association. |

**Final thresholds** (spec start-values; held up against the observed distribution and kept as-is):

| Threshold | Value | Meaning |
|---|---|---|
| `TAU_WITHIN` | **0.15** | Minimum \|within-block ρ\| for a real local signal. |
| `TAU_FULL` | **0.30** | Minimum \|pooled ρ\| for a zonal classification. |
| `TAU_BETWEEN` | **0.30** | Minimum \|between-block ρ\| for a zonal classification. |
| `UNSTABLE_MIN_FULL` | **0.20** | Sign-flip floor on \|rho_full\| for the unstable class. |

Sign-flip rule: an exact `0.0` in either year is treated as **non-flipping** (no defined reversal); both yearly ρ must be finite.

> **Threshold caveat (report as a sensitivity, not a hard boundary).** The within-block distribution is thin: max \|block_within\| = 0.327, median 0.060, p90 0.162, p95 0.202. `TAU_WITHIN = 0.15` sits near the 88th percentile, but because the strongest local signal is barely twice the cut, the `generalizable` class is intrinsically small and sensitive to the threshold. Sensitivity: `τ=0.10 → 234 generalizable`, `τ=0.15 → 96`, `τ=0.20 → 39`. A ±0.05 shift roughly doubles or halves the count (138 consistent rows fall in 0.10–0.15, 57 in 0.15–0.20). Only `generalizable` and `zonal_only` move with `TAU_WITHIN`; `unstable` is fixed at 404 because it is decided **before** the within-gate.

---

## 2. Counts per class

### Overall (3,072 rows)

| Class | Count | Share |
|---|---:|---:|
| weak | 2,327 | 75.7% |
| unstable | 404 | 13.2% |
| zonal_only | 245 | 8.0% |
| **generalizable** | **96** | **3.1%** |

### Per soil property (512 features each)

| Target | generalizable | unstable | zonal_only | weak |
|---|---:|---:|---:|---:|
| **pH** | 13 | 86 | 134 | 279 |
| **NO₃** | 44 | 123 | 9 | 336 |
| **K₂O** | 27 | 70 | 27 | 388 |
| **P₂O₅** | 8 | 33 | 25 | 446 |
| **SOC** | 4 | 15 | 13 | 480 |
| **S** | **0** | 77 | 37 | 398 |

---

## 3. Generalizable shortlists (the genuinely usable predictors)

For each property below, the table lists all `generalizable` features with `block_within` and both single-year ρ. A **robust** flag (★) marks features that additionally clear a pooled-strength gate — `|rho_full| ≥ 0.30` **and** within-block sign agreeing with the pooled sign — i.e. the subset that survives the validator's semantic critique (see §6). Only **31 of 96** generalizable rows are robust; **20** of the 96 are exact `s2_* / spectral_*` alias pairs (same values, different name), so the count of *distinct* usable predictors is smaller still.

> **Read this section conservatively.** `generalizable` as implemented requires only a within-year signal that does not sign-flip; it does **not** require pooled strength. Prefer the ★ robust rows for manuscript claims.

### pH — 13 generalizable (6 robust)

| ★ | feature | block_within | ρ 2022 | ρ 2023 | rho_full |
|---|---|---:|---:|---:|---:|
| ★ | l8_GNDVI_spring | −0.323 | −0.294 | −0.615 | −0.670 |
| ★ | l8_SAVI_spring | −0.302 | −0.388 | −0.406 | −0.482 |
| ★ | l8_NDVI_spring | −0.192 | −0.451 | −0.576 | −0.661 |
| ★ | glcm_glcm_nir_idm_late_summer | −0.186 | −0.733 | −0.300 | −0.327 |
| ★ | glcm_glcm_red_asm_spring | −0.168 | −0.629 | −0.293 | −0.397 |
| ★ | ts_l8_GNDVI_mean | −0.166 | −0.358 | −0.515 | −0.577 |
|   | glcm_ratio_asm_late_summer | −0.219 | −0.374 | −0.116 | −0.128 |
|   | glcm_ratio_ent_late_summer | +0.219 | +0.469 | +0.122 | +0.134 |
|   | topo_DEM | −0.195 | +0.723 | +0.321 | +0.382 |
|   | glcm_ratio_idm_late_summer | −0.181 | −0.701 | −0.191 | −0.265 |
|   | cs_SAVI_diff_spring | +0.172 | −0.707 | −0.103 | −0.258 |
|   | s2_S2REP_autumn | −0.161 | −0.174 | −0.297 | −0.206 |
|   | glcm_glcm_nir_idm_summer | −0.152 | −0.742 | −0.239 | −0.262 |

**Usable predictors for pH:** spring Landsat-8 vegetation indices (GNDVI, SAVI, NDVI) and their time-series mean are the strongest and most defensible (pooled ρ ≈ −0.48 to −0.67, consistent across both years).

### NO₃ — 44 generalizable (12 robust)

| ★ | feature | block_within | ρ 2022 | ρ 2023 | rho_full |
|---|---|---:|---:|---:|---:|
| ★ | l8_SR_B5_spring | −0.259 | −0.536 | −0.395 | −0.415 |
| ★ | s2_SAVI_spring | −0.207 | −0.587 | −0.286 | −0.431 |
| ★ | s2_EVI_spring | −0.207 | −0.577 | −0.264 | −0.406 |
| ★ | s2_IRECI_spring | −0.190 | −0.596 | −0.174 | −0.356 |
| ★ | s2_B8A_spring (= spectral_B8A_spring) | −0.176 | −0.334 | −0.318 | −0.341 |
| ★ | s2_B7_spring (= spectral_B7_spring) | −0.175 | −0.381 | −0.321 | −0.335 |
| ★ | s2_B6_spring (= spectral_B6_spring) | −0.175 | −0.373 | −0.320 | −0.316 |
| ★ | s2_B8_spring (= spectral_B8_spring) | −0.168 | −0.429 | −0.325 | −0.344 |
|   | glcm_ratio_contrast_summer | +0.320 | +0.472 | +0.183 | +0.273 |
|   | ts_l8_GNDVI_mean | +0.292 | −0.135 | −0.105 | −0.224 |
|   | ts_s2_GNDVI_mean (= ts_spectral_GNDVI_mean) | +0.283 | −0.249 | −0.148 | −0.236 |
|   | ts_l8_NDVI_mean | +0.282 | −0.172 | −0.125 | −0.165 |
|   | ts_l8_SAVI_mean | +0.269 | −0.149 | −0.125 | −0.185 |
|   | ts_s2_NDVI_mean (and SAVI/spectral aliases) | +0.248 | −0.122 | −0.115 | −0.114 |
|   | l8_SR_B3_spring | −0.243 | −0.177 | −0.251 | −0.186 |
|   | ts_s2_NDWI_mean | −0.243 | +0.244 | +0.129 | +0.213 |
|   | s2_B11_spring (= spectral_B11_spring) | −0.179 | −0.250 | −0.123 | −0.278 |
|   | s2_NDWI_spring | +0.164 | +0.575 | +0.108 | +0.299 |
|   | glcm_ratio_idm_late_summer | −0.150 | −0.344 | −0.257 | −0.280 |
|   | *(plus delta_* change features, glcm ratios, s2_B3/B5_summer — all with \|rho_full\|<0.30)* | | | | |

**Usable predictors for NO₃:** spring Sentinel-2 red-edge / NIR bands and indices (B6–B8A, SAVI, EVI, IRECI) and Landsat-8 spring SR_B5 — pooled ρ ≈ −0.32 to −0.43, both years negative. Several `ts_*` time-series-mean rows qualify as generalizable but have near-zero pooled ρ and within-block sign opposite to the pooled sign → not robust.

### K₂O — 27 generalizable (10 robust)

| ★ | feature | block_within | ρ 2022 | ρ 2023 | rho_full |
|---|---|---:|---:|---:|---:|
| ★ | s2_NBR_spring | +0.294 | +0.118 | +0.339 | +0.356 |
| ★ | s2_BSI_spring (= spectral_BSI_spring) | −0.271 | −0.237 | −0.477 | −0.478 |
| ★ | s2_B12_spring (= spectral_B12_spring) | −0.263 | −0.390 | −0.247 | −0.348 |
| ★ | s2_NDMI_spring | +0.246 | +0.106 | +0.389 | +0.399 |
| ★ | s2_MSI_spring | −0.246 | −0.106 | −0.389 | −0.399 |
| ★ | spectral_B11_B8_spring | −0.240 | −0.106 | −0.394 | −0.400 |
| ★ | s2_B11_spring (= spectral_B11_spring) | −0.184 | −0.509 | −0.229 | −0.333 |
|   | climate_MAP | −0.267 | +0.130 | +0.176 | +0.208 |
|   | glcm_glcm_nir_contrast_spring | +0.255 | +0.620 | +0.166 | +0.259 |
|   | s2_B3_late_summer (+ spectral alias) | +0.218 | +0.214 | +0.341 | +0.270 |
|   | s2_B5_late_summer (+ spectral alias) | +0.203 | +0.223 | +0.212 | +0.131 |
|   | glcm_glcm_red_contrast_spring | +0.200 | +0.588 | +0.135 | +0.226 |
|   | *(plus glcm asm/ent, s2_B2/B3/B4 autumn, PCA, cs_SAVI_diff — all \|rho_full\|<0.30)* | | | | |

**Usable predictors for K₂O:** spring Sentinel-2 SWIR-based moisture/bare-soil indices (BSI, NBR, NDMI, MSI, B11/B12) — pooled ρ ≈ ±0.33 to ±0.48, both years same sign.

### P₂O₅ — 8 generalizable (3 robust)

| ★ | feature | block_within | ρ 2022 | ρ 2023 | rho_full |
|---|---|---:|---:|---:|---:|
| ★ | glcm_glcm_nir_idm_summer | +0.162 | +0.412 | +0.402 | +0.410 |
| ★ | glcm_glcm_nir_ent_summer | −0.199 | −0.503 | −0.333 | −0.392 |
| ★ | glcm_glcm_nir_asm_summer | +0.198 | +0.503 | +0.300 | +0.369 |
|   | topo_plan_curvature | −0.205 | −0.108 | −0.116 | −0.105 |
|   | climate_MAT | −0.202 | +0.134 | +0.334 | +0.330 |
|   | climate_MAP | −0.182 | −0.429 | −0.413 | −0.189 |
|   | ts_s2_NDWI_slope | +0.182 | +0.138 | +0.112 | −0.005 |
|   | glcm_ratio_idm_late_summer | +0.160 | +0.587 | +0.228 | +0.290 |

**Usable predictors for P₂O₅:** summer NIR GLCM texture (IDM, ENT, ASM) — pooled ρ ≈ ±0.37 to ±0.41, both years same sign. The two `climate_*` rows have within-block sign opposite to pooled sign and are **not** robust.

### SOC — 4 generalizable, **0 robust**

| ★ | feature | block_within | ρ 2022 | ρ 2023 | rho_full |
|---|---|---:|---:|---:|---:|
|   | s2_B2_late_summer (= spectral_B2_late_summer) | +0.161 | −0.545 | −0.249 | −0.298 |
|   | ts_s2_MSI_cv | −0.152 | +0.493 | +0.166 | +0.233 |
|   | topo_aspect_sin | −0.162 | +0.635 | +0.106 | +0.100 |

**SOC has no robust generalizable predictor.** All four generalizable rows have weak pooled signal (\|rho_full\| ≤ 0.30), and three show within-block sign **opposite** to the pooled sign (Simpson-style reversal). SOC is best treated as **not locally predictable** from this feature set under within-field CV.

### S (sulfur) — **0 generalizable** (and 0 robust)

**Sulfur has no generalizable feature at all.** Its 512 rows split into 77 unstable, 37 zonal_only, 398 weak. This is consistent with S being a labile, year-unstable property: any apparent signal either reverses between years (unstable) or lives only between zones (zonal_only). **No usable local predictor exists for S in this dataset.**

---

## 4. Methods note (taxonomy ↔ literature)

This taxonomy operationalises spatially-aware predictor screening. Rather than discarding features by name (e.g. assuming any climate or DEM layer is "leaky"), each feature is screened by its behaviour under **block cross-validation**, following the spatial-CV predictor-selection logic of **Meyer et al. (2019)**, who showed that variable selection must be embedded inside a spatial resampling scheme or else apparently strong predictors merely encode spatial structure. The `block_within` vs `block_between` contrast is the **target-oriented / leave-location-out (LLTO)** diagnostic of **Meyer et al. (2018)**: a predictor whose association survives within a held-out block (`block_within`) reflects a transferable soil–signal relationship, whereas one that only appears between blocks (`block_between`, our `zonal_only`) reflects geographic stratification and will not generalise to new locations. The cross-year **sign-flip** test (our `unstable` class) extends this to the temporal axis, flagging relationships that reverse between acquisition years and would inflate apparent skill under non-temporal splits. We adopt the caveat of **Wadoux et al. (2021)** that cross-validation map-accuracy estimates are themselves design-dependent and can mislead when sampling is clustered; accordingly the `generalizable` threshold (`TAU_WITHIN = 0.15`) is reported as a **sensitivity** (§1), not a hard cut, and the thin within-block distribution is disclosed. **Crucially, season and feature-type are retained only as descriptive tags; leakage is *measured* — via cross-year sign-flip and within-block collapse — not *assumed* from a feature's name.**

> References (verify before citing in the manuscript):
> - Meyer, H., Reudenbach, C., Hengl, T., Katurji, M., Nauss, T. (2018). *Improving performance of spatio-temporal machine learning models using forward feature selection and target-oriented validation.* Environmental Modelling & Software, 101, 1–9.
> - Meyer, H., Reudenbach, C., Wöllauer, S., Nauss, T. (2019). *Importance of spatial predictor variable selection in machine learning applications — Moving from data reproduction to spatial prediction.* Ecological Modelling, 411, 108815.
> - Wadoux, A.M.J.-C., Heuvelink, G.B.M., de Bruin, S., Brus, D.J. (2021). *Spatial cross-validation is not the right way to evaluate map accuracy.* Ecological Modelling, 457, 109692.

---

## 5. Stable vs labile note (`cross_season_concern`)

Two **descriptive** tags accompany every row:

- **`cross_season`** — the feature was acquired in a season other than the sampling season (True = 1,692 rows; False = 1,380).
- **`property_lability`** — whether the soil property itself is temporally labile (labile = 1,024; stable = 2,048).

The composite flag **`cross_season_concern = cross_season AND labile`** is True on exactly **564 rows**, and these are **all and only** the labile properties:

| Target | cross_season_concern = True |
|---|---:|
| NO₃ | 282 |
| S | 282 |
| pH, SOC, K₂O, P₂O₅ | 0 |

**Rationale:** for a **labile** property (NO₃, S), a feature acquired in a different season measures the field in a genuinely different chemical state → a real value-mismatch concern. For a **stable** property (pH, SOC, K₂O, P₂O₅), a cross-season acquisition is simply the *same soil imaged at a different time* and is **not** flagged. This is a descriptive caution layered on top of the empirical class — it does **not** override `feature_class`.

---

## 6. Validation caveat (must be stated in the manuscript)

The taxonomy was independently re-implemented and recomputed from `feature_quality_cv.csv`: **zero of 3,072 rows differ** from the reference classifier, all four anchor cases match expectation, and every class invariant holds. There is **no mechanical/coding error**.

However, the validator raised a **semantic** caveat about the `generalizable` rule, which we reproduce here and have independently confirmed against the CSV:

- The `generalizable` rule checks `|block_within| ≥ 0.15` and `year_consistent = yes` but has **no pooled-strength gate**. As a result **63 of 96** generalizable rows have `|rho_full| < 0.30`, **25** have `< 0.15`, and **10** have `< 0.10` — i.e. near-zero pooled association is still labelled "usable".
- **26 of 96** generalizable rows have `block_within` of **opposite sign** to `rho_full` (Simpson-style reversal; e.g. NO₃ ~ `ts_s2_NDVI_mean`: within +0.248, full −0.114).
- `year_consistent` enforces only **sign** agreement, not **magnitude** stability: **23 of 96** generalizable rows have `|ρ2022 − ρ2023| > 0.3` (e.g. pH ~ `cs_SAVI_diff_spring`: −0.707 vs −0.103).

**Mitigation applied in this summary:** the ★ **robust** subset in §3 (`|rho_full| ≥ 0.30` AND within-block sign agreeing with pooled sign) isolates the **31 of 96** rows that survive all three concerns. Manuscript claims of "usable predictors" should rest on the ★ rows. Under this stricter reading, **SOC and S have no usable predictor**, pH/NO₃/K₂O retain a small but defensible spring-vegetation / SWIR / NIR-texture set, and P₂O₅ retains summer NIR-texture features only.
