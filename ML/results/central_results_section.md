# Central results section — empirical feature classification (manuscript-ready tables)

All values recomputed from `ML/results/feature_quality_tagged.csv` (512 features × 6 properties).
Classes are assigned from CV behaviour, not feature names.

## Table 7 — Empirical classification of RS features and resulting predictability, by soil property
|ρ|max(LC) = leakage-controlled univariate max; Farm-LOFO ρ = out-of-farm RF; counts = features per class; (robust) = generalizable features also passing |ρ_full|≥0.30 & within-sign==pooled-sign.

| Property | \|ρ\|max (LC) | Farm-LOFO ρ | generalizable (robust) | zonal_only | unstable | weak | Best transferable feature | Verdict |
|---|---|---|---|---|---|---|---|---|
| pH | 0.670 | 0.43 | 13 (6) | 134 | 86 | 279 | GNDVI_L8 spring (−0.67) | regionally + locally mappable |
| K₂O | 0.478 | 0.33 | 27 (10) | 27 | 70 | 388 | NBR_S2 spring (+0.36) | mappable (clay/SWIR) |
| P₂O₅ | 0.476 | 0.42 | 8 (3) | 25 | 33 | 446 | GLCM ENT_NIR summer (−0.39) | weakly mappable (texture) |
| NO₃ | 0.431 | 0.22 | 44 (12) | 9 | 123 | 336 | B5_NIR S2 spring (−0.41) | indirect (canopy-N proxy), weak out-of-farm |
| SOC | 0.368 | 0.25 | 4 (0) | 13 | 15 | 480 | — | regional gradient only |
| S | 0.309 | 0.04 | 0 (0) | 37 | 77 | 398 | — | unpredictable |

Total pool: weak 2327 (75.7%) · unstable 404 (13.2%) · zonal_only 245 (8.0%) · generalizable 96 (3.1%; 31 robust).
† For NO₃/SOC/S the present full-pool screen yields slightly higher |ρ| than the companion predictive study [18] (0.290/0.350/0.280), which additionally excludes canopy/management proxies for the labile nutrients under leakage-aware selection; both agree for pH/K₂O/P₂O₅ (0.670/0.478/0.476).

## Table 8 — Class definitions
| Class | Rule | Meaning |
|---|---|---|
| generalizable | \|within\|≥0.15 AND year-consistent | real local signal, stable across years |
| zonal_only | high \|ρ_full\| & \|between\|, \|within\|<0.15 | regional gradient only (mapping yes, local no) |
| unstable | ρ sign flips between 2022 and 2023 | non-transferable (year×location confound) |
| weak | none of the above | no exploitable association |

## Table 9 — GENERALIZABLE: top-5 (by within-block signal) and why
| Feature | Property | within / both-year ρ | Why it is a genuine local predictor |
|---|---|---|---|
| GNDVI_L8 spring | pH | −0.32 (−0.29 / −0.62) | On near-bare spring steppe GNDVI inverts to a soil-brightness index; surface reflectance tracks carbonate content, which co-sets pH. Mineralogy is year-stable → transferable. |
| GLCM contrast_Red summer | NO₃ | +0.32 (+0.47 / +0.18) | Within-field crop-vigour patchiness: better-fertilised, higher-mineralisation patches are greener/more textured and hold more residual nitrate. A local crop-N proxy (not robust: the regional sign opposes). |
| SAVI_L8 spring | pH | −0.30 (−0.39 / −0.41) | Soil-adjusted index over bare spring soil — same carbonate/brightness mechanism as GNDVI, with canopy correction. |
| NBR_S2 spring | K₂O | +0.29 (+0.12 / +0.34) | NBR uses SWIR (B12, ~2.2 µm) sensitive to clay-OH overtones; exchangeable K resides in the illitic clay/CEC fraction, so SWIR senses K-bearing mineralogy directly. |
| ts_GNDVI_L8 mean | NO₃ | +0.29 (−0.14 / −0.11) | Multi-season greenness mean as a within-field biomass/N proxy (positive within), with a negative regional arm — a Simpson split typical of labile, canopy-mediated nutrients. |

## Table 10 — ZONAL_ONLY: top-5 (by between-block / regional signal) and why
| Feature | Property | between / within | Why it is regional-only (not local) |
|---|---|---|---|
| S2REP spring | pH | −0.75 / +0.06 | Red-edge position senses canopy chlorophyll at the regional scale; it tracks the chernozem→kastanozem productivity gradient that co-varies with pH, but carries no within-field soil information. |
| GNDVI_S2 spring | K₂O | −0.74 / −0.01 | Regional greenness gradient aligns with the zonal clay/K gradient; within a field, greenness does not track exchangeable K. |
| ts_S2REP mean | pH | −0.72 / +0.06 | Multi-season red-edge mean = a smoothed regional productivity/climate gradient parallel to the latitudinal carbonate-leaching trend. |
| NDRE spring | pH | −0.71 / −0.03 | Red-edge chlorophyll index; its pH association is the macro-climatic vegetation gradient, not a soil chromophore (within-block ≈ 0). |
| NDWI_S2 spring | K₂O | +0.69 / −0.06 | Canopy/soil moisture index reflecting the regional rainfall gradient that drives K zonation; no sub-field content. |

## Table 11 — UNSTABLE: top-5 (by cross-year sign-flip) and why
| Feature | Property | ρ 2022 → 2023 | Why it flips (artefact, not real change) |
|---|---|---|---|
| Aspect (sin) | pH | +0.83 → −0.51 | Aspect is **time-invariant**; a sign reversal is physically impossible temporally. The 5 clustered 2022 farms sit on a different aspect–pH limb than the 15 disjoint 2023 farms → a year×location Simpson's paradox. |
| Aspect (cos) | SOC | −0.64 → +0.62 | Same: static topography cannot reverse in one year; the flip is the disjoint-region confound. |
| Slope | SOC | +0.66 → −0.48 | Slope is fixed and decadal SOC is ~constant over a year; the reversal can only come from the two years sampling different terrain–SOC relationships. |
| MAP (mean annual precip.) | SOC | +0.73 → −0.34 | A ~9 km climate covariate, near-constant within a farm; the 2022 and 2023 farm sets occupy opposite arms of the MAP–SOC gradient. |
| GS precipitation | P₂O₅ | −0.58 → +0.47 | Regional climate covariate flipping because the two campaigns span different climate ranges — temporal extrapolation the data cannot support. |

## Table 12 — WEAK: top-5 "false-strong" (high pooled ρ that fails) and why
| Feature | Property | ρ_full / within | Why it fails despite a strong pooled correlation |
|---|---|---|---|
| TWI | pH | −0.44 / +0.22 | Decent within-block signal, but the 2022 arm is absent (missing data) so cross-year stability cannot be confirmed → demoted; a "would-be" predictor left unvalidated by the design. |
| SR_B6 (NIR) spring | pH | −0.40 / −0.15 | Borderline local signal that neither clears the within threshold robustly nor replicates across years. |
| PCA-2 spring | K₂O | −0.39 / −0.26 | A composite with real within-block signal but a missing 2022 arm — unvalidated, hence weak rather than generalizable. |
| GLCM ENT_NIR spring | P₂O₅ | −0.38 / +0.00 | Strong pooled ρ carried almost entirely between blocks (within ≈ 0) yet just below the zonal threshold — a regional artefact masquerading as signal. |
| GLCM ASM_Red spring | P₂O₅ | +0.36 / +0.07 | Same: spring texture whose apparent strength is between-block; negligible local content. |

**Take-away for the section:** |ρ|max alone is misleading — the WEAK "false-strong" rows (ρ_full up to 0.44) and the UNSTABLE static-covariate flips show that pooled correlation conflates regional gradient, unreplicated design arms and genuine local signal. Only the GENERALIZABLE/robust tail (31 of 512) is transferable, and it is confined to spring bare-soil chromophores for the mineralogically anchored properties; SOC and S have none.
