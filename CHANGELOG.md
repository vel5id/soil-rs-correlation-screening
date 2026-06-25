# Changelog - manuscript edits

All edited/new content is highlighted GREEN in `article/manuscript_edited.docx`.

## Reproducibility re-audit follow-up (edits to `article/manuscript_corrected.docx`)
Independent recompute-from-raw-data audit; every number traced to `data/features/master_dataset_old.csv`.
- **Moran's I declustering moved to the true field id (farm + field_name, n = 103)** — it previously
  grouped on the reused `field_name` label (n = 81), the same artefact already corrected for the ICC.
  Under the true field id sulfur's declustered I is **0.49, not 0.15** (ΔI −0.27, z = 13.8, p ≈ 0), so
  its spatial autocorrelation is **real but purely regional (between-farm)**, not a sampling artefact —
  S remains unpredictable via the out-of-farm Farm-LOFO ρ = 0.04 and ~2 % between-field-within-farm
  variance, but the "artefact" framing is dropped. Updated: Tables 10 & 14 (all six properties' I_field,
  z_field, p_field, ΔI, n_eff_field, Ratio, Pattern; S Pattern Artifact→Mixed), abstract, §2.4.3, §3.3.2
  (n_eff_field S 802→368; threshold criterion |ΔI|>0.2→>0.08 so NO3 stays I_field; §4.6 n_eff range
  81–802→81–419), §3.6, §4.3, Conclusions. New reproducible generator:
  `spatial_analysis.declustered_morans_i_all()` (Table 14/10 were previously hard-coded, no code).
- **§4.2 stale field_name ICC removed.** pH/SOC/K2O "ICC of 0.54–0.71" (the old field_name grouping)
  → true-field **0.71–0.93**; "inter-farm variance 45–57%" → **50–87%** (Table 16); SOC "|ρ| = 0.35,
  ICC = 0.54" → **0.37 / 0.79**; R²equiv 0.12→0.14; declustered "I ≥ 0.50" → "≈ 0.50".
- **Table 13 single-|ρ| baseline aligned to the reproducible code** (`corrected_tables.py`, full-screen
  max): S 0.383→**0.418** (Δ −0.058), SOC 0.368→**0.373** (Δ −0.097); abstract, §3.5 ("gap < 0.025"→
  "≤ 0.06") and §4.1(iii) deltas updated to match.
- `README.md` field count made consistent (102 physical fields / 81 reused labels). `config.py`
  `ARTICLE_CLAIMS` stale inter-soil constants updated to the verified Table 6 values.
- Everything else reproduces to the digit (Tables 3–6, 15, 16, screening |ρ|max, Farm-LOFO ρ,
  confounding 42 %); all sampled DOIs (Crossref + Zenodo) resolve. NB: the **+0.83** in the Figure 12
  section below is the *pre-correction* field_name artefact value — the current manuscript/code report
  the corrected ρ = +0.26 (n.s., n = 6).

## New: adversarial decomposition of sulfur's apparent predictability (§4.3)
Added an explanatory paragraph in §4.3 answering "why does S show high indicators if it is not
optically sensed?". An adversarial battery (each test *trying* to rescue S) shows all three high
indicators (|ρ|max 0.42, Moran I 0.77, ICC 0.83) are the same **between-farm artefact**: per-feature
between-farm ρ = 0.53–0.78 but within-farm |ρ| ≤ 0.05; Farm-LOFO rank ρ = 0.04 with R² = 0.61
decomposing to R²between = 0.77 / R²within = −0.06; farm-restricted spatial-permutation p = 0.46
(vs 0.03 for pH); ~28 % pH-mediated; the only out-of-farm-recoverable signal is the soil-chemical
pH co-variation (lab pH → S ρ = 0.28), not spectral. Significant (n_eff = 143) but spatially
confounded — significance ≠ predictability. New reproducible module `math_statistics/adversarial_s.py`
→ `output/adversarial_s_decomposition.csv`; spatial-permutation row from `permutation_bootstrap.csv`.

## Textual differentiation from the companion paper (Agriculture 2026, 16(11), 1239)
Measured verbatim 8-gram overlap with the published paper = 6.5% (mostly shared
references/affiliation, which is normal). Localised same-study prose rewritten (facts unchanged, wording differentiated): study area, climate, relief/management,
properties & sampling, GEE extraction.

## New Section 3.3.3 - Empirical feature classification & modelling recommendations
- Table 7 (new central): per-property |rho|max (leakage-controlled) + Farm-LOFO rho +
  class counts (generalizable/zonal-only/unstable/weak, with robust) + verdict; footnote
  reconciles NO3/SOC/S with the companion study [18].
- Table 7a: class definitions. Tables 7b-7d: top-5 generalizable / zonal-only / unstable
  features with mechanism. Table 7e: ML feature recommendations (USE / USE-WITH-CAUTION /
  CONTROL-COVARIATE / EXCLUDE-UNSTABLE / EXCLUDE-TEMPORAL / DROP-WEAK).
- Old "Table 7" (top-5 single |rho|) demoted to Table S1.

## Section 3.2 - added a pH-hub partial-correlation paragraph
SOC-P2O5 is ~82% pH-mediated; fertiliser co-application links (NO3-P-K-S) are masked by
pH; basis for the RS confounding in 3.3.1/3.8.

## Section 4.5 (new) - leakage-aware synthesis (genuine vs artefactual signal)
Cites Roberts 2017, Meyer 2018/2019, Ploton 2020, Wadoux 2021. Old 4.5 Limitations -> 4.6.

## Consistency / fixes
- In-text hierarchy rewritten to leakage-controlled values (consistent with Table 7).
- Language fix ("focused on focusing on" -> corrected).
- Data Availability: concatenated URLs -> a single new-repo URL, independent of [18].

## Figure 12 - replaced with the corrected, leakage-controlled version
The previous Figure 12 plotted naive full-pool screening |rho| (pre-leakage-fix): it put sulfur
at the top (|rho| ~ 0.85, lowest ICC) and implied a spurious *negative* ICC-correlation slope,
contradicting the surrounding text (which already describes S as essentially unpredictable).
Replaced in place (`word/media/image12.png`) with a figure built from corrected values: ICC vs
out-of-farm Farm-LOFO rho (filled, coloured by verdict) with naive in-sample |rho| overlaid
(open circles) and the in-sample -> out-of-farm collapse shown as arrows. Out-of-farm
predictability rises with ICC (Spearman rho = +0.83, p = 0.04), whereas the naive-screening
relationship is not significant (p = 0.27); sulfur collapses from |rho| ~ 0.42 to ~ 0.04.
Caption and the Sec. 3.7 intro paragraph updated to match (green). Reproducible via
`python -m math_statistics.key_figure` (deterministic, byte-identical across runs).

## Open items for the author
- Push this repo to GitHub and fill the real URL + Zenodo DOI in Data Availability.
- Confirm the per-year sample/farm counts (2022: 174/5; 2023: 911/15) against sampling records.
- Decide final disposition of demoted Table S1 (keep as supplementary or drop).
