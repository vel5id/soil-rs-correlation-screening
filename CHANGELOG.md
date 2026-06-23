# Changelog - manuscript edits

All edited/new content is highlighted GREEN in `article/manuscript_edited.docx`.

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

## Open items for the author
- Push this repo to GitHub and fill the real URL + Zenodo DOI in Data Availability.
- Confirm the per-year sample/farm counts (2022: 174/5; 2023: 911/15) against sampling records.
- Decide final disposition of demoted Table S1 (keep as supplementary or drop).
