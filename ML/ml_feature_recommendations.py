"""
ml_feature_recommendations.py
=============================
Turns the empirical feature taxonomy (feature_quality_tagged.csv) into actionable
recommendations for predictive modelling: which features to USE as predictors,
which to EXCLUDE (harmful), and which to treat only as a regional CONTROL covariate
(not a local predictor). A direct, reproducible bridge from screening to modelling.

Mapping (per feature x target):
  USE-PREDICTOR        : robust generalizable  -> transferable local signal
  USE-WITH-CAUTION     : generalizable but not robust -> local but weak / canopy proxy
  CONTROL-COVARIATE    : zonal_only -> regional gradient; include ONLY under spatial
                         (farm-blocked) CV or as a stratifier you partial out, never
                         as a stand-alone local predictor (it drives spatial overfitting)
  EXCLUDE-UNSTABLE     : unstable -> sign flips across years (year x location confound)
  EXCLUDE-TEMPORAL     : cross_season_concern -> cross-season feature for a LABILE
                         property (NO3/S): value-mismatch leakage
  DROP-WEAK            : weak -> no exploitable association

Run:  python ML/ml_feature_recommendations.py
Output: ML/results/ml_feature_recommendations.csv + ML/results/ml_feature_whitelist.csv
"""
import pandas as pd
from pathlib import Path

RES = Path(__file__).parent / "results"
d = pd.read_csv(RES / "feature_quality_tagged.csv")
LABELS = ["pH", "K2O", "P2O5", "NO3", "SOC", "S"]


def recommend(r):
    c = r.feature_class
    if c == "generalizable":
        base = "USE-PREDICTOR" if bool(r.robust) else "USE-WITH-CAUTION"
    elif c == "zonal_only":
        base = "CONTROL-COVARIATE"
    elif c == "unstable":
        base = "EXCLUDE-UNSTABLE"
    else:
        base = "DROP-WEAK"
    # overlay the temporal-leakage warning only on labile (NO3/S) cross-season features
    # that would OTHERWISE be tempting to use/keep (not the already-dropped weak ones)
    if bool(r.cross_season_concern) and not bool(r.robust) and base in ("USE-WITH-CAUTION", "CONTROL-COVARIATE"):
        return "EXCLUDE-TEMPORAL"
    return base


d["ml_recommendation"] = d.apply(recommend, axis=1)
d.to_csv(RES / "ml_feature_recommendations.csv", index=False)

# whitelist = the concrete "use these" predictors (robust), per property
wl = d[d.ml_recommendation == "USE-PREDICTOR"][
    ["target", "feature", "group", "season", "rho_full", "block_within", "rho_2022", "rho_2023"]
].copy().sort_values(["target", "block_within"], key=lambda s: s.abs() if s.name == "block_within" else s)
wl.to_csv(RES / "ml_feature_whitelist.csv", index=False)

ORD = ["USE-PREDICTOR", "USE-WITH-CAUTION", "CONTROL-COVARIATE", "EXCLUDE-UNSTABLE",
       "EXCLUDE-TEMPORAL", "DROP-WEAK"]
print("=== recommendation counts per property (feature x target pairs) ===")
ct = d.groupby(["target", "ml_recommendation"]).size().unstack(fill_value=0)
ct = ct.reindex(columns=[c for c in ORD if c in ct.columns], fill_value=0)
ct = ct.reindex(index=[t for t in ["pH", "K2O", "P2O5", "NO3", "SOC", "S"] if t in ct.index])
print(ct.to_string())
print("\noverall:", d.ml_recommendation.value_counts().reindex(ORD).to_dict())

print("\n=== WHITELIST: robust 'USE-PREDICTOR' features per property ===")
for t in ["pH", "K2O", "P2O5", "NO3", "SOC", "S"]:
    sub = wl[wl.target == t]
    feats = ", ".join(f"{r.feature}({r.rho_full:+.2f})" for _, r in sub.iterrows()) or "— none —"
    print(f"  {t}: {feats}")

print("\n=== HARD-EXCLUDE example: time-invariant covariates that flip sign (never use raw) ===")
flip = d[(d.ml_recommendation == "EXCLUDE-UNSTABLE") &
         (d.group.isin(["Topographic", "Climatic"]))].copy()
flip["sw"] = (flip.rho_2022 - flip.rho_2023).abs()
for _, r in flip.sort_values("sw", ascending=False).head(6).iterrows():
    print(f"  {r.target} ~ {r.feature}: 2022 {r.rho_2022:+.2f} -> 2023 {r.rho_2023:+.2f}")
print(f"\nSaved -> {RES}/ml_feature_recommendations.csv ; ml_feature_whitelist.csv")
