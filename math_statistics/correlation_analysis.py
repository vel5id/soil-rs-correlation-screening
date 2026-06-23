"""
Full Spearman correlation analysis: soil properties vs all RS features.

Key verifications:
- pH ↔ L8 GNDVI spring:  ρ = -0.67
- pH ↔ MAP:              ρ = 0.66
- pH ↔ slope:            ρ = 0.55
- K  ↔ BSI spring:       ρ = -0.48
- P  ↔ GS_temp:          ρ = 0.48
- P  ↔ aspect_cos:       ρ = 0.47
- pH ↔ aspect_sin:       ρ = -0.47
- pH ↔ S2 GNDVI spring:  ρ ≈ -0.49..-0.52  (article: range for veg indices)
- SOC ↔ summer NDVI:     ρ ≈ 0.20-0.30

Also applies Benjamini-Hochberg FDR correction.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import (
    SOIL_TARGETS, SOIL_LABELS, ARTICLE_CLAIMS, ALPHA, OUTPUT_DIR,
    TOPO_COLS, CLIMATE_COLS, SEASONS, SEED,
)


def _get_rs_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all remote sensing / covariate feature columns."""
    exclude = {"id", "year", "farm", "field_name", "grid_id",
               "centroid_lon", "centroid_lat", "geometry_wkt",
               "protocol_number", "analysis_date", "sampling_date",
               "hu",  # hu is the raw humus, soc is derived
               # Soil lab micronutrients are MEASURED soil chemistry, not RS
               # covariates. They leak the target (soil-soil correlation) and
               # are only available for n=148 samples — must NOT enter the
               # RS feature screening pool.
               "cu", "fe", "mg", "mn", "mo", "zn"}
    return [c for c in df.columns
            if c not in exclude and c not in SOIL_TARGETS
            and df[c].dtype in ("float64", "int64")]


def compute_all_spearman(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations: each soil target vs each RS feature.

    Returns long-form DataFrame with columns:
      target, feature, rho, p_value, n, abs_rho
    """
    features = _get_rs_feature_columns(df)
    rows = []
    for target in SOIL_TARGETS:
        for feat in features:
            mask = df[[target, feat]].notna().all(axis=1)
            n = mask.sum()
            if n < 10:
                continue
            rho, p = stats.spearmanr(df.loc[mask, target], df.loc[mask, feat])
            rows.append({
                "target": target,
                "target_label": SOIL_LABELS[target],
                "feature": feat,
                "rho": rho,
                "abs_rho": abs(rho),
                "p_value": p,
                "n": n,
            })
    result = pd.DataFrame(rows)
    return result


def apply_bh_correction(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction per target."""
    df = corr_df.copy()
    df["p_adjusted"] = np.nan
    df["significant_bh"] = False

    for target in SOIL_TARGETS:
        mask = df["target"] == target
        pvals = df.loc[mask, "p_value"].values
        if len(pvals) == 0:
            continue
        reject, p_adj, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
        df.loc[mask, "p_adjusted"] = p_adj
        df.loc[mask, "significant_bh"] = reject

    return df


def top_correlations(corr_df: pd.DataFrame, n_top: int = 20) -> pd.DataFrame:
    """Top-N strongest correlations per soil target."""
    frames = []
    for target in SOIL_TARGETS:
        sub = corr_df[corr_df["target"] == target].nlargest(n_top, "abs_rho")
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def verify_article_claims(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Verify specific rho values stated in the article."""
    rows = []
    for claim_id, info in ARTICLE_CLAIMS.items():
        target, feature, article_rho = info["target"], info["feature"], info["rho"]

        # For inter-soil correlations, skip (handled in intercorrelation.py)
        if feature in SOIL_TARGETS:
            continue

        match = corr_df[(corr_df["target"] == target) & (corr_df["feature"] == feature)]
        if match.empty:
            rows.append({
                "claim": claim_id,
                "target": target,
                "feature": feature,
                "article_rho": article_rho,
                "computed_rho": np.nan,
                "difference": np.nan,
                "p_value": np.nan,
                "n": 0,
                "MATCH_within_0.05": False,
                "NOTE": "Feature not found in dataset",
            })
            continue

        row = match.iloc[0]
        diff = abs(row["rho"] - article_rho)
        rows.append({
            "claim": claim_id,
            "target": target,
            "feature": feature,
            "article_rho": article_rho,
            "computed_rho": round(row["rho"], 4),
            "difference": round(diff, 4),
            "p_value": row["p_value"],
            "n": row["n"],
            "MATCH_within_0.05": diff < 0.05,
            "NOTE": "",
        })
    return pd.DataFrame(rows)


def seasonal_comparison(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Compare spring vs summer correlations for vegetation indices.

    Article claims: spring NDVI/GNDVI/NDRE correlate stronger with pH than summer.
    """
    indices = ["NDVI", "GNDVI", "NDRE", "EVI", "SAVI"]
    rows = []
    for idx in indices:
        for target in SOIL_TARGETS:
            for prefix in ["s2_", "l8_"]:
                season_vals = {}
                for season in SEASONS:
                    feat = f"{prefix}{idx}_{season}"
                    match = corr_df[(corr_df["target"] == target) & (corr_df["feature"] == feat)]
                    if not match.empty:
                        season_vals[season] = match.iloc[0]["rho"]

                if len(season_vals) >= 2:
                    rows.append({
                        "target": target,
                        "index": f"{prefix}{idx}",
                        **{f"rho_{s}": round(v, 4) for s, v in season_vals.items()},
                        "spring_stronger_than_summer": (
                            abs(season_vals.get("spring", 0)) > abs(season_vals.get("summer", 0))
                            if "spring" in season_vals and "summer" in season_vals
                            else None
                        ),
                    })
    return pd.DataFrame(rows)


# ── Leakage-controlled screening (temporal alignment + out-of-farm) ──
# The plain whole-dataset |rho|max over the full pool is doubly inflated:
#   (1) TEMPORAL leakage — cross-season (summer/late-summer/autumn) spectral
#       features and GLCM textures postdate / mismatch the predominantly spring
#       (~75%) sampling and encode farm-level field state, not the soil property;
#   (2) SPATIAL pseudoreplication — with Moran's I 0.5-0.86 the 1085 samples are
#       not independent (effective n ~ 20 farms), so a feature that merely orders
#       farms wins the max-selection over hundreds of candidates.
# This is why the naive screen reports P2O5 = 0.525 (autumn GLCM texture) which
# collapses to ~0 under out-of-farm validation, instead of the defensible 0.476
# (growing-season temperature). The two functions below control both effects.

_CROSS_SEASON = ("summer", "late_summer", "autumn")


def _temporally_aligned_features(df: pd.DataFrame) -> list[str]:
    """RS features temporally valid for a predominantly spring-collected sample.

    Drops cross-season spectral features, GLCM textures (farm-confounded field
    state), and multi-season time-series / composite features; keeps spring
    spectral indices+bands and static topographic + climatic covariates. Base
    meta/target/micronutrient exclusions come from _get_rs_feature_columns.
    """
    keep = []
    for c in _get_rs_feature_columns(df):
        cl = c.lower()
        if any(s in cl for s in _CROSS_SEASON):
            continue
        if "glcm" in cl:
            continue
        if cl.startswith(("ts_", "range_", "delta_", "amp_", "cv_", "cs_")) \
                or "spectral_" in cl:
            continue
        keep.append(c)
    return keep


def _farm_lofo_rho(df: pd.DataFrame, target: str, features: list[str],
                   farm_col: str = "farm") -> tuple[float, float, int, int]:
    """Out-of-farm validation: multivariate RF leave-one-farm-out.

    Trains a Random Forest on N-1 farms and predicts the held-out farm (per-fold
    median imputation fit on the training farms only); returns (|rho|, R^2, n,
    n_farms) of the pooled out-of-fold predictions vs observations. A genuine
    association survives this; a spatial/temporal-leakage artefact collapses.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    sub = df.dropna(subset=[target]).copy()
    X = sub[features]
    y = sub[target].to_numpy()
    farms = sub[farm_col].to_numpy()
    preds = np.full(len(sub), np.nan)
    for fm in np.unique(farms):
        tr, te = farms != fm, farms == fm
        if tr.sum() < 10 or te.sum() < 1:
            continue
        med = X[tr].median()
        x_tr = X[tr].fillna(med).fillna(0.0)
        x_te = X[te].fillna(med).fillna(0.0)
        model = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.5,
            random_state=SEED, n_jobs=-1)
        model.fit(x_tr, y[tr])
        preds[te] = model.predict(x_te)
    v = np.isfinite(preds)
    rho, _ = stats.spearmanr(y[v], preds[v])
    return abs(rho), float(r2_score(y[v], preds[v])), int(v.sum()), int(len(np.unique(farms)))


def leakage_controlled_screening(df: pd.DataFrame,
                                 validate_out_of_farm: bool = True) -> pd.DataFrame:
    """Leakage-controlled |rho|max per soil property.

    Combines the two controls: the headline ``rho_max_aligned`` is the univariate
    Spearman |rho|max taken over the *temporally-aligned* pool only, and
    ``farm_lofo_rho`` is the multivariate out-of-farm (Farm-LOFO) check on that
    same pool. ``verdict`` flags properties whose aligned screening still does not
    generalise across farms (e.g. S).
    """
    pool = _temporally_aligned_features(df)
    rows = []
    for target in SOIL_TARGETS:
        best = {"rho": 0.0, "feature": None, "n": 0}
        pvals, feats = [], []
        for feat in pool:
            mask = df[[target, feat]].notna().all(axis=1)
            n = int(mask.sum())
            if n < 10:
                continue
            rho, p = stats.spearmanr(df.loc[mask, target], df.loc[mask, feat])
            if pd.isna(rho):
                continue
            pvals.append(p)
            feats.append(feat)
            if abs(rho) > best["rho"]:
                best = {"rho": abs(rho), "feature": feat, "n": n}
        p_adj = np.nan
        if pvals and best["feature"] is not None:
            _, adj, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
            p_adj = float(adj[feats.index(best["feature"])])

        row = {
            "target": target,
            "target_label": SOIL_LABELS[target],
            "rho_max_aligned": round(best["rho"], 3),
            "winner": best["feature"],
            "p_adj_bh": p_adj,
            "n": best["n"],
            "pool_size": len(pool),
        }
        if validate_out_of_farm and "farm" in df.columns:
            f_rho, f_r2, _, f_k = _farm_lofo_rho(df, target, pool)
            ratio = (f_rho / best["rho"]) if best["rho"] else 0.0
            row.update({
                "farm_lofo_rho": round(f_rho, 3),
                "farm_lofo_r2": round(f_r2, 3),
                "n_farms": f_k,
                "verdict": ("generalises" if ratio >= 0.5
                            else "weak" if ratio >= 0.25
                            else "does-not-generalise"),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run the full correlation analysis."""
    all_corr = compute_all_spearman(df)
    all_corr = apply_bh_correction(all_corr)
    top = top_correlations(all_corr)
    claims = verify_article_claims(all_corr)
    seasonal = seasonal_comparison(all_corr)

    leak_screen = leakage_controlled_screening(df)

    results = {
        "all_correlations": all_corr,
        "top_correlations": top,
        "article_claims_verification": claims,
        "seasonal_comparison": seasonal,
        "leakage_controlled_screening": leak_screen,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_DIR / "correlation_analysis.xlsx") as writer:
        top.to_excel(writer, sheet_name="top20_per_target", index=False)
        claims.to_excel(writer, sheet_name="claims_verification", index=False)
        seasonal.to_excel(writer, sheet_name="seasonal_comparison", index=False)
        leak_screen.to_excel(writer, sheet_name="leakage_controlled", index=False)
        # Full matrix too large for one sheet — save as CSV
    all_corr.to_csv(OUTPUT_DIR / "all_spearman_correlations.csv", index=False)
    leak_screen.to_csv(OUTPUT_DIR / "leakage_controlled_screening.csv", index=False)

    return results
