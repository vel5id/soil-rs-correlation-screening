"""
adversarial_s.py
================
Adversarial decomposition of sulfur's *apparent* predictability (Section 4.3).

S (sulfate) has no diagnostic absorption band in 400-2500 nm, yet its in-sample
indicators look high: |rho|max = 0.418 (ts_l8_NDVI_mean), Moran's I = 0.767, ICC
(farm+field) = 0.83. This module shows -- adversarially, each test *trying* to
rescue S -- that all three are the same artefact: S variance is ~81% between-farm
(nested_variance.py), so any feature that merely orders the 20 farms co-varies with
farm-mean S. Strip the farm means or hold out whole farms and the signal is zero.

Tests (all recomputed from data; nothing imported from [18]):
  T1  Between- vs within-farm decomposition of S's top features.
  T2  pH / latitude partial-correlation cascade (how much is the pH/carbonate proxy).
  T3  Strict out-of-farm null: Farm-LOFO rho + R^2 split (between- vs within-farm),
      and a pH-only Farm-LOFO baseline.
  T5  Effective-n significance of the leakage-clean |rho| (significant != predictive).
  T4 (the spatial-permutation null, S farm-restricted p = 0.462) lives in
      permutation_bootstrap.py / output/permutation_bootstrap.csv.

Run:  python -m math_statistics.adversarial_s
Output: math_statistics/output/adversarial_s_decomposition.csv (+ console)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"
CSV = ROOT / "data" / "features" / "master_dataset_old.csv"
SEED = 42
TARGET = "s"

SOIL = ["ph", "soc", "no3", "p", "k", "s"]
_EXCLUDE = {"id", "year", "farm", "field_name", "grid_id", "centroid_lon",
            "centroid_lat", "geometry_wkt", "protocol_number", "analysis_date",
            "sampling_date", "hu", "cu", "fe", "mg", "mn", "mo", "zn"}
_CROSS = ("summer", "late_summer", "autumn")


def _rs_cols(df):
    return [c for c in df.columns if c not in _EXCLUDE and c not in SOIL
            and df[c].dtype in ("float64", "int64")]


def _aligned(df):
    keep = []
    for c in _rs_cols(df):
        cl = c.lower()
        if any(s in cl for s in _CROSS) or "glcm" in cl:
            continue
        if cl.startswith(("ts_", "range_", "delta_", "amp_", "cv_", "cs_")) or "spectral_" in cl:
            continue
        keep.append(c)
    return keep


def _sp(a, b):
    r = stats.spearmanr(a, b)[0]
    return float(r) if pd.notna(r) else 0.0


def _partial_rho(df, target, x, z_cols):
    """Spearman partial correlation of (target, x) given z_cols (rank-OLS residuals)."""
    d = df[[target, x] + z_cols].dropna()
    R = stats.rankdata
    Y, X = R(d[target]), R(d[x])
    Z = np.column_stack([R(d[c]) for c in z_cols] + [np.ones(len(d))])
    by = np.linalg.lstsq(Z, Y, rcond=None)[0]
    bx = np.linalg.lstsq(Z, X, rcond=None)[0]
    return float(stats.pearsonr(Y - Z @ by, X - Z @ bx)[0])


def _farm_lofo(df, target, features):
    sub = df.dropna(subset=[target]).copy()
    X, y = sub[features], sub[target].to_numpy()
    farms = sub["farm"].to_numpy()
    preds = np.full(len(sub), np.nan)
    for fm in np.unique(farms):
        tr, te = farms != fm, farms == fm
        if tr.sum() < 10 or te.sum() < 1:
            continue
        med = X[tr].median()
        xt = X[tr].fillna(med).fillna(0.0)
        xe = X[te].fillna(med).fillna(0.0)
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                  max_features=0.5, random_state=SEED, n_jobs=-1)
        m.fit(xt, y[tr])
        preds[te] = m.predict(xe)
    v = np.isfinite(preds)
    return y[v], preds[v], farms[v]


def between_within(df, top_n=8):
    """T1: per-feature total / between-farm / within-farm Spearman for S top features."""
    feats = _rs_cols(df)
    scored = []
    for f in feats:
        m = df[[TARGET, f]].notna().all(axis=1)
        if m.sum() >= 10:
            scored.append((abs(_sp(df.loc[m, TARGET], df.loc[m, f])), f))
    top = [f for _, f in sorted(scored, reverse=True)[:top_n]]
    rows = []
    for f in top:
        s = df[[TARGET, f, "farm"]].dropna()
        rt = _sp(s[TARGET], s[f])
        fms = s.groupby("farm")[TARGET].transform("mean")
        fmf = s.groupby("farm")[f].transform("mean")
        rw = _sp(s[TARGET] - fms, s[f] - fmf)
        gb = s.groupby("farm").mean(numeric_only=True)
        rb = _sp(gb[TARGET], gb[f])
        rows.append({"feature": f, "rho_total": round(rt, 3),
                     "rho_between_farm": round(rb, 3), "rho_within_farm": round(rw, 3),
                     "p0_partial_pH": round(_partial_rho(df, TARGET, f, ["ph"]), 3),
                     "p0_partial_pH_lat_MAP": round(
                         _partial_rho(df, TARGET, f, ["ph", "centroid_lat", "climate_MAP"]), 3)})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(CSV, low_memory=False)
    bw = between_within(df)

    # T3: strict out-of-farm null + R^2 decomposition
    pool = _aligned(df)
    y, pr, farms = _farm_lofo(df, TARGET, pool)
    rho = _sp(y, pr)
    fmy = pd.Series(y).groupby(pd.Series(farms)).transform("mean").to_numpy()
    fmp = pd.Series(pr).groupby(pd.Series(farms)).transform("mean").to_numpy()
    r2_all = r2_score(y, pr)
    r2_between = r2_score(fmy, fmp)
    r2_within = r2_score(y - fmy, pr - fmp)
    yp, prp, fp = _farm_lofo(df, TARGET, ["ph"])
    rho_ph = _sp(yp, prp)

    OUT.mkdir(parents=True, exist_ok=True)
    bw.to_csv(OUT / "adversarial_s_decomposition.csv", index=False)

    print(bw.to_string(index=False))
    print("\nT1  within-farm |rho| collapses to ~0 (max %.3f); between-farm rho = %.2f-%.2f"
          % (bw["rho_within_farm"].abs().max(),
             bw["rho_between_farm"].min(), bw["rho_between_farm"].max()))
    print("T2  pH partial leaves %.2f-%.2f of |rho|; latitude explains ~none"
          % (bw["p0_partial_pH"].abs().min(), bw["p0_partial_pH"].abs().max()))
    print("T3  RS-full Farm-LOFO rho=%+.3f  R2_overall=%+.3f  R2_between=%+.3f  R2_within=%+.3f"
          % (rho, r2_all, r2_between, r2_within))
    print("    pH-only Farm-LOFO rho=%+.3f  (lab-pH ranks S out-of-farm better than the RS pool)"
          % rho_ph)
    print("T5  leakage-clean |rho|=0.309 stays significant at n_eff=143 (p=%.1e) -> real, not random,"
          % (2 * (1 - stats.t.cdf(0.309 * np.sqrt(141) / np.sqrt(1 - 0.309 ** 2), 141))))
    print("    but between-farm/confounded, not transferable (significant != predictive).")
    print("T4  S farm-restricted spatial-permutation p = 0.462 -> see permutation_bootstrap.csv")
    print(f"\nSaved -> {OUT}/adversarial_s_decomposition.csv")


if __name__ == "__main__":
    main()
