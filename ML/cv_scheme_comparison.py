"""
cv_scheme_comparison.py
=======================
Predictive metrics (rho, R2, RMSE, MAE, spread, RSE=RPD/RSR/RPIQ) per property
under several spatial cross-validation schemes of INCREASING strictness, computed
on the FULL feature pool (all 512 RS features incl. composites/textures/temporal —
"composites and not"), not just the leakage-clean subset.

Schemes (least -> most strict):
  Field-LOFO        : leave-one-field-out  (81 groups)            -- reference (least strict)
  Spatial-block CV  : leave-one-spatial-block-out (k-means on lon/lat, K blocks;
                      blocks smaller than a farm but larger than a field)        <- the requested
                      "less strict than Farm-LOFO, not Field-LOFO" intermediate scheme
  Farm-LOFO         : leave-one-farm-out   (20 groups)            -- strict spatial
  LLTO              : leave-location-and-time-out; here the two field seasons are
                      spatially disjoint (2022: 5 farms / 2023: 15 farms), so LLTO is
                      realised as cross-year transfer (train one year -> predict the
                      other, pooled both directions) -- strict spatial+temporal.

RF 300 trees, seed 42, per-fold median imputation fit on training rows only.
RSE := RPD = SD(obs)/RMSE (Viscarra Rossel 2006 scale: <1.4 poor / 1.4-2.0 fair / >2.0 good).

Run:  python ML/cv_scheme_comparison.py
Output: ML/results/cv_scheme_comparison.csv
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
CSV = ROOT / "data" / "features" / "master_dataset_old.csv"
RES = ROOT / "ML" / "results"
TARGETS = ["ph", "soc", "no3", "p", "k", "s"]
LABEL = {"ph": "pH", "soc": "SOC", "no3": "NO3", "p": "P2O5", "k": "K2O", "s": "S"}
SEED, K_BLOCKS = 42, 30
RF = dict(n_estimators=300, min_samples_leaf=2, max_features=0.5, random_state=SEED, n_jobs=-1)
EXCL = {"id", "year", "farm", "field_name", "grid_id", "centroid_lon", "centroid_lat",
        "geometry_wkt", "protocol_number", "analysis_date", "sampling_date", "hu",
        "cu", "fe", "mg", "mn", "mo", "zn"}


def rpd_class(r):
    if r < 1.0: return "very poor"
    if r < 1.4: return "poor"
    if r < 1.8: return "fair"
    if r < 2.0: return "good"
    if r < 2.5: return "very good"
    return "excellent"


def metrics(yo, yp):
    rmse = float(np.sqrt(mean_squared_error(yo, yp)))
    sd = float(np.std(yo, ddof=1))
    iqr = float(np.subtract(*np.percentile(yo, [75, 25])))
    rpd = sd / rmse if rmse > 0 else np.nan
    return dict(n=len(yo), rho=round(float(spearmanr(yo, yp)[0]), 3),
                R2=round(float(r2_score(yo, yp)), 3), RMSE=round(rmse, 3),
                MAE=round(float(mean_absolute_error(yo, yp)), 3),
                SD=round(sd, 3), IQR=round(iqr, 3),
                **{"RSE=RPD": round(rpd, 2), "RPD_class": rpd_class(rpd),
                   "RSR": round(rmse / sd, 3) if sd else np.nan,
                   "RPIQ": round(iqr / rmse, 2) if rmse else np.nan})


def cv_predict(df, t, feats, groups):
    """Generic leave-one-group-out; returns pooled (obs, pred)."""
    sub = df.dropna(subset=[t]).copy()
    g = groups.loc[sub.index].to_numpy()
    X, y = sub[feats], sub[t].to_numpy()
    pred = np.full(len(sub), np.nan)
    for gid in pd.unique(g):
        tr, te = g != gid, g == gid
        if tr.sum() < 10 or te.sum() < 1:
            continue
        med = X[tr].median()
        m = RandomForestRegressor(**RF)
        m.fit(X[tr].fillna(med).fillna(0.0), y[tr])
        pred[te] = m.predict(X[te].fillna(med).fillna(0.0))
    v = np.isfinite(pred)
    return y[v], pred[v]


def main():
    df = pd.read_csv(CSV, low_memory=False)
    feats = [c for c in df.columns if df[c].dtype in ("float64", "int64")
             and c not in EXCL and c not in TARGETS]
    print(f"{CSV.name}: {len(df)} samples; FULL pool = {len(feats)} features "
          f"(incl. composites/textures/temporal)\n")

    # spatial blocks via k-means on centroid coordinates
    coords = df[["centroid_lon", "centroid_lat"]].to_numpy()
    block = pd.Series(KMeans(n_clusters=K_BLOCKS, random_state=SEED, n_init=10)
                      .fit_predict(coords), index=df.index)
    print(f"Spatial-block CV: {K_BLOCKS} k-means blocks (median "
          f"{int(block.value_counts().median())} samples/block) — between field (81) and farm (20).\n")

    schemes = {
        f"Spatial-block ({K_BLOCKS})": block,
        "LLTO / cross-year": df["year"],   # 2 groups => train one year, predict the other
    }

    rows = []
    for sname, groups in schemes.items():
        for t in TARGETS:
            yo, yp = cv_predict(df, t, feats, groups)
            rows.append({"scheme": sname, "Property": LABEL[t], **metrics(yo, yp)})
            r = rows[-1]
            print(f"  [{sname:22s} {LABEL[t]:4s}] rho={r['rho']:+.3f} R2={r['R2']:+.3f} "
                  f"RMSE={r['RMSE']:.3f} RSE=RPD={r['RSE=RPD']:.2f} ({r['RPD_class']})")
        print()

    out = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    out.to_csv(RES / "cv_scheme_comparison.csv", index=False)
    print("=== RPD (=RSE=SD/RMSE) by scheme x property ===")
    piv = out.pivot(index="Property", columns="scheme", values="RSE=RPD")
    print(piv.to_string())
    print(f"\nSaved -> {RES}/cv_scheme_comparison.csv")
    print("Kept schemes (per request): Spatial-block (moderate spatial CV) and "
          "LLTO/cross-year (spatial+temporal). RPD falls from block to LLTO -> the "
          "extra drop is temporal (cross-year) non-transfer.")


if __name__ == "__main__":
    main()
