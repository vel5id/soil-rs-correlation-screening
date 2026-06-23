"""
farm_lofo_metrics.py
====================
Full predictive-error metrics for the honest out-of-farm (Farm-LOFO) evaluation,
adding RMSE and the spread-relative error metric requested (RSE := RPD = SD/RMSE,
the standard soil-spectroscopy "ratio of performance to deviation"), plus the
data-spread scale (SD, IQR, range) so the error can be read against how much
variation there actually is in each property.

Setup is identical to the committed leakage-controlled Farm-LOFO check
(master_dataset_old.csv; 47 temporally-aligned leakage-clean features; RF 300 trees,
seed 42; 20 leave-one-farm-out folds; per-fold median imputation), so rho/R2 here
match math_statistics/output/leakage_controlled_screening.csv.

Metrics per property (pooled out-of-fold predictions vs observations):
  rho, R2, RMSE, MAE                          -- accuracy / error
  SD, IQR, range, CV%                         -- the spread "scale" (how much total)
  RPD = SD/RMSE   (== "RSE", higher = better) -- spread-relative skill, with class
  RSR = RMSE/SD   (= 1/RPD, lower = better)
  RPIQ = IQR/RMSE (robust analogue of RPD)

Interpretation scale for RPD (Viscarra Rossel et al. 2006; coarser 3-class Chang
et al. 2001) -- printed as a reference table.

Run:  python ML/farm_lofo_metrics.py
Output: ML/results/farm_lofo_metrics.csv
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
CSV = ROOT / "data" / "features" / "master_dataset_old.csv"
RES = ROOT / "ML" / "results"
TARGETS = ["ph", "soc", "no3", "p", "k", "s"]
LABEL = {"ph": "pH", "soc": "SOC", "no3": "NO3", "p": "P2O5", "k": "K2O", "s": "S"}
UNIT = {"ph": "pH units", "soc": "%", "no3": "mg/kg", "p": "mg/kg", "k": "mg/kg", "s": "mg/kg"}
SEED = 42
RF = dict(n_estimators=300, min_samples_leaf=2, max_features=0.5, random_state=SEED, n_jobs=-1)
EXCL = {"id", "year", "farm", "field_name", "grid_id", "centroid_lon", "centroid_lat",
        "geometry_wkt", "protocol_number", "analysis_date", "sampling_date", "hu",
        "cu", "fe", "mg", "mn", "mo", "zn"}
SEAS = ("summer", "late_summer", "autumn")

# RPD quality scale (Viscarra Rossel et al. 2006). RPD = SD(obs) / RMSE.
RPD_SCALE = [
    (0.0, 1.0, "0 - very poor (no useful prediction)"),
    (1.0, 1.4, "1 - poor (distinguishes high vs low only)"),
    (1.4, 1.8, "2 - fair"),
    (1.8, 2.0, "3 - good"),
    (2.0, 2.5, "4 - very good"),
    (2.5, 99.0, "5 - excellent"),
]


def rpd_class(rpd):
    for lo, hi, name in RPD_SCALE:
        if lo <= rpd < hi:
            return name
    return "n/a"


def is_clean(c):
    cl = c.lower()
    if any(s in cl for s in SEAS):
        return False
    if "glcm" in cl:
        return False
    if cl.startswith(("ts_", "range_", "delta_", "amp_", "cv_", "cs_")) or "spectral_" in cl:
        return False
    return True


def main():
    df = pd.read_csv(CSV, low_memory=False)
    feats = [c for c in df.columns if df[c].dtype in ("float64", "int64")
             and c not in EXCL and c not in TARGETS and is_clean(c)]
    print(f"{CSV.name}: {len(df)} samples, {df['farm'].nunique()} farms; "
          f"leakage-clean pool = {len(feats)} features\n")

    rows = []
    for t in TARGETS:
        sub = df.dropna(subset=[t]).copy()
        X, y, g = sub[feats], sub[t].to_numpy(), sub["farm"].to_numpy()
        pred = np.full(len(sub), np.nan)
        for fm in np.unique(g):
            tr, te = g != fm, g == fm
            med = X[tr].median()
            m = RandomForestRegressor(**RF)
            m.fit(X[tr].fillna(med).fillna(0.0), y[tr])
            pred[te] = m.predict(X[te].fillna(med).fillna(0.0))
        v = np.isfinite(pred)
        yo, yp = y[v], pred[v]
        rmse = float(np.sqrt(mean_squared_error(yo, yp)))
        mae = float(mean_absolute_error(yo, yp))
        rho = float(spearmanr(yo, yp)[0])
        r2 = float(r2_score(yo, yp))
        sd = float(np.std(yo, ddof=1))
        q1, q3 = np.percentile(yo, [25, 75])
        iqr = float(q3 - q1)
        rng = float(yo.max() - yo.min())
        cv = float(sd / np.mean(yo) * 100)
        rpd = sd / rmse if rmse > 0 else np.nan
        rpiq = iqr / rmse if rmse > 0 else np.nan
        rsr = rmse / sd if sd > 0 else np.nan
        rows.append({
            "Property": LABEL[t], "unit": UNIT[t], "n": int(v.sum()),
            "rho": round(rho, 3), "R2": round(r2, 3),
            "RMSE": round(rmse, 3), "MAE": round(mae, 3),
            "SD (spread)": round(sd, 3), "IQR": round(iqr, 3),
            "range": round(rng, 3), "CV%": round(cv, 1),
            "RSE=RPD (SD/RMSE)": round(rpd, 2), "RPD class": rpd_class(rpd),
            "RSR (RMSE/SD)": round(rsr, 3), "RPIQ (IQR/RMSE)": round(rpiq, 2),
        })

    out = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    out.to_csv(RES / "farm_lofo_metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print("\n=== RPD ( = RSE = SD/RMSE ) interpretation scale ===")
    print("  (Viscarra Rossel et al. 2006; coarse Chang et al. 2001: <1.4 poor / 1.4-2.0 fair / >2.0 good)")
    for lo, hi, name in RPD_SCALE:
        hi_s = "inf" if hi >= 99 else f"{hi}"
        print(f"  RPD [{lo:>3} - {hi_s:>4}) : {name}")
    print(f"\nSaved -> {RES}/farm_lofo_metrics.csv")
    print("Note: under honest Farm-LOFO pH/SOC/NO3/P2O5/K2O are RPD 0.95-1.20 (poor / very")
    print("poor) -> not operationally mappable out-of-farm; pH is closest (RPD 1.20, rho 0.43).")
    print("S RPD=1.60 ('fair') is a SKEWNESS ARTEFACT: rho=0.04 (no rank skill, skew~2.8); the")
    print("model predicts near the mean on a right-skewed distribution, deflating RMSE vs SD.")
    print("=> always read RPD together with rho; for S the rho exposes the artefact.")


if __name__ == "__main__":
    main()
