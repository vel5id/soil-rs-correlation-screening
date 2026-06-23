"""
feature_quality_cv.py
=====================
PER-FEATURE quality table for all 500+ features: how much of each feature's
association with each soil property is GENUINE (survives spatial-block control and
replicates across years) vs spatial/temporal leakage. Lets you rank which features
are actually informative.

For each (feature, target) we report a robust decomposition (NO univariate model —
a per-fold univariate OLS over spatial blocks manufactures spurious rank structure;
we use rank-correlation components instead):

  rho_full        : whole-dataset Spearman (the raw screen; inflated by spatial structure)
  block_within    : Spearman after removing spatial-block means (k-means, K blocks) from
                    both ranks -> the signal that survives moderate spatial CV (Spatial-block)
  block_between    : Spearman of the K block-mean ranks -> the spatial (zonal) component
  rho_2022, rho_2023 : per-year Spearman -> temporal stability (the LLTO / cross-year view)
  year_consistent : same sign in both years AND min|rho_year| >= 0.10

A high-quality, generalisable feature has a non-trivial |block_within| AND consistent
per-year signs; a leakage feature has high rho_full / block_between but ~0 block_within
and flips sign across years.

Schemes kept per request: Spatial-block (within/between) and LLTO (cross-year).
Run:  python ML/feature_quality_cv.py
Output: ML/results/feature_quality_cv.csv (feature x target ~3072 rows) + console summary.
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
CSV = ROOT / "data" / "features" / "master_dataset_old.csv"
RES = ROOT / "ML" / "results"
TARGETS = ["ph", "soc", "no3", "p", "k", "s"]
LABEL = {"ph": "pH", "soc": "SOC", "no3": "NO3", "p": "P2O5", "k": "K2O", "s": "S"}
SEED, K_BLOCKS = 42, 30
EXCL = {"id", "year", "farm", "field_name", "grid_id", "centroid_lon", "centroid_lat",
        "geometry_wkt", "protocol_number", "analysis_date", "sampling_date", "hu",
        "cu", "fe", "mg", "mn", "mo", "zn"}
_CROSS = ("summer", "late_summer", "autumn")


def season_of(c):
    cl = c.lower()
    for s in ("late_summer", "summer", "spring", "autumn"):
        if s in cl:
            return s.replace("_", "-")
    return "static/multi"


def group_of(c):
    cl = c.lower()
    if "glcm" in cl: return "Texture"
    if cl.startswith("topo"): return "Topographic"
    if cl.startswith("climate"): return "Climatic"
    if cl.startswith(("ts_", "range_", "delta_", "amp_", "cv_")): return "Temporal"
    if cl.startswith(("s1_", "sar")): return "SAR"
    if "spectral_" in cl: return "Spectral-composite"
    return "Spectral-base"


def is_leaky(c):
    cl = c.lower()
    return (any(s in cl for s in _CROSS) or "glcm" in cl
            or cl.startswith(("ts_", "range_", "delta_", "amp_", "cv_", "cs_")) or "spectral_" in cl)


def main():
    df = pd.read_csv(CSV, low_memory=False)
    feats = [c for c in df.columns if df[c].dtype in ("float64", "int64")
             and c not in EXCL and c not in TARGETS]
    block_all = KMeans(n_clusters=K_BLOCKS, random_state=SEED, n_init=10).fit_predict(
        df[["centroid_lon", "centroid_lat"]].to_numpy())
    year_all = df["year"].to_numpy()
    print(f"{CSV.name}: {len(df)} samples; {len(feats)} features x {len(TARGETS)} targets "
          f"(spatial blocks K={K_BLOCKS})\n")

    def demean_by(codes, v):
        counts = np.bincount(codes)
        means = np.bincount(codes, weights=v) / counts
        return v - means[codes]

    rows = []
    for t in TARGETS:
        ym = df[t].notna().to_numpy()
        y = df.loc[ym, t].to_numpy(float)
        blk = block_all[ym]
        # recode blocks to 0..K-1 contiguous for bincount
        _, blk = np.unique(blk, return_inverse=True)
        yr = year_all[ym]
        ry = rankdata(y)
        ry_dev = demean_by(blk, ry)
        # block-mean ranks of y for between-block
        ymean_blk = np.bincount(blk, weights=ry) / np.bincount(blk)
        m22, m23 = yr == 2022, yr == 2023
        for f in feats:
            x = df.loc[ym, f].to_numpy(float)
            ok = np.isfinite(x)
            if ok.sum() < 50 or np.nanstd(x) < 1e-12:
                continue
            rho_full = spearmanr(y[ok], x[ok])[0]
            # within-block (rank residual correlation)
            rx = rankdata(np.where(ok, x, np.nanmedian(x[ok])))
            rx_dev = demean_by(blk, rx)
            sx, sy = rx_dev.std(), ry_dev.std()
            within = float(np.mean(rx_dev * ry_dev) / (sx * sy)) if sx > 1e-9 and sy > 1e-9 else np.nan
            xmean_blk = np.bincount(blk, weights=rx) / np.bincount(blk)
            between = spearmanr(xmean_blk, ymean_blk)[0]
            r22 = spearmanr(y[m22 & ok], x[m22 & ok])[0] if (m22 & ok).sum() > 20 else np.nan
            r23 = spearmanr(y[m23 & ok], x[m23 & ok])[0] if (m23 & ok).sum() > 20 else np.nan
            consistent = (np.isfinite(r22) and np.isfinite(r23)
                          and np.sign(r22) == np.sign(r23)
                          and min(abs(r22), abs(r23)) >= 0.10)
            rows.append({
                "feature": f, "group": group_of(f), "season": season_of(f),
                "leakage_suspect": "YES" if is_leaky(f) else "no", "target": LABEL[t],
                "rho_full": round(float(rho_full), 3),
                "block_within": round(within, 3) if np.isfinite(within) else np.nan,
                "block_between": round(float(between), 3) if np.isfinite(between) else np.nan,
                "rho_2022": round(float(r22), 3) if np.isfinite(r22) else np.nan,
                "rho_2023": round(float(r23), 3) if np.isfinite(r23) else np.nan,
                "year_consistent": "yes" if consistent else "no",
            })

    out = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    out.to_csv(RES / "feature_quality_cv.csv", index=False)

    print("=== TOP-8 features per property by |block_within| (genuine, spatially-controlled signal) ===")
    for t in TARGETS:
        sub = out[out.target == LABEL[t]].copy()
        sub["a"] = sub["block_within"].abs()
        top = sub.sort_values("a", ascending=False).head(8)
        print(f"\n{LABEL[t]}:")
        for _, r in top.iterrows():
            print(f"   within={r.block_within:+.3f} full={r.rho_full:+.3f} between={r.block_between:+.3f}  "
                  f"yr22={r.rho_2022} yr23={r.rho_2023} consist={r.year_consistent}  "
                  f"[{r.group},{r.season},leak={r.leakage_suspect}]  {r.feature}")

    print("\n=== mean |block_within| by feature GROUP (which families carry real signal) ===")
    g = out.assign(a=out["block_within"].abs()).groupby("group")["a"].agg(["mean", "max", "count"]).round(3)
    print(g.sort_values("mean", ascending=False).to_string())
    print("\n=== mean |block_within| by SEASON ===")
    s = out.assign(a=out["block_within"].abs()).groupby("season")["a"].agg(["mean", "max", "count"]).round(3)
    print(s.sort_values("mean", ascending=False).to_string())
    nconsist = (out["year_consistent"] == "yes").sum()
    print(f"\nFeature x target pairs temporally consistent (both years, |rho|>=0.10, same sign): "
          f"{nconsist}/{len(out)} ({100*nconsist/len(out):.0f}%)")
    print(f"Saved -> {RES}/feature_quality_cv.csv  ({len(out)} feature x target rows)")


if __name__ == "__main__":
    main()
