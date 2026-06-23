"""
permutation_bootstrap.py
========================
Reproducible false-positive controls for the screening — fixes audit finding M7
(the manuscript's toroidal-permutation p-values [162] and bootstrap CIs [161]/Fig 6
had no committed, traceable output).

1) MAX-STATISTIC PERMUTATION TEST. For each property, permute the target N times and
   recompute the MAXIMUM |rho| over the full 512-feature pool; the empirical
   p = (1 + #{perm_max >= observed_max}) / (N + 1) controls the multiple-comparison
   ("p-hacking") inflation inherent in screening hundreds of features. Two variants:
     - unrestricted permutation (exchangeable-under-null);
     - farm-restricted permutation (shuffle whole-farm target blocks) — a spatial-aware,
       more conservative control that respects the 20-farm clustering.
   (A toroidal-shift variant additionally needs the per-field raster grid, not stored
   in master_dataset_old.csv; the farm-restricted permutation is the committed proxy.)
2) BOOTSTRAP 95% CI of each property's winning |rho| (percentile, N resamples).

Deterministic: fixed seed. Vectorised via rank z-scores (Spearman = Pearson on ranks).
Run:  python -m math_statistics.permutation_bootstrap
Output: math_statistics/output/permutation_bootstrap.csv (+ console)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"
CSV = ROOT / "data" / "features" / "master_dataset_old.csv"
ALL = OUT / "all_spearman_correlations.csv"
TARGETS = ["ph", "k", "p", "no3", "soc", "s"]
LABEL = {"ph": "pH", "k": "K2O", "p": "P2O5", "no3": "NO3", "soc": "SOC", "s": "S"}
N_PERM, N_BOOT, SEED = 999, 1000, 42

EXCL = {"id", "year", "farm", "field_name", "grid_id", "centroid_lon", "centroid_lat",
        "geometry_wkt", "protocol_number", "analysis_date", "sampling_date", "hu",
        "cu", "fe", "mg", "mn", "mo", "zn"}


def zrank(a):
    r = rankdata(a)
    r = r - r.mean()
    sd = r.std()
    return r / sd if sd > 0 else r


def main():
    df = pd.read_csv(CSV, low_memory=False)
    feats = [c for c in df.columns if c not in EXCL and c not in TARGETS
             and df[c].dtype in ("float64", "int64")]
    # feature rank matrix (median-impute NaN for the null/bootstrap), z-scored
    F = df[feats].apply(lambda s: s.fillna(s.median())).to_numpy(dtype=float)
    Fz = np.apply_along_axis(zrank, 0, F)  # (n, k)
    allc = pd.read_csv(ALL)
    rng = np.random.default_rng(SEED)
    n = len(df)
    farms = df["farm"].to_numpy()
    farm_ids = np.unique(farms)

    rows = []
    for t in TARGETS:
        y = df[t].to_numpy(dtype=float)
        mask = ~np.isnan(y)
        yv = y[mask]
        Fzv = Fz[mask]
        nv = mask.sum()
        yz = zrank(yv)
        obs_max = float(np.max(np.abs(Fzv.T @ yz / nv)))

        # unrestricted permutation null of the max
        ge_un = 0
        for _ in range(N_PERM):
            yp = zrank(rng.permutation(yv))
            if np.max(np.abs(Fzv.T @ yp / nv)) >= obs_max - 1e-12:
                ge_un += 1
        p_un = (1 + ge_un) / (N_PERM + 1)

        # farm-restricted permutation (spatial-aware): decompose y into farm means
        # + within-farm deviations, then permute the BETWEEN-farm means across farms
        # while keeping each sample's within-farm deviation. This preserves within-farm
        # structure and tests whether the between-farm alignment with features is random
        # (handles unequal farm sizes correctly, unlike a raw block swap).
        fv = farms[mask]
        present = [f for f in farm_ids if (fv == f).any()]
        fmean = {f: yv[fv == f].mean() for f in present}
        dev = yv - np.array([fmean[f] for f in fv])
        mean_vals = np.array([fmean[f] for f in present])
        pos = {f: (fv == f) for f in present}
        ge_fr = 0
        for _ in range(N_PERM):
            perm_means = rng.permutation(mean_vals)
            new_mean = {f: perm_means[i] for i, f in enumerate(present)}
            yp = dev + np.array([new_mean[f] for f in fv])
            ypz = zrank(yp)
            if np.max(np.abs(Fzv.T @ ypz / nv)) >= obs_max - 1e-12:
                ge_fr += 1
        p_fr = (1 + ge_fr) / (N_PERM + 1)

        # bootstrap CI of the winning feature's |rho|
        win = allc[allc.target == t].sort_values("abs_rho", ascending=False).iloc[0]
        wf = win.feature
        xw = df[wf].to_numpy(dtype=float)
        m2 = mask & ~np.isnan(xw)
        xv, yv2 = xw[m2], y[m2]
        boots = np.empty(N_BOOT)
        nb = len(xv)
        for b in range(N_BOOT):
            idx = rng.integers(0, nb, nb)
            boots[b] = abs(spearmanr(xv[idx], yv2[idx])[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])

        rows.append({
            "Property": LABEL[t],
            "obs |rho|max": round(obs_max, 3),
            "p_perm (unrestricted)": round(p_un, 4),
            "p_perm (farm-restricted)": round(p_fr, 4),
            "winning feature": wf,
            "boot |rho| 95% CI": f"[{lo:.3f}, {hi:.3f}]",
        })
        print(f"  {LABEL[t]:4s} obs={obs_max:.3f}  p_un={p_un:.4f}  p_farm={p_fr:.4f}  "
              f"CI=[{lo:.3f},{hi:.3f}]  ({wf})")

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "permutation_bootstrap.csv", index=False)
    print(f"\nSaved -> {OUT}/permutation_bootstrap.csv (N_perm={N_PERM}, N_boot={N_BOOT}, seed={SEED})")


if __name__ == "__main__":
    main()
