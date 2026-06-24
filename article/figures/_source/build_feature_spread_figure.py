#!/usr/bin/env python3
"""Full-screening feature-spread figure: every RS feature as one point, by soil property.

Recomputes the complete per-feature Spearman screening from the raw master file
(reproducible; same exclusion logic as RHO_DISCREPANCY_REPORT.md), caches it to
math_statistics/output/feature_screening_full.csv, then draws the true spread of
|rho| for all 512 features per property, coloured by source group.

Leakage-suspect features (cross-season vs the ~75%-spring sample: summer / late-summer /
autumn windows and cross-season temporal aggregates) are drawn HOLLOW but kept; the
labelled top feature per property is the strongest *leakage-clean* one.

Run:  python scripts/build_feature_spread_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "features" / "master_dataset_old.csv"
CACHE = ROOT / "math_statistics" / "output" / "feature_screening_full.csv"
OUT = ROOT / "articles" / "article2_prediction" / "figures"

TARGETS = ["ph", "k", "p", "no3", "soc", "s"]
LABEL = {"ph": "pH", "k": "K$_2$O", "p": "P$_2$O$_5$", "no3": "NO$_3$", "soc": "SOC", "s": "S"}
EXCL = {"id", "year", "farm", "field_name", "grid_id", "centroid_lon", "centroid_lat",
        "geometry_wkt", "protocol_number", "analysis_date", "sampling_date", "hu"}

GROUP_COLORS = {
    "Spectral": "#4C78A8", "Climate": "#F28E2B", "Topographic": "#59A14F",
    "Temporal": "#B07AA1", "Texture (GLCM)": "#FF9DA7", "SAR": "#76B7B2",
    "Pedological": "#BAB0AC", "Other": "#9C755F",
}


def feature_group(f: str) -> str:
    f = str(f).lower()
    if f.startswith(("s2_", "l8_", "spectral_")):
        return "Spectral"
    if f.startswith("climate_"):
        return "Climate"
    if f.startswith("topo_"):
        return "Topographic"
    if f.startswith("glcm_"):
        return "Texture (GLCM)"
    if f.startswith("s1_"):
        return "SAR"
    if f.startswith(("sg_", "soilgrids")):
        return "Pedological"
    if "delta_" in f or "amp_" in f or f.startswith(("mean_", "ts_")):
        return "Temporal"
    return "Other"


def short(f: str) -> str:
    for p in ("glcm_glcm_", "glcm_", "s2_", "l8_", "s1_", "topo_", "climate_", "spectral_"):
        if str(f).startswith(p):
            return str(f)[len(p):]
    return str(f)


def leakage_suspect(f: str) -> bool:
    """Cross-season vs the ~75%-spring sample (RHO_DISCREPANCY_REPORT.md rule).

    Flags non-spring seasonal windows (summer/late-summer/autumn) and any
    cross-season aggregate/composite: time-series means, deltas, amplitudes,
    seasonal range/std, and trends.
    """
    f = str(f).lower()
    return any(k in f for k in (
        "summer", "late_summer", "autumn", "range",
        "ts_", "mean", "delta", "amp", "trend", "std"))


def _decorate(d: pd.DataFrame) -> pd.DataFrame:
    d["group"] = d["feature"].map(feature_group)
    d["short"] = d["feature"].map(short)
    d["leakage_suspect"] = d["feature"].map(leakage_suspect)
    return d


def compute_screening() -> pd.DataFrame:
    df = pd.read_csv(SRC, low_memory=False)
    feats = [c for c in df.columns
             if c not in EXCL and c not in TARGETS and pd.api.types.is_numeric_dtype(df[c])]
    print(f"feature pool = {len(feats)}")
    rows = []
    for t in TARGETS:
        s = df[t]
        rho = df[feats].corrwith(s, method="spearman")
        n = df[feats].notna().mul(s.notna(), axis=0).sum()
        for f in feats:
            r = rho.get(f)
            if pd.isna(r) or n[f] < 10:
                continue
            rows.append({"target": t, "feature": f, "rho": float(r),
                         "abs_rho": abs(float(r)), "n": int(n[f])})
    out = _decorate(pd.DataFrame(rows))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, index=False)
    return out


def load() -> pd.DataFrame:
    if CACHE.exists():
        return _decorate(pd.read_csv(CACHE))
    return compute_screening()


def build(d: pd.DataFrame) -> None:
    d = d.copy()
    present = [g for g in GROUP_COLORS if (d["group"] == g).any()]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(3, 2, figsize=(13, 9.2), sharex=True)
    axes = axes.ravel()
    for ax, t in zip(axes, TARGETS):
        sub = d[d["target"] == t]
        clean = sub[~sub["leakage_suspect"]]
        susp = sub[sub["leakage_suspect"]]

        # IQR band + median, as a quiet reference for the distribution
        q1, med, q3 = sub["abs_rho"].quantile([0.25, 0.5, 0.75])
        ax.fill_betweenx([-0.32, 0.32], q1, q3, color="#ECECEC", zorder=1)
        ax.plot([med, med], [-0.32, 0.32], color="#AAAAAA", lw=1.2, zorder=1)

        ax.scatter(susp["abs_rho"].to_numpy(), rng.uniform(-0.30, 0.30, len(susp)),
                   s=22, facecolors="none", edgecolors=susp["group"].map(GROUP_COLORS).tolist(),
                   alpha=0.5, linewidths=0.6, zorder=2)
        ax.scatter(clean["abs_rho"].to_numpy(), rng.uniform(-0.30, 0.30, len(clean)),
                   s=18, c=clean["group"].map(GROUP_COLORS).tolist(), alpha=0.7,
                   linewidths=0, zorder=3)

        if len(clean):
            top = clean.nlargest(1, "abs_rho").iloc[0]
            ax.scatter([top["abs_rho"]], [0], s=70, facecolors="none", edgecolors="#222222",
                       linewidths=1.3, zorder=4)
            ax.annotate(f"{top['short']}  ({top['rho']:+.2f})", (top["abs_rho"], 0.40),
                        ha="center", va="bottom", fontsize=8, color="#222222")

        ax.set_title(LABEL[t], fontsize=13, loc="left", fontweight="bold")
        ax.set_ylim(-0.62, 0.62)
        ax.set_yticks([])
        ax.set_xlim(0, 0.72)
        ax.grid(True, axis="x", alpha=0.2)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.text(0.715, -0.55, f"{len(sub)} features", ha="right", va="bottom",
                fontsize=7.5, color="#999999")

    for ax in axes[-2:]:
        ax.set_xlabel("|Spearman ρ|  (one point = one RS feature)")

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=7,
                      markerfacecolor=GROUP_COLORS[g], markeredgecolor="none", label=g)
               for g in present]
    handles += [
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               markerfacecolor="none", markeredgecolor="#555555", label="leakage-suspect"),
        Line2D([], [], marker="o", linestyle="none", markersize=9,
               markerfacecolor="none", markeredgecolor="#222222", markeredgewidth=1.3,
               label="strongest clean"),
    ]
    fig.legend(handles=handles, title="Source group", frameon=False, fontsize=8.5,
               title_fontsize=9, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Spread of individual RS features, per soil property (full screening, 512 each)",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    png = OUT / "fig_corr_features_spread_full.png"
    pdf = OUT / "fig_corr_features_spread_full.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    with Image.open(png) as img:
        img.save(png, format="PNG", dpi=(300, 300), optimize=True)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    rep = d.groupby("target").apply(
        lambda g: pd.Series({
            "n": len(g),
            "n_suspect": int(g["leakage_suspect"].sum()),
            "max_all": g["abs_rho"].max(),
            "max_clean": g.loc[~g["leakage_suspect"], "abs_rho"].max(),
        }), include_groups=False)
    print(rep.to_string())


def main() -> None:
    build(load())


if __name__ == "__main__":
    main()
