#!/usr/bin/env python3
"""Feature-level preview figures: emphasise individual RS features and their metrics.

Two candidate styles, both built from the per-feature screening
(math_statistics/output/correlation_analysis.xlsx, sheet 'top20_per_target'):
  A) ranked named-feature small-multiples (one panel per soil property)
  B) strip / spread of |rho| across individual features, coloured by source group

Style matches scripts/build_article1_figures_en.py (Tableau palette, 300 DPI).
Run:  python scripts/build_feature_level_previews.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
XL = ROOT / "math_statistics" / "output" / "correlation_analysis.xlsx"
OUT = ROOT / "articles" / "article2_prediction" / "figures"

TARGETS = ["ph", "k", "p", "no3", "soc", "s"]
LABEL = {"ph": "pH", "k": "K$_2$O", "p": "P$_2$O$_5$", "no3": "NO$_3$", "soc": "SOC", "s": "S"}

GROUP_COLORS = {
    "Spectral": "#4C78A8",
    "Climate": "#F28E2B",
    "Topographic": "#59A14F",
    "Temporal": "#B07AA1",
    "Texture (GLCM)": "#FF9DA7",
    "SAR": "#76B7B2",
    "Other": "#9C755F",
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
    if "delta_" in f or "amp_" in f or f.startswith(("mean_", "ts_")):
        return "Temporal"
    return "Other"


def short(f: str) -> str:
    for p in ("glcm_glcm_", "glcm_", "s2_", "l8_", "s1_", "topo_", "climate_", "spectral_"):
        if str(f).startswith(p):
            return str(f)[len(p):]
    return str(f)


def load() -> pd.DataFrame:
    d = pd.read_excel(XL, sheet_name="top20_per_target")
    d["group"] = d["feature"].map(feature_group)
    d["short"] = d["feature"].map(short)
    return d


def fig_a_ranked(d: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    top_n = 12
    for ax, t in zip(axes, TARGETS):
        sub = d[d["target"] == t].nlargest(top_n, "abs_rho").sort_values("abs_rho")
        colors = [GROUP_COLORS[g] for g in sub["group"]]
        ax.barh(sub["short"], sub["abs_rho"], color=colors)
        for y, (_, r) in enumerate(sub.iterrows()):
            ax.text(r["abs_rho"] + 0.008, y, f"{r['rho']:+.2f}", va="center", ha="left",
                    fontsize=7.5, color="#555555")
        ax.set_title(LABEL[t], fontsize=12)
        ax.set_xlim(0, min(1.0, sub["abs_rho"].max() * 1.22))
        ax.set_xlabel("|Spearman ρ|", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, axis="x", alpha=0.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    present = [g for g in GROUP_COLORS if (d["group"] == g).any()]
    handles = [Patch(color=GROUP_COLORS[g], label=g) for g in present]
    fig.legend(handles, present, loc="lower center", ncol=len(present), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Variant A — individual RS features ranked by |ρ|, per soil property (top 12)",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, OUT / "fig_corr_features_ranked.png")


def fig_b_spread(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    order = [LABEL[t] for t in TARGETS]
    d = d.copy()
    d["prop"] = d["target"].map(LABEL)
    sns.stripplot(data=d, x="prop", y="abs_rho", hue="group", order=order,
                  palette=GROUP_COLORS, jitter=0.28, size=7, alpha=0.85,
                  edgecolor="white", linewidth=0.4, ax=ax)
    # label the single strongest feature per property
    for i, t in enumerate(TARGETS):
        top = d[d["target"] == t].nlargest(1, "abs_rho").iloc[0]
        ax.annotate(top["short"], (i, top["abs_rho"]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=7.5, color="#333333")
    ax.set_xlabel("")
    ax.set_ylabel("|Spearman ρ| (individual feature)")
    ax.set_ylim(0, 0.78)
    ax.set_title("Variant B — spread of individual RS features by property (top-20 screened)",
                 fontsize=13)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Source group", frameon=False, fontsize=8.5, title_fontsize=9,
              loc="upper right", ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    _save(fig, OUT / "fig_corr_features_spread.png")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    with Image.open(path) as img:
        img.save(path, format="PNG", dpi=(300, 300), optimize=True)
    print(f"Wrote {path}")


def main() -> None:
    d = load()
    fig_a_ranked(d)
    fig_b_spread(d)


if __name__ == "__main__":
    main()
