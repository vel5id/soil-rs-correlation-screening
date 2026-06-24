#!/usr/bin/env python3
"""Render Table 7 (feature predictability taxonomy) as a two-panel figure.

Reads the committed taxonomy table and recomputes everything from it (no values are
hardcoded):
  - left panel  : per-property composition of the 512 candidate RS features into
                  robustness tiers (robust / generalises / zonal-only / unstable / weak)
  - right panel : dumbbell from the in-sample |rho|max ceiling to the out-of-farm
                  Farm-LOFO rho, showing how much signal survives leaving the farm

Data source : ML/results/key_table_taxonomy.csv  (== manuscript Table 7).
              NOTE: the Farm-LOFO column here is the taxonomy's own out-of-farm rho;
              do NOT substitute ML/results/farm_lofo_all_targets.csv (different model /
              dataset, e.g. S=0.317 vs 0.038).
Verdict text: manuscript wording (Table 7), kept verbatim.
Style       : matches scripts/build_article1_figures_en.py (Tableau palette, 300 DPI
              PNG + vector PDF).

Run:  python scripts/build_table7_predictability_figure.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ML" / "results" / "key_table_taxonomy.csv"
OUT_DIR = ROOT / "articles" / "article2_prediction" / "figures"
TOTAL_FEATURES = 512

# Ordinal good -> bad tier colours, drawn from the article Tableau palette.
C_ROBUST = "#59A14F"    # generalises (robust)
C_GENERAL = "#4C78A8"   # generalises (not robust)
C_ZONAL = "#F28E2B"     # zonal-only (regional)
C_UNSTABLE = "#E15759"  # unstable
C_WEAK = "#D6D3CC"      # weak / no signal (background track)
C_DARK = "#3B3B3B"      # dumbbell markers / connectors

# Manuscript verdict wording (Table 7), keyed by property label.
VERDICTS = {
    "pH": "regionally + locally mappable",
    "K2O": "mappable (clay/SWIR)",
    "P2O5": "weakly mappable (texture)",
    "NO3": "indirect (canopy-N proxy)",
    "SOC": "regional gradient only",
    "S": "unpredictable",
}

# Pretty subscripts for the y labels.
DISPLAY = {
    "pH": "pH", "K2O": "K$_2$O", "P2O5": "P$_2$O$_5$",
    "NO3": "NO$_3$", "SOC": "SOC", "S": "S",
}


def _parse_general(cell: str) -> tuple[int, int]:
    """'13 (6)' -> (general_total=13, robust=6)."""
    m = re.match(r"\s*(\d+)\s*\((\d+)\)", str(cell))
    if not m:
        raise ValueError(f"cannot parse 'general.(robust)' cell: {cell!r}")
    return int(m.group(1)), int(m.group(2))


def load_table() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df.columns = [c.strip() for c in df.columns]
    gen = df["general.(robust)"].map(_parse_general)
    df["general_total"] = [g[0] for g in gen]
    df["robust"] = [g[1] for g in gen]
    df["general_nonrobust"] = df["general_total"] - df["robust"]
    df["rho_max"] = df["|rho|max(LC)"].astype(float)
    df["farm_lofo"] = df["Farm-LOFO rho"].astype(float)
    for col in ("zonal_only", "unstable", "weak"):
        df[col] = df[col].astype(int)
    df["informative"] = df["general_total"] + df["zonal_only"] + df["unstable"]

    # integrity check: the four tiers must account for the full candidate pool
    tot = df["general_total"] + df["zonal_only"] + df["unstable"] + df["weak"]
    bad = df.loc[tot != TOTAL_FEATURES, "Property"].tolist()
    if bad:
        raise AssertionError(f"tier counts do not sum to {TOTAL_FEATURES} for: {bad}")

    return df.sort_values("rho_max", ascending=False).reset_index(drop=True)


def build_figure(df: pd.DataFrame) -> plt.Figure:
    n = len(df)
    ys = list(range(n))[::-1]  # highest rho on top

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.5, 5.4),
        gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.06},
    )

    # ---- left panel: stacked tier composition over the 512-feature pool ----
    for yi, (_, r) in zip(ys, df.iterrows()):
        axL.barh(yi, TOTAL_FEATURES, color=C_WEAK, height=0.6, zorder=1)
        left = 0.0
        for val, col in (
            (r.robust, C_ROBUST),
            (r.general_nonrobust, C_GENERAL),
            (r.zonal_only, C_ZONAL),
            (r.unstable, C_UNSTABLE),
        ):
            axL.barh(yi, val, left=left, color=col, height=0.6, zorder=2)
            left += val
        axL.text(left + 8, yi, f"{int(r.informative)} informative · {int(r.robust)} robust",
                 va="center", ha="left", fontsize=8, color="#555555")

    axL.set_xlim(0, TOTAL_FEATURES)
    axL.set_ylim(-0.6, n - 0.4)
    axL.set_yticks([])
    axL.set_xlabel("Number of RS features (of 512)")
    axL.set_title("Feature tiers", fontsize=11)
    axL.grid(True, axis="x", alpha=0.2)
    for s in ("top", "right", "left"):
        axL.spines[s].set_visible(False)

    # two-line left labels: property name + verdict
    trans = mtransforms.blended_transform_factory(axL.transAxes, axL.transData)
    for yi, p in zip(ys, df["Property"]):
        axL.text(-0.015, yi + 0.13, DISPLAY[p], transform=trans, ha="right", va="bottom",
                 fontsize=12, fontweight="bold", color="#222222")
        axL.text(-0.015, yi - 0.13, VERDICTS[p], transform=trans, ha="right", va="top",
                 fontsize=7.5, color="#777777", style="italic")

    # ---- right panel: dumbbell |rho|max (ceiling) -> Farm-LOFO rho (out-of-farm) ----
    for yi, (_, r) in zip(ys, df.iterrows()):
        axR.plot([r.farm_lofo, r.rho_max], [yi, yi], color="#9A9A9A", lw=2, zorder=1)
        axR.scatter(r.rho_max, yi, s=70, facecolors="white", edgecolors=C_DARK,
                    linewidths=1.6, zorder=3)
        axR.scatter(r.farm_lofo, yi, s=70, color=C_DARK, zorder=3)
        axR.text(r.rho_max + 0.013, yi, f"{r.rho_max:.2f}", va="center", ha="left",
                 fontsize=8, color="#555555")
        axR.text(r.farm_lofo - 0.013, yi, f"{r.farm_lofo:.2f}", va="center", ha="right",
                 fontsize=8, color="#555555")

    axR.set_xlim(0, 0.75)
    axR.set_ylim(-0.6, n - 0.4)
    axR.set_yticks([])
    axR.set_xlabel("Spearman ρ")
    axR.set_title("In-sample max → out-of-farm (Farm-LOFO)", fontsize=11)
    axR.grid(True, axis="x", alpha=0.2)
    for s in ("top", "right", "left"):
        axR.spines[s].set_visible(False)

    # ---- legends: two centred rows below the figure (tiers, then markers) ----
    tier_handles = [plt.Rectangle((0, 0), 1, 1, color=c)
                    for c in (C_ROBUST, C_GENERAL, C_ZONAL, C_UNSTABLE, C_WEAK)]
    tier_labels = ["generalises (robust)", "generalises", "zonal-only", "unstable", "weak (no signal)"]
    fig.legend(tier_handles, tier_labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=5, frameon=False, fontsize=8.5, handlelength=1.1, columnspacing=1.6)

    marker_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor="white", markeredgecolor=C_DARK, markeredgewidth=1.6),
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=C_DARK),
    ]
    fig.legend(marker_handles, [r"in-sample $|\rho|_{\max}$ (ceiling)", "Farm-LOFO ρ (out-of-farm)"],
               loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False,
               fontsize=8.5, columnspacing=2.0)

    fig.suptitle("Remote-sensing predictability of soil properties: feature tiers and ρ shrinkage",
                 fontsize=13, y=0.99)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.88, bottom=0.10, wspace=0.06)
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_table()
    fig = build_figure(df)

    png = OUT_DIR / "fig_feature_tiers_predictability.png"
    pdf = OUT_DIR / "fig_feature_tiers_predictability.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    with Image.open(png) as img:
        img.save(png, format="PNG", dpi=(300, 300), optimize=True)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    cols = ["Property", "rho_max", "farm_lofo", "robust", "general_total",
            "zonal_only", "unstable", "weak", "informative"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
