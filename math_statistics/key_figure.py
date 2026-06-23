"""Key figure: between-field structure (ICC) vs out-of-farm predictability.

Headline result of the paper. For each soil property it plots the between-field
variance share (ICC) against two correlation metrics with the best RS feature:

  * naive in-sample screening |rho| over all 512 features (open circles), and
  * the honest out-of-farm Farm-LOFO rho (filled, coloured by verdict).

The honest predictability rises with ICC (Spearman ~ +0.83): properties with
more between-field structure are genuinely more learnable from field-resolution
remote sensing. Naive screening sits uniformly above it (the inflation gap), and
the gap is largest for sulfur -- the property with the LOWEST between-field
structure (ICC = 0.17) yet a high in-sample |rho| that collapses to ~0 out of
farm. A high in-sample correlation for a property that is mostly within-field
noise is the signature of leakage, not signal.

Run:
    python -m math_statistics.key_figure

Reads only committed inputs (master_dataset_old.csv, all_spearman_correlations.csv,
leakage_controlled_screening.csv); ICC is recomputed via variance_decomposition.
Deterministic -> byte-reproducible.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import FEATURES_CSV, OUTPUT_DIR
from . import variance_decomposition

SHORT = {"ph": "pH", "soc": "SOC", "no3": "NO₃", "p": "P₂O₅", "k": "K₂O", "s": "S"}
LAB2T = {"pH (KCl)": "ph", "SOC, %": "soc", "NO₃, mg/kg": "no3", "NO3, mg/kg": "no3",
         "P₂O₅, mg/kg": "p", "P2O5, mg/kg": "p", "K₂O, mg/kg": "k",
         "K2O, mg/kg": "k", "S, mg/kg": "s"}
VERDICT_COLOR = {"generalises": "#2c7fb8", "weak": "#f4a300", "does-not-generalise": "#d7301f"}
# label offsets (points) to avoid marker overlap, per property
OFFSET = {"ph": (8, -4), "k": (8, 4), "soc": (8, 6), "p": (8, -6), "no3": (-26, 6), "s": (10, 2)}


def build():
    df = pd.read_csv(FEATURES_CSV)
    vd = variance_decomposition.run(df)["decomposition"].copy()
    vd["target"] = vd["Property"].map(LAB2T)
    icc = vd.set_index("target")["ICC"]

    allc = pd.read_csv(OUTPUT_DIR / "all_spearman_correlations.csv")
    abscol = "abs_rho" if "abs_rho" in allc.columns else [c for c in allc.columns if "abs" in c.lower()][0]
    naive = allc.groupby("target")[abscol].max()

    lc = pd.read_csv(OUTPUT_DIR / "leakage_controlled_screening.csv").set_index("target")

    targets = ["ph", "k", "soc", "p", "no3", "s"]
    X = np.array([icc[t] for t in targets])
    Ynaive = np.array([naive[t] for t in targets])
    Yfl = np.array([lc.loc[t, "farm_lofo_rho"] for t in targets])
    verd = [lc.loc[t, "verdict"] for t in targets]

    r_fl = spearmanr(X, Yfl).statistic
    r_na = spearmanr(X, Ynaive).statistic

    fig, ax = plt.subplots(figsize=(8.6, 6.4))

    # inflation arrows: naive -> out-of-farm
    for x, yn, yf in zip(X, Ynaive, Yfl):
        ax.annotate("", xy=(x, yf), xytext=(x, yn),
                    arrowprops=dict(arrowstyle="-|>", color="0.6", lw=1.3, alpha=0.85))

    # naive screening (open circles)
    ax.scatter(X, Ynaive, s=80, facecolors="none", edgecolors="0.45", linewidths=1.4, zorder=4)

    # out-of-farm Farm-LOFO (filled, coloured by verdict) + trend
    for x, yf, v, t in zip(X, Yfl, verd, targets):
        ax.scatter(x, yf, s=180, color=VERDICT_COLOR.get(v, "0.3"),
                   edgecolors="black", linewidths=1.0, zorder=5)
        dx, dy = OFFSET[t]
        ax.annotate(SHORT[t], (x, yf), xytext=(dx, dy), textcoords="offset points",
                    fontsize=11.5, fontweight="bold")
    b, a = np.polyfit(X, Yfl, 1)
    xs = np.linspace(X.min() - 0.03, X.max() + 0.03, 50)
    ax.plot(xs, a + b * xs, color=VERDICT_COLOR["generalises"], lw=2, alpha=0.8, zorder=3)

    ax.set_xlabel("ICC — between-field variance share", fontsize=12)
    ax.set_ylabel("Spearman ρ with best RS feature", fontsize=12)
    ax.set_title("Between-field structure governs out-of-farm predictability", fontsize=13.5)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.03, float(Ynaive.max()) * 1.14)
    ax.set_xlim(0.10, 0.78)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="0.45",
               markersize=9, label="Naïve in-sample screening |ρ| (512 features)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VERDICT_COLOR["generalises"],
               markeredgecolor="black", markersize=11, label="Out-of-farm ρ — generalises"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VERDICT_COLOR["weak"],
               markeredgecolor="black", markersize=11, label="Out-of-farm ρ — weak"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VERDICT_COLOR["does-not-generalise"],
               markeredgecolor="black", markersize=11, label="Out-of-farm ρ — does not generalise"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True)

    txt = (f"Spearman ρ(ICC, out-of-farm) = {r_fl:+.2f}\n"
           f"Spearman ρ(ICC, in-sample)   = {r_na:+.2f}")
    ax.text(0.985, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    out = OUTPUT_DIR / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "00_key_icc_predictability.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "00_key_icc_predictability.tiff", dpi=600, format="tiff",
                bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print("targets   :", targets)
    print("ICC       :", [round(float(v), 3) for v in X])
    print("naive |rho|:", [round(float(v), 3) for v in Ynaive])
    print("farmlofo rho:", [round(float(v), 3) for v in Yfl])
    print("verdict   :", verd)
    print(f"rho(ICC, out-of-farm) = {r_fl:+.3f}   rho(ICC, in-sample) = {r_na:+.3f}")
    print("saved -> 00_key_icc_predictability.png / .tiff")


if __name__ == "__main__":
    build()
