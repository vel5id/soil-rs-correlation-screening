"""Key figure: in-sample screening |rho| inflates over honest out-of-farm predictability.

For each soil property the figure contrasts two correlation metrics with the best
RS feature, ordered by honest predictability:

  * naive in-sample screening |rho| over all 512 features (open circles), and
  * the honest out-of-farm Farm-LOFO rho (filled, coloured by verdict),

with an arrow marking the in-sample -> out-of-farm collapse. Naive screening sits
uniformly above the honest value, and the collapse is largest for the labile /
weakly-structured properties: sulfur's in-sample |rho| ~ 0.42 falls to rho ~ 0.04
out of farm, and NO3 collapses similarly. A high in-sample correlation that does
not survive out-of-farm validation is the signature of leakage, not signal -- the
reason this study screens features under out-of-farm control.

NOTE (correction): an earlier version of this figure placed a between-field ICC on
the x-axis and reported Spearman rho(ICC, out-of-farm) = +0.83 (p=0.04). That ICC
was computed by grouping on the raw ``field_name`` label, which is reused across
farms (81 labels vs 103 true fields), and the +0.83 was an artefact of that
pooling. Under the true field id (farm + field_name) the ICC range is narrow
(0.70-0.93) and the ICC-vs-predictability correlation is +0.26 (n.s., n=6); there
is no statistically supported ICC->predictability law. The robust, reproducible
message is the in-sample -> out-of-farm inflation shown here. The per-property ICC
is reported alongside each marker as context, not as a predictor.

Run:
    python -m math_statistics.key_figure

Reads only committed inputs (master_dataset_old.csv, all_spearman_correlations.csv,
leakage_controlled_screening.csv); ICC is recomputed via variance_decomposition
(true field id). Deterministic -> byte-reproducible.
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
    icc = vd.set_index("target")["ICC"]  # true field id (farm + field_name)

    allc = pd.read_csv(OUTPUT_DIR / "all_spearman_correlations.csv")
    abscol = "abs_rho" if "abs_rho" in allc.columns else [c for c in allc.columns if "abs" in c.lower()][0]
    naive = allc.groupby("target")[abscol].max()

    lc = pd.read_csv(OUTPUT_DIR / "leakage_controlled_screening.csv").set_index("target")

    # order properties by honest out-of-farm predictability (descending)
    targets = sorted(["ph", "k", "soc", "p", "no3", "s"],
                     key=lambda t: float(lc.loc[t, "farm_lofo_rho"]), reverse=True)
    Ynaive = np.array([naive[t] for t in targets])
    Yfl = np.array([lc.loc[t, "farm_lofo_rho"] for t in targets])
    verd = [lc.loc[t, "verdict"] for t in targets]
    Icc = np.array([icc[t] for t in targets])

    # honest context stats (no significant law at n=6)
    r_fl = spearmanr(Icc, Yfl).statistic
    p_fl = spearmanr(Icc, Yfl).pvalue

    xpos = np.arange(len(targets))
    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    # inflation arrows: naive -> out-of-farm
    for x, yn, yf in zip(xpos, Ynaive, Yfl):
        ax.annotate("", xy=(x, yf), xytext=(x, yn),
                    arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.6, alpha=0.9))
        ax.annotate(f"−{yn - yf:.2f}", xy=(x, (yn + yf) / 2), xytext=(7, 0),
                    textcoords="offset points", fontsize=8.5, color="0.4", va="center")

    # naive screening (open circles) and out-of-farm (filled, coloured by verdict)
    ax.scatter(xpos, Ynaive, s=90, facecolors="none", edgecolors="0.4", linewidths=1.6,
               zorder=4, label="Naïve in-sample screening |ρ| (512 features)")
    for x, yf, v in zip(xpos, Yfl, verd):
        ax.scatter(x, yf, s=200, color=VERDICT_COLOR.get(v, "0.3"),
                   edgecolors="black", linewidths=1.0, zorder=5)

    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{SHORT[t]}\nICC={icc[t]:.2f}" for t in targets], fontsize=10.5)
    ax.set_ylabel("Spearman ρ with best RS feature", fontsize=12)
    ax.set_title("In-sample screening |ρ| inflates over honest out-of-farm predictability",
                 fontsize=13)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(-0.03, float(Ynaive.max()) * 1.15)
    ax.axhline(0.0, color="0.7", lw=0.8)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="0.4",
               markersize=9, label="Naïve in-sample screening |ρ| (512 features)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VERDICT_COLOR["generalises"],
               markeredgecolor="black", markersize=11, label="Out-of-farm Farm-LOFO ρ — generalises"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VERDICT_COLOR["weak"],
               markeredgecolor="black", markersize=11, label="Out-of-farm ρ — weak"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VERDICT_COLOR["does-not-generalise"],
               markeredgecolor="black", markersize=11, label="Out-of-farm ρ — does not generalise"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, frameon=True)

    txt = (f"ICC (true field id) range {Icc.min():.2f}–{Icc.max():.2f}\n"
           f"Spearman ρ(ICC, out-of-farm) = {r_fl:+.2f} (p={p_fl:.2f}, n=6, n.s.)")
    ax.text(0.015, 0.04, txt, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    out = OUTPUT_DIR / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "00_key_icc_predictability.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "00_key_icc_predictability.tiff", dpi=600, format="tiff",
                bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print("targets (by out-of-farm ρ):", targets)
    print("ICC (true field id):", [round(float(v), 3) for v in Icc])
    print("naive |rho|:", [round(float(v), 3) for v in Ynaive])
    print("farmlofo rho:", [round(float(v), 3) for v in Yfl])
    print("verdict   :", verd)
    print(f"rho(ICC, out-of-farm) = {r_fl:+.3f} (p={p_fl:.3f}, n=6) -> no significant law")
    print("saved -> 00_key_icc_predictability.png / .tiff")


if __name__ == "__main__":
    build()
