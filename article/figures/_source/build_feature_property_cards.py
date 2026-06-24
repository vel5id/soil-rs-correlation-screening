#!/usr/bin/env python3
"""One standalone 'property card' per soil property.

Each card = the spread of that property's 512 screened RS features (|rho|), with
leakage-suspect features hollow, the strongest leakage-clean feature circled, a
top-5 leakage-clean list, a facts line, and a one-line behaviour note.

All numbers come from existing artefacts (no fabrication):
  - spread / top features : math_statistics/output/feature_screening_full.csv
                            (built by scripts/build_feature_spread_figure.py)
  - |rho|max, Farm-LOFO, tiers, verdict : ML/results/key_table_taxonomy.csv
  - between-field ICC      : math_statistics/output/variance_decomposition.xlsx
Behaviour notes condense the manuscript discussion / RHO_DISCREPANCY_REPORT.md.

Run:  python scripts/build_feature_property_cards.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "math_statistics" / "output" / "feature_screening_full.csv"
TAX = ROOT / "ML" / "results" / "key_table_taxonomy.csv"
VAR = ROOT / "math_statistics" / "output" / "variance_decomposition.xlsx"
OUT = ROOT / "articles" / "article2_prediction" / "figures" / "cards"

TARGETS = ["ph", "k", "p", "no3", "soc", "s"]
LABEL = {"ph": "pH", "k": "K$_2$O", "p": "P$_2$O$_5$", "no3": "NO$_3$", "soc": "SOC", "s": "S"}
TAX_KEY = {"pH": "ph", "K2O": "k", "P2O5": "p", "NO3": "no3", "SOC": "soc", "S": "s"}
ICC_KEY = {"pH (KCl)": "ph", "SOC, %": "soc", "NO₃, mg/kg": "no3",
           "P₂O₅, mg/kg": "p", "K₂O, mg/kg": "k", "S, mg/kg": "s"}

VERDICT = {
    "ph": "regionally + locally mappable", "k": "mappable (clay/SWIR)",
    "p": "weakly mappable (texture)", "no3": "indirect (canopy-N proxy)",
    "soc": "regional gradient only", "s": "unpredictable",
}
BEHAVIOUR = {
    "ph": "Strongest, spatially structured signal — spring canopy indices (GNDVI/NDVI), MAP and "
          "slope; passes the spatial-permutation test (p<0.001). Mappable regionally and within-field.",
    "k": "Local soil control (latitudinal Δ<10%) carried by clay/SWIR features (BSI, SWIR bands). "
         "Mappable, though out-of-farm ρ drops to 0.33.",
    "p": "Weakly mappable: the raw maximum is autumn GLCM texture (temporal-leakage artefact); the "
         "leakage-clean driver is climate (GS_temp, ρ=0.476). Model error ≈ one supply class.",
    "no3": "Indirect — spring canopy proxies (SAVI/NDRE) reflect biomass/management, not direct "
           "nitrate sensing; out-of-farm signal is weak (ρ=0.22).",
    "soc": "Regional gradient via slope/topography, but ~40% of it is the latitudinal moisture "
           "gradient; weak out-of-farm (ρ=0.25).",
    "s": "Essentially unpredictable — no diagnostic absorption bands (400–2500 nm); ICC=0.17; the "
         "raw screening maxima are temporal-leakage artefacts.",
}

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
    """Cross-season vs the ~75%-spring sample (must match build_feature_spread_figure.py)."""
    f = str(f).lower()
    return any(k in f for k in (
        "summer", "late_summer", "autumn", "range",
        "ts_", "mean", "delta", "amp", "trend", "std"))


def load_facts() -> dict:
    tax = pd.read_csv(TAX)
    tax.columns = [c.strip() for c in tax.columns]
    var = pd.read_excel(VAR, sheet_name="decomposition")
    icc = {ICC_KEY[r.Property]: float(r.ICC) for _, r in var.iterrows() if r.Property in ICC_KEY}
    facts = {}
    for _, r in tax.iterrows():
        t = TAX_KEY.get(str(r["Property"]).strip())
        if t is None:
            continue
        m = re.match(r"\s*(\d+)\s*\((\d+)\)", str(r["general.(robust)"]))
        gen, rob = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        facts[t] = {
            "rho_max": float(r["|rho|max(LC)"]), "farm_lofo": float(r["Farm-LOFO rho"]),
            "gen": gen, "rob": rob, "zonal": int(r["zonal_only"]),
            "unstable": int(r["unstable"]), "weak": int(r["weak"]), "icc": icc.get(t, float("nan")),
        }
    return facts


def load_screen() -> pd.DataFrame:
    d = pd.read_csv(SCREEN)
    d["group"] = d["feature"].map(feature_group)
    d["short"] = d["feature"].map(short)
    d["leakage_suspect"] = d["feature"].map(leakage_suspect)
    return d


def card(t: str, d: pd.DataFrame, f: dict, rng: np.random.Generator) -> None:
    sub = d[d["target"] == t]
    clean = sub[~sub["leakage_suspect"]]
    susp = sub[sub["leakage_suspect"]]
    present = [g for g in GROUP_COLORS if (sub["group"] == g).any()]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.05, right=0.985)

    q1, med, q3 = sub["abs_rho"].quantile([0.25, 0.5, 0.75])
    ax.fill_betweenx([-0.32, 0.32], q1, q3, color="#ECECEC", zorder=1)
    ax.plot([med, med], [-0.32, 0.32], color="#AAAAAA", lw=1.2, zorder=1)

    ax.scatter(susp["abs_rho"].to_numpy(), rng.uniform(-0.30, 0.30, len(susp)), s=24,
               facecolors="none", edgecolors=susp["group"].map(GROUP_COLORS).tolist(),
               alpha=0.5, linewidths=0.6, zorder=2)
    ax.scatter(clean["abs_rho"].to_numpy(), rng.uniform(-0.30, 0.30, len(clean)), s=20,
               c=clean["group"].map(GROUP_COLORS).tolist(), alpha=0.75, linewidths=0, zorder=3)

    top5 = clean.nlargest(5, "abs_rho")
    champ = top5.iloc[0]
    champ_abs = float(champ["abs_rho"])

    # clean ceiling + champion marker
    ax.axvline(champ_abs, color="#444444", ls=(0, (5, 3)), lw=1.0, alpha=0.7, zorder=2)
    ax.scatter([champ_abs], [0], s=80, facecolors="none", edgecolors="#222222",
               linewidths=1.4, zorder=4)
    ax.annotate(champ["short"], (champ_abs, 0), xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold")
    ax.text(champ_abs, -0.585, "clean max", color="#444444", fontsize=7.5, ha="center", va="bottom")

    # leakage-suspect features sitting ABOVE the clean ceiling (top 3)
    leaked_above = susp[susp["abs_rho"] > champ_abs].nlargest(3, "abs_rho").reset_index(drop=True)
    for j, r in leaked_above.iterrows():
        ax.axvline(float(r["abs_rho"]), color="#B5402F", ls=(0, (4, 3)), lw=1.0, alpha=0.85, zorder=2)
        ax.text(float(r["abs_rho"]), 0.50, str(j + 1), color="#B5402F", fontsize=8.5,
                ha="center", va="bottom", fontweight="bold")

    lines = "\n".join(f"{r['short']:<22.22s} {r['rho']:+.2f}" for _, r in top5.iterrows())
    ax.text(0.985, 0.96, "Top leakage-clean features\n" + lines, transform=ax.transAxes,
            ha="right", va="top", family="monospace", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="#DDDDDD", alpha=0.92))

    if len(leaked_above):
        lk = "\n".join(f"{j + 1}  {r['short']:<20.20s} {r['rho']:+.2f}"
                       for j, r in leaked_above.iterrows())
        ax.text(0.985, 0.05, "Above clean max — leaked (excluded)\n" + lk, transform=ax.transAxes,
                ha="right", va="bottom", family="monospace", fontsize=8, color="#B5402F",
                bbox=dict(boxstyle="round", fc="white", ec="#E8C9C3", alpha=0.93))

    ax.set_xlim(0, 0.72)
    ax.set_ylim(-0.62, 0.62)
    ax.set_yticks([])
    ax.set_xlabel("|Spearman ρ|   (one point = one of 512 screened RS features)")
    ax.grid(True, axis="x", alpha=0.2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=7,
                      markerfacecolor=GROUP_COLORS[g], markeredgecolor="none", label=g)
               for g in present]
    handles += [
        Line2D([], [], marker="o", linestyle="none", markersize=7, markerfacecolor="none",
               markeredgecolor="#555555", label="leakage-suspect"),
        Line2D([], [], marker="o", linestyle="none", markersize=9, markerfacecolor="none",
               markeredgecolor="#222222", markeredgewidth=1.4, label="strongest clean"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower left", ncol=2)

    facts = (f"|ρ|max (clean) = {f['rho_max']:.2f}      Farm-LOFO ρ = {f['farm_lofo']:.2f}      "
             f"between-field ICC = {f['icc']:.2f}      "
             f"tiers: {f['gen']} generalise ({f['rob']} robust) / {f['zonal']} zonal / "
             f"{f['unstable']} unstable / {f['weak']} weak")
    fig.text(0.05, 0.93, f"{LABEL[t]}  —  {VERDICT[t]}", fontsize=16, fontweight="bold", va="top")
    fig.text(0.05, 0.85, facts, fontsize=9.5, color="#555555", va="top")
    fig.text(0.05, 0.11, "Behaviour:  " + BEHAVIOUR[t], fontsize=9.5, va="top", wrap=True)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"card_{t}.png"
    fig.savefig(png, dpi=300)
    fig.savefig(OUT / f"card_{t}.pdf")
    plt.close(fig)
    with Image.open(png) as img:
        img.save(png, format="PNG", dpi=(300, 300), optimize=True)
    print(f"Wrote {png}")


def main() -> None:
    d = load_screen()
    facts = load_facts()
    rng = np.random.default_rng(0)
    for t in TARGETS:
        card(t, d, facts[t], rng)


if __name__ == "__main__":
    main()
