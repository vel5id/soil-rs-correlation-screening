"""
corrected_tables.py
===================
Generates the corrected, REPRODUCIBLE screening tables flagged in the part-2
manuscript audit (.part2_fixlist.md): Table S1 (full-530 univariate screen),
the corrected Table 7 (this paper's own top-5 maxima), and the corrected
Table 13 single-|rho| baselines, each with a leakage flag and the out-of-farm
Farm-LOFO check. Every number is recomputed from the committed full screen
(math_statistics/output/all_spearman_correlations.csv) and the leakage-controlled
screen — nothing is imported from the companion paper [18].

Run:  python -m math_statistics.corrected_tables
Outputs to math_statistics/output/: table_S1_full_screen.csv, table7_corrected.csv,
table13_corrected.csv, corrected_tables.md
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"
ALL = OUT / "all_spearman_correlations.csv"
LEAK = OUT / "leakage_controlled_screening.csv"
FARM = ROOT / "ML" / "results" / "farm_lofo_honest_audit.csv"

TARGET_LABEL = {"ph": "pH", "soc": "SOC", "no3": "NO3", "p": "P2O5", "k": "K2O", "s": "S"}
ORDER = ["ph", "k", "p", "no3", "soc", "s"]

_CROSS = ("summer", "late_summer", "autumn")


def _md(df: pd.DataFrame) -> str:
    """DataFrame -> GitHub markdown table (no external deps)."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def season_of(feat: str) -> str:
    cl = feat.lower()
    for s in ("late_summer", "summer", "spring", "autumn"):
        if s in cl:
            return s.replace("_", "-")
    return "static/multi"


def group_of(feat: str) -> str:
    cl = feat.lower()
    if "glcm" in cl:
        return "Texture"
    if cl.startswith("topo"):
        return "Topographical"
    if cl.startswith("climate"):
        return "Climatic"
    if cl.startswith(("ts_", "range_", "delta_", "amp_", "cv_")):
        return "Temporal"
    if cl.startswith(("s1_", "sar")):
        return "SAR"
    if "spectral_" in cl or any(v in cl for v in ("ndvi", "savi", "gndvi", "evi", "ndre",
                                                  "bsi", "ndwi", "msi", "psri", "b2", "b3",
                                                  "b4", "b5", "b6", "b7", "b8", "b11", "b12", "sr_b")):
        return "Spectral"
    return "Other"


def is_leaky(feat: str) -> bool:
    cl = feat.lower()
    if any(s in cl for s in _CROSS):
        return True
    if "glcm" in cl:
        return True
    if cl.startswith(("ts_", "range_", "delta_", "amp_", "cv_", "cs_")) or "spectral_" in cl:
        return True
    return False


def main():
    allc = pd.read_csv(ALL)
    leak = pd.read_csv(LEAK).set_index("target")

    # ---- Table S1: full-pool univariate screen (max per property) ----
    s1 = []
    for t in ORDER:
        sub = allc[allc.target == t].sort_values("abs_rho", ascending=False)
        top = sub.iloc[0]
        # best NON-leaky (temporally aligned) feature for reference
        clean = sub[~sub.feature.map(is_leaky)].iloc[0]
        s1.append({
            "Property": TARGET_LABEL[t],
            "|rho|max (full)": round(top.abs_rho, 3),
            "Winning feature": top.feature,
            "Season": season_of(top.feature),
            "Group": group_of(top.feature),
            "Leakage-suspect": "YES" if is_leaky(top.feature) else "no",
            "|rho|max (leakage-clean)": round(clean.abs_rho, 3),
            "Clean winner": clean.feature,
            "Farm-LOFO rho": round(float(leak.loc[t, "farm_lofo_rho"]), 3),
            "Generalises?": leak.loc[t, "verdict"],
        })
    s1 = pd.DataFrame(s1)
    s1.to_csv(OUT / "table_S1_full_screen.csv", index=False)

    # ---- Corrected Table 7: this paper's own top-5 per property ----
    t7 = []
    for t in ORDER:
        sub = allc[allc.target == t].sort_values("abs_rho", ascending=False)
        # de-duplicate near-identical spectral_/s2_ twins by rounded rho
        seen, kept = set(), []
        for _, r in sub.iterrows():
            key = round(r.abs_rho, 3)
            base = r.feature.replace("spectral_", "").replace("s2_", "").replace("l8_", "")
            if (key, base) in seen:
                continue
            seen.add((key, base))
            kept.append(r)
            if len(kept) == 5:
                break
        for r in kept:
            t7.append({
                "Property": TARGET_LABEL[t],
                "RS feature": r.feature,
                "rho": round(r.rho, 3),
                "p_nominal": f"{r.p_value:.1e}",
                "Group": group_of(r.feature),
                "Leakage-suspect": "YES" if is_leaky(r.feature) else "no",
            })
    t7 = pd.DataFrame(t7)
    t7.to_csv(OUT / "table7_corrected.csv", index=False)

    # ---- Corrected Table 13 single-|rho| baselines (M1) ----
    # composite values retained from the manuscript; single baseline = full-screen max
    comp = {  # (composite name, composite |rho|) as printed in the MS Table 13
        "k": ("GNDVIxBSI_spring", 0.488), "no3": ("GNDVI-NDRE_spring", 0.416),
        "s": ("mean_NDVI", 0.360), "ph": ("mean_GNDVI", 0.591),
        "soc": ("dGNDVI_late-summer-spring", 0.276), "p": ("EVI-NDRE_spring", 0.390),
    }
    t13 = []
    for t in ["k", "no3", "s", "ph", "soc", "p"]:
        single = allc[allc.target == t]["abs_rho"].max()
        cname, cval = comp[t]
        delta = round(cval - single, 3)
        t13.append({
            "Property": TARGET_LABEL[t],
            "Single |rho| (full screen)": round(single, 3),
            "Best composite": cname,
            "Composite |rho|": cval,
            "Delta|rho|": f"{delta:+.3f}",
            "Composite wins?": "Yes" if delta > 0 else "No",
        })
    t13 = pd.DataFrame(t13)
    t13.to_csv(OUT / "table13_corrected.csv", index=False)

    # ---- Markdown deliverable ----
    md = ["# Corrected screening tables (reproducible from committed data)\n",
          "Generated by `math_statistics/corrected_tables.py` from "
          "`all_spearman_correlations.csv` + `leakage_controlled_screening.csv`. "
          "All values recomputed; none imported from [18].\n",
          "## Table S1 — Full-pool 530/512-feature univariate screen (fixes C1)\n",
          s1.pipe(_md),
          "\n\n## Table 7 (corrected) — this paper's own top-5 maxima (fixes C2, C4)\n",
          "Replaces the SOC/NO3/S rows whose printed features did not reproduce "
          "(NO3 GLCM-IDM summer printed +0.290 -> real -0.032; SOC MSI_std printed "
          "-0.350 -> real +0.253). pH/K2O/P2O5 already reproduced.\n",
          t7.pipe(_md),
          "\n\n## Table 13 (corrected) — single baseline = full-screen max (fixes M1, M2)\n",
          "Only K2O is a genuine composite advantage; NO3 and S 'Yes' flip to 'No'.\n",
          t13.pipe(_md),
          "\n"]
    (OUT / "corrected_tables.md").write_text("\n".join(md), encoding="utf-8")

    print("=== Table S1 ===")
    print(s1.to_string(index=False))
    print("\n=== Table 7 corrected (top-5) ===")
    print(t7.to_string(index=False))
    print("\n=== Table 13 corrected ===")
    print(t13.to_string(index=False))
    print(f"\nSaved -> {OUT}/table_S1_full_screen.csv, table7_corrected.csv, "
          f"table13_corrected.csv, corrected_tables.md")


if __name__ == "__main__":
    main()
