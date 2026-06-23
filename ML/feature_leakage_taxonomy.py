"""Empirical feature taxonomy for the per-(feature, target) CV diagnostics.

This module replaces the name-based ``leakage_suspect`` flag in
``ML/results/feature_quality_cv.csv`` with an *empirical* classification derived
from the cross-validation correlation statistics actually measured for each
(feature, target) pair.

Rationale
---------
The original ``leakage_suspect`` column was assigned heuristically from feature
*names* (e.g. acquisition season). That conflates two unrelated things:

  * whether a feature is a real, year-stable, spatially-controlled local signal,
  * whether a feature happens to be acquired in a season that, for a *labile*
    soil property, introduces a value-mismatch concern.

Here we separate them. The single ``feature_class`` label is decided purely from
the empirical correlation behaviour (within-block / full / between-block
strengths, year consistency, sign stability across years). Season/feature-type
and lability information are kept as *descriptive* tags, not as leakage flags.

Classification (one class per row, strict precedence)
-----------------------------------------------------
Let ``af = |rho_full|``, ``aw = |block_within|``, ``ab = |block_between|``,
``consistent = (year_consistent == "yes")`` and ``sign_flip`` be True when
``rho_2022`` and ``rho_2023`` are both finite and have opposite signs.

  1. generalizable : aw >= TAU_WITHIN and consistent
        Real, spatially-controlled, year-stable local signal.
  2. unstable      : sign_flip and af >= 0.20
        The correlation reverses sign between years -- the dominant real defect
        in this dataset.
  3. zonal_only    : af >= TAU_FULL and aw < TAU_WITHIN and ab >= TAU_BETWEEN
        Works only as a regional (between-block) gradient; usable for mapping
        but not for local within-field prediction.
  4. weak          : everything else.

Descriptive tags (NOT leakage flags)
------------------------------------
  * ``cross_season``           : season in {summer, late-summer, autumn}.
  * ``property_lability``      : "stable" for {pH, SOC, K2O, P2O5},
                                 "labile" for {NO3, S}.
  * ``cross_season_concern``   : cross_season AND labile. For a labile property a
                                 cross-season acquisition is a genuine value
                                 mismatch (the soil value may have changed); for
                                 a stable property it is not (same soil, just a
                                 different acquisition time).

The module is deterministic: no randomness, no I/O side effects beyond writing
the single output CSV. The original input CSV is never modified or overwritten.

Run
---
    python ML/feature_leakage_taxonomy.py

Produces ``ML/results/feature_quality_tagged.csv`` with all original columns
plus: feature_class, cross_season, property_lability, cross_season_concern.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Thresholds. Calibrated against the observed distribution of |block_within|,
# |rho_full| and |block_between| in feature_quality_cv.csv; the spec start
# values held up against the data, so they are kept.
TAU_WITHIN: float = 0.15  # within-block |rho| needed to call a signal "local"
TAU_FULL: float = 0.30  # pooled |rho| needed to call a signal non-trivial
TAU_BETWEEN: float = 0.30  # between-block |rho| needed to call a signal "zonal"

# Minimum pooled |rho| for a year sign-flip to count as a real defect (rather
# than two near-zero correlations whose signs happen to differ).
UNSTABLE_MIN_FULL: float = 0.20

# "Robust" gate on top of generalizable (validation finding): a generalizable row
# is only ROBUST if its pooled correlation is also non-trivial AND the within-block
# sign agrees with the pooled sign (no Simpson reversal). Manuscript claims should
# rest on the robust subset, not on all generalizable rows.
ROBUST_MIN_FULL: float = 0.30

# Descriptive groupings.
CROSS_SEASON_SEASONS: frozenset[str] = frozenset({"summer", "late-summer", "autumn"})
STABLE_PROPERTIES: frozenset[str] = frozenset({"pH", "SOC", "K2O", "P2O5"})
LABILE_PROPERTIES: frozenset[str] = frozenset({"NO3", "S"})

# Paths (resolved relative to this file so it runs from any cwd).
_THIS_DIR = Path(__file__).resolve().parent
INPUT_CSV = _THIS_DIR / "results" / "feature_quality_cv.csv"
OUTPUT_CSV = _THIS_DIR / "results" / "feature_quality_tagged.csv"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _is_finite(x: float) -> bool:
    """True iff x is a real, finite number (not NaN/inf/None)."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _sign_flip(rho_2022: float, rho_2023: float) -> bool:
    """True iff both yearly correlations are finite and have opposite signs.

    Exact zeros are treated as non-flipping (no defined sign reversal).
    """
    if not (_is_finite(rho_2022) and _is_finite(rho_2023)):
        return False
    a, b = float(rho_2022), float(rho_2023)
    if a == 0.0 or b == 0.0:
        return False
    return (a > 0) != (b > 0)


def classify_row(row: pd.Series) -> str:
    """Return the empirical ``feature_class`` for a single (feature, target) row.

    Precedence: generalizable > unstable > zonal_only > weak.
    """
    af = abs(float(row["rho_full"])) if _is_finite(row["rho_full"]) else 0.0
    aw = abs(float(row["block_within"])) if _is_finite(row["block_within"]) else 0.0
    ab = abs(float(row["block_between"])) if _is_finite(row["block_between"]) else 0.0
    consistent = str(row["year_consistent"]).strip().lower() == "yes"
    flip = _sign_flip(row["rho_2022"], row["rho_2023"])

    # 1. Real, spatially-controlled, year-stable local signal.
    if aw >= TAU_WITHIN and consistent:
        return "generalizable"
    # 2. Correlation reverses across years -- dominant real defect.
    if flip and af >= UNSTABLE_MIN_FULL:
        return "unstable"
    # 3. Regional gradient only: strong pooled + between, weak within.
    if af >= TAU_FULL and aw < TAU_WITHIN and ab >= TAU_BETWEEN:
        return "zonal_only"
    # 4. Everything else.
    return "weak"


def _is_robust(row: pd.Series) -> bool:
    """True iff a generalizable row also passes the pooled-strength + sign gate.

    Requires feature_class == "generalizable", |rho_full| >= ROBUST_MIN_FULL, and
    the within-block correlation sign agreeing with the pooled sign.
    """
    if str(row.get("feature_class")) != "generalizable":
        return False
    if not (_is_finite(row["rho_full"]) and _is_finite(row["block_within"])):
        return False
    f, w = float(row["rho_full"]), float(row["block_within"])
    if abs(f) < ROBUST_MIN_FULL or f == 0.0 or w == 0.0:
        return False
    return (f > 0) == (w > 0)


def _property_lability(target: str) -> str:
    """Map a soil property to its lability class."""
    t = str(target).strip()
    if t in STABLE_PROPERTIES:
        return "stable"
    if t in LABILE_PROPERTIES:
        return "labile"
    return "unknown"


def tag_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the new taxonomy columns appended.

    Original columns are preserved in their original order; new columns are
    appended at the end.
    """
    out = df.copy()
    out["feature_class"] = out.apply(classify_row, axis=1)
    out["robust"] = out.apply(_is_robust, axis=1)
    out["cross_season"] = out["season"].astype(str).str.strip().isin(CROSS_SEASON_SEASONS)
    out["property_lability"] = out["target"].map(_property_lability)
    out["cross_season_concern"] = out["cross_season"] & (out["property_lability"] == "labile")
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(input_csv: Path = INPUT_CSV, output_csv: Path = OUTPUT_CSV) -> pd.DataFrame:
    """Read, classify, write, and return the tagged DataFrame."""
    df = pd.read_csv(input_csv)
    tagged = tag_dataframe(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tagged.to_csv(output_csv, index=False)
    return tagged


if __name__ == "__main__":
    tagged = main()

    # Deterministic summary to stdout (no effect on the output CSV).
    print(f"Input : {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Rows  : {len(tagged)}")
    print(
        f"Thresholds: TAU_WITHIN={TAU_WITHIN}, TAU_FULL={TAU_FULL}, "
        f"TAU_BETWEEN={TAU_BETWEEN}, UNSTABLE_MIN_FULL={UNSTABLE_MIN_FULL}"
    )

    print("\n== feature_class counts (overall) ==")
    print(tagged["feature_class"].value_counts().to_string())

    n_gen = int((tagged["feature_class"] == "generalizable").sum())
    n_rob = int(tagged["robust"].sum())
    print(f"\ngeneralizable={n_gen}; of these ROBUST (|rho_full|>={ROBUST_MIN_FULL} & "
          f"within-sign agrees) = {n_rob}")
    print("robust per target:",
          tagged[tagged["robust"]]["target"].value_counts().to_dict())

    print("\n== feature_class counts per target ==")
    ct = (
        tagged.groupby(["target", "feature_class"]).size().unstack(fill_value=0)
    )
    print(ct.to_string())

    print("\n== anchor rows ==")
    anchors = [
        ("pH", "l8_GNDVI_spring"),
        ("pH", "climate_MAP"),
        ("SOC", "topo_slope"),
        ("pH", "s2_BSI_spring"),
    ]
    for tgt, feat in anchors:
        sel = tagged[(tagged["target"] == tgt) & (tagged["feature"] == feat)]
        cls = sel["feature_class"].iloc[0] if len(sel) else "<missing>"
        print(f"  {tgt:4s} ~ {feat:20s} -> {cls}")
