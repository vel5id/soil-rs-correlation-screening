"""
Variance decomposition: between-field vs within-field (Section 3.6.1 of article v2).

The grouping variable is the TRUE field identity = farm + field_name. The raw
``field_name`` column is NOT globally unique (labels "1", "2", "3"... are reused
across farms): there are 81 distinct field_name labels but 103 distinct
(farm, field_name) physical fields, and a single label such as "1" spans up to
7 farms located ~500 km apart in different soil zones. Grouping on field_name
alone pools physically distinct fields into one group, inflating the within-group
variance and collapsing the between-field ICC (most severely for sulfur:
ICC 0.17 under the label-only grouping vs 0.83 under the true field id). Always
group on the (farm, field_name) key.

For the full nested farm / field-in-farm / within decomposition (Table 16), see
``nested_variance.py``.
"""

import pandas as pd
import numpy as np

from .config import SOIL_TARGETS, SOIL_LABELS, OUTPUT_DIR

# True field identity. field_name alone is reused across farms (81 labels vs
# 103 real fields), so it must be qualified by farm.
FIELD_ID = "field_uid"


def _add_field_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a globally-unique field id (farm + field_name)."""
    out = df.copy()
    out[FIELD_ID] = out["farm"].astype(str) + "__" + out["field_name"].astype(str)
    return out


def decompose_variance(df: pd.DataFrame, group_col: str = FIELD_ID) -> pd.DataFrame:
    """One-way variance decomposition for each soil property.

    Total variance = Between-group variance + Within-group variance
    SS_total = SS_between + SS_within

    Groups are the true physical fields (farm + field_name), not the raw,
    non-unique ``field_name`` label (see module docstring).
    """
    if group_col == FIELD_ID and FIELD_ID not in df.columns:
        df = _add_field_uid(df)
    rows = []
    for col in SOIL_TARGETS:
        valid = df[[group_col, col]].dropna()
        if valid.empty:
            continue

        grand_mean = valid[col].mean()
        ss_total = ((valid[col] - grand_mean) ** 2).sum()

        ss_between = 0
        ss_within = 0
        groups = valid.groupby(group_col)
        for _, group in groups:
            n_g = len(group)
            group_mean = group[col].mean()
            ss_between += n_g * (group_mean - grand_mean) ** 2
            ss_within += ((group[col] - group_mean) ** 2).sum()

        n_groups = groups.ngroups
        n_total = len(valid)

        # Percentages
        pct_between = (ss_between / ss_total * 100) if ss_total > 0 else 0
        pct_within = (ss_within / ss_total * 100) if ss_total > 0 else 0

        # Intraclass Correlation Coefficient (ICC)
        ms_between = ss_between / max(n_groups - 1, 1)
        ms_within = ss_within / max(n_total - n_groups, 1)
        n_mean = n_total / n_groups  # average group size
        icc = (ms_between - ms_within) / (ms_between + (n_mean - 1) * ms_within) if (ms_between + (n_mean - 1) * ms_within) > 0 else 0

        rows.append({
            "Property": SOIL_LABELS[col],
            "n_samples": n_total,
            "n_fields": n_groups,
            "SS_total": round(ss_total, 2),
            "SS_between": round(ss_between, 2),
            "SS_within": round(ss_within, 2),
            "Pct_between": round(pct_between, 1),
            "Pct_within": round(pct_within, 1),
            "ICC": round(icc, 4),
        })
    return pd.DataFrame(rows)


def verify_article_claims(decomp: pd.DataFrame) -> pd.DataFrame:
    """Sanity-check the decomposition against the corrected, reproducible values.

    Reference values are the between/within shares under the true field id
    (farm + field_name), recomputed from master_dataset_old.csv. They replace the
    earlier label-only-grouping targets (pH 73.2 % between-field etc.), which were
    an artefact of pooling distinct fields under shared labels.
    """
    checks = []

    # pH: ~93.3 % between-field, ~6.7 % within-field (true field id)
    ph_row = decomp[decomp["Property"] == SOIL_LABELS["ph"]]
    if not ph_row.empty:
        pct = ph_row.iloc[0]["Pct_between"]
        checks.append({
            "Claim": "pH: ~93% between-field variance",
            "Reference_value": 93.3,
            "Computed_value": pct,
            "Difference": round(abs(pct - 93.3), 1),
            "MATCH_within_5pct": abs(pct - 93.3) < 5,
        })

    # SOC within-field ~19.9 %
    soc_row = decomp[decomp["Property"] == SOIL_LABELS["soc"]]
    if not soc_row.empty:
        pct_soc = soc_row.iloc[0]["Pct_within"]
        checks.append({
            "Claim": "SOC within-field variance ~20%",
            "Reference_value": 19.9,
            "Computed_value": pct_soc,
            "Difference": round(abs(pct_soc - 19.9), 1),
            "MATCH_within_5pct": abs(pct_soc - 19.9) < 5,
        })

    return pd.DataFrame(checks)


def run(df: pd.DataFrame) -> dict:
    """Run variance decomposition analysis."""
    decomp = decompose_variance(df)
    claims = verify_article_claims(decomp)

    results = {
        "decomposition": decomp,
        "claims_verification": claims,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_DIR / "variance_decomposition.xlsx") as writer:
        decomp.to_excel(writer, sheet_name="decomposition", index=False)
        claims.to_excel(writer, sheet_name="claims_verification", index=False)

    return results
