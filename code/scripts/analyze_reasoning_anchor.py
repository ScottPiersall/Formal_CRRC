#!/usr/bin/env python
"""Preregistered analysis of the reasoning-capable external anchor.

Runs only after every intended row has been attempted. Nothing here chooses a
threshold, a metric or an exclusion after seeing an outcome: the crossing rule,
the reachability rule, the tie policy, the curve-exclusion rule and the
interpretation categories all come from the frozen preregistration.

Curve exclusion is mechanical. A curve is dropped when a strictness level has no
valid primary margin -- a truncated reasoning trace, or a scoring failure -- and
never because of what a margin turned out to be.

Writes, all additive, under ``artifacts/reasoning_anchor/``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import day2, day3, metrics, provenance  # noqa: E402
from formalcrrc import reasoning_anchor as ra  # noqa: E402
from formalcrrc import reasoning_anchor_bootstrap as rab  # noqa: E402
from formalcrrc import reasoning_anchor_config as rc  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    DECISION_THRESHOLD,
    FAMILIES,
    MVR_TOLERANCE,
    MVR_TOLERANCE_STRICT,
    N_STRICTNESS,
    NO_CROSSING_INDEX,
)
from formalcrrc.day2_bootstrap import crossing_index  # noqa: E402


# --------------------------------------------------------------------------
# Section 14 -- completeness
# --------------------------------------------------------------------------


def completeness_report(rows: pd.DataFrame) -> dict[str, Any]:
    attempted = int(len(rows))
    truncated = int(rows["reasoning_truncated"].sum())
    scored = int(rows["margin"].notna().sum())
    other_failures = int(
        ((~rows["reasoning_truncated"]) & rows["margin"].isna()).sum()
    )
    completeness = ra.curve_completeness(rows)
    rate = truncated / attempted if attempted else 0.0
    return {
        "rows_attempted": attempted,
        "rows_with_successful_reasoning": int(rows["think_end_reached"].sum()),
        "rows_truncated": truncated,
        "rows_other_failure": other_failures,
        "rows_scored": scored,
        "truncation_rate": rate,
        "truncation_warning": rate > rc.TRUNCATION_WARNING_FRACTION,
        "truncation_warning_flag": (
            rc.TRUNCATION_WARNING_FLAG
            if rate > rc.TRUNCATION_WARNING_FRACTION
            else None
        ),
        "curve_exclusion_rule": rc.CURVE_EXCLUSION_RULE,
        **completeness.to_dict(),
    }


# --------------------------------------------------------------------------
# Sections 15-16 -- per-curve metrics
# --------------------------------------------------------------------------


def artifact_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per complete artifact curve."""
    ordered = rows.sort_values(["artifact_id", "strictness_index"])
    records: list[dict[str, Any]] = []
    for artifact_id, group in ordered.groupby("artifact_id", sort=True):
        levels = group["strictness_index"].to_numpy()
        if not np.array_equal(levels, np.arange(N_STRICTNESS)):
            raise ValueError(f"{artifact_id}: expected strictness 0..8 exactly once")
        margins = group["margin"].to_numpy(dtype=float)
        truth = group["formal_truth"].to_numpy(dtype=int)
        j_star = int(group["true_first_fail_index"].iloc[0])
        probabilities = day2.sigmoid(margins)

        j_hat = int(crossing_index(margins[None, :])[0])
        reach = day3.reachable_crossings(margins)
        rc_count = len(reach)
        decisions = (margins >= 0.0).astype(int)

        records.append(
            {
                "model_id": rc.MODEL_ID,
                "artifact_id": artifact_id,
                "family": group["family"].iloc[0],
                "latent_level": int(group["latent_level"].iloc[0]),
                "true_first_fail_index": j_star,
                # primary localisation
                "j_hat_raw": j_hat,
                "tce_raw": abs(j_hat - j_star),
                # primary ordering
                "translation_reachable": j_star in reach,
                "nontrivial_boundary": 1 <= j_star <= N_STRICTNESS - 1,
                "reachable_crossings": ",".join(str(j) for j in reach),
                "reachable_count": rc_count,
                "rc_is_ten": rc_count == 10,
                "prefix_record_rate": day3.prefix_record_rate(margins),
                # secondary, row-level reduced to the curve
                "artifact_accuracy": float(np.mean(decisions == truth)),
                "artifact_brier": float(np.mean((probabilities - truth) ** 2)),
                # shape
                "mvr_practical": metrics.monotonicity_violation_rate(
                    probabilities, MVR_TOLERANCE
                ),
                "mvr_zero_tolerance": metrics.monotonicity_violation_rate(
                    probabilities, MVR_TOLERANCE_STRICT
                ),
                "mvm": metrics.monotonicity_violation_magnitude(probabilities),
                "margin_mvr": day2.margin_monotonicity_violation_rate(margins),
                "span": day2.response_span(margins),
                "margin_spearman": day2.margin_spearman(margins),
                # reasoning descriptors
                "mean_reasoning_tokens": float(
                    group["reasoning_token_count"].mean()
                ),
                "total_reasoning_tokens": int(group["reasoning_token_count"].sum()),
                # ties, descriptive
                **day3.tie_diagnostics(margins, j_star),
            }
        )
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Sections 17-18 -- references and contextual comparison
# --------------------------------------------------------------------------


def comparison_table(mean_tce: float, unreachable_pct: float, accuracy: float) -> pd.DataFrame:
    """Structural references, the original panel, and the anchor -- kept apart."""
    rows: list[dict[str, Any]] = [
        {
            "row_kind": "structural reference",
            "name": "Fixed center / no input",
            "inference_mode": "no model input",
            "tce": rc.BASELINE_FIXED_CENTER_TCE,
            "accuracy_pct": 100.0 * rc.BASELINE_FIXED_CENTER_ACC,
            "unreachable_pct": None,
        },
        {
            "row_kind": "structural reference",
            "name": "Uniform random crossing",
            "inference_mode": "no model input",
            "tce": rc.BASELINE_RANDOM_CROSSING_TCE,
            "accuracy_pct": 100.0 * rc.BASELINE_RANDOM_CROSSING_ACC,
            "unreachable_pct": None,
        },
        {
            "row_kind": "structural reference",
            "name": "Always met / majority class",
            "inference_mode": "no model input",
            "tce": rc.BASELINE_ALWAYS_MET_TCE,
            "accuracy_pct": 100.0 * rc.BASELINE_MAJORITY_ACC,
            "unreachable_pct": None,
        },
    ]
    for name, tce in rc.DAY3_TCE.items():
        rows.append(
            {
                "row_kind": "original Day-3 judge",
                "name": name,
                "inference_mode": "immediate next-token logit",
                "tce": tce,
                "accuracy_pct": None,
                "unreachable_pct": rc.DAY3_UNREACHABLE_PCT[name],
            }
        )
    rows.append(
        {
            "row_kind": "reasoning anchor",
            "name": rc.MODEL_SHORT_NAME,
            "inference_mode": "reason-then-score (Stage R + Stage D)",
            "tce": mean_tce,
            "accuracy_pct": 100.0 * accuracy,
            "unreachable_pct": unreachable_pct,
        }
    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Section 20 -- the prespecified interpretation
# --------------------------------------------------------------------------


def apply_interpretation(unreachable_pct: float) -> dict[str, Any]:
    """Exactly the categories fixed before inference. No new threshold."""
    low, high = rc.DAY3_UNREACHABLE_RANGE
    if unreachable_pct <= 0.0:
        outcome, statement = (
            "C",
            "The reasoning-capable anchor does not reproduce Day-3 boundary "
            "unreachability under this reason-then-score protocol.",
        )
    elif low <= unreachable_pct <= high:
        outcome, statement = (
            "A",
            "The reasoning-capable anchor exhibits boundary unreachability of "
            "the same order as the original FormalCRRC panel.",
        )
    elif unreachable_pct < low:
        outcome, statement = (
            "B",
            "The reasoning-capable anchor reduces ordering failure relative to "
            "every original Day-3 judge, but does not eliminate the FormalCRRC "
            "phenomenon.",
        )
    else:
        outcome, statement = (
            "A-above",
            "The reasoning-capable anchor exhibits boundary unreachability "
            "above the original FormalCRRC panel range.",
        )
    return {
        "outcome_category": outcome,
        "statement": statement,
        "unreachable_pct": unreachable_pct,
        "original_panel_range_pct": [low, high],
        "note": (
            "TCE is reported separately and is not collapsed with TRR into a "
            "single success or failure label."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = pathlib.Path(args.root)

    prereg = json.loads((root / rc.PREREG_JSON_PATH).read_text(encoding="utf-8"))
    if not provenance.verify_checksum_file(
        root / rc.PREREG_JSON_PATH, root / rc.PREREG_SHA_PATH
    ):
        raise SystemExit("REFUSING TO ANALYSE: preregistration checksum mismatch.")

    rows = pd.read_parquet(root / rc.RAW_ROWS_PATH)
    print(f"loaded {len(rows)} rows")

    completeness = completeness_report(rows)
    print(
        f"complete curves {completeness['n_complete_curves']}, "
        f"excluded {completeness['n_excluded_curves']}, "
        f"truncation {completeness['truncation_rate']:.4%}"
    )

    usable = ra.complete_rows(rows)
    if usable.empty:
        raise SystemExit("no complete curves; nothing to analyse")
    curves = artifact_metrics(usable)
    curves.to_parquet(root / rc.METRICS_PATH, index=False)

    nontrivial = curves[curves["nontrivial_boundary"]]
    mean_tce = float(curves["tce_raw"].mean())
    trr = float(nontrivial["translation_reachable"].mean())
    unreachable_pct = 100.0 * (1.0 - trr)
    accuracy = float(curves["artifact_accuracy"].mean())
    brier = float(curves["artifact_brier"].mean())

    boot = rab.bootstrap_anchor(curves)
    rab.to_frame(boot).to_parquet(root / rc.BOOTSTRAP_PATH, index=False)

    family_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        block = curves[curves["family"] == family]
        if block.empty:
            continue
        block_nt = block[block["nontrivial_boundary"]]
        family_rows.append(
            {
                "family": family,
                "n_curves": int(len(block)),
                "n_nontrivial": int(len(block_nt)),
                "tce": float(block["tce_raw"].mean()),
                "trr": float(block_nt["translation_reachable"].mean()),
                "unreachable_pct": 100.0
                * (1.0 - float(block_nt["translation_reachable"].mean())),
                "accuracy": float(block["artifact_accuracy"].mean()),
                "brier": float(block["artifact_brier"].mean()),
                "mean_reasoning_tokens": float(block["mean_reasoning_tokens"].mean()),
                "mean_reachable_count": float(block["reachable_count"].mean()),
            }
        )
    family_frame = pd.DataFrame(family_rows)
    family_frame.to_parquet(root / rc.FAMILY_SUMMARY_PATH, index=False)

    tables = root / rc.TABLES_DIR
    tables.mkdir(parents=True, exist_ok=True)

    reasoning_tokens = usable["reasoning_token_count"].to_numpy(dtype=float)
    table_a = pd.DataFrame(
        [
            {
                "model": rc.MODEL_SHORT_NAME,
                "complete_curves": int(len(curves)),
                "mean_reasoning_tokens": float(reasoning_tokens.mean()),
                "median_reasoning_tokens": float(np.median(reasoning_tokens)),
                "truncation_rate_pct": 100.0 * completeness["truncation_rate"],
                "tce": mean_tce,
                "tce_ci_low": boot["overall"]["mean_tce"]["ci_low"],
                "tce_ci_high": boot["overall"]["mean_tce"]["ci_high"],
                "accuracy_pct": 100.0 * accuracy,
                "brier": brier,
                "trr_nontrivial": trr,
                "trr_ci_low": boot["overall"]["trr_nontrivial"]["ci_low"],
                "trr_ci_high": boot["overall"]["trr_nontrivial"]["ci_high"],
                "unreachable_pct": unreachable_pct,
            }
        ]
    )
    table_a.to_csv(tables / "table_a_primary.csv", index=False)
    comparison_table(mean_tce, unreachable_pct, accuracy).to_csv(
        tables / "table_b_comparison.csv", index=False
    )
    family_frame.to_csv(tables / "table_c_family.csv", index=False)

    rc_ten = float(curves["rc_is_ten"].mean())
    summary: dict[str, Any] = {
        "study": rc.STUDY_NAME,
        "anchor_description": rc.ANCHOR_DESCRIPTION,
        "model": prereg["model"],
        "protocol": rc.PROTOCOL,
        "preregistration_sha256": provenance.sha256_file(root / rc.PREREG_JSON_PATH),
        "preregistration_frozen_at": prereg["frozen_at"],
        "amendments": prereg["amendments"],
        "completeness": completeness,
        "primary": {
            "mean_tce": mean_tce,
            "mean_tce_ci": [
                boot["overall"]["mean_tce"]["ci_low"],
                boot["overall"]["mean_tce"]["ci_high"],
            ],
            "trr_nontrivial": trr,
            "trr_ci": [
                boot["overall"]["trr_nontrivial"]["ci_low"],
                boot["overall"]["trr_nontrivial"]["ci_high"],
            ],
            "unreachable_fraction": 1.0 - trr,
            "unreachable_pct": unreachable_pct,
            "n_complete_curves": int(len(curves)),
            "n_nontrivial_curves": int(len(nontrivial)),
        },
        "secondary": {
            "accuracy": accuracy,
            "brier": brier,
            "mvr_practical": float(curves["mvr_practical"].mean()),
            "mvr_zero_tolerance": float(curves["mvr_zero_tolerance"].mean()),
            "mvm": float(curves["mvm"].mean()),
            "mean_reachable_count": float(curves["reachable_count"].mean()),
            "prefix_record_rate": float(curves["prefix_record_rate"].mean()),
            "reasoning_tokens": {
                "mean": float(reasoning_tokens.mean()),
                "median": float(np.median(reasoning_tokens)),
                "p05": float(np.percentile(reasoning_tokens, 5)),
                "p95": float(np.percentile(reasoning_tokens, 95)),
                "min": float(reasoning_tokens.min()),
                "max": float(reasoning_tokens.max()),
            },
        },
        "scalar_readout_diagnostic": {
            "rc_equals_ten_fraction": rc_ten,
            "rc_equals_ten_count": int(curves["rc_is_ten"].sum()),
            "provenance": (
                "inherited from the post-study step-function proposition; not an "
                "original Day-3 endpoint"
            ),
            "necessity": (
                "RC = 10 is a necessary consequence of the strict scalar "
                "latent-estimation / readout model"
            ),
            "caveat": (
                "RC = 10 does not prove the model actually uses that "
                "representation; necessity is not sufficiency"
            ),
        },
        "reasoning_length_associations": ra.reasoning_length_association(curves),
        "baselines": {
            "fixed_center_tce": rc.BASELINE_FIXED_CENTER_TCE,
            "random_crossing_tce": rc.BASELINE_RANDOM_CROSSING_TCE,
            "always_met_tce": rc.BASELINE_ALWAYS_MET_TCE,
            "always_not_met_tce": rc.BASELINE_ALWAYS_NOT_MET_TCE,
            "fixed_center_accuracy_pct": 100.0 * rc.BASELINE_FIXED_CENTER_ACC,
            "random_crossing_accuracy_pct": 100.0 * rc.BASELINE_RANDOM_CROSSING_ACC,
            "majority_accuracy_pct": 100.0 * rc.BASELINE_MAJORITY_ACC,
        },
        "day3_comparison": {
            "tce": rc.DAY3_TCE,
            "unreachable_pct": rc.DAY3_UNREACHABLE_PCT,
            "status": rc.COMPARISON_STATUS,
        },
        "interpretation": apply_interpretation(unreachable_pct),
        "bootstrap": boot,
        "family": family_rows,
        "analysis_code": provenance.source_manifest(rc.ANALYSIS_CODE_PATHS, root),
        "generated_at": provenance.utc_now(),
    }
    provenance.write_json(root / rc.SUMMARY_PATH, summary)

    print()
    print(f"mean TCE            {mean_tce:.3f}")
    print(f"nontrivial TRR      {trr:.4f}")
    print(f"unreachable         {unreachable_pct:.2f}%")
    print(f"accuracy            {100 * accuracy:.2f}%")
    print(f"Brier               {brier:.4f}")
    print(f"mean RC             {float(curves['reachable_count'].mean()):.3f}")
    print(f"RC = 10             {100 * rc_ten:.2f}%")
    print(f"outcome category    {summary['interpretation']['outcome_category']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
