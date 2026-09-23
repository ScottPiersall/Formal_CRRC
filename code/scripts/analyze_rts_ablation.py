#!/usr/bin/env python
"""Preregistered analysis of the Qwen2.5 reason-then-score protocol ablation.

Runs only after every intended row has been attempted. The crossing rule, the
reachability rule, the tie policy, the tie-versus-inversion decomposition, the
curve-exclusion rule and the interpretation categories all come from the frozen
preregistration; nothing here is chosen after seeing an outcome.

Condition O is recomputed from the frozen Day-3 raw scores by this study's own
code, so agreement with the frozen metrics table is a real check rather than a
restatement. Condition O files are opened read-only and never written.

Where the reason-then-score unreachable count is zero, the analysis reports an
exact Clopper-Pearson interval alongside the bootstrap, because a bootstrap over
an identically-zero statistic returns [0, 0] and that is a fact about resampling,
not about the population.
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
from formalcrrc import rts_ablation as ra  # noqa: E402
from formalcrrc import rts_ablation_bootstrap as rab  # noqa: E402
from formalcrrc import rts_ablation_config as rc  # noqa: E402
from formalcrrc import day3_config as d3c  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    FAMILIES,
    MVR_TOLERANCE,
    MVR_TOLERANCE_STRICT,
    N_STRICTNESS,
)
from formalcrrc.day2_bootstrap import crossing_index  # noqa: E402


# --------------------------------------------------------------------------
# Phase 14 -- completeness
# --------------------------------------------------------------------------


def completeness_report(rows: pd.DataFrame) -> dict[str, Any]:
    attempted = int(len(rows))
    reached = int(rows["marker_reached"].sum())
    truncated = int(rows["reasoning_truncated"].sum())
    premature = int(rows["premature_label_observed"].sum())
    scored = int(rows["margin"].notna().sum())
    other = int(((rows["marker_reached"]) & (rows["margin"].isna())).sum())
    completeness = ra.curve_completeness(rows)
    rate = (attempted - reached) / attempted if attempted else 0.0
    return {
        "rows_attempted": attempted,
        "rows_marker_reached": reached,
        "rows_truncated": truncated,
        "rows_scoring_failures": other,
        "rows_scored": scored,
        "rows_premature_label": premature,
        "premature_label_rate": premature / attempted if attempted else 0.0,
        "marker_failure_rate": rate,
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
# Per-curve metrics for Condition R, paired to Condition O
# --------------------------------------------------------------------------


def build_paired_curves(
    rows: pd.DataFrame, comparator: pd.DataFrame
) -> pd.DataFrame:
    """One row per complete artifact curve carrying both conditions."""
    ordered = rows.sort_values(["artifact_id", "strictness_index"])
    comparator = comparator.set_index("artifact_id")
    records: list[dict[str, Any]] = []
    for artifact_id, group in ordered.groupby("artifact_id", sort=True):
        levels = group["strictness_index"].to_numpy()
        if not np.array_equal(levels, np.arange(N_STRICTNESS)):
            raise ValueError(f"{artifact_id}: expected strictness 0..8 exactly once")
        if artifact_id not in comparator.index:
            raise ValueError(f"{artifact_id} has no frozen Day-3 counterpart")
        o = comparator.loc[artifact_id]

        margins = group["margin"].to_numpy(dtype=float)
        truth = group["formal_truth"].to_numpy(dtype=int)
        target = int(group["true_first_fail_index"].iloc[0])
        probabilities = day2.sigmoid(margins)
        j_hat = int(crossing_index(margins[None, :])[0])
        reach = day3.reachable_crossings(margins)
        decisions = (margins >= 0.0).astype(int)
        generated = group["generated_token_count"].to_numpy(dtype=float)

        records.append(
            {
                "artifact_id": artifact_id,
                "family": group["family"].iloc[0],
                "latent_level": int(group["latent_level"].iloc[0]),
                "true_first_fail_index": target,
                "nontrivial_boundary": 1 <= target <= N_STRICTNESS - 1,
                # Condition R
                "j_hat_r": j_hat,
                "tce_r": abs(j_hat - target),
                "reachable_r": bool(target in reach),
                "rc_r": len(reach),
                "prefix_record_rate_r": day3.prefix_record_rate(margins),
                "failure_type_r": ra.classify_failure(margins, target),
                "accuracy_r": float(np.mean(decisions == truth)),
                "brier_r": float(np.mean((probabilities - truth) ** 2)),
                "mvr_practical_r": metrics.monotonicity_violation_rate(
                    probabilities, MVR_TOLERANCE
                ),
                "mvr_zero_r": metrics.monotonicity_violation_rate(
                    probabilities, MVR_TOLERANCE_STRICT
                ),
                "mvm_r": metrics.monotonicity_violation_magnitude(probabilities),
                "span_r": day2.response_span(margins),
                "rc_is_ten_r": len(reach) == 10,
                "mean_generated_tokens": float(generated.mean()),
                "total_generated_tokens": int(generated.sum()),
                "premature_labels_in_curve": int(
                    group["premature_label_observed"].sum()
                ),
                # Condition O, frozen
                "j_hat_o": int(o["j_hat_o"]),
                "tce_o": int(o["tce_o"]),
                "reachable_o": bool(o["translation_reachable_o"]),
                "rc_o": int(o["reachable_count_o"]),
                "prefix_record_rate_o": float(o["prefix_record_rate_o"]),
                "failure_type_o": str(o["failure_type_o"]),
                "accuracy_o": float(o["accuracy_o"]),
                "brier_o": float(o["brier_o"]),
                "rc_is_ten_o": int(o["reachable_count_o"]) == 10,
                # paired
                "delta_tce": abs(j_hat - target) - int(o["tce_o"]),
                "delta_reachable": int(target in reach) - int(o["translation_reachable_o"]),
            }
        )
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Phase 16 -- unreachable-failure decomposition
# --------------------------------------------------------------------------


def decomposition(curves: pd.DataFrame) -> pd.DataFrame:
    """Counts of reachable / tie-only / strict-inversion, per condition."""
    nontrivial = curves[curves["nontrivial_boundary"]]
    rows: list[dict[str, Any]] = []
    for condition, column in (
        (rc.CONDITION_ORIGINAL, "failure_type_o"),
        (rc.CONDITION_REASONING, "failure_type_r"),
    ):
        counts = nontrivial[column].value_counts()
        total = int(len(nontrivial))
        unreachable = total - int(counts.get(rc.FAILURE_NONE, 0))
        rows.append(
            {
                "condition": condition,
                "n_nontrivial": total,
                "n_reachable": int(counts.get(rc.FAILURE_NONE, 0)),
                "n_unreachable": unreachable,
                "unreachable_pct": 100.0 * unreachable / total if total else float("nan"),
                "n_strict_inversion": int(counts.get(rc.FAILURE_STRICT_INVERSION, 0)),
                "n_tie_only": int(counts.get(rc.FAILURE_TIE_ONLY, 0)),
                "strict_inversion_pct_of_unreachable": (
                    100.0 * int(counts.get(rc.FAILURE_STRICT_INVERSION, 0)) / unreachable
                    if unreachable
                    else float("nan")
                ),
                "tie_only_pct_of_unreachable": (
                    100.0 * int(counts.get(rc.FAILURE_TIE_ONLY, 0)) / unreachable
                    if unreachable
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Phase 20 -- prespecified interpretation
# --------------------------------------------------------------------------


def apply_interpretation(delta_trr: dict[str, Any]) -> dict[str, Any]:
    low, high = delta_trr["ci_low"], delta_trr["ci_high"]
    if low > 0.0:
        category, statement = (
            "increase",
            "Elicited reason-then-score increases true-boundary reachability "
            "for the same Qwen2.5 judge.",
        )
    elif high < 0.0:
        category, statement = (
            "decrease",
            "Elicited reason-then-score decreases true-boundary reachability "
            "for the same Qwen2.5 judge.",
        )
    else:
        category, statement = (
            "uncertain",
            "The experiment does not establish a directional protocol-associated "
            "change in true-boundary reachability.",
        )
    return {
        "category": category,
        "statement": statement,
        "delta_trr": delta_trr["point"],
        "delta_trr_ci": [low, high],
        "unit_of_intervention": rc.INTERVENTION_IS_COMPOSITE,
    }


def anchor_positioning(trr_r: float, trr_o: float) -> dict[str, str]:
    """Section 21.1 -- where the ablation lands relative to the anchor."""
    anchor = rc.ANCHOR_TRR
    if trr_r >= anchor - 1e-12:
        statement = (
            "The previous panel-versus-anchor difference cannot be attributed "
            "primarily to model capability alone; the inference/readout protocol "
            "materially contributes."
        )
        category = "approaches_anchor"
    elif abs(trr_r - trr_o) < 0.05:
        statement = (
            "Applying elicited reason-then-score to Qwen2.5 does not reproduce "
            "the Qwen3 anchor's complete reachability, strengthening the "
            "interpretation that model capability and training differences "
            "contribute materially to the anchor result."
        )
        category = "near_original"
    else:
        statement = (
            "Both protocol and model-regime differences are consistent with the "
            "observed panel-versus-anchor contrast."
        )
        category = "intermediate"
    return {
        "category": category,
        "statement": statement,
        "caveat": (
            "These are descriptive causal constraints, not identification of a "
            "single causal mechanism. " + rc.ANCHOR_COMPARISON_STATUS
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
    print(f"loaded {len(rows)} reason-then-score rows")

    # Condition O, recomputed read-only from the frozen record.
    raw_o = pd.read_parquet(
        root / "artifacts/day3/raw_scores/qwen2_5_14b_instruct.parquet"
    )
    dataset = pd.read_parquet(root / d3c.DATASET_PATH)
    comparator = ra.recover_comparator(raw_o, dataset)
    frozen_metrics = pd.read_parquet(root / d3c.METRICS_PATH)
    frozen_metrics = frozen_metrics[frozen_metrics["model_id"] == rc.MODEL_ID]
    comparator_verdict = ra.verify_comparator(comparator, frozen_metrics)
    if comparator_verdict["status"] != "PASS":
        raise SystemExit(f"comparator recovery failed: {comparator_verdict}")
    print(
        f"comparator recovered: TCE {comparator_verdict['mean_tce']:.4f}  "
        f"TRR {comparator_verdict['trr']:.4f}"
    )

    completeness = completeness_report(rows)
    print(
        f"complete curves {completeness['n_complete_curves']}, "
        f"excluded {completeness['n_excluded_curves']}, "
        f"marker failures {completeness['marker_failure_rate']:.4%}"
    )

    usable = ra.complete_rows(rows)
    if usable.empty:
        raise SystemExit("no complete curves; nothing to analyse")
    curves = build_paired_curves(usable, comparator)
    curves.to_parquet(root / rc.METRICS_PATH, index=False)
    curves.to_parquet(root / rc.PAIRED_CURVES_PATH, index=False)

    nontrivial = curves[curves["nontrivial_boundary"]]
    trr_r = float(nontrivial["reachable_r"].mean())
    trr_o = float(nontrivial["reachable_o"].mean())
    unreachable_r = 100.0 * (1.0 - trr_r)
    unreachable_o = 100.0 * (1.0 - trr_o)
    tce_r = float(curves["tce_r"].mean())
    tce_o = float(curves["tce_o"].mean())

    boot = rab.bootstrap_ablation(curves)
    rab.to_frame(boot).to_parquet(root / rc.BOOTSTRAP_PATH, index=False)

    decomp = decomposition(curves)
    decomp.to_parquet(root / rc.FAILURE_DECOMP_PATH, index=False)

    # Exact interval for a zero unreachable count.
    n_unreachable_r = int((~nontrivial["reachable_r"]).sum())
    n_nontrivial = int(len(nontrivial))
    cp_low, cp_high = ra.clopper_pearson(
        n_unreachable_r, n_nontrivial, rc.CLOPPER_PEARSON_LEVEL
    )
    exact_interval = {
        "n_unreachable": n_unreachable_r,
        "n_nontrivial": n_nontrivial,
        "observed_fraction": n_unreachable_r / n_nontrivial if n_nontrivial else None,
        "clopper_pearson_low": cp_low,
        "clopper_pearson_high": cp_high,
        "level": rc.CLOPPER_PEARSON_LEVEL,
        "applies": n_unreachable_r == 0,
        "rule": rc.ZERO_COUNT_RULE,
    }

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
                "tce_o": float(block["tce_o"].mean()),
                "tce_r": float(block["tce_r"].mean()),
                "delta_tce": float(block["delta_tce"].mean()),
                "trr_o": float(block_nt["reachable_o"].mean()),
                "trr_r": float(block_nt["reachable_r"].mean()),
                "delta_trr": float(
                    block_nt["reachable_r"].mean() - block_nt["reachable_o"].mean()
                ),
                "unreachable_pct_o": 100.0 * (1 - float(block_nt["reachable_o"].mean())),
                "unreachable_pct_r": 100.0 * (1 - float(block_nt["reachable_r"].mean())),
                "accuracy_o": float(block["accuracy_o"].mean()),
                "accuracy_r": float(block["accuracy_r"].mean()),
                "brier_o": float(block["brier_o"].mean()),
                "brier_r": float(block["brier_r"].mean()),
                "mean_generated_tokens": float(block["mean_generated_tokens"].mean()),
            }
        )
    family_frame = pd.DataFrame(family_rows)
    family_frame.to_parquet(root / rc.FAMILY_SUMMARY_PATH, index=False)

    # ---------------- tables ----------------
    tables = root / rc.TABLES_DIR
    tables.mkdir(parents=True, exist_ok=True)

    table_a = pd.DataFrame(
        [
            {
                "condition": "Qwen2.5 immediate (frozen Day 3)",
                "readout": "immediate next-token logit",
                "tce": tce_o,
                "tce_ci_low": boot["immediate"]["mean_tce"]["ci_low"],
                "tce_ci_high": boot["immediate"]["mean_tce"]["ci_high"],
                "trr": trr_o,
                "unreachable_pct": unreachable_o,
                "accuracy_pct": 100.0 * float(curves["accuracy_o"].mean()),
                "brier": float(curves["brier_o"].mean()),
                "mean_rc": float(curves["rc_o"].mean()),
            },
            {
                "condition": "Qwen2.5 reason-then-score",
                "readout": "elicited reasoning + conditional sequence likelihood",
                "tce": tce_r,
                "tce_ci_low": boot["reason_then_score"]["mean_tce"]["ci_low"],
                "tce_ci_high": boot["reason_then_score"]["mean_tce"]["ci_high"],
                "trr": trr_r,
                "unreachable_pct": unreachable_r,
                "accuracy_pct": 100.0 * float(curves["accuracy_r"].mean()),
                "brier": float(curves["brier_r"].mean()),
                "mean_rc": float(curves["rc_r"].mean()),
            },
        ]
    )
    table_a.to_csv(tables / "table_a_within_model.csv", index=False)

    table_b = pd.DataFrame(
        [
            {
                "effect": name,
                "point": block["point"],
                "ci_low": block["ci_low"],
                "ci_high": block["ci_high"],
                "excludes_zero": (block["ci_low"] > 0) or (block["ci_high"] < 0),
            }
            for name, block in boot["paired"].items()
        ]
    )
    table_b.to_csv(tables / "table_b_paired_effects.csv", index=False)
    decomp.to_csv(tables / "table_c_failure_mechanism.csv", index=False)

    table_d = pd.DataFrame(
        [
            {
                "row_kind": "within-model ablation",
                "model": rc.MODEL_SHORT_NAME,
                "protocol": "immediate next-token (frozen Day 3)",
                "tce": tce_o,
                "trr": trr_o,
                "unreachable_pct": unreachable_o,
            },
            {
                "row_kind": "within-model ablation",
                "model": rc.MODEL_SHORT_NAME,
                "protocol": "elicited reason-then-score (this study)",
                "tce": tce_r,
                "trr": trr_r,
                "unreachable_pct": unreachable_r,
            },
            {
                "row_kind": "external context",
                "model": "Qwen3-30B-A3B-Thinking",
                "protocol": "native reason-then-score (anchor study)",
                "tce": rc.ANCHOR_TCE,
                "trr": rc.ANCHOR_TRR,
                "unreachable_pct": rc.ANCHOR_UNREACHABLE_PCT,
            },
        ]
    )
    table_d.to_csv(tables / "table_d_model_protocol_context.csv", index=False)
    family_frame.to_csv(tables / "table_e_family.csv", index=False)

    generated = usable["generated_token_count"].to_numpy(dtype=float)
    interpretation = apply_interpretation(boot["paired"]["delta_trr"])
    positioning = anchor_positioning(trr_r, trr_o)

    summary: dict[str, Any] = {
        "study": rc.STUDY_NAME,
        "design": prereg["design"],
        "model": prereg["model"],
        "preregistration_sha256": provenance.sha256_file(root / rc.PREREG_JSON_PATH),
        "preregistration_frozen_at": prereg["frozen_at"],
        "amendments": prereg["amendments"],
        "completeness": completeness,
        "comparator_recovery": comparator_verdict,
        "primary": {
            "trr_immediate": trr_o,
            "trr_reason_then_score": trr_r,
            "unreachable_pct_immediate": unreachable_o,
            "unreachable_pct_reason_then_score": unreachable_r,
            "delta_trr": boot["paired"]["delta_trr"],
            "n_nontrivial_curves": n_nontrivial,
            "exact_interval": exact_interval,
        },
        "tce": {
            "immediate": tce_o,
            "reason_then_score": tce_r,
            "delta_tce": boot["paired"]["delta_tce"],
            "immediate_ci": [
                boot["immediate"]["mean_tce"]["ci_low"],
                boot["immediate"]["mean_tce"]["ci_high"],
            ],
            "reason_then_score_ci": [
                boot["reason_then_score"]["mean_tce"]["ci_low"],
                boot["reason_then_score"]["mean_tce"]["ci_high"],
            ],
        },
        "failure_decomposition": decomp.to_dict("records"),
        "secondary": {
            "accuracy_o": float(curves["accuracy_o"].mean()),
            "accuracy_r": float(curves["accuracy_r"].mean()),
            "brier_o": float(curves["brier_o"].mean()),
            "brier_r": float(curves["brier_r"].mean()),
            "mvr_practical_r": float(curves["mvr_practical_r"].mean()),
            "mvr_zero_r": float(curves["mvr_zero_r"].mean()),
            "mvm_r": float(curves["mvm_r"].mean()),
            "mean_rc_o": float(curves["rc_o"].mean()),
            "mean_rc_r": float(curves["rc_r"].mean()),
            "rc_is_ten_o": int(curves["rc_is_ten_o"].sum()),
            "rc_is_ten_r": int(curves["rc_is_ten_r"].sum()),
            "prefix_record_rate_o": float(curves["prefix_record_rate_o"].mean()),
            "prefix_record_rate_r": float(curves["prefix_record_rate_r"].mean()),
            "generated_tokens": {
                "mean": float(generated.mean()),
                "median": float(np.median(generated)),
                "p05": float(np.percentile(generated, 5)),
                "p95": float(np.percentile(generated, 95)),
                "min": float(generated.min()),
                "max": float(generated.max()),
            },
        },
        "reasoning_length_associations": ra_length_associations(curves),
        "baselines": {
            "fixed_center_tce": rc.BASELINE_FIXED_CENTER_TCE,
            "random_crossing_tce": rc.BASELINE_RANDOM_CROSSING_TCE,
            "always_met_tce": rc.BASELINE_ALWAYS_MET_TCE,
            "always_not_met_tce": rc.BASELINE_ALWAYS_NOT_MET_TCE,
            "fixed_center_accuracy_pct": 100.0 * rc.BASELINE_FIXED_CENTER_ACC,
            "random_crossing_accuracy_pct": 100.0 * rc.BASELINE_RANDOM_CROSSING_ACC,
            "majority_accuracy_pct": 100.0 * rc.BASELINE_MAJORITY_ACC,
        },
        "anchor_context": {
            "model": rc.ANCHOR_MODEL,
            "tce": rc.ANCHOR_TCE,
            "trr": rc.ANCHOR_TRR,
            "unreachable_pct": rc.ANCHOR_UNREACHABLE_PCT,
            "status": rc.ANCHOR_COMPARISON_STATUS,
            "positioning": positioning,
        },
        "interpretation": interpretation,
        "bootstrap": boot,
        "family": family_rows,
        "analysis_code": provenance.source_manifest(rc.ANALYSIS_CODE_PATHS, root),
        "generated_at": provenance.utc_now(),
    }
    provenance.write_json(root / rc.SUMMARY_PATH, summary)

    d = boot["paired"]["delta_trr"]
    print()
    print(f"TRR immediate        {trr_o:.4f}   unreachable {unreachable_o:.2f}%")
    print(f"TRR reason-then-score{trr_r:.4f}   unreachable {unreachable_r:.2f}%")
    print(f"delta TRR            {d['point']:+.4f}  CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
    print(f"TCE  {tce_o:.3f} -> {tce_r:.3f}   delta {boot['paired']['delta_tce']['point']:+.3f}")
    if exact_interval["applies"]:
        print(
            f"exact 0/{n_nontrivial}: Clopper-Pearson upper bound "
            f"{100 * cp_high:.3f}%"
        )
    print(f"interpretation       {interpretation['category']}")
    print(f"anchor positioning   {positioning['category']}")
    return 0


def ra_length_associations(curves: pd.DataFrame) -> list[dict[str, Any]]:
    """Descriptive only; trace length is chosen by the model."""
    return ra_length_helper(curves)


def ra_length_helper(curves: pd.DataFrame) -> list[dict[str, Any]]:
    from formalcrrc.reasoning_anchor import reasoning_length_association

    return reasoning_length_association(
        curves,
        length_column="mean_generated_tokens",
        outcome_columns=("tce_r", "reachable_r"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
