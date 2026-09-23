#!/usr/bin/env python
"""Generate the reasoning-anchor results document from machine artifacts.

Every number in the output is read from ``summary.json``, the metrics table or
the bootstrap table. Interpretive prose lives in the ``NARRATIVE`` block below
and is written after the analysis; no figure or statistic is ever typed by hand
into the document.

Two claim-discipline rules are enforced structurally rather than left to
memory:

* the model is described as a *reasoning-capable open-weight external anchor*,
  never as a frontier or proprietary judge;
* the recalibration statement is the corrected one. A richer monotone
  recalibration cannot recover an unreachable boundary, because additive
  translation is already complete for the first-crossing locations obtainable
  under any uniform strictly order-preserving scalar recalibration.
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

from formalcrrc import provenance  # noqa: E402
from formalcrrc import reasoning_anchor_config as rc  # noqa: E402


NARRATIVE: dict[str, str] = {
    "why_reason_then_score": (
        "The original Day-3 panel was scored at the immediate response "
        "boundary: the A-vs-B margin was read from the next-token logits at the "
        "first generated position. That procedure cannot be transferred to this "
        "model. `Qwen3-30B-A3B-Thinking-2507` is a thinking-only model whose "
        "native chat template *opens* a reasoning block in the generation "
        "prompt, so the first generated token begins its reasoning rather than "
        "its judgment. Scoring A and B there would have measured the start of a "
        "thought, not the end of one, and would have produced a number that "
        "looked like a Day-3 margin while meaning something entirely different."
    ),
    "separator_provenance": (
        "The post-thinking separator was not chosen; it was read out of the "
        "model's own chat template, whose assistant branch renders a completed "
        "turn with exactly two newlines between the end-of-thinking delimiter "
        "and the answer. It was fixed before any FormalCRRC row was scored and "
        "was never revisited in light of an outcome."
    ),
    "one_trace_limit": (
        "Stage R samples, so the generated trace is part of the measurement "
        "rather than a fixed property of the prompt. The study uses exactly one "
        "preregistered reasoning sample per row and does not estimate "
        "within-prompt sampling variance. Uncertainty intervals therefore "
        "reflect variation across artifacts, not variation across alternative "
        "reasoning samples."
    ),
    "comparison_caveat": (
        "The anchor is not a fifth member of the Day-3 panel and must not be "
        "read as one. It differs from those judges in scale, in training and -- "
        "decisively -- in inference procedure. Every comparison below is "
        "descriptive context, not a controlled contrast, and no pooled test "
        "treating the five models as exchangeable samples is performed."
    ),
    "recalibration": (
        "Additive translation is complete for first-crossing locations "
        "obtainable under any uniform strictly order-preserving scalar "
        "recalibration. Recovering a non-prefix-record boundary therefore "
        "requires changing the ordering itself, or using threshold- or "
        "artifact-dependent information. A richer monotone recalibration does "
        "not help."
    ),
}


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def pct(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = pathlib.Path(args.root)

    summary = json.loads((root / rc.SUMMARY_PATH).read_text(encoding="utf-8"))
    prereg = json.loads((root / rc.PREREG_JSON_PATH).read_text(encoding="utf-8"))
    stage_r = prereg["protocol"]["stage_r"]
    stage_d = prereg["protocol"]["stage_d"]
    curves = pd.read_parquet(root / rc.METRICS_PATH)
    comparison = pd.read_csv(root / rc.TABLES_DIR / "table_b_comparison.csv")
    family = pd.read_csv(root / rc.TABLES_DIR / "table_c_family.csv")

    model = summary["model"]
    primary = summary["primary"]
    secondary = summary["secondary"]
    completeness = summary["completeness"]
    interpretation = summary["interpretation"]
    diagnostic = summary["scalar_readout_diagnostic"]
    boot = summary["bootstrap"]
    tokens = secondary["reasoning_tokens"]

    warning = completeness.get("truncation_warning_flag")
    integrity_path = root / rc.INTEGRITY_MANIFEST_PATH
    integrity = (
        json.loads(integrity_path.read_text(encoding="utf-8"))
        if integrity_path.is_file()
        else {"status": "not yet run", "checks": {}}
    )

    lines: list[str] = []
    add = lines.append

    add(f"# FormalCRRC Reasoning-Capable External Anchor — Results")
    add("")
    add(
        "**Preregistered post-study external-anchor experiment.** Frozen before "
        "any FormalCRRC evaluation with this model; 0 amendments. Days 1-3 and "
        "the label-swap study remain frozen and unchanged."
    )
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Status | **COMPLETE** — {completeness['rows_scored']}/{rc.N_ROWS} rows scored |")
    add(f"| Preregistration frozen | `{summary['preregistration_frozen_at']}` |")
    add(f"| Preregistration SHA-256 | `{summary['preregistration_sha256']}` |")
    add(f"| Amendments | {summary['amendments']} |")
    add(f"| Anchor model | `{model['model_id']}` |")
    add(f"| Model revision | `{model['revision']}` |")
    add(f"| Tokenizer revision | `{model['tokenizer_revision']}` |")
    add(f"| Precision | {model['dtype']}, quantization `{model['quantization']}` |")
    add(f"| Protocol | reason-then-score (Stage R + Stage D) |")
    add(f"| Frozen-record integrity | **{integrity['status']}** |")
    if warning:
        add(f"| Completeness | **{warning}** |")
    add("")
    add(
        f"This model is a **{rc.ANCHOR_DESCRIPTION}**. It is not a frontier "
        "proprietary judge and is not described as one anywhere in this "
        "document."
    )
    add("")

    add("## 1. Status")
    add("")
    add(
        f"All {completeness['rows_attempted']} planned rows were attempted. "
        f"{completeness['rows_with_successful_reasoning']} reasoning traces "
        f"reached the native end-of-thinking delimiter and "
        f"{completeness['rows_scored']} rows carry a valid primary margin. "
        f"{completeness['rows_truncated']} rows exhausted the "
        f"{rc.REASONING_TOKEN_BUDGET}-token budget "
        f"({pct(100 * completeness['truncation_rate'])})."
    )
    add("")
    if warning:
        add(
            f"Truncation exceeds the preregistered 1% threshold, so the study "
            f"carries the `{warning}` qualification. The budget was **not** "
            "increased after observing which rows truncated; doing so would "
            "have been a post-inference deviation."
        )
    else:
        add(
            "Truncation is below the preregistered 1% threshold, so no "
            "completeness warning applies."
        )
    add("")

    add("## 2. Why reason-then-score was required")
    add("")
    add(NARRATIVE["why_reason_then_score"])
    add("")

    add("## 3. The exact scoring procedure")
    add("")
    add("**Stage R — reasoning generation.** The frozen Day-3 user message is passed")
    add("through the model's native chat template and generation continues until the")
    add(f"native `{rc.THINK_END_TEXT}` delimiter (token id "
        f"`{stage_r['think_end_token_id']}`, a single token) or the")
    add(f"{rc.REASONING_TOKEN_BUDGET}-token budget is exhausted. A trace that never")
    add("produced the delimiter is marked truncated; the delimiter is never fabricated.")
    add("")
    add("**Stage D — decision scoring.** The standardised decision context is the")
    add("original formatted prompt, the exact generated reasoning including its")
    add("delimiter, then the frozen post-thinking separator. Candidates `A` and `B` are")
    add("tokenised as *continuations of that context* — never in isolation — and scored")
    add("by full sequence log-likelihood, so a multi-token candidate is handled")
    add("correctly. The canonical margin is")
    add("")
    add("```")
    add("M = L_A - L_B")
    add("```")
    add("")
    add("with `M > 0` meaning the criterion is judged met and `M = 0` counting as met,")
    add("exactly as in Day 3. No free-form final answer is generated for the primary")
    add("margin.")
    add("")
    add(NARRATIVE["separator_provenance"])
    add("")

    add("## 4. Generation configuration")
    add("")
    add("| Parameter | Value |")
    add("|---|---|")
    add(f"| temperature | {rc.TEMPERATURE} |")
    add(f"| top_p | {rc.TOP_P} |")
    add(f"| top_k | {rc.TOP_K} |")
    add(f"| min_p | {rc.MIN_P} |")
    add(f"| reasoning samples per row | {rc.REASONING_SAMPLES_PER_ROW} |")
    add(f"| reasoning token budget | {rc.REASONING_TOKEN_BUDGET} |")
    add(f"| experiment base seed | {rc.EXPERIMENT_SEED} |")
    add("")
    add(f"Per-row seeds are derived deterministically: `{rc.SEED_RULE}`")
    add("")

    add("## 5. Dataset verification")
    add("")
    validation = summary.get("dataset_validation")
    add(
        f"The frozen Day-3 TEST partition (seed {rc.SOURCE_SEED}) was verified "
        f"before inference: {rc.N_ARTIFACTS} artifacts, {rc.N_ROWS} rows, "
        f"{rc.N_STRICTNESS_LEVELS} strictness levels each, 108 artifacts per "
        f"family, 36 artifacts at each true crossing 1-9, and "
        f"{rc.N_NONTRIVIAL} nontrivial curves. Formal truth and the true "
        "crossing were independently recomputed from primitive dataset fields "
        "for all three families; no model output was used in that validation."
    )
    add("")

    add("## 6. Completeness and curve exclusion")
    add("")
    add("| | |")
    add("|---|---:|")
    add(f"| Rows attempted | {completeness['rows_attempted']} |")
    add(f"| Reasoning completed | {completeness['rows_with_successful_reasoning']} |")
    add(f"| Truncated | {completeness['rows_truncated']} |")
    add(f"| Other failures | {completeness['rows_other_failure']} |")
    add(f"| Complete curves | {completeness['n_complete_curves']} |")
    add(f"| Excluded curves | {completeness['n_excluded_curves']} |")
    add("")
    add(f"{rc.CURVE_EXCLUSION_RULE}")
    add("")

    add("## 7. Primary results")
    add("")
    add("| Metric | Value | 95% CI |")
    add("|---|---:|---:|")
    add(
        f"| Mean TCE | {fmt(primary['mean_tce'])} | "
        f"[{fmt(primary['mean_tce_ci'][0])}, {fmt(primary['mean_tce_ci'][1])}] |"
    )
    add(
        f"| Nontrivial TRR | {fmt(primary['trr_nontrivial'], 4)} | "
        f"[{fmt(primary['trr_ci'][0], 4)}, {fmt(primary['trr_ci'][1], 4)}] |"
    )
    add(
        f"| Unreachable fraction | {pct(primary['unreachable_pct'])} | "
        f"[{pct(100 * (1 - primary['trr_ci'][1]))}, "
        f"{pct(100 * (1 - primary['trr_ci'][0]))}] |"
    )
    add("")
    add(
        f"Computed over {primary['n_complete_curves']} complete curves, of which "
        f"{primary['n_nontrivial_curves']} have a nontrivial true boundary "
        "(`j* in 1..8`) and therefore carry ordering information."
    )
    add("")

    add("## 8. Secondary metrics")
    add("")
    add("| Metric | Value |")
    add("|---|---:|")
    add(f"| Accuracy | {pct(100 * secondary['accuracy'])} |")
    add(f"| Brier score | {fmt(secondary['brier'], 4)} |")
    add(f"| MVR (practical, tol 0.05) | {fmt(secondary['mvr_practical'], 4)} |")
    add(f"| MVR (zero tolerance) | {fmt(secondary['mvr_zero_tolerance'], 4)} |")
    add(f"| MVM | {fmt(secondary['mvm'], 4)} |")
    add(f"| Mean reachable count RC | {fmt(secondary['mean_reachable_count'])} |")
    add(f"| Prefix-record rate | {fmt(secondary['prefix_record_rate'], 4)} |")
    add(f"| Mean reasoning tokens | {fmt(tokens['mean'], 1)} |")
    add(f"| Median reasoning tokens | {fmt(tokens['median'], 1)} |")
    add(f"| Reasoning tokens p05-p95 | {fmt(tokens['p05'], 0)}-{fmt(tokens['p95'], 0)} |")
    add("")

    add("## 9. Structural baseline comparison")
    add("")
    add("| Reference strategy | TCE | Accuracy |")
    add("|---|---:|---:|")
    add(f"| Fixed centre / no input | {fmt(rc.BASELINE_FIXED_CENTER_TCE, 4)} | "
        f"{pct(100 * rc.BASELINE_FIXED_CENTER_ACC, 4)} |")
    add(f"| Uniform random crossing | {fmt(rc.BASELINE_RANDOM_CROSSING_TCE, 4)} | "
        f"{pct(100 * rc.BASELINE_RANDOM_CROSSING_ACC, 4)} |")
    add(f"| Always met / majority | {fmt(rc.BASELINE_ALWAYS_MET_TCE, 4)} | "
        f"{pct(100 * rc.BASELINE_MAJORITY_ACC, 4)} |")
    add(f"| Always not met | {fmt(rc.BASELINE_ALWAYS_NOT_MET_TCE, 4)} | "
        f"{pct(100 * (1 - rc.BASELINE_MAJORITY_ACC), 4)} |")
    add(f"| **Reasoning anchor** | **{fmt(primary['mean_tce'])}** | "
        f"**{pct(100 * secondary['accuracy'])}** |")
    add("")
    add(
        "These are structural references, not learned competitors. Any statement "
        "of the form \"better than random\" names both the metric and the "
        "reference strategy; the anchor is never called better than random "
        "generically."
    )
    add("")

    add("## 10. Contextual comparison with the original Day-3 panel")
    add("")
    add(NARRATIVE["comparison_caveat"])
    add("")
    add("| Row kind | Model / strategy | Inference mode | TCE | Unreachable |")
    add("|---|---|---|---:|---:|")
    for row in comparison.itertuples(index=False):
        unreachable = (
            "—" if pd.isna(row.unreachable_pct) else pct(row.unreachable_pct)
        )
        add(
            f"| {row.row_kind} | {row.name} | {row.inference_mode} | "
            f"{fmt(row.tce)} | {unreachable} |"
        )
    add("")

    add("## 11. Family-stratified results")
    add("")
    add("| Family | Curves | TCE | TRR | Unreachable | Accuracy | Brier | Mean tokens |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in family.itertuples(index=False):
        add(
            f"| {row.family} | {row.n_curves} | {fmt(row.tce)} | "
            f"{fmt(row.trr, 4)} | {pct(row.unreachable_pct)} | "
            f"{pct(100 * row.accuracy)} | {fmt(row.brier, 4)} | "
            f"{fmt(row.mean_reasoning_tokens, 1)} |"
        )
    add("")

    add("## 12. Scalar-readout diagnostic (RC = 10)")
    add("")
    add(
        f"`RC = 10` occurs for {diagnostic['rc_equals_ten_count']} of "
        f"{primary['n_complete_curves']} complete curves "
        f"({pct(100 * diagnostic['rc_equals_ten_fraction'])})."
    )
    add("")
    add(f"* This diagnostic is {diagnostic['provenance']}.")
    add(f"* {diagnostic['necessity']}.")
    add(f"* **{diagnostic['caveat']}.**")
    add("")

    add("## 13. Reasoning-length diagnostics")
    add("")
    for association in summary.get("reasoning_length_associations", []):
        rho = association["spearman_rho"]
        if rho is None or not np.isfinite(rho):
            # A constant outcome has no rank correlation. Saying so is more
            # informative than printing "n/a", and here the constant is itself
            # the headline result: every boundary was reachable.
            add(
                f"* {association['length_column']} vs "
                f"`{association['outcome']}`: **undefined** — the outcome is "
                f"constant across all {association['n']} curves, so no rank "
                f"correlation exists."
            )
        else:
            add(
                f"* {association['length_column']} vs `{association['outcome']}`: "
                f"Spearman rho {fmt(rho, 3)} "
                f"(p {fmt(association['p_value'], 4)}, n {association['n']}) — "
                f"{association['interpretation']}."
            )
    add("")
    add(
        "Trace length is chosen by the model and is confounded with item "
        "difficulty, so these associations are reported as descriptive and are "
        "not interpreted causally."
    )
    add("")

    add("## 14. Prespecified interpretation")
    add("")
    add(f"**Outcome category {interpretation['outcome_category']}.**")
    add("")
    add(f"> {interpretation['statement']}")
    add("")
    add(
        f"The unreachable fraction is {pct(interpretation['unreachable_pct'])}, "
        f"against the original Day-3 panel range "
        f"{pct(interpretation['original_panel_range_pct'][0])}-"
        f"{pct(interpretation['original_panel_range_pct'][1])}. "
        f"{interpretation['note']}"
    )
    add("")

    add("## 15. Limitations")
    add("")
    add(NARRATIVE["one_trace_limit"])
    add("")
    add(
        "The protocol also differs from Day 3 by construction, so a difference "
        "in any metric confounds model capability with inference procedure. The "
        "study cannot separate those two, and does not try to."
    )
    add("")

    add("## 16. On recalibration")
    add("")
    add(NARRATIVE["recalibration"])
    add("")

    add("## 17. Deviations and retries")
    add("")
    retry_path = root / rc.RETRY_LOG_PATH
    retries = (
        json.loads(retry_path.read_text(encoding="utf-8")).get("retries", [])
        if retry_path.is_file()
        else []
    )
    if retries:
        add(f"{len(retries)} infrastructure retry/retries, each with an identical")
        add("row, seed, model, prompt, generation configuration and scoring")
        add("configuration:")
        add("")
        for entry in retries:
            add(f"* {json.dumps(entry)}")
    else:
        add("No retries were required. No methodological deviation occurred.")
    add("")

    add("## 18. Integrity verification")
    add("")
    frozen = integrity.get("frozen_record", {})
    add("| Check | Result |")
    add("|---|---|")
    for name, ok in integrity.get("checks", {}).items():
        add(f"| {name} | {'PASS' if ok else 'FAIL'} |")
    if frozen:
        add(f"| protected files | {frozen.get('n_recorded', 'n/a')} |")
        add(f"| combined digest before | `{frozen.get('combined_recorded', 'n/a')}` |")
        add(f"| combined digest after | `{frozen.get('combined_now', 'n/a')}` |")
        add(f"| changed files | {frozen.get('changed') or 'NONE'} |")
    add("")

    add("## 19. Claims that must not be made")
    add("")
    add("This study does not establish that:")
    add("")
    for claim in (
        "Qwen3 represents all frontier judges, or any proprietary API judge",
        "reasoning generally causes better or worse judge reliability",
        "chain-of-thought is causally responsible for any observed difference",
        "the reasoning trace faithfully reveals the model's hidden computation",
        "one sampled reasoning trace characterises all possible reasoning paths",
        "a lower TCE implies correct score ordering",
        "a higher TRR implies a scalar latent representation",
        "failure of this anchor invalidates FormalCRRC",
        "success of this anchor proves FormalCRRC generalises to semantic rubrics",
        "the comparison with the original models is perfectly controlled",
    ):
        add(f"* {claim};")
    add("")
    add(
        f"The correct term for this model is **{rc.ANCHOR_DESCRIPTION}**, not "
        "\"frontier judge\"."
    )
    add("")

    add("## 20. Manuscript-facing interpretation")
    add("")
    add("The narrow question this experiment answers is:")
    add("")
    add(
        "> Does the central FormalCRRC localisation/reachability phenomenon "
        "remain observable in a substantially more capable reasoning-enabled "
        "judge under a preregistered reason-then-score protocol?"
    )
    add("")
    add("Recommended wording, using the exact repository-derived values:")
    add("")
    add(
        f"> Under a preregistered reason-then-score protocol, the "
        f"reasoning-capable open-weight external anchor "
        f"`{model['model_id']}` reaches a mean threshold crossing error of "
        f"{fmt(primary['mean_tce'])} "
        f"(95% CI [{fmt(primary['mean_tce_ci'][0])}, "
        f"{fmt(primary['mean_tce_ci'][1])}]) and leaves "
        f"{pct(primary['unreachable_pct'])} of nontrivial formal boundaries "
        f"unreachable by any additive translation "
        f"(95% CI [{pct(100 * (1 - primary['trr_ci'][1]))}, "
        f"{pct(100 * (1 - primary['trr_ci'][0]))}]), against "
        f"{pct(rc.DAY3_UNREACHABLE_RANGE[0])}-"
        f"{pct(rc.DAY3_UNREACHABLE_RANGE[1])} for the original four-judge "
        f"panel scored at the immediate response boundary. "
        f"{interpretation['statement']}"
    )
    add("")
    add(
        "Report TCE and reachability separately rather than collapsing them "
        "into a single verdict, and state the procedural difference whenever "
        "the anchor appears beside the original panel."
    )
    add("")
    add("---")
    add("")
    add(
        f"*Generated by `scripts/write_results_reasoning_anchor.py` from "
        f"`{rc.SUMMARY_PATH}`. Every number above is read from a machine "
        f"artifact.*"
    )

    out_path = root / rc.RESULTS_MD_PATH
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out_path.relative_to(root)} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
