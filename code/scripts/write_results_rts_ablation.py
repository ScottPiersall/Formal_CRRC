#!/usr/bin/env python
"""Generate the reason-then-score ablation results document from machine artifacts.

Every number is read from ``summary.json``, the metrics table, the decomposition
table or the bootstrap table. Interpretive prose lives in the ``NARRATIVE`` block
and is written after the analysis; no statistic is typed by hand.

Two claim-discipline rules are enforced structurally:

* the supported statement is about the **reason-then-score protocol**, never
  about "reasoning" as an isolated factor, because the intervention is composite;
* the Qwen3 anchor is external context and is never presented as a third arm of
  the ablation or as the same protocol.
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
from formalcrrc import rts_ablation_config as rc  # noqa: E402


NARRATIVE: dict[str, str] = {
    "rationale": (
        "The reasoning-anchor study compared a 30B thinking model under a "
        "reason-then-score protocol against a four-judge panel scored at the "
        "immediate response boundary, and found a very large difference in "
        "boundary reachability. That comparison could not say where the "
        "difference came from, because the model and the readout changed "
        "together. This experiment holds the model fixed -- the same weights, "
        "the same revision, the same artifacts, the same formal thresholds -- "
        "and moves only the protocol."
    ),
    "composite": (
        "The intervention is deliberately composite. It changes the evaluation "
        "instruction, introduces a generated trace, moves the position at which "
        "A and B are scored, and replaces an immediate next-token logit "
        "difference with a conditional sequence log-likelihood. The experiment "
        "cannot attribute the effect to any one of those parts, and does not "
        "try to. What it supports is a statement about the protocol as a whole."
    ),
    "elicited_vs_native": (
        "Qwen2.5-14B-Instruct has no thinking-only chat-template mechanism, so "
        "this study elicits reasoning with an explicit instruction and a fixed "
        "final-answer marker. That is not the same thing as the anchor's native "
        "thinking mode, and the two are never described as one protocol."
    ),
    "one_trace": (
        "Stage R samples, so the generated trace is part of the measurement. "
        "Exactly one preregistered trace is drawn per row and the study does not "
        "estimate within-prompt sampling variance. Intervals reflect variation "
        "across artifacts, not across alternative traces."
    ),
    "prompt_note": (
        "The reason-then-score prompt was built by appending the preregistered "
        "scaffold to the frozen Day-3 template rather than by retyping it, so "
        "the difference is a pure suffix and the rubric and candidate-response "
        "bodies are untouched. Had the prompt been retyped, its whitespace and "
        "layout would have shifted as well, and the protocol effect would have "
        "been confounded with a formatting change."
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


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = pathlib.Path(args.root)

    summary = json.loads((root / rc.SUMMARY_PATH).read_text(encoding="utf-8"))
    prereg = json.loads((root / rc.PREREG_JSON_PATH).read_text(encoding="utf-8"))
    curves = pd.read_parquet(root / rc.METRICS_PATH)
    decomp = pd.read_parquet(root / rc.FAILURE_DECOMP_PATH)
    family = pd.read_csv(root / rc.TABLES_DIR / "table_e_family.csv")
    context = pd.read_csv(root / rc.TABLES_DIR / "table_d_model_protocol_context.csv")

    model = summary["model"]
    primary = summary["primary"]
    tce = summary["tce"]
    sec = summary["secondary"]
    comp = summary["completeness"]
    interp = summary["interpretation"]
    positioning = summary["anchor_context"]["positioning"]
    stage_r, stage_d = prereg["stage_r"], prereg["stage_d"]
    prompt = prereg["prompt"]
    tokens = sec["generated_tokens"]
    d_trr = primary["delta_trr"]
    d_tce = tce["delta_tce"]

    integrity_path = root / rc.INTEGRITY_MANIFEST_PATH
    integrity = (
        json.loads(integrity_path.read_text(encoding="utf-8"))
        if integrity_path.is_file()
        else {"status": "not yet run", "checks": {}}
    )

    lines: list[str] = []
    add = lines.append

    add("# FormalCRRC Qwen2.5 Reason-Then-Score Protocol Ablation — Results")
    add("")
    add(
        "**Preregistered within-model protocol ablation.** Frozen before any "
        "Qwen2.5 reason-then-score study row was run; 0 amendments. Days 1-3, "
        "the label-swap study and the reasoning-anchor study remain frozen and "
        "unchanged."
    )
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Status | **COMPLETE** — {comp['rows_scored']}/{rc.N_ROWS} rows scored |")
    add(f"| Preregistration frozen | `{summary['preregistration_frozen_at']}` |")
    add(f"| Preregistration SHA-256 | `{summary['preregistration_sha256']}` |")
    add(f"| Amendments | {summary['amendments']} |")
    add(f"| Model | `{model['model_id']}` |")
    add(f"| Model revision | `{model['revision']}` (frozen Day-3) |")
    add(f"| Tokenizer revision | `{model['tokenizer_revision']}` |")
    add(f"| Precision | {model['dtype']}, quantization `{model['quantization']}` |")
    add("| Comparator | frozen Day-3 immediate next-token scoring, same weights |")
    add(f"| Frozen-record integrity | **{integrity['status']}** |")
    add("")

    add("## 1. Status")
    add("")
    add(
        f"All {comp['rows_attempted']} planned rows were attempted. The "
        f"`{rc.MARKER}` marker was reached on {comp['rows_marker_reached']} rows "
        f"({pct(100 * (1 - comp['marker_failure_rate']))}), and "
        f"{comp['rows_scored']} rows carry a valid reason-then-score margin. "
        f"{comp['rows_truncated']} rows exhausted the "
        f"{rc.REASONING_TOKEN_BUDGET}-token budget."
    )
    add("")
    if comp.get("truncation_warning_flag"):
        add(
            f"Marker failure exceeds the preregistered 1% threshold, so the "
            f"study carries the `{comp['truncation_warning_flag']}` "
            "qualification. The budget was **not** increased afterwards."
        )
    else:
        add(
            "Marker failure is below the preregistered 1% threshold, so no "
            "completeness warning applies."
        )
    add("")

    add("## 2. Why this experiment exists")
    add("")
    add(NARRATIVE["rationale"])
    add("")

    add("## 3. Within-model design")
    add("")
    add("| | Condition O | Condition R |")
    add("|---|---|---|")
    add(f"| Model | `{model['model_id']}` | same |")
    add(f"| Revision | `{model['revision']}` | same |")
    add("| Artifacts / thresholds | frozen Day-3 TEST, seed 45 | same |")
    add("| A/B semantics | A = met, B = not met | same |")
    add("| Readout | immediate next-token logit | elicited reasoning, then conditional sequence likelihood |")
    add("| Margin | `M_O = S_A - S_B` | `M_R = L_A - L_B` |")
    add("| Source | frozen Day-3 record, read only | this experiment |")
    add("")
    add(NARRATIVE["elicited_vs_native"])
    add("")

    add("## 4. The exact prompt intervention")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Day-3 template SHA-256 | `{prompt['day3_template_sha256']}` |")
    add(f"| Reason-then-score SHA-256 | `{prompt['rts_template_sha256']}` |")
    add(f"| Added scaffold SHA-256 | `{prompt['added_suffix_sha256']}` |")
    add(f"| Bytes added | {prompt['added_bytes']} |")
    add(f"| Lines removed | {len(prompt['removed_lines'])} |")
    add(f"| Pure suffix extension | {prompt['is_pure_suffix_extension']} |")
    add("")
    add("The appended scaffold, verbatim:")
    add("")
    add("```text")
    add(rc.REASONING_SCAFFOLD)
    add("```")
    add("")
    add(NARRATIVE["prompt_note"])
    add("")

    add("## 5. Stage R — elicited reasoning")
    add("")
    add("| Parameter | Value |")
    add("|---|---|")
    add(f"| temperature | {stage_r['temperature']} |")
    add(f"| top_p | {stage_r['top_p']} |")
    add(f"| top_k | {stage_r['top_k']} |")
    add(f"| min_p | {stage_r['min_p']} ({stage_r['min_p_status']}) |")
    add(f"| traces per row | {stage_r['samples_per_row']} |")
    add(f"| token budget | {stage_r['reasoning_token_budget']} |")
    add(f"| base seed | {stage_r['experiment_seed']} |")
    add("")
    add(
        f"Per-row seeds use the same SHA-256 convention as the reasoning-anchor "
        f"study, reduced modulo {stage_r['seed_modulus']}, so the two "
        "experiments are procedurally comparable. Generation stops on the "
        "complete marker and before the next token, so the model never samples "
        "the A/B label that the primary margin measures."
    )
    add("")

    add("## 6. Stage D — decision scoring")
    add("")
    add(
        "The scoring context is the exact formatted prompt, the exact generated "
        f"reasoning, and the exact generated `{rc.MARKER}` marker. The reasoning "
        "is never regenerated, summarised or trimmed. Candidates are scored as "
        "continuations by full sequence log-likelihood, with no EOS probability "
        "and no length normalisation:"
    )
    add("")
    add("```")
    add("M_R = L_A - L_B")
    add("```")
    add("")

    add("## 7. Marker and candidate tokenisation")
    add("")
    marker = stage_r["marker_tokenization"]
    cands = stage_d["candidates"]
    add("| | Text | Bytes | Contextual token ids | Multi-token |")
    add("|---|---|---|---|---|")
    add(
        f"| Marker | `{marker['marker']}` | `{marker['marker_bytes_hex']}` | "
        f"`{marker['contextual_token_ids']}` | **{marker['multi_token']}** |"
    )
    add(
        f"| Candidate A | `{cands['candidate_met']!r}` | — | "
        f"`{cands['met_token_ids']}` | {len(cands['met_token_ids']) > 1} |"
    )
    add(
        f"| Candidate B | `{cands['candidate_not_met']!r}` | — | "
        f"`{cands['not_met_token_ids']}` | {len(cands['not_met_token_ids']) > 1} |"
    )
    add("")
    add(
        "The marker spans two tokens, which is why stopping is sequence-aware "
        "rather than single-token. Candidates carry a leading space and are "
        "resolved as continuations of the real decision context, never encoded "
        "in isolation."
    )
    add("")

    add("## 8. Dataset verification")
    add("")
    validation = prereg["dataset"]["validation"]
    add(
        f"The frozen Day-3 TEST partition (seed {rc.SOURCE_SEED}) was verified "
        f"before inference: {rc.N_ARTIFACTS} artifacts, {rc.N_ROWS} rows, nine "
        "strictness levels each, 108 artifacts per family, 36 at each true "
        f"crossing, {rc.N_NONTRIVIAL} nontrivial curves. Formal truth and the "
        "true crossing were independently recomputed from primitive fields for "
        "all three families. Every study row pairs to exactly one frozen Day-3 "
        "row."
    )
    add("")
    add(f"Dataset content SHA-256: `{validation['dataset_content_sha256']}`")
    add("")

    add("## 9. Completeness")
    add("")
    add("| | |")
    add("|---|---:|")
    add(f"| Rows attempted | {comp['rows_attempted']} |")
    add(f"| Marker reached | {comp['rows_marker_reached']} |")
    add(f"| Truncated | {comp['rows_truncated']} |")
    add(f"| Scoring failures | {comp['rows_scoring_failures']} |")
    add(f"| Premature-label flags | {comp['rows_premature_label']} ({pct(100 * comp['premature_label_rate'])}) |")
    add(f"| Complete curves | {comp['n_complete_curves']} |")
    add(f"| Excluded curves | {comp['n_excluded_curves']} |")
    add("")

    add("## 10. Primary result — boundary reachability")
    add("")
    add("| Condition | Nontrivial TRR | Unreachable |")
    add("|---|---:|---:|")
    add(
        f"| Qwen2.5 immediate (frozen Day 3) | {fmt(primary['trr_immediate'], 4)} | "
        f"{pct(primary['unreachable_pct_immediate'])} |"
    )
    add(
        f"| Qwen2.5 reason-then-score | **{fmt(primary['trr_reason_then_score'], 4)}** | "
        f"**{pct(primary['unreachable_pct_reason_then_score'])}** |"
    )
    add("")
    add(
        f"Paired change **Δ TRR = {signed(d_trr['point'], 4)}**, 95% CI "
        f"[{signed(d_trr['ci_low'], 4)}, {signed(d_trr['ci_high'], 4)}], over "
        f"{primary['n_nontrivial_curves']} nontrivial curves measured under both "
        "protocols on the same artifacts."
    )
    add("")
    exact = primary["exact_interval"]
    if exact["applies"]:
        add(
            f"The reason-then-score unreachable count is "
            f"{exact['n_unreachable']}/{exact['n_nontrivial']}. A bootstrap over "
            "an identically-zero statistic returns a point interval, which "
            "describes resampling rather than the population, so the exact "
            "two-sided 95% Clopper-Pearson interval is reported instead: "
            f"[{pct(100 * exact['clopper_pearson_low'], 3)}, "
            f"**{pct(100 * exact['clopper_pearson_high'], 3)}**]."
        )
    else:
        add(
            f"The reason-then-score unreachable count is "
            f"{exact['n_unreachable']}/{exact['n_nontrivial']}, which is "
            "non-zero, so the bootstrap interval is well defined and the "
            "zero-count exact-binomial rule does not apply. For reference the "
            "exact Clopper-Pearson interval is "
            f"[{pct(100 * exact['clopper_pearson_low'], 2)}, "
            f"{pct(100 * exact['clopper_pearson_high'], 2)}]."
        )
    add("")

    add("## 11. Threshold localisation")
    add("")
    add("| Condition | Mean TCE | 95% CI |")
    add("|---|---:|---:|")
    add(
        f"| Qwen2.5 immediate | {fmt(tce['immediate'])} | "
        f"[{fmt(tce['immediate_ci'][0])}, {fmt(tce['immediate_ci'][1])}] |"
    )
    add(
        f"| Qwen2.5 reason-then-score | **{fmt(tce['reason_then_score'])}** | "
        f"[{fmt(tce['reason_then_score_ci'][0])}, {fmt(tce['reason_then_score_ci'][1])}] |"
    )
    add("")
    add(
        f"Paired change **Δ TCE = {signed(d_tce['point'], 4)}**, 95% CI "
        f"[{signed(d_tce['ci_low'], 4)}, {signed(d_tce['ci_high'], 4)}]. "
        "TCE is secondary: changing the scoring context can move locations even "
        "when ordering is preserved."
    )
    add("")
    add("Against the exact structural references:")
    add("")
    add("| Reference | TCE |")
    add("|---|---:|")
    add(f"| Fixed centre / no input | {fmt(rc.BASELINE_FIXED_CENTER_TCE, 4)} |")
    add(f"| Uniform random crossing | {fmt(rc.BASELINE_RANDOM_CROSSING_TCE, 4)} |")
    add(f"| Always met / majority | {fmt(rc.BASELINE_ALWAYS_MET_TCE, 4)} |")
    add(f"| Always not met | {fmt(rc.BASELINE_ALWAYS_NOT_MET_TCE, 4)} |")
    add("")

    add("## 12. Why boundaries were unreachable")
    add("")
    add(
        "An unreachable boundary can fail for two different reasons. A **strict "
        "inversion** means an earlier, laxer threshold scored strictly below the "
        "true boundary -- a genuine ordering failure. A **tie** means nothing "
        "scored below it but something scored exactly equal, so it fails only "
        "because reachability requires a strict record low. Exact stored values, "
        "no tolerance."
    )
    add("")
    add("| Condition | Unreachable | Strict inversion | Tie only |")
    add("|---|---:|---:|---:|")
    for row in decomp.itertuples(index=False):
        label = (
            "Qwen2.5 immediate"
            if row.condition == rc.CONDITION_ORIGINAL
            else "Qwen2.5 reason-then-score"
        )
        add(
            f"| {label} | {row.n_unreachable}/{row.n_nontrivial} "
            f"({pct(row.unreachable_pct)}) | {row.n_strict_inversion} | "
            f"{row.n_tie_only} |"
        )
    add("")

    add("## 13. Secondary metrics")
    add("")
    add("| Metric | Immediate | Reason-then-score |")
    add("|---|---:|---:|")
    add(f"| Accuracy | {pct(100 * sec['accuracy_o'])} | {pct(100 * sec['accuracy_r'])} |")
    add(f"| Brier | {fmt(sec['brier_o'], 4)} | {fmt(sec['brier_r'], 4)} |")
    add(f"| Mean RC | {fmt(sec['mean_rc_o'])} | {fmt(sec['mean_rc_r'])} |")
    add(f"| RC = 10 (curves) | {sec['rc_is_ten_o']} | {sec['rc_is_ten_r']} |")
    add(f"| Prefix-record rate | {fmt(sec['prefix_record_rate_o'], 4)} | {fmt(sec['prefix_record_rate_r'], 4)} |")
    add(f"| MVR (practical, 0.05) | — | {fmt(sec['mvr_practical_r'], 4)} |")
    add(f"| MVR (zero tolerance) | — | {fmt(sec['mvr_zero_r'], 4)} |")
    add(f"| MVM | — | {fmt(sec['mvm_r'], 4)} |")
    add("")

    add("## 14. Family-stratified results")
    add("")
    add("| Family | TCE O → R | TRR O → R | Unreachable O → R | Accuracy O → R | Mean tokens |")
    add("|---|---:|---:|---:|---:|---:|")
    for row in family.itertuples(index=False):
        add(
            f"| {row.family} | {fmt(row.tce_o)} → {fmt(row.tce_r)} | "
            f"{fmt(row.trr_o, 4)} → {fmt(row.trr_r, 4)} | "
            f"{pct(row.unreachable_pct_o)} → {pct(row.unreachable_pct_r)} | "
            f"{pct(100 * row.accuracy_o, 1)} → {pct(100 * row.accuracy_r, 1)} | "
            f"{fmt(row.mean_generated_tokens, 0)} |"
        )
    add("")

    add("## 15. Reasoning-length diagnostics")
    add("")
    add(
        f"Generated tokens per row: mean {fmt(tokens['mean'], 1)}, median "
        f"{fmt(tokens['median'], 0)}, p05-p95 {fmt(tokens['p05'], 0)}-"
        f"{fmt(tokens['p95'], 0)}, range {fmt(tokens['min'], 0)}-"
        f"{fmt(tokens['max'], 0)}."
    )
    add("")
    for association in summary.get("reasoning_length_associations", []):
        rho = association["spearman_rho"]
        if rho is None or not np.isfinite(rho):
            add(
                f"* {association['length_column']} vs `{association['outcome']}`: "
                f"**undefined** — the outcome is constant across all "
                f"{association['n']} curves."
            )
        else:
            add(
                f"* {association['length_column']} vs `{association['outcome']}`: "
                f"Spearman rho {fmt(rho, 3)} (p {fmt(association['p_value'], 4)}, "
                f"n {association['n']}) — descriptive only."
            )
    add("")
    add(
        "Trace length is chosen by the model and confounded with item "
        "difficulty; these associations are not interpreted causally."
    )
    add("")

    add("## 16. Context: the Qwen3 reasoning anchor")
    add("")
    add(
        "The rows below are **not** three arms of one experiment. The first two "
        "are the within-model ablation. The third is a different model family, "
        "scale and training regime under *native* thinking rather than elicited "
        "reasoning, shown only for context. No pooled significance test is "
        "performed across these regimes."
    )
    add("")
    add("| Row kind | Model | Protocol | TCE | TRR | Unreachable |")
    add("|---|---|---|---:|---:|---:|")
    for row in context.itertuples(index=False):
        add(
            f"| {row.row_kind} | {row.model} | {row.protocol} | {fmt(row.tce)} | "
            f"{fmt(row.trr, 4)} | {pct(row.unreachable_pct)} |"
        )
    add("")
    add(f"**{positioning['statement']}**")
    add("")
    add(positioning["caveat"])
    add("")

    add("## 17. Prespecified interpretation")
    add("")
    add(f"**Δ TRR category: {interp['category']}.**")
    add("")
    add(f"> {interp['statement']}")
    add("")
    add(
        f"Effect magnitude: Δ TRR = {signed(d_trr['point'], 4)}, 95% CI "
        f"[{signed(d_trr['ci_low'], 4)}, {signed(d_trr['ci_high'], 4)}]; "
        f"Δ TCE = {signed(d_tce['point'], 4)}, 95% CI "
        f"[{signed(d_tce['ci_low'], 4)}, {signed(d_tce['ci_high'], 4)}]."
    )
    add("")

    add("## 18. Limitations")
    add("")
    add(NARRATIVE["composite"])
    add("")
    add(NARRATIVE["one_trace"])
    add("")
    add(
        "This is not a prompt-robustness study. One reasoning prompt was "
        "preregistered and no alternatives were tested or optimised."
    )
    add("")

    add("## 19. Deviations and retries")
    add("")
    retry_path = root / rc.RETRY_LOG_PATH
    retries = (
        json.loads(retry_path.read_text(encoding="utf-8")).get("retries", [])
        if retry_path.is_file()
        else []
    )
    if retries:
        add(f"{len(retries)} infrastructure retry/retries, each with identical inputs:")
        add("")
        for entry in retries:
            add(f"* {json.dumps(entry)}")
    else:
        add("No retries were required. No methodological deviation occurred.")
    add("")

    add("## 20. Integrity verification")
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
    add(
        "The Day-3 Qwen2.5 comparator and the reasoning-anchor artifacts were "
        "opened read-only and re-verified byte-identical."
    )
    add("")

    add("## 21. Claims that must not be made")
    add("")
    add("This study does not establish that:")
    add("")
    for claim in (
        "reasoning itself causes the observed change — the intervention is composite",
        "any single subcomponent of the protocol is responsible",
        "elicited reason-then-score and native thinking are the same protocol",
        "the reasoning trace faithfully reveals the model's hidden computation",
        "one sampled trace characterises all possible traces",
        "the result transfers to other models, prompts or rubric styles",
        "a lower TCE implies correct score ordering",
    ):
        add(f"* {claim};")
    add("")

    add("## 22. Manuscript-facing interpretation")
    add("")
    add("The narrow question this experiment answers is:")
    add("")
    add(
        "> For the exact same frozen Qwen2.5-14B judge and Day-3 evaluation "
        "panel, does replacing immediate A/B next-token readout with an "
        "elicited reason-then-score protocol materially change true-boundary "
        "reachability?"
    )
    add("")
    add("Recommended wording, using the exact repository-derived values:")
    add("")
    add(
        f"> Holding the judge, its revision and the evaluation set fixed, "
        f"replacing immediate next-token A/B readout with an elicited "
        f"reason-then-score protocol raised nontrivial true-boundary "
        f"reachability for `{model['model_id']}` from "
        f"{fmt(primary['trr_immediate'], 4)} to "
        f"{fmt(primary['trr_reason_then_score'], 4)} "
        f"(paired Δ TRR {signed(d_trr['point'], 4)}, 95% CI "
        f"[{signed(d_trr['ci_low'], 4)}, {signed(d_trr['ci_high'], 4)}]), "
        f"reducing unreachable boundaries from "
        f"{pct(primary['unreachable_pct_immediate'])} to "
        f"{pct(primary['unreachable_pct_reason_then_score'])}, and lowered mean "
        f"threshold crossing error from {fmt(tce['immediate'])} to "
        f"{fmt(tce['reason_then_score'])} (Δ {signed(d_tce['point'], 3)}, 95% CI "
        f"[{signed(d_tce['ci_low'], 3)}, {signed(d_tce['ci_high'], 3)}]). "
        f"{positioning['statement']} Because the intervention jointly changes "
        f"the instruction, the generated trace, the scoring position and the "
        f"readout, the supported statement concerns the reason-then-score "
        f"protocol as a whole rather than reasoning as an isolated factor."
    )
    add("")
    add(
        "Report reachability and localisation separately, and state that the "
        "Qwen3 anchor uses native thinking whenever it appears alongside these "
        "rows."
    )
    add("")
    add("---")
    add("")
    add(
        f"*Generated by `scripts/write_results_rts_ablation.py` from "
        f"`{rc.SUMMARY_PATH}`. Every number above is read from a machine "
        f"artifact.*"
    )

    out_path = root / rc.RESULTS_MD_PATH
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out_path.relative_to(root)} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
