#!/usr/bin/env python
"""Freeze the FormalCRRC Day-1 preregistration (Phase D).

Builds ``artifacts/day1/preregistration.json`` from the central configuration
object, so the frozen document and the executing code cannot drift apart, and
writes ``artifacts/day1/preregistration.sha256``.

Refuses to overwrite an existing preregistration. Once frozen the document may
only be corrected through a numbered, checksummed amendment that preserves the
original; see ``docs/PREREGISTRATION_DAY1.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import predicates, prompts, provenance  # noqa: E402
from formalcrrc.bootstrap import BOOTSTRAP_METRICS  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    ANALYSIS_CODE_PATHS,
    BOOTSTRAP_CI,
    BOOTSTRAP_METHOD,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    BOOTSTRAP_STRATIFY_BY,
    BOOTSTRAP_UNIT,
    DATASET_MANIFEST_PATH,
    DECISION_THRESHOLD,
    FAMILIES,
    INFERENCE_BATCH_SIZE,
    INFERENCE_ORDER_SALT,
    LABEL_MET,
    LABEL_NOT_MET,
    LIST_SIZE,
    MARKERS_PER_RESPONSE,
    MODEL_DTYPE,
    MODEL_IDS,
    MODEL_ORDER_SALT,
    MODEL_PROVENANCE_PATH,
    MVR_TOLERANCE,
    MVR_TOLERANCE_STRICT,
    N_ARTIFACTS,
    N_INSTANCES,
    N_LATENT_LEVELS,
    N_ROWS,
    N_STRICTNESS,
    NO_CROSSING_INDEX,
    NUMERIC_TARGET_MAX,
    NUMERIC_TARGET_MIN,
    PREREG_JSON_PATH,
    PREREG_SHA_PATH,
    PROMPT_TEMPLATE,
    SCALE_MAX,
    SCORING_FALLBACK,
    SCORING_LABEL_RESOLUTION,
    SCORING_PRIMARY,
    SEED,
)

RESEARCH_QUESTION = (
    "When evaluation criteria differ only by a formally controlled decision "
    "threshold, do LLM judges exhibit the monotone counterfactual response "
    "function implied by the rubric?"
)

HYPOTHESES = [
    {
        "id": "H1",
        "statement": (
            "For a fixed artifact evaluated under an ordered family of formal "
            "thresholds, judge probability P_met is non-increasing in the "
            "strictness index s."
        ),
        "operationalisation": (
            "Artifact-level MVR at tolerance 0.05 is the primary measure; a "
            "judge consistent with H1 has MVR near 0."
        ),
        "directional": True,
    },
    {
        "id": "H2",
        "statement": (
            "The judge's 0.5 crossing point coincides with the formal first-fail "
            "index j* known by construction."
        ),
        "operationalisation": "Artifact-level TCE; a judge consistent with H2 has TCE near 0.",
        "directional": True,
    },
    {
        "id": "H3",
        "statement": (
            "Curve fidelity and aggregate binary accuracy can diverge: a judge "
            "may score high accuracy while violating monotonicity or crossing at "
            "the wrong threshold."
        ),
        "operationalisation": (
            "Joint reporting of accuracy against MVR/MVM/TCE. This is an "
            "observational contrast, not a directional prediction."
        ),
        "directional": False,
    },
    {
        "id": "H4",
        "statement": (
            "Curve fidelity may depend on the predicate family, in particular "
            "between count-based and arithmetic-tolerance predicates."
        ),
        "operationalisation": "All primary metrics reported per family; no averaging away.",
        "directional": False,
    },
]

NOVELTY_BOUNDARIES = {
    "claimed_contribution": (
        "An experimental protocol in which a fixed evaluated artifact is paired "
        "with a programmatically parameterized family of evaluation thresholds, "
        "producing a counterfactual rubric response curve whose monotonicity, "
        "threshold crossing and sensitivity can be compared with behaviour known "
        "exactly by construction."
    ),
    "explicitly_not_claimed": [
        "first rubric perturbation method",
        "first counterfactual evaluation of LLM judges",
        "first strict-vs-lenient rubric experiment",
        "first multi-level strictness experiment",
        "first judge robustness curve",
        "first monotonicity analysis of an LLM evaluator",
        "first threshold-based judge evaluation",
        "first use of controlled rubric changes",
        "first continuous LLM judge reliability measure",
    ],
    "collision_areas_constraining_claims": [
        "policy/rubric-threshold invariance work (Weng, Feng & Xie, 2026)",
        "FlexGuard / FlexBench",
        "RubricRobustness",
        "IRT-based analyses of LLM judge reliability",
        "rubric counterfactual/reversal work (Bagaria et al.)",
        "RuVerBench",
        "formal or certified judge robustness work such as 'Certifying the Judge'",
    ],
    "bibliographic_status": (
        "Collision areas are recorded by research direction only. Exact "
        "bibliographic metadata is not asserted in this document and must be "
        "verified from authoritative public sources before being recorded "
        "anywhere in the repository."
    ),
    "day1_does_not_establish": [
        "human alignment",
        "clinical reliability",
        "HealthBench reliability",
        "general semantic rubric reliability",
        "reward-model correctness",
        "causal reasoning ability",
        "robustness to arbitrary natural-language rubrics",
        "reliability of free-form semantic relaxation",
        "general superiority of any model family",
    ],
}

EXCLUSION_RULES = {
    "artifact_exclusions": "None. All 324 artifacts enter every analysis.",
    "threshold_exclusions": "None. All 9 strictness levels enter every analysis.",
    "row_exclusions": (
        "A row is excluded only if its scoring produced a non-finite label "
        "score, which would be an implementation defect rather than a result. "
        "Any such exclusion must be reported with an exact count and the "
        "affected prompt_ids, and blocks the affected model's analysis."
    ),
    "model_exclusions": (
        "A model is excluded only when it is inaccessible before any FormalCRRC "
        "study outcome has been generated for it, and is then recorded as "
        "BLOCKED_BY_MODEL_ACCESS. No model may be dropped, added or substituted "
        "on the basis of observed FormalCRRC performance."
    ),
    "no_filtering_of_hard_artifacts": True,
    "no_selection_of_latent_levels": True,
}

FAILURE_HANDLING = {
    "incomplete_model_run": (
        "If a model's raw scores do not contain all 2916 prompts exactly once, "
        "that model's analysis is BLOCKED and reported as such. Partial results "
        "are not analysed as if complete."
    ),
    "model_access_failure": (
        "Recorded as BLOCKED_BY_MODEL_ACCESS. No substitute model is chosen. "
        "A three-model panel is never described as the specified four-model panel."
    ),
    "implementation_defect": (
        "Stop the affected analysis, document the defect, preserve the defective "
        "artifacts, create a numbered preregistration amendment if the protocol "
        "is affected, and rerun only under an explicitly documented correction. "
        "Low judge performance is never treated as a defect."
    ),
    "no_outcome_driven_tuning": (
        "After inference begins, no change may be made to thresholds, strictness "
        "levels, the 0.05 MVR tolerance, the 0.5 crossing boundary, the prompt, "
        "the answer labels, the model panel, the predicate families, the "
        "bootstrap units, or the metric definitions; and no post-hoc "
        "calibration, temperature fitting, isotonic regression, monotonicity "
        "enforcement, curve correction, selective rerunning or run averaging is "
        "permitted."
    ),
}

FIGURE_PLAN = [
    {
        "id": "figure1",
        "title": "FormalCRRC concept",
        "content": (
            "Schematic: fixed artifact -> ordered rubric thresholds -> judge "
            "probabilities -> response curve."
        ),
        "path": "figures/day1/figure1_concept.png",
    },
    {
        "id": "figure2",
        "title": "Boundary-aligned response curves",
        "content": (
            "Mean P_met against signed distance d = s - j* from the true "
            "crossing, per judge, with uncertainty."
        ),
        "path": "figures/day1/figure2_boundary_aligned_curves.png",
    },
    {
        "id": "figure3",
        "title": "MVR distributions",
        "content": "Per-model distribution of artifact-level MVR.",
        "path": "figures/day1/figure3_mvr_distributions.png",
    },
    {
        "id": "figure4",
        "title": "TCE distributions",
        "content": "Per-model distribution of threshold crossing error.",
        "path": "figures/day1/figure4_tce_distributions.png",
    },
    {
        "id": "figure5",
        "title": "Model x predicate-family summary",
        "content": "Heatmap of primary metrics by model and predicate family.",
        "path": "figures/day1/figure5_model_family_heatmap.png",
    },
]

STOPPING_RULE = {
    "rule": (
        "Inference stops when every preregistered judge has produced exactly "
        "2916 scored prompts, or has been formally marked "
        "BLOCKED_BY_MODEL_ACCESS. The preregistered analysis is then run exactly "
        "once."
    ),
    "no_interim_analysis": (
        "No monotonicity, crossing, accuracy or ranking summary is computed or "
        "displayed while inference is running. Console output during inference "
        "is restricted to operational information."
    ),
    "no_data_dependent_extension": (
        "The design is not extended, repeated or enlarged on the basis of "
        "observed results. No Day-2 experiment follows automatically."
    ),
    "success_criterion": (
        "Day 1 succeeds when dataset integrity passes, the preregistration was "
        "frozen before outcomes, intended runs completed or were transparently "
        "blocked, probability extraction is valid, the analysis matches this "
        "document, and all results are reported regardless of direction. No "
        "favourable-effect threshold is defined."
    ),
}


def build_document(root: Path) -> dict:
    """Assemble the preregistration from the central configuration."""
    dataset_manifest_path = root / DATASET_MANIFEST_PATH
    if not dataset_manifest_path.is_file():
        raise SystemExit(
            "dataset manifest not found; run scripts/generate_day1_dataset.py first"
        )
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))

    model_provenance_path = root / MODEL_PROVENANCE_PATH
    revisions: dict[str, str | None] = {m: None for m in MODEL_IDS}
    scoring_methods: dict[str, str | None] = {m: None for m in MODEL_IDS}
    panel_status: dict[str, str] = {m: "NOT_CHECKED" for m in MODEL_IDS}
    if model_provenance_path.is_file():
        provenance_doc = json.loads(
            model_provenance_path.read_text(encoding="utf-8")
        )
        for model_id, record in provenance_doc.get("models", {}).items():
            if model_id in revisions:
                revisions[model_id] = record.get("revision")
                scoring_methods[model_id] = (
                    record.get("label_tokenization", {}) or {}
                ).get("scoring_method")
                panel_status[model_id] = record.get("status", "NOT_CHECKED")

    return {
        "document": "FormalCRRC Day-1 preregistration",
        "version": 1,
        "frozen_at": provenance.utc_now(),
        "amendments": [],
        "research_question": RESEARCH_QUESTION,
        "hypotheses": HYPOTHESES,
        "novelty_boundaries": NOVELTY_BOUNDARIES,
        "design": {
            "type": "within-artifact counterfactual threshold ladder",
            "fixed_factor": "candidate_response (byte-identical across thresholds)",
            "manipulated_factor": "formal decision threshold printed in the rubric",
            "strictness_index_range": [0, SCALE_MAX],
            "n_strictness_levels": N_STRICTNESS,
            "most_permissive_strictness": 0,
            "most_strict_strictness": SCALE_MAX,
            "ground_truth_source": (
                "arithmetic predicate over construction-time quantities; no "
                "language model, validator, consensus panel or human annotation "
                "participates in labelling"
            ),
            "human_involvement": "none",
        },
        "predicate_families": {
            "coverage": {
                "code": "A",
                "latent_quantity": "m, number of Required Items present (0..8)",
                "rubric_threshold": "at least s",
                "raw_threshold_formula": "s",
                "truth": "1[m >= s]",
                "true_first_fail_index": "m + 1, or 9 when m = 8",
                "list_size": LIST_SIZE,
                "markers_per_response": MARKERS_PER_RESPONSE,
            },
            "max_violation": {
                "code": "B",
                "latent_quantity": "v, number of Prohibited Items present (0..8)",
                "rubric_threshold": "at most 8 - s",
                "raw_threshold_formula": "8 - s",
                "truth": "1[v <= 8 - s]",
                "true_first_fail_index": "9 - v",
                "list_size": LIST_SIZE,
                "markers_per_response": MARKERS_PER_RESPONSE,
            },
            "numeric_tolerance": {
                "code": "C",
                "latent_quantity": "e = |V - T|, absolute error (0..8)",
                "rubric_threshold": "at most 8 - s",
                "raw_threshold_formula": "8 - s",
                "truth": "1[|V - T| <= 8 - s]",
                "true_first_fail_index": "9 - e",
                "target_range": [NUMERIC_TARGET_MIN, NUMERIC_TARGET_MAX],
                "error_signs": "alternating by instance index; 0 when e = 0",
            },
        },
        "threshold_values": {
            family: {
                str(s): predicates.raw_threshold(family, s)
                for s in range(N_STRICTNESS)
            }
            for family in FAMILIES
        },
        "true_first_fail_index_by_cell": dataset_manifest[
            "true_first_fail_index_by_cell"
        ],
        "dataset": {
            "n_families": len(FAMILIES),
            "families": list(FAMILIES),
            "n_latent_levels": N_LATENT_LEVELS,
            "n_instances_per_cell": N_INSTANCES,
            "n_artifacts": N_ARTIFACTS,
            "n_strictness_levels": N_STRICTNESS,
            "n_rubric_response_pairs": N_ROWS,
            "path": "artifacts/day1/formal_dataset.parquet",
            "content_sha256": dataset_manifest["hashes"]["dataset_content_sha256"],
            "file_sha256": dataset_manifest["hashes"]["dataset_file_sha256"],
            "prompt_manifest_file_sha256": dataset_manifest["hashes"][
                "prompt_manifest_file_sha256"
            ],
            "rubric_template_sha256": dataset_manifest["hashes"][
                "rubric_template_sha256"
            ],
        },
        "seeds": {
            "master_seed": SEED,
            "dataset_construction": (
                "numpy.default_rng([SEED, family_index, latent_level, "
                "instance_index]) / PCG64"
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "inference_order_salt": INFERENCE_ORDER_SALT,
            "model_order_salt": MODEL_ORDER_SALT,
        },
        "judge_models": {
            "model_ids": list(MODEL_IDS),
            "panel_size": len(MODEL_IDS),
            "dtype": MODEL_DTYPE,
            "quantisation": "none",
            "fine_tuning": "none",
            "adapters": "none",
            "calibration_layer": "none",
            "prompt_optimisation": "none",
            "resolved_revisions": revisions,
            "resolved_scoring_methods": scoring_methods,
            "panel_status_at_freeze": panel_status,
            "run_order": prompts.model_run_order(),
            "selection_rule": (
                "The panel is fixed in advance and may not be changed on the "
                "basis of observed FormalCRRC performance."
            ),
        },
        "prompt": {
            "template": PROMPT_TEMPLATE,
            "template_sha256": prompts.prompt_template_sha256(),
            "message_structure": "single user message; no system message",
            "chat_template": "each model's native chat template, rendered hash recorded",
            "label_met": LABEL_MET,
            "label_not_met": LABEL_NOT_MET,
            "prohibited": [
                "chain-of-thought prompting",
                "few-shot examples",
                "self-explanation requests",
                "model-specific reasoning instructions",
                "persona or role framing",
                "temperature or verbosity variation between models",
            ],
            "only_study_variable_text": "the numeric rubric threshold",
        },
        "scoring": {
            "label_token_resolution": {
                "rule": SCORING_LABEL_RESOLUTION,
                "definition": (
                    "The scored token ids for a label are the continuation ids "
                    "encode(rendered_prompt + label) minus encode(rendered_prompt), "
                    "both with add_special_tokens=False. Encoding the bare label "
                    "string is NOT used: a SentencePiece tokenizer returns a "
                    "word-boundary-marked token for a string encoded in "
                    "isolation, which is a different id from the one that "
                    "actually continues a prompt."
                ),
                "reference_prompt": (
                    "a message with the study prompt's chat-template structure "
                    "and placeholder content (<RUBRIC>, <RESPONSE>), so the "
                    "resolution depends on no rubric, threshold or candidate "
                    "response"
                ),
                "stability_verification": (
                    "scripts/verify_label_tokens.py confirms, by tokenizer "
                    "inspection alone and with no model call, that the resolved "
                    "ids are identical for the first 25 prompts of the frozen "
                    "inference order; recorded in "
                    "artifacts/day1/label_token_check.json"
                ),
                "observed_naive_encoding_mismatch": (
                    "mistralai/Mistral-7B-Instruct-v0.3 encodes 'A' as the "
                    "word-boundary token '_A' (id 1098) but continues its prompt "
                    "with 'A' (id 29509). Detected before this preregistration "
                    "was frozen and before any study inference."
                ),
            },
            "primary_rule": SCORING_PRIMARY,
            "primary_formula": "P_met = exp(z_A) / (exp(z_A) + exp(z_B))",
            "primary_condition": (
                "both 'A' and 'B' continue the rendered prompt with exactly one "
                "token under the model's own tokenizer"
            ),
            "fallback_rule": SCORING_FALLBACK,
            "fallback_formula": "P_met = exp(l_A) / (exp(l_A) + exp(l_B))",
            "fallback_condition": (
                "single-token continuation fails for at least one label; applied "
                "per model, decided from tokenizer behaviour before any study "
                "prompt is scored"
            ),
            "sampling": "none; no generation, no temperature",
            "recorded_per_row": [
                "raw_score_met",
                "raw_score_not_met",
                "p_met",
                "scoring_method",
                "n_prompt_tokens",
                "rendered_prompt_sha256",
            ],
            "selection_rule": (
                "The scoring rule is determined by tokenisation alone and may "
                "never be chosen on the basis of the curves it produces."
            ),
        },
        "inference": {
            "batch_size": INFERENCE_BATCH_SIZE,
            "batching_rationale": (
                "one prompt per forward pass, so no score depends on padding or "
                "batch composition"
            ),
            "order": (
                "prompts are sorted by sha256(INFERENCE_ORDER_SALT|prompt_id), "
                "frozen in artifacts/day1/prompt_manifest.parquet before any GPU "
                "inference; family, latent level and strictness are therefore "
                "interleaved"
            ),
            "model_order": (
                "models are sorted by sha256(MODEL_ORDER_SALT|model_id)"
            ),
            "independence": "each rubric-response pair is scored independently",
            "console_output": (
                "operational only: progress, processed count, failures, timing, "
                "memory. No accuracy, MVR, MVM, TCE, ranking or curve summary is "
                "printed during inference."
            ),
        },
        "primary_metrics": {
            "MVR": {
                "definition": (
                    "MVR_x = (1/8) * sum_{s=0}^{7} 1[P_x(s+1) > P_x(s) + delta]"
                ),
                "tolerance_delta": MVR_TOLERANCE,
                "unit": "artifact",
            },
            "MVM": {
                "definition": (
                    "MVM_x = (1/8) * sum_{s=0}^{7} max(0, P_x(s+1) - P_x(s))"
                ),
                "unit": "artifact",
            },
            "TCE": {
                "definition": "TCE_x = |j_hat_x - j*_x|",
                "predicted_boundary": (
                    "j_hat_x = min{s : P_x(s) < 0.5}, or 9 if never below"
                ),
                "true_boundary": "j*_x = min{s : Y_x(s) = 0}, or 9 if never 0",
                "decision_threshold": DECISION_THRESHOLD,
                "no_crossing_index": NO_CROSSING_INDEX,
                "unit": "artifact",
                "reported": [
                    "mean",
                    "median",
                    "distribution",
                    "exact-boundary rate (TCE = 0)",
                    "rate within one threshold step (TCE <= 1)",
                ],
            },
        },
        "secondary_metrics": {
            "MVR_zero_tolerance": {
                "definition": "MVR at delta = 0",
                "tolerance_delta": MVR_TOLERANCE_STRICT,
            },
            "binary_accuracy": "mean 1[(P_met >= 0.5) == Y] over all cells",
            "brier_score": "mean (P_met - Y)^2 over all cells",
            "boundary_relative_curve": (
                "mean P_met as a function of signed distance d = s - j*"
            ),
            "spearman_threshold_sensitivity": (
                "within-artifact Spearman association between s and P_met; "
                "expected direction non-positive; undefined (nan) for a constant "
                "curve, in which case the artifact is omitted from the mean and "
                "the omitted count is reported. Descriptive support only, never "
                "a replacement for MVR/TCE."
            ),
            "family_specific_performance": (
                "every metric reported separately for coverage, max_violation "
                "and numeric_tolerance; family-specific failures are not hidden "
                "behind overall averages"
            ),
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": BOOTSTRAP_UNIT,
            "cluster_rule": "all 9 thresholds of an artifact are carried together",
            "stratify_by": BOOTSTRAP_STRATIFY_BY,
            "interval_method": BOOTSTRAP_METHOD,
            "coverage": BOOTSTRAP_CI,
            "metrics": list(BOOTSTRAP_METRICS),
            "paired_comparisons": (
                "judge-vs-judge differences use the identical resampled artifact "
                "sets for both judges; difference estimates and 95% intervals "
                "are reported"
            ),
            "prohibited": [
                "treating the 2916 threshold cells as independent observations",
                "asymptotic standard errors that ignore within-artifact dependence",
                "selecting any hyperparameter from bootstrap outcomes",
            ],
        },
        "analysis_code_paths": list(ANALYSIS_CODE_PATHS),
        "exclusion_rules": EXCLUSION_RULES,
        "failure_handling": FAILURE_HANDLING,
        "figure_plan": FIGURE_PLAN,
        "stopping_rule": STOPPING_RULE,
        "source_manifest_at_freeze": provenance.source_manifest(
            ANALYSIS_CODE_PATHS, root=root
        ),
        "git_at_freeze": provenance.git_state(root),
    }


def _bullets(items) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_markdown(document: dict, digest: str) -> str:
    """Render the human-readable preregistration from the frozen JSON.

    Both files are generated from the same object, so the prose and the
    machine-readable document cannot drift apart.
    """
    models = document["judge_models"]
    scoring = document["scoring"]
    primary = document["primary_metrics"]
    boot = document["bootstrap"]
    novelty = document["novelty_boundaries"]

    revision_rows = "\n".join(
        f"| `{model_id}` | `{models['resolved_revisions'].get(model_id) or 'UNRESOLVED'}` "
        f"| {models['resolved_scoring_methods'].get(model_id) or '—'} "
        f"| {models['panel_status_at_freeze'].get(model_id, 'NOT_CHECKED')} |"
        for model_id in models["model_ids"]
    )

    threshold_rows = "\n".join(
        "| "
        + str(s)
        + " | "
        + " | ".join(
            str(document["threshold_values"][family][str(s)])
            for family in document["dataset"]["families"]
        )
        + " |"
        for s in range(N_STRICTNESS)
    )

    hypothesis_rows = "\n".join(
        f"| **{h['id']}** | {h['statement']} | {h['operationalisation']} |"
        for h in document["hypotheses"]
    )

    figure_rows = "\n".join(
        f"| {f['id']} | {f['title']} | {f['content']} | `{f['path']}` |"
        for f in document["figure_plan"]
    )

    return f"""# FormalCRRC Day-1 preregistration

**Frozen:** {document['frozen_at']}
**SHA-256:** `{digest}`
**Machine-readable copy:** [`artifacts/day1/preregistration.json`](../artifacts/day1/preregistration.json)
**Checksum:** [`artifacts/day1/preregistration.sha256`](../artifacts/day1/preregistration.sha256)
**Amendments:** {len(document['amendments'])}

This document was frozen **before any FormalCRRC study inference**. It is
generated from the same object as the JSON copy, so the two cannot disagree.

> Once frozen, this preregistration is never silently edited. A genuine
> implementation defect is corrected by a **numbered amendment** that explains
> the defect, explains why the correction is outcome-independent, preserves the
> original unchanged, and carries its own checksum. **No amendment may be
> justified by weak, ugly, or unexpected scientific results.**

---

## 1. Research question

> {document['research_question']}

## 2. Hypotheses

| | Statement | Operationalisation |
|---|---|---|
{hypothesis_rows}

## 3. Novelty boundaries

**Claimed contribution.** {novelty['claimed_contribution']}

**Explicitly not claimed:**

{_bullets(novelty['explicitly_not_claimed'])}

**Collision areas that constrain the claim:**

{_bullets(novelty['collision_areas_constraining_claims'])}

{novelty['bibliographic_status']}

**Day 1 does not establish:**

{_bullets(novelty['day1_does_not_establish'])}

## 4. Design

- Type: {document['design']['type']}
- Fixed factor: {document['design']['fixed_factor']}
- Manipulated factor: {document['design']['manipulated_factor']}
- Strictness index: {document['design']['strictness_index_range'][0]} … {document['design']['strictness_index_range'][1]} ({document['design']['n_strictness_levels']} levels)
- Ground truth: {document['design']['ground_truth_source']}
- Human involvement: **{document['design']['human_involvement']}**

## 5. Dataset dimensions

| | |
|---|---|
| Predicate families | {document['dataset']['n_families']} — {', '.join(document['dataset']['families'])} |
| Latent levels per family | {document['dataset']['n_latent_levels']} |
| Instances per cell | {document['dataset']['n_instances_per_cell']} |
| **Artifacts** | **{document['dataset']['n_artifacts']}** |
| Thresholds per artifact | {document['dataset']['n_strictness_levels']} |
| **Rubric-response pairs** | **{document['dataset']['n_rubric_response_pairs']}** |
| Dataset content SHA-256 | `{document['dataset']['content_sha256']}` |
| Dataset file SHA-256 | `{document['dataset']['file_sha256']}` |
| Prompt manifest SHA-256 | `{document['dataset']['prompt_manifest_file_sha256']}` |

## 6. Predicate families

| Family | Latent quantity | Rubric threshold | Formal truth | j* |
|---|---|---|---|---|
| **A** coverage | {document['predicate_families']['coverage']['latent_quantity']} | {document['predicate_families']['coverage']['rubric_threshold']} | `{document['predicate_families']['coverage']['truth']}` | {document['predicate_families']['coverage']['true_first_fail_index']} |
| **B** max violation | {document['predicate_families']['max_violation']['latent_quantity']} | {document['predicate_families']['max_violation']['rubric_threshold']} | `{document['predicate_families']['max_violation']['truth']}` | {document['predicate_families']['max_violation']['true_first_fail_index']} |
| **C** numeric tolerance | {document['predicate_families']['numeric_tolerance']['latent_quantity']} | {document['predicate_families']['numeric_tolerance']['rubric_threshold']} | `{document['predicate_families']['numeric_tolerance']['truth']}` | {document['predicate_families']['numeric_tolerance']['true_first_fail_index']} |

### Threshold values printed in the rubric

| strictness s | coverage | max_violation | numeric_tolerance |
|---|---|---|---|
{threshold_rows}

## 7. Seeds

| | |
|---|---|
| Master seed | `{document['seeds']['master_seed']}` |
| Dataset construction | `{document['seeds']['dataset_construction']}` |
| Bootstrap seed | `{document['seeds']['bootstrap_seed']}` |
| Inference order salt | `{document['seeds']['inference_order_salt']}` |
| Model order salt | `{document['seeds']['model_order_salt']}` |

## 8. Judge models

dtype `{models['dtype']}`; quantisation {models['quantisation']}; fine-tuning
{models['fine_tuning']}; adapters {models['adapters']}; calibration layer
{models['calibration_layer']}; prompt optimisation {models['prompt_optimisation']}.

| Model | Revision at freeze | Scoring rule | Status at freeze |
|---|---|---|---|
{revision_rows}

{models['selection_rule']}

Run order (hash-derived): {', '.join(f'`{m}`' for m in models['run_order'])}

## 9. Prompt

SHA-256 `{document['prompt']['template_sha256']}` — {document['prompt']['message_structure']}; {document['prompt']['chat_template']}.

```
{document['prompt']['template']}
```

Answer labels: `{document['prompt']['label_met']}` = met, `{document['prompt']['label_not_met']}` = not met.
Only study-variable text: {document['prompt']['only_study_variable_text']}.

Prohibited:

{_bullets(document['prompt']['prohibited'])}

## 10. Probability extraction

**Primary rule — `{scoring['primary_rule']}`**

`{scoring['primary_formula']}`

Applied when {scoring['primary_condition']}.

**Fallback rule — `{scoring['fallback_rule']}`**

`{scoring['fallback_formula']}`

Applied when {scoring['fallback_condition']}.

Sampling: {scoring['sampling']}. Recorded per row: {', '.join(f'`{c}`' for c in scoring['recorded_per_row'])}.

{scoring['selection_rule']}

## 11. Inference

- Batch size {document['inference']['batch_size']} — {document['inference']['batching_rationale']}
- Order: {document['inference']['order']}
- Model order: {document['inference']['model_order']}
- {document['inference']['independence']}
- Console output: {document['inference']['console_output']}

## 12. Primary metrics

**MVR** — `{primary['MVR']['definition']}` with δ = {primary['MVR']['tolerance_delta']}, per {primary['MVR']['unit']}.

**MVM** — `{primary['MVM']['definition']}`, per {primary['MVM']['unit']}.

**TCE** — `{primary['TCE']['definition']}` where `{primary['TCE']['predicted_boundary']}` and `{primary['TCE']['true_boundary']}`, decision threshold {primary['TCE']['decision_threshold']}.

Reported: {', '.join(primary['TCE']['reported'])}.

## 13. Secondary metrics

{chr(10).join(f'- **{k}** — {v if isinstance(v, str) else v.get("definition", "")}' for k, v in document['secondary_metrics'].items())}

## 14. Bootstrap

| | |
|---|---|
| Replicates | {boot['replicates']} |
| Seed | {boot['seed']} |
| Resampling unit | `{boot['unit']}` |
| Cluster rule | {boot['cluster_rule']} |
| Stratification | {boot['stratify_by']} |
| Interval method | {boot['interval_method']} at {boot['coverage']:.0%} |
| Metrics | {', '.join(boot['metrics'])} |

{boot['paired_comparisons']}

Prohibited:

{_bullets(boot['prohibited'])}

## 15. Exclusion rules

{chr(10).join(f'- **{k}** — {v}' for k, v in document['exclusion_rules'].items())}

## 16. Failure handling

{chr(10).join(f'- **{k}** — {v}' for k, v in document['failure_handling'].items())}

## 17. Figure plan

| | Title | Content | Path |
|---|---|---|---|
{figure_rows}

Which models and families appear is fixed by the design, never by observed
performance.

## 18. Stopping rule

{chr(10).join(f'- **{k}** — {v}' for k, v in document['stopping_rule'].items())}

## 19. Analysis code

{_bullets(f'`{p}`' for p in document['analysis_code_paths'])}

Source tree SHA-256 at freeze: `{document['source_manifest_at_freeze']['tree_sha256']}`

Git at freeze: commit `{(document['git_at_freeze'].get('commit') or 'unavailable')}`, clean = {document['git_at_freeze'].get('clean')}

---

## Amendment log

No amendments. Any future amendment is appended here as a numbered section,
preserving everything above unchanged, and is separately checksummed.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = Path(args.root)

    target = root / PREREG_JSON_PATH
    sha_path = root / PREREG_SHA_PATH
    if target.exists():
        print(
            f"REFUSING to overwrite the frozen preregistration at {PREREG_JSON_PATH}.\n"
            "A frozen preregistration may only be corrected by a numbered, "
            "checksummed amendment that preserves the original.",
            flush=True,
        )
        return 1

    document = build_document(root)
    provenance.write_json(target, document)
    digest = provenance.write_checksum_file(target, sha_path)

    markdown_path = root / "docs/PREREGISTRATION_DAY1.md"
    markdown_path.write_text(render_markdown(document, digest), encoding="utf-8")

    print(f"wrote {PREREG_JSON_PATH}", flush=True)
    print(f"wrote {PREREG_SHA_PATH}", flush=True)
    print("wrote docs/PREREGISTRATION_DAY1.md", flush=True)
    print(f"preregistration sha256: {digest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
