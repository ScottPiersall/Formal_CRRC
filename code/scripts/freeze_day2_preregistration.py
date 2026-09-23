#!/usr/bin/env python
"""Freeze the FormalCRRC Day-2 preregistration (Phase E).

Builds ``artifacts/day2/preregistration.json`` from the Day-2 configuration
object and the already-frozen partition manifests, writes
``artifacts/day2/preregistration.sha256``, and renders the human-readable
``docs/PREREGISTRATION_DAY2.md`` from the same object so the two cannot drift.

Refuses to overwrite an existing preregistration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import day2_config as d2c  # noqa: E402
from formalcrrc import prompts, provenance  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    BOOTSTRAP_CI,
    BOOTSTRAP_METHOD,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DECISION_THRESHOLD,
    FAMILIES,
    INFERENCE_BATCH_SIZE,
    LABEL_MET,
    LABEL_NOT_MET,
    MODEL_DTYPE,
    MODEL_IDS,
    N_ARTIFACTS,
    N_ROWS,
    N_STRICTNESS,
    NO_CROSSING_INDEX,
    PROMPT_TEMPLATE,
)

RESEARCH_QUESTIONS = {
    "primary": (
        "Does correcting a judge's global decision-margin bias using an "
        "independent formal calibration set improve threshold localization on a "
        "fresh held-out FormalCRRC test set without modifying the underlying "
        "threshold-response shape?"
    ),
    "secondary": (
        "After global bias is removed, how much threshold-localization error "
        "remains?"
    ),
    "additional": (
        "Is a single global bias parameter sufficient, or does bias depend "
        "systematically on predicate family?"
    ),
}

HYPOTHESES = [
    {
        "id": "H1",
        "label": "Global bias correction",
        "statement": (
            "An independently fitted global intercept will improve held-out "
            "threshold localization for at least some judges."
        ),
        "quantity": "delta_TCE_G = TCE_raw - TCE_global, positive means improvement",
        "reporting": (
            "per judge, with paired two-stage bootstrap intervals and the "
            "fraction of replicates above zero"
        ),
        "note": "Not all judges are required to improve.",
        "status": "primary",
    },
    {
        "id": "H2",
        "label": "Residual error",
        "statement": (
            "Even after global intercept correction, nontrivial "
            "threshold-localization error may remain."
        ),
        "quantity": "TCE_global reported directly",
        "reporting": "mean, median, distribution, exact and within-one-level rates",
        "note": (
            "No threshold for what counts as 'nontrivial' is defined, before or "
            "after seeing the data."
        ),
        "status": "primary",
    },
    {
        "id": "H3",
        "label": "Shape invariance",
        "statement": (
            "Because global calibration adds a constant in margin space, every "
            "margin-based shape metric is mathematically unchanged."
        ),
        "quantity": "MMVR, MMVM, SPAN and margin Spearman before and after shifting",
        "reporting": "maximum absolute deviation, verified to numerical tolerance",
        "note": (
            "This is an identity, not a finding. A material deviation is an "
            "implementation defect."
        ),
        "status": "primary",
    },
    {
        "id": "H4",
        "label": "Predicate-family bias",
        "statement": (
            "Family-specific intercept correction may outperform the single "
            "global correction."
        ),
        "quantity": "delta_TCE_F = TCE_global - TCE_family",
        "reporting": "per judge and per family, with bootstrap intervals",
        "note": "Secondary.",
        "status": "secondary",
    },
]

NOVELTY_BOUNDARIES = {
    "claimed_contribution": (
        "FormalCRRC uses formally controlled threshold-response curves to "
        "decompose evaluator failure into curve-location bias and residual "
        "curve-shape/boundary error under a held-out, shape-preserving "
        "calibration protocol."
    ),
    "object_of_interest": "the decomposition, not the invention of calibration",
    "calibration_method_novelty": (
        "None claimed. Intercept-only logistic calibration is standard and is "
        "used here precisely because it is the weakest correction that can move "
        "location without touching shape."
    ),
    "explicitly_not_claimed": [
        "first probability calibration of an LLM",
        "first logit calibration",
        "first intercept-only calibration",
        "first calibration of LLM-as-a-Judge",
        "first study of LLM response bias",
        "first use of held-out calibration data",
        "first use of counterfactual curves",
        "first threshold robustness test",
    ],
    "day2_does_not_establish": [
        "human alignment",
        "clinical reliability",
        "natural-language rubric reliability",
        "HealthBench reliability",
        "calibration on arbitrary tasks",
        "general superiority of any model",
        "that intercept calibration should be deployed",
        "that calibrated probabilities are true probabilities",
        "that semantic rubric failures reduce to global bias",
    ],
}

EXCLUSION_RULES = {
    "artifact_exclusions": "None. All 324 CAL and all 324 TEST artifacts are used.",
    "threshold_exclusions": "None. All 9 strictness levels are used.",
    "row_exclusions": (
        "A row is excluded only if its scoring produced a non-finite label "
        "score, which would be an implementation defect rather than a result. "
        "Any exclusion is reported with an exact count and prompt_ids, and "
        "blocks the affected model."
    ),
    "model_exclusions": (
        "A model is excluded only if its exact Day-1 revision is inaccessible or "
        "its scoring conditions no longer match the Day-1 record, and is then "
        "recorded as BLOCKED_BY_MODEL_ACCESS. No substitution."
    ),
    "family_c_candidate_coincidence": (
        "Family-C candidate responses are determined by the reported value, a "
        "bounded integer, so byte-identical candidate strings recur across "
        "independently seeded partitions. These are NOT filtered. The (target, "
        "reported) pairs and the rendered prompts are disjoint by construction, "
        "so no evaluated item is shared; the coincidence count is reported."
    ),
    "no_filtering_of_hard_artifacts": True,
    "no_selection_of_families": True,
    "no_data_dependent_exclusion": True,
}

CALIBRATION_PROHIBITIONS = [
    "fitting alpha using TEST",
    "fitting alpha using any Day-1 outcome",
    "choosing among multiple alpha estimators based on TEST",
    "fitting separate offsets after inspecting which family performs poorly",
    "changing the decision threshold from 0.5",
    "fitting a slope",
    "temperature scaling",
    "isotonic regression",
    "Platt scaling with a free slope",
    "monotone regression",
    "per-artifact calibration",
    "per-threshold calibration",
    "per-latent-level calibration",
    "learned embeddings",
    "neural calibration",
    "prompt optimization",
]

TAXONOMY = {
    "mislocated_but_orderly": "MMVR = 0 and TCE_raw > 1",
    "locally_unstable": "MMVR > 0",
    "correctly_localized": "TCE_raw = 0",
    "correctable_by_global_bias": "TCE_raw > 1 and TCE_global <= 1",
    "residual_localization_failure": "TCE_global > 1",
    "note": (
        "Descriptive categories, predeclared before inference. They are counted "
        "and reported, never used to select favourable subsets."
    ),
}

FIGURE_PLAN = [
    {
        "id": "figure1",
        "title": "Bias-vs-shape concept",
        "content": (
            "A raw margin curve, the same curve shifted by a global intercept, "
            "and the resulting change of decision-boundary location with "
            "identical shape; shows that a vertical shift cannot remove a local "
            "reversal."
        ),
        "path": "figures/day2/figure1_bias_vs_shape.png",
    },
    {
        "id": "figure2",
        "title": "Estimated global offsets",
        "content": (
            "One alpha per judge with bootstrap uncertainty, family offsets "
            "shown as secondary markers."
        ),
        "path": "figures/day2/figure2_global_offsets.png",
    },
    {
        "id": "figure3",
        "title": "Raw versus globally corrected TCE",
        "content": "Paired artifact-level TCE for each judge.",
        "path": "figures/day2/figure3_paired_tce.png",
    },
    {
        "id": "figure4",
        "title": "Boundary-aligned held-out curves",
        "content": (
            "Per judge on TEST: raw probability curve, globally adjusted curve, "
            "and the formal boundary."
        ),
        "path": "figures/day2/figure4_boundary_aligned.png",
    },
    {
        "id": "figure5",
        "title": "Residual localization error",
        "content": "TCE distributions after global correction.",
        "path": "figures/day2/figure5_residual_tce.png",
    },
    {
        "id": "figure6",
        "title": "Model x family decomposition",
        "content": "Heatmap of raw TCE, global TCE, family TCE and MMVR.",
        "path": "figures/day2/figure6_model_family.png",
    },
]

STOPPING_RULE = {
    "rule": (
        "Inference stops when every judge has produced exactly 2916 scored "
        "prompts on CALIBRATION and 2916 on TEST, or has been marked "
        "BLOCKED_BY_MODEL_ACCESS. Calibration is fitted only then, and the "
        "preregistered analysis is run exactly once."
    ),
    "no_interim_analysis": (
        "No intercept, TCE, accuracy, MVR, family result, boundary curve or "
        "ranking is computed or displayed while inference is running."
    ),
    "no_data_dependent_extension": (
        "The design is not extended, repeated or enlarged on the basis of "
        "observed results. No Day-3 experiment follows automatically."
    ),
    "success_criterion": (
        "Day 2 succeeds when Day-1 immutability passes, partitions are disjoint "
        "and integral, the preregistration was frozen before outcomes, intended "
        "runs completed or were transparently blocked, calibration used "
        "CALIBRATION only, shape invariance holds, and all results are reported "
        "regardless of direction. No favourable-effect threshold is defined."
    ),
}

NO_OUTCOME_DRIVEN_TUNING = [
    "changing seeds",
    "changing datasets",
    "changing calibration or test sizes",
    "changing models or revisions",
    "changing the prompt or answer labels",
    "changing the 0.5 decision boundary",
    "changing predicate families",
    "fitting slope parameters",
    "introducing a third calibration method",
    "filtering artifacts",
    "removing difficult families",
    "changing bootstrap units",
    "choosing a model-specific procedure",
    "rerunning only unfavourable examples",
]


def build_document(root: Path) -> dict:
    """Assemble the Day-2 preregistration from frozen inputs."""
    manifests = {}
    for partition, path in d2c.MANIFEST_PATHS.items():
        candidate = root / path
        if not candidate.is_file():
            raise SystemExit(
                f"{path} not found; run scripts/generate_day2_datasets.py first"
            )
        manifests[partition] = json.loads(candidate.read_text(encoding="utf-8"))

    day1_prereg = json.loads(
        (root / "artifacts/day1/preregistration.json").read_text(encoding="utf-8")
    )
    day1_models = json.loads(
        (root / "artifacts/day1/model_provenance.json").read_text(encoding="utf-8")
    )["models"]

    day2_provenance_path = root / d2c.MODEL_PROVENANCE_PATH
    day2_status = {m: "NOT_CHECKED" for m in MODEL_IDS}
    if day2_provenance_path.is_file():
        document = json.loads(day2_provenance_path.read_text(encoding="utf-8"))
        for model_id in MODEL_IDS:
            day2_status[model_id] = document.get("models", {}).get(model_id, {}).get(
                "status", "NOT_CHECKED"
            )

    baseline_path = root / d2c.DAY1_BASELINE_PATH
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.is_file()
        else {}
    )
    findings_path = root / d2c.DAY1_AUDIT_FINDINGS_PATH
    findings = (
        json.loads(findings_path.read_text(encoding="utf-8"))
        if findings_path.is_file()
        else {}
    )

    return {
        "document": "FormalCRRC Day-2 preregistration",
        "experiment": "Separating global judge bias from threshold-response shape",
        "version": 1,
        "frozen_at": provenance.utc_now(),
        "amendments": [],
        "relationship_to_day1": {
            "statement": (
                "Day 1 motivates Day 2. Day 2 is a new, separately preregistered "
                "experiment on new data, not a modification, rerun or repair of "
                "Day 1."
            ),
            "day1_results_status": "unchanged and final",
            "day1_preregistration_sha256": (
                (root / "artifacts/day1/preregistration.sha256")
                .read_text(encoding="utf-8")
                .split()[0]
            ),
            "day1_baseline_manifest_sha256": baseline.get("combined_sha256"),
            "day1_baseline_n_files": baseline.get("n_files"),
            "day1_audit": {
                "errata_required": findings.get(
                    "issue_a_boundary_curve_claim", {}
                ).get("errata_required"),
                "preregistered_files_changed": findings.get(
                    "issue_b_source_provenance", {}
                ).get("preregistered_files_changed"),
                "all_runs_pinned_to_freeze": findings.get(
                    "issue_b_source_provenance", {}
                ).get("all_runs_pinned_to_freeze"),
                "figure_hash_defect_present": findings.get(
                    "issue_c_figure_integrity", {}
                ).get("defect_present"),
                "documents": [
                    "docs/DAY1_ERRATA.md",
                    "docs/DAY1_PROVENANCE_AUDIT.md",
                    "docs/DAY1_INTEGRITY_ADDENDUM.md",
                ],
            },
            "day1_source_extensions": d2c.DAY1_SOURCE_EXTENSIONS,
            "day1_dataset_content_sha256": d2c.DAY1_DATASET_CONTENT_SHA256,
        },
        "research_questions": RESEARCH_QUESTIONS,
        "hypotheses": HYPOTHESES,
        "novelty_boundaries": NOVELTY_BOUNDARIES,
        "design": {
            "type": "held-out shape-preserving intercept calibration",
            "unit": "artifact",
            "calibration_partition": "CAL, seed 43, intercept fitting only",
            "test_partition": "TST, seed 44, held-out evaluation only",
            "core_object": "M(s) = S_A - S_B, the two-label decision margin",
            "probability": "P_raw = sigmoid(M)",
            "invariance": (
                "M'(s+1) - M'(s) = M(s+1) - M(s) for any intercept, so location "
                "may change and shape may not"
            ),
        },
        "datasets": {
            partition: {
                "seed": manifest["seed"],
                "role": manifest["role"],
                "n_artifacts": manifest["dimensions"]["n_artifacts"],
                "n_rows": manifest["dimensions"]["n_rows"],
                "path": d2c.DATASET_PATHS[partition],
                "content_sha256": manifest["hashes"]["dataset_content_sha256"],
                "file_sha256": manifest["hashes"]["dataset_file_sha256"],
                "prompt_manifest_file_sha256": manifest["hashes"][
                    "prompt_manifest_file_sha256"
                ],
                "order_salt": manifest["inference_order"]["salt"],
            }
            for partition, manifest in manifests.items()
        },
        "dataset_totals": {
            "n_artifacts_per_partition": N_ARTIFACTS,
            "n_rows_per_partition": N_ROWS,
            "n_prompts_per_judge": 2 * N_ROWS,
            "n_prompts_across_panel": 2 * N_ROWS * len(MODEL_IDS),
        },
        "partition_disjointness": manifests[d2c.PARTITION_CALIBRATION][
            "cross_partition"
        ],
        "data_generating_process": {
            "families": list(FAMILIES),
            "identical_to_day1": (
                "same three predicate families, same nine strictness levels, "
                "same latent-level distribution, same candidate formatting, same "
                "formal truth equations, same threshold interpretation, same "
                "prompt semantics"
            ),
            "family_c_uniqueness_rule": (
                "Family C draws an integer target from a bounded range, so "
                "independent draws can coincide. Generation carries a forbidden "
                "set of (target, reported) pairs -- Day 1's, then Day 1's plus "
                "CAL's -- and redraws on collision. This guarantees that no "
                "rendered prompt is shared between Day 1, CAL and TEST. It is a "
                "documented, seed-deterministic, outcome-independent deviation "
                "from independent sampling with replacement, affecting family-C "
                "targets only."
            ),
            "partition_tag_is_not_model_facing": (
                "The partition tag appears only in artifact_id and prompt_id. "
                "Prompts contain the rubric and candidate response alone."
            ),
        },
        "judge_models": {
            "policy": (
                "Exactly the Day-1 panel at exactly the Day-1 revisions, read "
                "from the immutable Day-1 provenance. No upgrade, no branch "
                "re-resolution, no substitution, no quantisation, no "
                "fine-tuning, no adapters."
            ),
            "model_ids": list(MODEL_IDS),
            "dtype": MODEL_DTYPE,
            "revisions": {
                model_id: day1_models[model_id]["revision"] for model_id in MODEL_IDS
            },
            "scoring_methods": {
                model_id: day1_models[model_id]["label_tokenization"]["scoring_method"]
                for model_id in MODEL_IDS
            },
            "label_token_ids": {
                model_id: {
                    "met": day1_models[model_id]["label_tokenization"]["met_token_ids"],
                    "not_met": day1_models[model_id]["label_tokenization"][
                        "not_met_token_ids"
                    ],
                }
                for model_id in MODEL_IDS
            },
            "status_at_freeze": day2_status,
            "run_order": prompts.model_run_order(),
        },
        "prompt": {
            "reused_from_day1_byte_identical": True,
            "template_sha256": prompts.prompt_template_sha256(),
            "day1_template_sha256": day1_prereg["prompt"]["template_sha256"],
            "template": PROMPT_TEMPLATE,
            "label_met": LABEL_MET,
            "label_not_met": LABEL_NOT_MET,
            "prohibited": day1_prereg["prompt"]["prohibited"],
        },
        "scoring": {
            "reused_from_day1": True,
            "label_token_resolution": day1_prereg["scoring"]["label_token_resolution"],
            "primary_rule": day1_prereg["scoring"]["primary_rule"],
            "fallback_rule": day1_prereg["scoring"]["fallback_rule"],
            "margin_definition": "M = S_A - S_B",
            "probability": "P_raw = sigmoid(M), identical to the Day-1 normalisation",
            "margins_from_raw_scores": (
                "Margins are computed from the recorded raw label scores, never "
                "reconstructed from rounded probabilities."
            ),
            "recorded_per_row": [
                "raw_score_met",
                "raw_score_not_met",
                "margin",
                "p_met_raw",
                "scoring_method",
                "n_prompt_tokens",
                "rendered_prompt_sha256",
                "revision",
            ],
        },
        "inference": {
            "batch_size": INFERENCE_BATCH_SIZE,
            "order": (
                "prompts are sorted by sha256(partition salt|prompt_id), frozen "
                "in the partition prompt manifests before any GPU inference"
            ),
            "calibration_order_salt": d2c.CALIBRATION_ORDER_SALT,
            "test_order_salt": d2c.TEST_ORDER_SALT,
            "model_order_salt": d2c.MODEL_ORDER_SALT,
            "console_output": (
                "operational only: processed rows, failures, timing, GPU memory. "
                "No accuracy, TCE, fitted alpha, ranking, MVR, family result or "
                "boundary curve before all intended raw runs complete."
            ),
        },
        "calibration": {
            "primary_model": d2c.CALIBRATION_PRIMARY,
            "primary_equation": "M_G(s) = M(s) + alpha_J",
            "primary_free_parameters_per_judge": d2c.CALIBRATION_N_PARAMETERS_PRIMARY,
            "secondary_model": d2c.CALIBRATION_SECONDARY,
            "secondary_equation": "M_F(s) = M(s) + alpha_{J,F}",
            "secondary_free_parameters_per_judge": (
                d2c.CALIBRATION_N_PARAMETERS_SECONDARY
            ),
            "slope": d2c.CALIBRATION_SLOPE,
            "objective": d2c.CALIBRATION_OBJECTIVE,
            "solver": d2c.CALIBRATION_SOLVER,
            "fitted_on": "CALIBRATION partition only",
            "never_fitted_on": ["TEST partition", "Day-1 data", "any outcome"],
            "interpretation": (
                "alpha > 0: the raw judge is globally too pessimistic about "
                "'criterion met'; alpha < 0: globally too permissive; alpha near "
                "0: little global offset. This is a decision-margin location "
                "parameter under this formal task, not a psychological or "
                "ideological property."
            ),
            "prohibited": CALIBRATION_PROHIBITIONS,
        },
        "primary_metrics": {
            "delta_TCE_global": {
                "definition": "TCE_raw - TCE_global, per artifact, on TEST only",
                "j_hat_raw": "min{s : P_raw(s) < 0.5}, or 9",
                "j_hat_global": "min{s : sigmoid(M(s) + alpha) < 0.5}, or 9",
                "decision_threshold": DECISION_THRESHOLD,
                "no_crossing_index": NO_CROSSING_INDEX,
                "reported": [
                    "mean raw TCE",
                    "mean corrected TCE",
                    "mean paired improvement",
                    "medians",
                    "exact boundary rate",
                    "within-one-level rate",
                    "full TCE distribution",
                ],
            },
            "alpha": {
                "definition": "the fitted global intercept per judge",
                "reported": "point estimate and bootstrap interval",
            },
            "TCE_global": {
                "definition": "residual localisation error after global correction"
            },
        },
        "shape_metrics": {
            "computed_on": "raw decision margins",
            "MMVR": "(1/8) sum_{s=0}^{7} 1[M(s+1) > M(s)]",
            "MMVM": "(1/8) sum_{s=0}^{7} max(0, M(s+1) - M(s))",
            "SPAN": "M(0) - M(8)",
            "margin_spearman": "within-artifact Spearman of s against M(s)",
            "invariance_claim": (
                "All four are exactly or numerically invariant under any global "
                "or family-specific intercept. Verified empirically; a material "
                "deviation is an implementation defect."
            ),
            "day1_continuity": (
                "The Day-1 probability-based MVR may be reproduced descriptively "
                "on Day-2 raw scores, but is not the primary Day-2 shape metric."
            ),
        },
        "secondary_metrics": {
            "reported_for": ["RAW", "GLOBAL", "FAMILY"],
            "accuracy": "mean 1[(P_met >= 0.5) == Y]",
            "brier": "mean (P_met - Y)^2",
            "exact_boundary_rate": "mean 1[TCE = 0]",
            "within_one_level_rate": "mean 1[TCE <= 1]",
            "never_cross_rate": "artifacts with P(s) >= 0.5 at every threshold",
            "always_below_rate": "artifacts with P(0) < 0.5",
            "family_breakdown": (
                "every metric reported per predicate family; family effects are "
                "never averaged away"
            ),
        },
        "taxonomy": TAXONOMY,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "stages": d2c.BOOTSTRAP_STAGES,
            "calibration_stream": d2c.BOOTSTRAP_CALIBRATION_STREAM,
            "test_stream": d2c.BOOTSTRAP_TEST_STREAM,
            "unit": "artifact_id",
            "cluster_rule": (
                "all 9 thresholds of an artifact move together in both stages"
            ),
            "stratify_by": "family",
            "refit_inside_replicate": True,
            "interval_method": BOOTSTRAP_METHOD,
            "coverage": BOOTSTRAP_CI,
            "paired_comparisons": (
                "judge-versus-judge comparisons reuse identical CAL and TEST "
                "resamples"
            ),
            "reported": [
                "point estimate",
                "95% percentile interval",
                "fraction of replicates greater than zero",
            ],
            "multiple_comparison_correction": (
                "None. Model-specific reporting is descriptive, and no "
                "correction may be added after results are seen."
            ),
            "prohibited": [
                "treating threshold rows as independent observations",
                "resampling TEST without refitting alpha",
                "selecting any hyperparameter from bootstrap outcomes",
            ],
        },
        "exclusion_rules": EXCLUSION_RULES,
        "failure_handling": {
            "incomplete_model_run": (
                "If a judge's raw scores do not contain all 2916 prompts exactly "
                "once on both partitions, that judge is BLOCKED and reported as "
                "such."
            ),
            "model_access_failure": (
                "Recorded as BLOCKED_BY_MODEL_ACCESS. No substitution. A partial "
                "panel is never described as the four-model panel."
            ),
            "day1_immutability_failure": (
                "If any Day-1 fingerprint changes, DAY1_IMMUTABILITY = FAIL and "
                "DAY2_ANALYSIS = BLOCKED. Files are not silently restored."
            ),
            "implementation_defect": (
                "Stop the affected stage, preserve the defective output, "
                "document it, decide whether the protocol is affected, amend if "
                "necessary, and rerun only after an outcome-independent "
                "correction. Poor calibration performance is not a defect."
            ),
            "no_outcome_driven_tuning": NO_OUTCOME_DRIVEN_TUNING,
        },
        "figure_plan": FIGURE_PLAN,
        "stopping_rule": STOPPING_RULE,
        "analysis_code_paths": list(d2c.ANALYSIS_CODE_PATHS),
        "source_manifest_at_freeze": provenance.source_manifest(
            d2c.ANALYSIS_CODE_PATHS, root=root
        ),
        "git_at_freeze": provenance.git_state(root),
    }


def _bullets(items) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_markdown(document: dict, digest: str) -> str:
    """Render the human-readable preregistration from the frozen JSON."""
    models = document["judge_models"]
    calibration = document["calibration"]
    boot = document["bootstrap"]
    datasets = document["datasets"]
    novelty = document["novelty_boundaries"]
    rel = document["relationship_to_day1"]

    hypothesis_rows = "\n".join(
        f"| **{h['id']}** | {h['label']} | {h['statement']} | `{h['quantity']}` | {h['status']} |"
        for h in document["hypotheses"]
    )
    model_rows = "\n".join(
        f"| `{m}` | `{models['revisions'][m]}` | {models['scoring_methods'][m]} | "
        f"A={models['label_token_ids'][m]['met']} B={models['label_token_ids'][m]['not_met']} | "
        f"{models['status_at_freeze'][m]} |"
        for m in models["model_ids"]
    )
    dataset_rows = "\n".join(
        f"| {partition} | {block['seed']} | {block['role']} | {block['n_artifacts']} | "
        f"{block['n_rows']} | `{block['content_sha256'][:16]}…` |"
        for partition, block in datasets.items()
    )
    figure_rows = "\n".join(
        f"| {f['id']} | {f['title']} | {f['content']} | `{f['path']}` |"
        for f in document["figure_plan"]
    )
    disjoint = document["partition_disjointness"]

    return f"""# FormalCRRC Day-2 preregistration

## Separating global judge bias from threshold-response shape

**Frozen:** {document['frozen_at']}
**SHA-256:** `{digest}`
**Machine-readable copy:** [`artifacts/day2/preregistration.json`](../artifacts/day2/preregistration.json)
**Checksum:** [`artifacts/day2/preregistration.sha256`](../artifacts/day2/preregistration.sha256)
**Amendments:** {len(document['amendments'])}

Frozen **before any Day-2 study inference**. Generated from the same object as
the JSON copy, so the two cannot disagree.

> Once frozen, this document is never silently edited. A genuine implementation
> defect is corrected by a numbered amendment that explains the defect, explains
> why the correction is outcome-independent, preserves the original unchanged,
> and carries its own checksum. **Weak or unexpected scientific outcomes never
> justify an amendment.**

---

## 1. Relationship to Day 1

{rel['statement']}

Day-1 results are **{rel['day1_results_status']}**. Day 2 is not a repair of
Day 1 and does not reinterpret any Day-1 outcome.

| | |
|---|---|
| Day-1 preregistration SHA-256 | `{rel['day1_preregistration_sha256']}` |
| Day-1 baseline fingerprint | `{rel['day1_baseline_manifest_sha256']}` over {rel['day1_baseline_n_files']} files |
| Day-1 dataset content SHA-256 | `{rel['day1_dataset_content_sha256']}` |
| Day-1 errata required | {rel['day1_audit']['errata_required']} |
| Preregistered Day-1 analysis files changed post-freeze | {rel['day1_audit']['preregistered_files_changed'] or 'none'} |
| All Day-1 inference runs pinned to the freeze | {rel['day1_audit']['all_runs_pinned_to_freeze']} |
| Day-1 figure-hash defect present | {rel['day1_audit']['figure_hash_defect_present']} |

Supporting audits: {', '.join(f'`{d}`' for d in rel['day1_audit']['documents'])}.

**Day-1 source extensions.** Day 2 extends shared modules in a
backward-compatible way; Day-1 outputs are unaffected and the Day-1 dataset
still reproduces content hash `{rel['day1_dataset_content_sha256']}`, which the
test suite asserts permanently.

{chr(10).join(f'- `{path}` — {note}' for path, note in rel['day1_source_extensions'].items())}

## 2. Research questions

**Primary.** {document['research_questions']['primary']}

**Secondary.** {document['research_questions']['secondary']}

**Additional.** {document['research_questions']['additional']}

## 3. Hypotheses

| | Label | Statement | Quantity | Status |
|---|---|---|---|---|
{hypothesis_rows}

## 4. Novelty boundaries

**Claimed contribution.** {novelty['claimed_contribution']}

The object of interest is **{novelty['object_of_interest']}**.
{novelty['calibration_method_novelty']}

**Explicitly not claimed:**

{_bullets(novelty['explicitly_not_claimed'])}

**Day 2 does not establish:**

{_bullets(novelty['day2_does_not_establish'])}

## 5. The formal object

The two-label decision margin is

```text
M(s) = S_A - S_B          P_raw(s) = sigmoid(M(s))
```

where `S_A` is the raw score for answer A (criterion met). Margins are taken
from the recorded raw label scores, never reconstructed from rounded
probabilities.

The primary intervention is a single additive constant:

```text
M_G(s) = M(s) + alpha_J        slope fixed at {calibration['slope']}
```

Because the same constant is added at every level,

```text
M_G(s+1) - M_G(s) = M(s+1) - M(s)
```

so the correction can move a curve relative to the decision boundary but cannot
change margin ordering, margin differences, local reversals, total span, or
Spearman ordering. **Location can change; shape cannot.** That invariance is the
experimental control.

## 6. Calibration

| | |
|---|---|
| Primary model | `{calibration['primary_model']}` — {calibration['primary_equation']} |
| Free parameters per judge | {calibration['primary_free_parameters_per_judge']} |
| Secondary model | `{calibration['secondary_model']}` — {calibration['secondary_equation']} |
| Free parameters per judge | {calibration['secondary_free_parameters_per_judge']} |
| Objective | {calibration['objective']} |
| Solver | `{calibration['solver']}` |
| Fitted on | **{calibration['fitted_on']}** |
| Never fitted on | {', '.join(calibration['never_fitted_on'])} |

Interpretation: {calibration['interpretation']}

**Prohibited:**

{_bullets(calibration['prohibited'])}

## 7. Data

| Partition | Seed | Role | Artifacts | Rows | Content hash |
|---|---|---|---|---|---|
{dataset_rows}

Totals: {document['dataset_totals']['n_prompts_per_judge']} prompts per judge,
{document['dataset_totals']['n_prompts_across_panel']} across the panel.

### Disjointness, verified before freezing

| Check | CAL vs TEST | CAL vs Day 1 | TEST vs Day 1 |
|---|---|---|---|
| artifact_id overlap | {disjoint['artifact_id_overlap']['cal_vs_test']} | {disjoint['artifact_id_overlap']['cal_vs_day1']} | {disjoint['artifact_id_overlap']['test_vs_day1']} |
| rendered prompt overlap | {disjoint['rendered_prompt_overlap']['cal_vs_test']} | {disjoint['rendered_prompt_overlap']['cal_vs_day1']} | {disjoint['rendered_prompt_overlap']['test_vs_day1']} |
| family-C (target, reported) pair overlap | {disjoint['family_c_target_pair_overlap']['cal_vs_test']} | {disjoint['family_c_target_pair_overlap']['cal_vs_day1']} | {disjoint['family_c_target_pair_overlap']['test_vs_day1']} |
| family-C candidate coincidences | {disjoint['candidate_response_overlap_by_family']['numeric_tolerance']['cal_vs_test']} | {disjoint['candidate_response_overlap_by_family']['numeric_tolerance']['cal_vs_day1']} | {disjoint['candidate_response_overlap_by_family']['numeric_tolerance']['test_vs_day1']} |

{disjoint['family_c_candidate_coincidence_note']}

{document['data_generating_process']['family_c_uniqueness_rule']}

{document['data_generating_process']['partition_tag_is_not_model_facing']}

## 8. Judge panel

{models['policy']}

| Model | Revision | Scoring rule | Label tokens | Status at freeze |
|---|---|---|---|---|
{model_rows}

dtype `{models['dtype']}`. Run order: {', '.join(f'`{m}`' for m in models['run_order'])}.

## 9. Prompt

Reused from Day 1 **byte-identically**, SHA-256
`{document['prompt']['template_sha256']}` (Day 1:
`{document['prompt']['day1_template_sha256']}`).

```
{document['prompt']['template']}
```

Prohibited: {', '.join(document['prompt']['prohibited'])}.

## 10. Inference

- Batch size {document['inference']['batch_size']}
- {document['inference']['order']}
- Calibration salt `{document['inference']['calibration_order_salt']}`
- Test salt `{document['inference']['test_order_salt']}`
- Console output: {document['inference']['console_output']}

## 11. Primary metrics

**delta_TCE_G** — {document['primary_metrics']['delta_TCE_global']['definition']}, with
`{document['primary_metrics']['delta_TCE_global']['j_hat_raw']}` and
`{document['primary_metrics']['delta_TCE_global']['j_hat_global']}` at the
preregistered {DECISION_THRESHOLD} boundary.

Reported: {', '.join(document['primary_metrics']['delta_TCE_global']['reported'])}.

**alpha_J** and **TCE_global** are reported directly.

## 12. Shape metrics

Computed on {document['shape_metrics']['computed_on']}:

| | |
|---|---|
| MMVR | `{document['shape_metrics']['MMVR']}` |
| MMVM | `{document['shape_metrics']['MMVM']}` |
| SPAN | `{document['shape_metrics']['SPAN']}` |
| Spearman | {document['shape_metrics']['margin_spearman']} |

{document['shape_metrics']['invariance_claim']}

## 13. Secondary metrics

Reported for {', '.join(document['secondary_metrics']['reported_for'])}: accuracy,
Brier score, exact-boundary rate, within-one-level rate, never-cross rate,
always-below rate, each broken down by predicate family.

## 14. Bias-vs-shape taxonomy

{chr(10).join(f'- **{k}** — {v}' for k, v in TAXONOMY.items() if k != 'note')}

{TAXONOMY['note']}

## 15. Bootstrap

| | |
|---|---|
| Replicates | {boot['replicates']} |
| Seed | {boot['seed']} |
| Unit | `{boot['unit']}` |
| Cluster rule | {boot['cluster_rule']} |
| Stratification | {boot['stratify_by']} |
| Refit inside each replicate | {boot['refit_inside_replicate']} |
| Interval | {boot['interval_method']} at {boot['coverage']:.0%} |

{boot['stages']}

{boot['paired_comparisons']}

Multiple comparisons: {boot['multiple_comparison_correction']}

Prohibited:

{_bullets(boot['prohibited'])}

## 16. Exclusion rules

{chr(10).join(f'- **{k}** — {v}' for k, v in document['exclusion_rules'].items())}

## 17. Failure handling

{chr(10).join(f'- **{k}** — {v}' for k, v in document['failure_handling'].items() if k != 'no_outcome_driven_tuning')}

Once inference begins, the following are prohibited:

{_bullets(document['failure_handling']['no_outcome_driven_tuning'])}

## 18. Figure plan

| | Title | Content | Path |
|---|---|---|---|
{figure_rows}

## 19. Stopping rule

{chr(10).join(f'- **{k}** — {v}' for k, v in document['stopping_rule'].items())}

## 20. Analysis code

{_bullets(f'`{p}`' for p in document['analysis_code_paths'])}

Source tree SHA-256 at freeze:
`{document['source_manifest_at_freeze']['tree_sha256']}`

Git at freeze: commit
`{document['git_at_freeze'].get('commit') or 'unavailable'}`, clean =
{document['git_at_freeze'].get('clean')}

The Day-2 integrity manifest hashes **exactly this file list**, so the freeze
digest and the final digest are directly comparable — the list mismatch Day 1
exhibited cannot recur.

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

    target = root / d2c.PREREG_JSON_PATH
    sha_path = root / d2c.PREREG_SHA_PATH
    if target.exists():
        print(
            f"REFUSING to overwrite the frozen preregistration at "
            f"{d2c.PREREG_JSON_PATH}.\nA frozen preregistration may only be "
            "corrected by a numbered, checksummed amendment.",
            flush=True,
        )
        return 1

    document = build_document(root)
    provenance.write_json(target, document)
    digest = provenance.write_checksum_file(target, sha_path)

    markdown = root / "docs/PREREGISTRATION_DAY2.md"
    markdown.write_text(render_markdown(document, digest), encoding="utf-8")

    print(f"wrote {d2c.PREREG_JSON_PATH}", flush=True)
    print(f"wrote {d2c.PREREG_SHA_PATH}", flush=True)
    print("wrote docs/PREREGISTRATION_DAY2.md", flush=True)
    print(f"preregistration sha256: {digest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
