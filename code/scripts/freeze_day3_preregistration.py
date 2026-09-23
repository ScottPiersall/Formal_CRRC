#!/usr/bin/env python
"""Freeze the FormalCRRC Day-3 preregistration (Phase E).

Builds ``artifacts/day3/preregistration.json`` from the Day-3 configuration and
the already-frozen dataset manifest, writes
``artifacts/day3/preregistration.sha256``, and renders
``docs/PREREGISTRATION_DAY3.md`` from the same object so the two cannot drift.

Refuses to overwrite an existing preregistration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import day3_config as d3c  # noqa: E402
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
        "What is the minimum threshold-crossing error achievable when each "
        "individual judge-response curve may undergo an arbitrary additive "
        "translation but its shape is held exactly fixed?"
    ),
    "secondary_reachability": (
        "What fraction of formal decision boundaries are exactly reachable by "
        "translation alone?"
    ),
    "secondary_variation": (
        "How does the translation-imposed localization floor vary across judge "
        "models and predicate families?"
    ),
    "secondary_hierarchy": (
        "How much additional localization capacity is gained when translation "
        "flexibility increases from judge-level to family-level to "
        "artifact-level?"
    ),
}

HYPOTHESES = [
    {
        "id": "H1",
        "label": "Translation reachability is incomplete",
        "statement": (
            "For nontrivial formal boundaries j* in 1..8, not every boundary is "
            "expected to satisfy the strict prefix-record-low condition."
        ),
        "quantity": "nontrivial TRR, and OTCE",
        "note": "No favourable threshold is defined, before or after.",
        "status": "primary",
    },
    {
        "id": "H2",
        "label": "Artifact-level oracle lowers localization error",
        "statement": (
            "Because alpha = 0 is admissible, OTCE <= TCE_raw for every "
            "artifact. This is guaranteed; the empirical question is the size of "
            "the remaining OTCE."
        ),
        "quantity": "delta_TCE_A = TCE_raw - OTCE",
        "note": "The inequality is asserted computationally on every artifact.",
        "status": "primary",
    },
    {
        "id": "H3",
        "label": "Residual translation-limited error remains",
        "statement": (
            "OTCE > 0 means no additive translation places that curve's crossing "
            "exactly at the formal boundary; OTCE > 1 means none places it even "
            "within one strictness level."
        ),
        "quantity": "SLR1 = P(OTCE > 0), SLR2 = P(OTCE > 1)",
        "note": "Reported directly, with no post-hoc threshold.",
        "status": "primary",
    },
    {
        "id": "H4",
        "label": "Translation flexibility forms nested bounds",
        "statement": (
            "Since G subset F subset A, the optimised means must satisfy "
            "TCE_G >= TCE_F >= TCE_A."
        ),
        "quantity": "the three oracle means",
        "note": "A violation is an implementation defect, not a finding.",
        "status": "primary",
    },
    {
        "id": "H5",
        "label": "Shape limitation varies across predicate families",
        "statement": (
            "OTCE, TRR and reachable-crossing count may differ by predicate "
            "family."
        ),
        "quantity": "all primary metrics, reported per family",
        "note": "Family differences are never averaged away.",
        "status": "primary",
    },
]

NOVELTY_BOUNDARIES = {
    "claimed_contribution": (
        "FormalCRRC uses exact shape-preserving oracle translation bounds to "
        "distinguish threshold-localization error that can in principle be "
        "explained by response-curve location from residual error imposed by the "
        "curve's ordering itself."
    ),
    "object_of_interest": "the diagnostic decomposition",
    "mathematics_novelty": (
        "None claimed. Adding a constant, the order-preservation of a "
        "translation, oracle bounds as a technique, and per-example calibration "
        "are all standard."
    ),
    "explicitly_not_claimed": [
        "first oracle calibration",
        "first oracle bound",
        "first additive margin correction",
        "first per-example calibration",
        "first threshold-response experiment",
        "first monotonicity test",
        "first counterfactual judge evaluation",
        "first analysis of model bias",
        "first study of response-curve geometry",
    ],
    "permitted_phrasings": [
        "Under FormalCRRC's additive-translation model, these curves possess a "
        "nonzero shape-imposed localization floor.",
        "The correct formal boundary is unreachable by translation for X% of "
        "nontrivial artifacts.",
    ],
    "forbidden_phrasings": [
        "We prove LLM judges have bad shape.",
        "Judge X is shape-reliable.",
        "Calibration cannot fix LLM judges.",
    ],
    "day3_does_not_establish": [
        "human alignment",
        "clinical reliability",
        "natural-language rubric reliability",
        "HealthBench reliability",
        "that any judge is reliable or unreliable in general",
        "that calibration is futile or that another transformation would fail",
        "that the oracle should be deployed",
    ],
}

EXCLUSION_RULES = {
    "artifact_exclusions": "NONE. All 324 artifacts are used.",
    "threshold_exclusions": "NONE. All 9 strictness levels are used.",
    "family_exclusions": "NONE.",
    "model_exclusions": (
        "Only exact-revision access failure or a scoring-condition mismatch "
        "detected before study outcomes, recorded as BLOCKED_BY_MODEL_ACCESS. No "
        "substitution."
    ),
    "row_exclusions": (
        "Only a non-finite raw score, which blocks the affected model's analysis "
        "and is reported with an exact count and prompt_ids."
    ),
    "never_filtered": [
        "non-monotone curves",
        "flat curves",
        "extreme curves",
        "hard artifacts",
        "tied margins",
        "unusual model outputs",
    ],
    "rationale": "Those are precisely the objects Day 3 measures.",
}

NO_OUTCOME_DRIVEN_TUNING = [
    "changing the seed",
    "changing dataset size",
    "changing predicate families",
    "changing the model panel or revisions",
    "changing the prompt or scoring",
    "changing the 0.5 decision boundary",
    "introducing margin tolerances",
    "changing strict to non-strict inequalities",
    "changing the oracle definition",
    "introducing slope fitting",
    "isotonic regression",
    "temperature scaling",
    "monotonic regression",
    "curve smoothing",
    "artifact filtering",
    "rerunning only bad curves",
    "adding models",
    "adding semantic tasks",
]

FIGURE_PLAN = [
    {
        "id": "figure1",
        "title": "Translation reachability concept",
        "content": (
            "One margin curve with its strict prefix-record-low points, the "
            "reachable crossing positions, an unreachable true crossing, and the "
            "zero line under translation; makes visible that translation moves "
            "the crossing but cannot reorder margins."
        ),
        "path": "figures/day3/figure1_reachability_concept.png",
    },
    {
        "id": "figure2",
        "title": "Raw versus oracle-artifact TCE",
        "content": "Per judge: raw TCE distribution and OTCE distribution.",
        "path": "figures/day3/figure2_raw_vs_otce.png",
    },
    {
        "id": "figure3",
        "title": "Translation reachability",
        "content": "Nontrivial TRR by judge and by predicate family.",
        "path": "figures/day3/figure3_reachability.png",
    },
    {
        "id": "figure4",
        "title": "Reachable crossing geometry",
        "content": (
            "Heatmap of the fraction of artifacts for which each true boundary "
            "j* is translation-reachable, by judge."
        ),
        "path": "figures/day3/figure4_crossing_geometry.png",
    },
    {
        "id": "figure5",
        "title": "Oracle hierarchy",
        "content": (
            "RAW, ORACLE-GLOBAL, ORACLE-FAMILY, ORACLE-ARTIFACT mean TCE per "
            "judge, with bootstrap uncertainty."
        ),
        "path": "figures/day3/figure5_oracle_hierarchy.png",
    },
    {
        "id": "figure6",
        "title": "Shape floor by family",
        "content": "Judge x family heatmap of OTCE, TRR, reachable count and MMVR.",
        "path": "figures/day3/figure6_shape_floor.png",
    },
]

STOPPING_RULE = {
    "rule": (
        "Inference stops when every judge has produced exactly 2916 scored "
        "prompts, or has been marked BLOCKED_BY_MODEL_ACCESS. The oracle "
        "analysis begins only then and is run exactly once."
    ),
    "no_interim_analysis": (
        "No TCE, OTCE, TRR, record-low count, ranking, family result or oracle "
        "shift is computed or displayed while inference is running."
    ),
    "no_data_dependent_extension": (
        "The design is not extended, repeated or enlarged on the basis of "
        "observed results. No Day-4 experiment follows automatically."
    ),
    "no_stronger_oracle": (
        "The translation class is fixed at M'(s) = M(s) + alpha. Slope-plus- "
        "intercept, affine maps, temperature scaling, nonlinear monotone maps, "
        "isotonic regression, monotone envelopes, spline warping, "
        "threshold-specific offsets, curve editing, sorting, rearrangement and "
        "learned shape correction are all out of scope. If a substantial "
        "residual remains, a stronger transformation is a future research "
        "question and is not started now."
    ),
    "success_criterion": (
        "Day 3 succeeds when prior-experiment immutability passes, the dataset "
        "is disjoint and integral, the preregistration was frozen before "
        "outcomes, the reachability theorem agrees with independent enumeration "
        "on every artifact, the oracle nesting inequality holds, intended runs "
        "completed or were transparently blocked, and all results are reported "
        "regardless of direction. No favourable-effect threshold is defined."
    ),
}


def build_document(root: Path) -> dict:
    """Assemble the Day-3 preregistration from frozen inputs."""
    manifest_path = root / d3c.DATASET_MANIFEST_PATH
    if not manifest_path.is_file():
        raise SystemExit(
            f"{d3c.DATASET_MANIFEST_PATH} not found; run "
            "scripts/generate_day3_dataset.py first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    day1_prereg = json.loads(
        (root / "artifacts/day1/preregistration.json").read_text(encoding="utf-8")
    )
    day1_models = json.loads(
        (root / "artifacts/day1/model_provenance.json").read_text(encoding="utf-8")
    )["models"]

    provenance_path = root / d3c.MODEL_PROVENANCE_PATH
    status = {m: "NOT_CHECKED" for m in MODEL_IDS}
    if provenance_path.is_file():
        document = json.loads(provenance_path.read_text(encoding="utf-8"))
        for model_id in MODEL_IDS:
            status[model_id] = document.get("models", {}).get(model_id, {}).get(
                "status", "NOT_CHECKED"
            )

    prior_path = root / d3c.PRIOR_MANIFEST_PATH
    prior = (
        json.loads(prior_path.read_text(encoding="utf-8"))
        if prior_path.is_file()
        else {}
    )

    return {
        "document": "FormalCRRC Day-3 preregistration",
        "experiment": "Translation limits of threshold-response curves",
        "subtitle": "Oracle shape-preserving localization bounds",
        "version": 1,
        "frozen_at": provenance.utc_now(),
        "amendments": [],
        "relationship_to_prior": {
            "statement": (
                "Day 1 motivates Day 2, and Day 2 motivates Day 3. Day 3 is a "
                "new, separately preregistered experiment on new data. It is not "
                "a repair of Day 1, not a repair of Day 2, not another "
                "calibration experiment, and not a stronger post-hoc calibrator."
            ),
            "prior_results_status": "unchanged and final",
            "day1_preregistration_sha256": (
                (root / "artifacts/day1/preregistration.sha256")
                .read_text(encoding="utf-8")
                .split()[0]
            ),
            "day2_preregistration_sha256": (
                (root / "artifacts/day2/preregistration.sha256")
                .read_text(encoding="utf-8")
                .split()[0]
            ),
            "prior_manifest_sha256": prior.get("combined_sha256"),
            "prior_manifest_n_files": prior.get("n_files"),
            "day1_dataset_content_sha256": d3c.DAY1_DATASET_CONTENT_SHA256,
        },
        "research_questions": RESEARCH_QUESTIONS,
        "hypotheses": HYPOTHESES,
        "novelty_boundaries": NOVELTY_BOUNDARIES,
        "formal_model": {
            "margin": "M_x(s) = S_A - S_B, from the recorded raw label scores",
            "probability": "P_x(s) = sigmoid(M_x(s))",
            "translation_class": d3c.TRANSLATION_CLASS,
            "predicted_crossing": (
                "j_hat_x(alpha) = min{s : M_x(s) + alpha < 0}, or 9 if none"
            ),
            "equivalence": "identical to P_x(s) < 0.5",
            "true_crossing": "j*_x = min{s : Y_x(s) = 0}, or 9 if every level passes",
            "decision_threshold": DECISION_THRESHOLD,
            "no_crossing_index": NO_CROSSING_INDEX,
            "invariance": (
                "M'(s+1) - M'(s) = M(s+1) - M(s), so translation cannot change "
                "margin ordering, local reversals, pairwise differences, span, "
                "Spearman ordering or record-low positions; it changes only "
                "where the fixed curve intersects zero"
            ),
        },
        "reachability_theorem": {
            "condition": d3c.REACHABILITY_CONDITION,
            "always_reachable": list(d3c.ALWAYS_REACHABLE),
            "proof": (
                "To place the first negative level at j, an intercept must "
                "satisfy M(t) + alpha >= 0 for all t < j and M(j) + alpha < 0, "
                "hence alpha >= -min_{t<j} M(t) and alpha < -M(j). Such an "
                "interval is non-empty iff -min_{t<j} M(t) < -M(j), i.e. iff "
                "M(j) < min_{t<j} M(t). Crossing 0 is reachable with a "
                "sufficiently negative alpha and crossing 9 with a sufficiently "
                "positive one."
            ),
            "reachable_set": (
                "R_x = {0, 9} union {j in 1..8 : M_x(j) < min_{t<j} M_x(t)}"
            ),
            "two_implementations": (
                "Method A applies the prefix-record-low rule directly. Method B "
                "enumerates candidate intercepts explicitly and collects the "
                "achieved crossings. They must agree on every artifact; a "
                "disagreement is an implementation defect and blocks the "
                "analysis."
            ),
            "tie_policy": d3c.TIE_POLICY,
        },
        "primary_metrics": {
            "OTCE": {
                "definition": "OTCE_x = min_{j in R_x} |j - j*_x|",
                "meaning": (
                    "the best threshold localisation any additive translation "
                    "could achieve for this individual curve"
                ),
                "guarantee": "OTCE_x <= TCE_raw_x, asserted on every artifact",
                "oracle": (
                    "uses formal truth deliberately; a diagnostic lower bound, "
                    "not a deployable calibration procedure"
                ),
            },
            "TRR": {
                "definition": "TR_x = 1[j*_x in R_x]; TRR = mean over artifacts",
                "nontrivial": (
                    "restricted to j*_x in 1..8, because j* = 9 is always "
                    "reachable and would otherwise inflate the headline number"
                ),
                "both_preregistered": True,
            },
            "SLR": {
                "SLR1": "P(OTCE > 0)",
                "SLR2": "P(OTCE > 1)",
                "terminology": (
                    "translation-limited residual, or shape-limited under "
                    "additive translation; never universal 'model shape failure'"
                ),
            },
        },
        "secondary_metrics": {
            "RC": "reachable crossing count |R_x|, between 2 and 10",
            "PRR": (
                "prefix record rate: fraction of levels 1..8 that are strict "
                "prefix record lows; descriptive, never a substitute for OTCE or "
                "TRR"
            ),
            "shape_metrics": [
                "MMVR",
                "MMVM",
                "SPAN",
                "margin Spearman",
            ],
            "shape_metric_associations": (
                "Spearman association between OTCE and each of MMVR, MMVM, SPAN, "
                "PRR and RC, by judge and family. Descriptive; not interpreted "
                "causally."
            ),
        },
        "oracle_hierarchy": {
            "classes": list(d3c.ORACLE_CLASSES),
            "global": "one truth-optimised intercept per judge",
            "family": "one truth-optimised intercept per judge and predicate family",
            "artifact": "one arbitrary intercept per artifact; its minimum is OTCE",
            "nesting": "G subset F subset A, so TCE_G >= TCE_F >= TCE_A",
            "assertion": "the inequality is checked computationally; a violation is a defect",
            "not_a_variance_decomposition": (
                "the differences are nested operational bounds, not an "
                "algebraically additive decomposition"
            ),
            "not_deployment": (
                "all three use formal truth and are upper bounds on what "
                "increasingly flexible location-only explanations can accomplish"
            ),
            "enumeration": d3c.ORACLE_ENUMERATION,
            "tie_break": list(d3c.ORACLE_TIE_BREAK),
            "no_gradient_descent": (
                "mean TCE is piecewise constant in alpha, so the optimum is found "
                "by finite enumeration of breakpoints, not by descent"
            ),
        },
        "taxonomy": {
            **d3c.TAXONOMY,
            "thresholds_fixed_before_inference": True,
            "note": (
                "Descriptive categories reported as counts only. Never used to "
                "select favourable subsets."
            ),
        },
        "dataset": {
            "partition": manifest["partition"],
            "partition_tag": manifest["partition_tag"],
            "seed": manifest["seed"],
            "role": manifest["role"],
            "n_artifacts": manifest["dimensions"]["n_artifacts"],
            "n_rows": manifest["dimensions"]["n_rows"],
            "path": d3c.DATASET_PATH,
            "content_sha256": manifest["hashes"]["dataset_content_sha256"],
            "file_sha256": manifest["hashes"]["dataset_file_sha256"],
            "prompt_manifest_file_sha256": manifest["hashes"][
                "prompt_manifest_file_sha256"
            ],
            "generator": (
                "identical to Days 1 and 2: same three predicate families, nine "
                "strictness levels, latent balance, candidate formatting, rubric "
                "templates, truth functions and response-length controls. No new "
                "family, no semantic rubric, no adversarial variant."
            ),
            "family_c_uniqueness_rule": manifest["family_c_uniqueness_rule"],
        },
        "disjointness": manifest["disjointness"],
        "judge_models": {
            "policy": (
                "Exactly the Day-1/Day-2 panel at exactly the Day-1 revisions, "
                "read from the immutable Day-1 provenance. No upgrade, no branch "
                "re-resolution, no substitution, no quantisation, no "
                "fine-tuning, no adapters."
            ),
            "model_ids": list(MODEL_IDS),
            "dtype": MODEL_DTYPE,
            "revisions": {m: day1_models[m]["revision"] for m in MODEL_IDS},
            "scoring_methods": {
                m: day1_models[m]["label_tokenization"]["scoring_method"]
                for m in MODEL_IDS
            },
            "label_token_ids": {
                m: {
                    "met": day1_models[m]["label_tokenization"]["met_token_ids"],
                    "not_met": day1_models[m]["label_tokenization"][
                        "not_met_token_ids"
                    ],
                }
                for m in MODEL_IDS
            },
            "status_at_freeze": status,
            "continuity_check": (
                "the same non-study smoke prompt used in Days 1 and 2; a material "
                "difference in raw scores sets MODEL_CONTINUITY = FAIL"
            ),
        },
        "prompt": {
            "reused_byte_identically": True,
            "template_sha256": prompts.prompt_template_sha256(),
            "prior_template_sha256": day1_prereg["prompt"]["template_sha256"],
            "template": PROMPT_TEMPLATE,
            "label_met": LABEL_MET,
            "label_not_met": LABEL_NOT_MET,
            "prohibited": day1_prereg["prompt"]["prohibited"],
        },
        "scoring": {
            "reused_from_prior_days": True,
            "label_token_resolution": day1_prereg["scoring"]["label_token_resolution"],
            "primary_rule": day1_prereg["scoring"]["primary_rule"],
            "fallback_rule": day1_prereg["scoring"]["fallback_rule"],
            "recorded_per_row": [
                "raw_score_met",
                "raw_score_not_met",
                "margin",
                "p_met",
                "model_id",
                "revision",
                "scoring_method",
                "rendered_prompt_sha256",
                "n_prompt_tokens",
                "artifact_id",
                "strictness_index",
            ],
            "no_calibration_during_inference": True,
            "no_generation_or_sampling": True,
        },
        "inference": {
            "batch_size": INFERENCE_BATCH_SIZE,
            "order_salt": d3c.INFERENCE_ORDER_SALT,
            "model_order_salt": d3c.MODEL_ORDER_SALT,
            "order": (
                "prompts sorted by sha256(salt|prompt_id), frozen in the prompt "
                "manifest before any GPU inference; family, latent level and "
                "strictness are interleaved"
            ),
            "console_output": (
                "operational only: counts, timing, node, GPU, failures, memory. "
                "No TCE, OTCE, TRR, record lows, rankings, family results or "
                "oracle shifts before all intended raw runs complete."
            ),
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "artifact_id",
            "cluster_rule": "all 9 thresholds of an artifact are carried together",
            "stratify_by": "family",
            "oracle_global": "refit the TCE-minimising global shift inside each replicate",
            "oracle_family": "refit family-specific shifts inside each replicate",
            "oracle_artifact": "OTCE recomputed from each resampled curve",
            "paired_comparisons": "identical artifact resamples across judges",
            "interval_method": BOOTSTRAP_METHOD,
            "coverage": BOOTSTRAP_CI,
            "prohibited": [
                "treating threshold rows as independent observations",
                "holding a truth-optimised shift fixed while resampling",
                "selecting any hyperparameter from bootstrap outcomes",
            ],
        },
        "randomness_policy": {
            "dataset_seed": d3c.SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ordering": "named deterministic hash salts",
            "no_stochastic_model_generation": True,
            "no_seed_sweep": True,
            "no_discarded_seed": True,
            "no_seed_replacement": True,
        },
        "prior_replication": {
            "policy": (
                "Raw Day-3 metrics are reported alongside the recorded Day-1 and "
                "Day-2 TEST values descriptively. There is no replication gate: "
                "Day-3 data is not modified if it fails to reproduce prior "
                "values. A failed replication is a result."
            )
        },
        "exclusion_rules": EXCLUSION_RULES,
        "failure_handling": {
            "prior_immutability_failure": (
                "If any prior hash changed, PRIOR_EXPERIMENT_IMMUTABILITY = FAIL "
                "and DAY3_ANALYSIS = BLOCKED. Files are not silently restored."
            ),
            "reachability_disagreement": (
                "If Method A and Method B disagree on any artifact, "
                "ORACLE_IMPLEMENTATION = FAIL and DAY3_ANALYSIS = BLOCKED."
            ),
            "nesting_violation": (
                "A violation of TCE_G >= TCE_F >= TCE_A is an implementation "
                "defect and blocks the analysis."
            ),
            "incomplete_model_run": (
                "If a judge's raw scores do not contain all 2916 prompts exactly "
                "once, that judge is BLOCKED and reported as such."
            ),
            "infrastructure_failure": (
                "A node that fails before producing output is documented and the "
                "job resubmitted outcome-independently, per existing cluster "
                "policy. Scientific outcomes are never inspected to choose nodes "
                "or reruns."
            ),
            "implementation_defect": (
                "Stop the affected stage, preserve the defective output, "
                "document it, decide whether the protocol is affected, amend if "
                "necessary, and rerun only after an outcome-independent "
                "correction. Poor OTCE is not a defect."
            ),
            "no_outcome_driven_tuning": NO_OUTCOME_DRIVEN_TUNING,
        },
        "figure_plan": FIGURE_PLAN,
        "stopping_rule": STOPPING_RULE,
        "analysis_code_paths": list(d3c.ANALYSIS_CODE_PATHS),
        "source_manifest_at_freeze": provenance.source_manifest(
            d3c.ANALYSIS_CODE_PATHS, root=root
        ),
        "git_at_freeze": provenance.git_state(root),
    }


def _bullets(items) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_markdown(document: dict, digest: str) -> str:
    """Render the human-readable preregistration from the frozen JSON."""
    models = document["judge_models"]
    theorem = document["reachability_theorem"]
    hierarchy = document["oracle_hierarchy"]
    primary = document["primary_metrics"]
    boot = document["bootstrap"]
    novelty = document["novelty_boundaries"]
    rel = document["relationship_to_prior"]
    data = document["dataset"]
    disjoint = document["disjointness"]

    hypothesis_rows = "\n".join(
        f"| **{h['id']}** | {h['label']} | {h['statement']} | `{h['quantity']}` |"
        for h in document["hypotheses"]
    )
    model_rows = "\n".join(
        f"| `{m}` | `{models['revisions'][m]}` | {models['scoring_methods'][m]} | "
        f"A={models['label_token_ids'][m]['met']} "
        f"B={models['label_token_ids'][m]['not_met']} | "
        f"{models['status_at_freeze'][m]} |"
        for m in models["model_ids"]
    )
    figure_rows = "\n".join(
        f"| {f['id']} | {f['title']} | {f['content']} | `{f['path']}` |"
        for f in document["figure_plan"]
    )
    overlap_rows = "\n".join(
        f"| {label} | "
        + " | ".join(
            str(disjoint[key][name])
            for name in ("day1", "day2_calibration", "day2_test")
        )
        + " |"
        for label, key in (
            ("artifact_id overlap", "artifact_id_overlap"),
            ("rendered prompt overlap", "rendered_prompt_overlap"),
            ("family-C (target, reported) pair overlap", "family_c_target_pair_overlap"),
        )
    )

    return f"""# FormalCRRC Day-3 preregistration

## Translation limits of threshold-response curves
### Oracle shape-preserving localization bounds

**Frozen:** {document['frozen_at']}
**SHA-256:** `{digest}`
**Machine-readable copy:** [`artifacts/day3/preregistration.json`](../artifacts/day3/preregistration.json)
**Checksum:** [`artifacts/day3/preregistration.sha256`](../artifacts/day3/preregistration.sha256)
**Amendments:** {len(document['amendments'])}

Frozen **before any Day-3 study inference**. Generated from the same object as
the JSON copy, so the two cannot disagree.

> Once frozen, this document is never silently edited. A genuine implementation
> defect is corrected by a numbered amendment that preserves the original and
> carries its own checksum. **Unexpected results never justify an amendment.**

---

## 1. Relationship to Days 1 and 2

{rel['statement']}

Prior results are **{rel['prior_results_status']}**.

| | |
|---|---|
| Day-1 preregistration SHA-256 | `{rel['day1_preregistration_sha256']}` |
| Day-2 preregistration SHA-256 | `{rel['day2_preregistration_sha256']}` |
| Prior-experiment fingerprint | `{rel['prior_manifest_sha256']}` over {rel['prior_manifest_n_files']} files |
| Day-1 dataset content SHA-256 | `{rel['day1_dataset_content_sha256']}` |

See [`DAY3_RATIONALE.md`](DAY3_RATIONALE.md) for the progression.

## 2. Research questions

**Primary.** {document['research_questions']['primary']}

**Secondary.** {document['research_questions']['secondary_reachability']}

**Secondary.** {document['research_questions']['secondary_variation']}

**Secondary.** {document['research_questions']['secondary_hierarchy']}

## 3. Hypotheses

| | Label | Statement | Quantity |
|---|---|---|---|
{hypothesis_rows}

## 4. Novelty boundaries

**Claimed contribution.** {novelty['claimed_contribution']}

The object of interest is **{novelty['object_of_interest']}**.
{novelty['mathematics_novelty']}

**Explicitly not claimed:**

{_bullets(novelty['explicitly_not_claimed'])}

**Permitted phrasings:**

{_bullets(novelty['permitted_phrasings'])}

**Forbidden phrasings:**

{_bullets(novelty['forbidden_phrasings'])}

## 5. The formal model

```text
M(s) = S_A - S_B            P(s) = sigmoid(M(s))
j_hat(alpha) = min{{s : M(s) + alpha < 0}}, or 9
j*           = min{{s : Y(s) = 0}},          or 9
```

The admissible class is `{document['formal_model']['translation_class']}` and
nothing stronger. {document['formal_model']['invariance']}

## 6. Reachability theorem

A crossing at level `j` in 1..8 is reachable by some additive translation **iff**

```text
{theorem['condition']}
```

{theorem['proof']}

The complete reachable set is

```text
{theorem['reachable_set']}
```

**Two implementations.** {theorem['two_implementations']}

**Ties.** {theorem['tie_policy']}

## 7. Primary metrics

**OTCE** — `{primary['OTCE']['definition']}` — {primary['OTCE']['meaning']}.
Guarantee: {primary['OTCE']['guarantee']}. {primary['OTCE']['oracle']}.

**TRR** — `{primary['TRR']['definition']}`. The **nontrivial** rate is
{primary['TRR']['nontrivial']}. Both are preregistered.

**SLR** — `{primary['SLR']['SLR1']}` and `{primary['SLR']['SLR2']}`.
Terminology: {primary['SLR']['terminology']}.

## 8. Secondary metrics

| | |
|---|---|
| RC | {document['secondary_metrics']['RC']} |
| PRR | {document['secondary_metrics']['PRR']} |
| Shape | {', '.join(document['secondary_metrics']['shape_metrics'])} |

{document['secondary_metrics']['shape_metric_associations']}

## 9. Oracle hierarchy

| Class | Freedom |
|---|---|
| **G** global | {hierarchy['global']} |
| **F** family | {hierarchy['family']} |
| **A** artifact | {hierarchy['artifact']} |

{hierarchy['nesting']}. {hierarchy['assertion']}.

These are {hierarchy['not_a_variance_decomposition']}, and
{hierarchy['not_deployment']}.

**Exact optimisation.** {hierarchy['no_gradient_descent']}.
{hierarchy['enumeration']}.

Tie-break, in order: {', '.join(hierarchy['tie_break'])}.

## 10. Translation taxonomy

{chr(10).join(f'- **{k}** — {v}' for k, v in document['taxonomy'].items() if isinstance(v, str))}

{document['taxonomy']['note']}

## 11. Dataset

| | |
|---|---|
| Partition | {data['partition']} (tag `{data['partition_tag']}`) |
| Seed | {data['seed']} |
| Role | {data['role']} |
| Artifacts | {data['n_artifacts']} |
| Rubric-response pairs | {data['n_rows']} per judge |
| Content SHA-256 | `{data['content_sha256']}` |

{data['generator']}

### Disjointness from every prior evaluated set

| Check | Day 1 | Day-2 CAL | Day-2 TEST |
|---|---|---|---|
{overlap_rows}

{data['family_c_uniqueness_rule']}

Family-C candidate strings may coincide because the reported value comes from a
bounded integer space; those coincidences are recorded, not filtered.

## 12. Judge panel

{models['policy']}

| Model | Revision | Scoring rule | Label tokens | Status at freeze |
|---|---|---|---|---|
{model_rows}

dtype `{models['dtype']}`. Continuity: {models['continuity_check']}.

## 13. Prompt

Reused **byte-identically**, SHA-256
`{document['prompt']['template_sha256']}` (prior days:
`{document['prompt']['prior_template_sha256']}`).

```
{document['prompt']['template']}
```

Prohibited: {', '.join(document['prompt']['prohibited'])}.

## 14. Inference

- Batch size {document['inference']['batch_size']}
- Order salt `{document['inference']['order_salt']}`
- {document['inference']['order']}
- Console output: {document['inference']['console_output']}

## 15. Bootstrap

| | |
|---|---|
| Replicates | {boot['replicates']} |
| Seed | {boot['seed']} |
| Unit | `{boot['unit']}` |
| Cluster rule | {boot['cluster_rule']} |
| Stratification | {boot['stratify_by']} |
| Interval | {boot['interval_method']} at {boot['coverage']:.0%} |

Oracle refitting: global — {boot['oracle_global']}; family —
{boot['oracle_family']}; artifact — {boot['oracle_artifact']}.
{boot['paired_comparisons']}.

Prohibited:

{_bullets(boot['prohibited'])}

## 16. Exclusion rules

{chr(10).join(f'- **{k}** — {v}' for k, v in document['exclusion_rules'].items() if isinstance(v, str))}

Never filtered: {', '.join(document['exclusion_rules']['never_filtered'])}.
{document['exclusion_rules']['rationale']}

## 17. Failure handling

{chr(10).join(f'- **{k}** — {v}' for k, v in document['failure_handling'].items() if isinstance(v, str))}

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

The Day-3 integrity manifest hashes **exactly this file list**, so the freeze
digest and the final digest are directly comparable.

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

    target = root / d3c.PREREG_JSON_PATH
    sha_path = root / d3c.PREREG_SHA_PATH
    if target.exists():
        print(
            f"REFUSING to overwrite the frozen preregistration at "
            f"{d3c.PREREG_JSON_PATH}.\nA frozen preregistration may only be "
            "corrected by a numbered, checksummed amendment.",
            flush=True,
        )
        return 1

    document = build_document(root)
    provenance.write_json(target, document)
    digest = provenance.write_checksum_file(target, sha_path)

    markdown = root / "docs/PREREGISTRATION_DAY3.md"
    markdown.write_text(render_markdown(document, digest), encoding="utf-8")

    print(f"wrote {d3c.PREREG_JSON_PATH}", flush=True)
    print(f"wrote {d3c.PREREG_SHA_PATH}", flush=True)
    print("wrote docs/PREREGISTRATION_DAY3.md", flush=True)
    print(f"preregistration sha256: {digest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
