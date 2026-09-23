"""Preregistered constants for the FormalCRRC Qwen2.5 Reason-Then-Score Ablation.

Kept in its own module so this study adds no constant to any prior configuration
and touches no file a previous preregistration pinned by hash. Everything here
is serialised verbatim into
``artifacts/reason_then_score_ablation/preregistration.json`` and may not change
after the freeze except through a numbered, checksummed amendment.

The design is a **within-model** ablation. Condition O is the frozen Day-3
record for this exact model and revision, read only and never rerun. Condition R
runs the same weights over the same artifacts under an elicited
reason-then-score protocol. Holding the model fixed is the whole point: the
reasoning-anchor study could not separate model capability from readout
procedure, and this one can speak to that confound because only the protocol
moves.

What moves is a *composite*: the instruction text, the presence of a generated
trace, where A/B is scored, and immediate-logit versus conditional
sequence-likelihood readout. The supported claim is therefore about the
reason-then-score protocol, never about "reasoning" as an isolated factor.
"""

from __future__ import annotations

from typing import Final

from formalcrrc.config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED  # noqa: F401
from formalcrrc.config import PROMPT_TEMPLATE as _DAY3_PROMPT_TEMPLATE

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

STUDY_NAME: Final[str] = "FormalCRRC Qwen2.5 Reason-Then-Score Protocol Ablation"

CONDITION_ORIGINAL: Final[str] = "immediate"
CONDITION_REASONING: Final[str] = "reason_then_score"

#: What the intervention is, stated so the freeze records it verbatim.
INTERVENTION: Final[str] = (
    "replacing immediate next-token A/B readout with an elicited "
    "reason-then-score protocol, holding model, revision, dataset, artifacts, "
    "formal thresholds and A/B semantics fixed"
)

#: The composite nature of the intervention, restated in the results document.
INTERVENTION_IS_COMPOSITE: Final[str] = (
    "The protocol intervention jointly changes the evaluation instruction, the "
    "generation of an intermediate trace, the position at which A/B is scored, "
    "and the immediate-next-token versus conditional sequence-likelihood "
    "readout. The supported causal statement is the effect of the "
    "reason-then-score protocol, not the effect of reasoning itself."
)

# --------------------------------------------------------------------------
# Model -- recovered from frozen Day-3 provenance, never floated
# --------------------------------------------------------------------------

MODEL_ID: Final[str] = "Qwen/Qwen2.5-14B-Instruct"
MODEL_SLUG: Final[str] = "qwen2_5_14b_instruct"
MODEL_SHORT_NAME: Final[str] = "Qwen2.5-14B"

#: Read from artifacts/day3/model_provenance.json and asserted by test.
#: A floating Hugging Face revision is not acceptable: the comparator is the
#: frozen Day-3 record for exactly these weights.
DAY3_REVISION: Final[str] = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
DAY3_TOKENIZER_REVISION: Final[str] = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
DAY3_CHAT_TEMPLATE_SHA256: Final[str] = (
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)

MODEL_DTYPE: Final[str] = "bfloat16"
QUANTIZATION: Final[str] = "none"

FORBIDDEN_QUANTIZATION: Final[tuple[str, ...]] = (
    "fp8", "gptq", "awq", "bitsandbytes", "gguf", "int8", "int4",
)

# --------------------------------------------------------------------------
# Why the Qwen3 native protocol cannot be copied literally
# --------------------------------------------------------------------------

PROTOCOL_RATIONALE: Final[str] = (
    "Qwen2.5-14B-Instruct has no thinking-only chat-template mechanism, so this "
    "ablation cannot use a native end-of-thinking boundary without inventing "
    "model behaviour that was never part of Qwen2.5. The model is instead given "
    "an explicit reasoning instruction and generates until a fixed preregistered "
    "final-answer marker. What is evaluated is therefore elicited "
    "reason-then-score, not native thinking, and the two must never be described "
    "as the same protocol."
)

PROTOCOL: Final[str] = "elicited_reason_then_score"

# --------------------------------------------------------------------------
# The reason-then-score prompt
# --------------------------------------------------------------------------

#: The fixed reasoning scaffold appended to the frozen Day-3 semantic content.
#: Frozen before inference and never paraphrased after observing any output.
REASONING_SCAFFOLD: Final[str] = (
    "Before giving the final label, reason about whether the stated criterion "
    "is met.\n"
    "Do not give the final A or B label until the reasoning is complete.\n"
    "When the reasoning is complete, write exactly:\n"
    "FINAL:\n"
    "and then give the label."
)

#: Separator between the frozen Day-3 body and the appended scaffold.
SCAFFOLD_SEPARATOR: Final[str] = "\n\n"

#: The complete reason-then-score template, derived **programmatically** from
#: the frozen Day-3 template rather than retyped.
#:
#: The governing draft quotes the Day-3 prompt in a flattened form -- rubric and
#: criterion on one line, no blank lines between sections -- which is not what
#: the frozen template actually contains. Retyping that flattened rendering
#: would have changed the prompt's whitespace and layout in addition to adding
#: the scaffold, confounding the protocol ablation with a formatting change.
#: Appending to the frozen template instead makes the diff purely additive,
#: which is what the draft asks for: "preserved as closely as possible", "adds
#: only a fixed reasoning scaffold". The scaffold text itself is verbatim.
RTS_PROMPT_TEMPLATE: Final[str] = (
    _DAY3_PROMPT_TEMPLATE + SCAFFOLD_SEPARATOR + REASONING_SCAFFOLD
)

#: Recorded so the freeze pins how the template was constructed.
PROMPT_CONSTRUCTION: Final[str] = (
    "RTS_PROMPT_TEMPLATE = formalcrrc.config.PROMPT_TEMPLATE + '\\n\\n' + "
    "REASONING_SCAFFOLD. Derived programmatically from the frozen Day-3 "
    "template; never retyped. The diff against Day 3 is therefore purely "
    "additive and the rubric and candidate-response bodies are untouched."
)

# --------------------------------------------------------------------------
# Final-answer marker
# --------------------------------------------------------------------------

MARKER: Final[str] = "FINAL:"

#: Generation stops immediately after the complete marker, before the model can
#: emit the label itself. The primary margin must come from Stage-D scoring, not
#: from a token the model sampled.
MARKER_STOP_RULE: Final[str] = (
    "Sequence-aware stop on the complete marker string. Generation halts after "
    "the last token of the marker and before the next generated token, so the "
    "model never produces the A/B label used for the primary margin."
)

# --------------------------------------------------------------------------
# Stage R -- elicited reasoning generation
# --------------------------------------------------------------------------

TEMPERATURE: Final[float] = 0.6
TOP_P: Final[float] = 0.95
TOP_K: Final[int] = 20
MIN_P: Final[float] = 0.0
REASONING_SAMPLES_PER_ROW: Final[int] = 1
REASONING_TOKEN_BUDGET: Final[int] = 8192
EXPERIMENT_SEED: Final[int] = 42

#: Identical rule to the reasoning-anchor study, reused deliberately so the two
#: experiments are procedurally comparable.
SEED_RULE: Final[str] = (
    "seed_r = int.from_bytes("
    "sha256(f'{base_seed}|{artifact_id}|{strictness_index}'"
    ".encode('utf-8')).digest()[:8], 'big') % 2147483647. "
    "Same convention as the reasoning-anchor study. SHA-256 is used because "
    "Python's built-in hash() is process-randomised and would not reproduce."
)
SEED_MODULUS: Final[int] = 2147483647

GENERATION_PARAMETER_SOURCE: Final[str] = (
    "chosen for cross-study procedural comparability with the reasoning-anchor "
    "experiment; not claimed to be optimised for Qwen2.5"
)

# --------------------------------------------------------------------------
# Stage D -- final decision scoring
# --------------------------------------------------------------------------

#: Candidates carry a leading ASCII space, because that is what follows the
#: marker in natural continuation. They are tokenised in context, never alone.
CANDIDATE_MET: Final[str] = " A"
CANDIDATE_NOT_MET: Final[str] = " B"

CANDIDATE_TOKENIZATION: Final[str] = "prompt_continuation_difference"
SCORING_METHOD: Final[str] = "sequence_loglikelihood"
CANONICAL_MARGIN: Final[str] = "M_R = L_A - L_B"

#: Explicitly excluded, because both would change what the margin means.
NO_EOS_PROBABILITY: Final[bool] = True
NO_LENGTH_NORMALIZATION: Final[bool] = True

#: Unchanged from Day 3, restated so the freeze records them.
CROSSING_CONVENTION: Final[str] = (
    "j_hat = min{s : M(s) < 0}, else 9. M = 0 counts as 'criterion met'."
)
DECISION_CONVENTION: Final[str] = "D(s) = 1[M(s) >= 0]"
REACHABILITY_CONDITION: Final[str] = (
    "M(j) < min_{t < j} M(t)  (strict prefix record low); crossings 0 and 9 "
    "always reachable"
)
TIE_POLICY: Final[str] = (
    "Exact stored float64 comparison under the strict relation. No tolerance is "
    "introduced anywhere, including in the tie-versus-inversion decomposition."
)

# --------------------------------------------------------------------------
# Unreachable-failure decomposition
# --------------------------------------------------------------------------

FAILURE_STRICT_INVERSION: Final[str] = "strict_inversion"
FAILURE_TIE_ONLY: Final[str] = "tie_only"
FAILURE_NONE: Final[str] = "reachable"

FAILURE_RULE: Final[str] = (
    "For an unreachable nontrivial boundary j*, classify strict_inversion when "
    "some earlier threshold t < j* has M(t) < M(j*), and tie_only when no such "
    "strict inversion exists but some earlier threshold has M(t) = M(j*). "
    "Strict inversion takes precedence when both occur, because an actual "
    "ordering inversion blocks reachability independently of any tie. Exact "
    "stored values; no tolerance."
)

# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------

SOURCE_PARTITION: Final[str] = "DAY3_TEST"
SOURCE_SEED: Final[int] = 45
N_ARTIFACTS: Final[int] = 324
N_STRICTNESS_LEVELS: Final[int] = 9
N_ROWS: Final[int] = 2916
N_NONTRIVIAL: Final[int] = 288

# --------------------------------------------------------------------------
# Completeness
# --------------------------------------------------------------------------

TRUNCATION_WARNING_FRACTION: Final[float] = 0.01
TRUNCATION_WARNING_FLAG: Final[str] = "REASONING_COMPLETENESS_WARNING"

CURVE_EXCLUSION_RULE: Final[str] = (
    "A curve is excluded from curve-level TCE/TRR if and only if one or more of "
    "its nine strictness levels lacks a valid reason-then-score margin. No "
    "exclusion depends on the value of any margin or on model performance."
)

#: A row that emits a standalone A/B label before the marker is flagged but not
#: excluded, provided the marker is still reached and a margin is computable.
PREMATURE_LABEL_FLAG: Final[str] = "premature_label_observed"

# --------------------------------------------------------------------------
# Frozen comparator -- Condition O, read only
# --------------------------------------------------------------------------

#: Re-read from machine artifacts at analysis time; these are the expected
#: values and are asserted, never substituted for the recomputation.
COMPARATOR_TCE: Final[float] = 3.373
COMPARATOR_TRR: Final[float] = 0.4792
COMPARATOR_UNREACHABLE_PCT: Final[float] = 52.08

COMPARATOR_SOURCE: Final[str] = (
    "artifacts/day3/raw_scores/qwen2_5_14b_instruct.parquet, read only and "
    "re-verified byte-identical after the ablation"
)

# --------------------------------------------------------------------------
# Structural reference baselines
# --------------------------------------------------------------------------

BASELINE_FIXED_CENTER_TCE: Final[float] = 20.0 / 9.0
BASELINE_RANDOM_CROSSING_TCE: Final[float] = 80.0 / 27.0
BASELINE_ALWAYS_MET_TCE: Final[float] = 4.0
BASELINE_ALWAYS_NOT_MET_TCE: Final[float] = 5.0
BASELINE_FIXED_CENTER_ACC: Final[float] = 61.0 / 81.0
BASELINE_RANDOM_CROSSING_ACC: Final[float] = 163.0 / 243.0
BASELINE_MAJORITY_ACC: Final[float] = 5.0 / 9.0

# --------------------------------------------------------------------------
# Qwen3 anchor -- external context only
# --------------------------------------------------------------------------

ANCHOR_MODEL: Final[str] = "Qwen/Qwen3-30B-A3B-Thinking-2507"
ANCHOR_TCE: Final[float] = 0.401
ANCHOR_TRR: Final[float] = 1.0
ANCHOR_UNREACHABLE_PCT: Final[float] = 0.0

ANCHOR_COMPARISON_STATUS: Final[str] = (
    "external context only. The anchor uses native thinking under a different "
    "model family, scale and training regime; this study uses elicited "
    "reason-then-score on Qwen2.5. The two protocols are not identical and no "
    "pooled significance test across these model/protocol regimes is performed."
)

# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

BOOTSTRAP_SCHEME: Final[str] = (
    "artifact-curve clustered resampling with replacement; all nine strictness "
    "levels of a curve stay together; for paired original-versus-reasoning "
    "comparisons the two conditions of one artifact are drawn jointly; family "
    "strata preserved for family analyses; no row-level bootstrap"
)
BOOTSTRAP_CI_DESCRIPTION: Final[str] = "percentile, 95%"

#: A zero-count result gets an exact interval, not a degenerate bootstrap one.
ZERO_COUNT_RULE: Final[str] = (
    "If the reason-then-score unreachable count is zero over all nontrivial "
    "curves, report the observed count out of N together with an exact "
    "two-sided 95% Clopper-Pearson interval for the unreachable probability, "
    "and lead with its upper bound. A bootstrap interval of [0, 0] must not be "
    "presented as population uncertainty; it is retained in machine artifacts "
    "for completeness only."
)
CLOPPER_PEARSON_LEVEL: Final[float] = 0.95

# --------------------------------------------------------------------------
# Paths -- all additive
# --------------------------------------------------------------------------

ARTIFACT_DIR: Final[str] = "artifacts/reason_then_score_ablation"
FIGURE_DIR: Final[str] = "figures/reason_then_score_ablation"

PREREG_MD_PATH: Final[str] = "docs/PREREGISTRATION_REASON_THEN_SCORE_ABLATION.md"
PREREG_JSON_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration.json"
PREREG_SHA_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration.sha256"
PREREG_FROZEN_COPY_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration_frozen.md"

FROZEN_FINGERPRINT_PATH: Final[str] = f"{ARTIFACT_DIR}/frozen_record_fingerprint.json"
MODEL_PROVENANCE_PATH: Final[str] = f"{ARTIFACT_DIR}/model_provenance.json"
PREFLIGHT_PATH: Final[str] = f"{ARTIFACT_DIR}/preflight.json"
READINESS_PATH: Final[str] = f"{ARTIFACT_DIR}/readiness.json"
PROMPT_DIFF_PATH: Final[str] = f"{ARTIFACT_DIR}/prompt_diff.json"
COMPARATOR_PATH: Final[str] = f"{ARTIFACT_DIR}/comparator_recovery.json"
GENERATION_CONFIG_PATH: Final[str] = f"{ARTIFACT_DIR}/generation_config.json"

RAW_ROWS_PATH: Final[str] = f"{ARTIFACT_DIR}/raw_rows.parquet"
PAIRED_CURVES_PATH: Final[str] = f"{ARTIFACT_DIR}/paired_curves.parquet"
METRICS_PATH: Final[str] = f"{ARTIFACT_DIR}/metrics_by_artifact.parquet"
FAILURE_DECOMP_PATH: Final[str] = f"{ARTIFACT_DIR}/failure_decomposition.parquet"
BOOTSTRAP_PATH: Final[str] = f"{ARTIFACT_DIR}/bootstrap_results.parquet"
FAMILY_SUMMARY_PATH: Final[str] = f"{ARTIFACT_DIR}/family_summary.parquet"
TABLES_DIR: Final[str] = f"{ARTIFACT_DIR}/tables"
SUMMARY_PATH: Final[str] = f"{ARTIFACT_DIR}/summary.json"
INTEGRITY_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/integrity_manifest.json"
RETRY_LOG_PATH: Final[str] = f"{ARTIFACT_DIR}/retry_log.json"

RESULTS_MD_PATH: Final[str] = "docs/RESULTS_REASON_THEN_SCORE_ABLATION.md"

# --------------------------------------------------------------------------
# Frozen record that must remain byte-identical
# --------------------------------------------------------------------------

PROTECTED_TREES: Final[tuple[str, ...]] = (
    "artifacts/day1",
    "artifacts/day2",
    "artifacts/day3",
    "artifacts/labelswap",
    "artifacts/reasoning_anchor",
    "artifacts/poststudy_baselines",
    "artifacts/poststudy_theory",
    "figures/day1",
    "figures/day2",
    "figures/day3",
    "figures/labelswap",
    "figures/reasoning_anchor",
)

PROTECTED_DOCS: Final[tuple[str, ...]] = (
    "docs/PREREGISTRATION_DAY1.md",
    "docs/PREREGISTRATION_DAY2.md",
    "docs/PREREGISTRATION_DAY3.md",
    "docs/PREREGISTRATION_LABEL_SWAP.md",
    "docs/PREREGISTRATION_REASONING_ANCHOR.md",
    "docs/RESULTS_DAY1.md",
    "docs/RESULTS_DAY2.md",
    "docs/RESULTS_DAY3.md",
    "docs/RESULTS_LABEL_SWAP.md",
    "docs/RESULTS_REASONING_ANCHOR.md",
    "docs/DAY1_ERRATA.md",
    "docs/DAY1_PROVENANCE_AUDIT.md",
    "docs/DAY1_INTEGRITY_ADDENDUM.md",
    "docs/DAY2_RATIONALE.md",
    "docs/DAY3_RATIONALE.md",
    "docs/NOVELTY_BOUNDARIES.md",
    "docs/NOVELTY_BOUNDARIES_DAY2.md",
    "docs/NOVELTY_BOUNDARIES_DAY3.md",
    "docs/LINEAGE.md",
    "docs/FORMALCRRC_DAY1_SPEC.md",
    "docs/POSTSTUDY_BASELINE_AUDIT.md",
)

#: Read by this study and never written.
READ_ONLY_INPUTS: Final[tuple[str, ...]] = (
    "artifacts/day3/test_dataset.parquet",
    "artifacts/day3/test_manifest.json",
    "artifacts/day3/prompt_manifest.parquet",
    "artifacts/day3/model_provenance.json",
    "artifacts/day3/metrics_by_artifact.parquet",
    "artifacts/day3/raw_scores/qwen2_5_14b_instruct.parquet",
    "artifacts/reasoning_anchor/summary.json",
)

ANALYSIS_CODE_PATHS: Final[tuple[str, ...]] = (
    "src/formalcrrc/config.py",
    "src/formalcrrc/predicates.py",
    "src/formalcrrc/prompts.py",
    "src/formalcrrc/scoring.py",
    "src/formalcrrc/provenance.py",
    "src/formalcrrc/day2.py",
    "src/formalcrrc/day3.py",
    "src/formalcrrc/rts_ablation.py",
    "src/formalcrrc/rts_ablation_bootstrap.py",
    "src/formalcrrc/rts_ablation_config.py",
    "scripts/preflight_rts_ablation.py",
    "scripts/freeze_rts_ablation_preregistration.py",
    "scripts/run_rts_ablation_inference.py",
    "scripts/analyze_rts_ablation.py",
    "scripts/figures_rts_ablation.py",
    "scripts/verify_rts_ablation_integrity.py",
)
