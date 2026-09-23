"""Preregistered constants for the FormalCRRC Reasoning-Capable External Anchor Study.

Kept in its own module so this study adds no constant to the Day-1, Day-2, Day-3
or label-swap configurations, and touches no file any prior preregistration
pinned by hash. Everything here is serialised verbatim into
``artifacts/reasoning_anchor/preregistration.json`` and may not change after the
freeze except through a numbered, checksummed amendment.

The study reuses the frozen Day-3 TEST partition, the Day-3 user-message
content, the Day-3 crossing convention, the Day-3 reachability rule and the
Day-3 tie policy **unchanged**. What differs is the measurement procedure: the
anchor is a thinking-only reasoning model, so its A/B decision is scored *after*
it has reasoned, not at the first token following the prompt.
"""

from __future__ import annotations

from typing import Final

from formalcrrc.config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED  # noqa: F401

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

STUDY_NAME: Final[str] = "FormalCRRC Reasoning-Capable External Anchor Study"

#: Namespace prefix for this study's own outputs. Artifact ids stay exactly as
#: Day 3 wrote them, so pairing with the frozen record is by identity.
CONDITION_TAG: Final[str] = "RANCHOR"

#: What this model is, and is not. Used verbatim in the results document.
ANCHOR_DESCRIPTION: Final[str] = "reasoning-capable open-weight external anchor"

# --------------------------------------------------------------------------
# The anchor model
# --------------------------------------------------------------------------

MODEL_ID: Final[str] = "Qwen/Qwen3-30B-A3B-Thinking-2507"
MODEL_SLUG: Final[str] = "qwen3_30b_a3b_thinking_2507"
MODEL_SHORT_NAME: Final[str] = "Qwen3-30B-A3B-Thinking"

#: Official non-quantised weights. Any quantisation is a different study.
MODEL_DTYPE: Final[str] = "bfloat16"
QUANTIZATION: Final[str] = "none"

#: Forbidden load paths. Presence of any of these stops the experiment.
FORBIDDEN_QUANTIZATION: Final[tuple[str, ...]] = (
    "fp8",
    "gptq",
    "awq",
    "bitsandbytes",
    "gguf",
    "int8",
    "int4",
)

# --------------------------------------------------------------------------
# Why the Day-3 procedure cannot be reused unchanged
# --------------------------------------------------------------------------

SCORING_RATIONALE: Final[str] = (
    "The Day-3 panel was scored at the immediate response boundary, "
    "M = S_A - S_B on the first generated token. This anchor's native chat "
    "template opens a thinking block in the generation prompt, so the first "
    "generated token begins its reasoning rather than its judgment. Scoring "
    "A/B there would measure the start of reasoning, not the final decision."
)

#: Two-stage measurement. Stage R generates, Stage D scores.
PROTOCOL: Final[str] = "reason_then_score"

# --------------------------------------------------------------------------
# Stage R -- reasoning generation
# --------------------------------------------------------------------------

TEMPERATURE: Final[float] = 0.6
TOP_P: Final[float] = 0.95
TOP_K: Final[int] = 20
MIN_P: Final[float] = 0.0
REASONING_SAMPLES_PER_ROW: Final[int] = 1
REASONING_TOKEN_BUDGET: Final[int] = 8192

#: Experiment-level base seed. Per-row seeds derive from it deterministically.
EXPERIMENT_SEED: Final[int] = 42

#: The exact per-row seed rule. Documented here because it is preregistered.
SEED_RULE: Final[str] = (
    "seed_r = int.from_bytes("
    "sha256(f'{base_seed}|{artifact_id}|{strictness_index}'"
    ".encode('utf-8')).digest()[:8], 'big') % 2147483647. "
    "SHA-256 is used because Python's built-in hash() is process-randomised "
    "and would not reproduce across runs. The reduction modulus is 2**31 - 1, "
    "the largest signed 32-bit prime, which keeps the seed inside the range "
    "every sampler accepts."
)
SEED_MODULUS: Final[int] = 2147483647

# --------------------------------------------------------------------------
# Stage D -- final decision scoring
# --------------------------------------------------------------------------

#: The native end-of-thinking delimiter. Its token id is resolved from the
#: official tokenizer before freeze and recorded in model provenance.
THINK_END_TEXT: Final[str] = "</think>"

#: Post-thinking separator, taken from the model's own chat template, whose
#: assistant branch renders a completed turn as
#: thinking-open, reasoning, newline, thinking-close, blank line, answer.
#: It is therefore model-native and was fixed before any study inference; it
#: was not selected by comparing FormalCRRC outcomes.
POST_THINK_SEPARATOR: Final[str] = "\n\n"
POST_THINK_SEPARATOR_BYTES: Final[str] = "0x0a 0x0a"
POST_THINK_SEPARATOR_SOURCE: Final[str] = (
    "Qwen3 native chat template, assistant branch: the completed assistant "
    "turn is rendered as the thinking-open delimiter, a newline, the reasoning "
    "content stripped of leading and trailing newlines, a newline, the "
    "thinking-close delimiter, then exactly two newlines, then the answer."
)

#: Candidate final answers. Semantics identical to Day 3; not label-swapped.
CANDIDATE_MET: Final[str] = "A"
CANDIDATE_NOT_MET: Final[str] = "B"

#: Candidates are tokenised as continuations of the real decision context,
#: never in isolation, and scored by full sequence log-likelihood so that a
#: multi-token candidate is handled correctly.
CANDIDATE_TOKENIZATION: Final[str] = "prompt_continuation_difference"
SCORING_METHOD: Final[str] = "sequence_loglikelihood"

CANONICAL_MARGIN: Final[str] = "M = L_A - L_B"

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
    "Exact stored float64 comparison under the strict relation. No tolerance "
    "is introduced. Tie frequencies are recorded descriptively and the rule is "
    "not changed after observing them."
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

#: Above this truncation fraction the study is reported with a warning banner.
TRUNCATION_WARNING_FRACTION: Final[float] = 0.01
TRUNCATION_WARNING_FLAG: Final[str] = "REASONING_COMPLETENESS_WARNING"

#: Curve exclusion is mechanical: a curve is dropped only when a threshold has
#: no valid primary margin. Never because of what the margin turned out to be.
CURVE_EXCLUSION_RULE: Final[str] = (
    "A curve is excluded from curve-level TCE/TRR if and only if one or more "
    "of its nine strictness levels lacks a valid primary margin (reasoning "
    "truncated, or scoring failed). No exclusion depends on the value of any "
    "margin or on model performance."
)

# --------------------------------------------------------------------------
# Structural reference baselines (exact, previously established)
# --------------------------------------------------------------------------

BASELINE_FIXED_CENTER_TCE: Final[float] = 20.0 / 9.0
BASELINE_RANDOM_CROSSING_TCE: Final[float] = 80.0 / 27.0
BASELINE_ALWAYS_MET_TCE: Final[float] = 4.0
BASELINE_ALWAYS_NOT_MET_TCE: Final[float] = 5.0

BASELINE_FIXED_CENTER_ACC: Final[float] = 61.0 / 81.0
BASELINE_RANDOM_CROSSING_ACC: Final[float] = 163.0 / 243.0
BASELINE_MAJORITY_ACC: Final[float] = 5.0 / 9.0

# --------------------------------------------------------------------------
# Frozen Day-3 contextual comparison values (read-only, never recomputed)
# --------------------------------------------------------------------------

DAY3_TCE: Final[dict[str, float]] = {
    "Qwen2.5-14B": 3.373,
    "Llama-3.1-8B": 4.904,
    "Mistral-7B": 2.938,
    "Gemma-2-9B": 3.361,
}

DAY3_UNREACHABLE_PCT: Final[dict[str, float]] = {
    "Qwen2.5-14B": 52.08,
    "Llama-3.1-8B": 72.92,
    "Mistral-7B": 66.67,
    "Gemma-2-9B": 39.58,
}

DAY3_UNREACHABLE_RANGE: Final[tuple[float, float]] = (39.58, 72.92)

COMPARISON_STATUS: Final[str] = (
    "descriptive only; the anchor uses a reason-then-score protocol rather "
    "than the original immediate next-token procedure, so no pooled hypothesis "
    "test treating the five models as exchangeable is performed"
)

# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

BOOTSTRAP_SCHEME: Final[str] = (
    "artifact-curve clustered resampling with replacement; all nine strictness "
    "levels of a curve stay together; family strata preserved for "
    "family-stratified analyses; no row-level bootstrap"
)
BOOTSTRAP_CI_DESCRIPTION: Final[str] = "percentile, 95%"

# --------------------------------------------------------------------------
# Paths -- all additive
# --------------------------------------------------------------------------

ARTIFACT_DIR: Final[str] = "artifacts/reasoning_anchor"
FIGURE_DIR: Final[str] = "figures/reasoning_anchor"

PREREG_MD_PATH: Final[str] = "docs/PREREGISTRATION_REASONING_ANCHOR.md"
PREREG_JSON_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration.json"
PREREG_SHA_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration.sha256"
PREREG_FROZEN_COPY_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration_frozen.md"

FROZEN_FINGERPRINT_PATH: Final[str] = f"{ARTIFACT_DIR}/frozen_record_fingerprint.json"
MODEL_PROVENANCE_PATH: Final[str] = f"{ARTIFACT_DIR}/model_provenance.json"
PREFLIGHT_PATH: Final[str] = f"{ARTIFACT_DIR}/preflight.json"
READINESS_PATH: Final[str] = f"{ARTIFACT_DIR}/readiness.json"
GENERATION_CONFIG_PATH: Final[str] = f"{ARTIFACT_DIR}/generation_config.json"

RAW_ROWS_PATH: Final[str] = f"{ARTIFACT_DIR}/raw_rows.parquet"
REASONING_TRACES_PATH: Final[str] = f"{ARTIFACT_DIR}/reasoning_traces.parquet"
METRICS_PATH: Final[str] = f"{ARTIFACT_DIR}/metrics_by_artifact.parquet"
BOOTSTRAP_PATH: Final[str] = f"{ARTIFACT_DIR}/bootstrap_results.parquet"
FAMILY_SUMMARY_PATH: Final[str] = f"{ARTIFACT_DIR}/family_summary.parquet"
TABLES_DIR: Final[str] = f"{ARTIFACT_DIR}/tables"
SUMMARY_PATH: Final[str] = f"{ARTIFACT_DIR}/summary.json"
INTEGRITY_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/integrity_manifest.json"
RETRY_LOG_PATH: Final[str] = f"{ARTIFACT_DIR}/retry_log.json"
FINAL_CONSOLE_PATH: Final[str] = f"{ARTIFACT_DIR}/final_console_block.txt"

RESULTS_MD_PATH: Final[str] = "docs/RESULTS_REASONING_ANCHOR.md"

# --------------------------------------------------------------------------
# Frozen record that must remain byte-identical
# --------------------------------------------------------------------------

PROTECTED_TREES: Final[tuple[str, ...]] = (
    "artifacts/day1",
    "artifacts/day2",
    "artifacts/day3",
    "artifacts/labelswap",
    "artifacts/poststudy_baselines",
    "artifacts/poststudy_theory",
    "figures/day1",
    "figures/day2",
    "figures/day3",
    "figures/labelswap",
)

PROTECTED_DOCS: Final[tuple[str, ...]] = (
    "docs/PREREGISTRATION_DAY1.md",
    "docs/PREREGISTRATION_DAY2.md",
    "docs/PREREGISTRATION_DAY3.md",
    "docs/PREREGISTRATION_LABEL_SWAP.md",
    "docs/RESULTS_DAY1.md",
    "docs/RESULTS_DAY2.md",
    "docs/RESULTS_DAY3.md",
    "docs/RESULTS_LABEL_SWAP.md",
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

#: Day-3 outputs this study reads and must never write.
DAY3_READ_ONLY: Final[tuple[str, ...]] = (
    "artifacts/day3/test_dataset.parquet",
    "artifacts/day3/test_manifest.json",
    "artifacts/day3/prompt_manifest.parquet",
    "artifacts/day3/metrics_by_artifact.parquet",
)

#: Source files whose content defines this study's outcomes.
ANALYSIS_CODE_PATHS: Final[tuple[str, ...]] = (
    "src/formalcrrc/config.py",
    "src/formalcrrc/predicates.py",
    "src/formalcrrc/prompts.py",
    "src/formalcrrc/scoring.py",
    "src/formalcrrc/provenance.py",
    "src/formalcrrc/day2.py",
    "src/formalcrrc/day3.py",
    "src/formalcrrc/reasoning_anchor.py",
    "src/formalcrrc/reasoning_anchor_bootstrap.py",
    "src/formalcrrc/reasoning_anchor_config.py",
    "scripts/preflight_reasoning_anchor.py",
    "scripts/freeze_reasoning_anchor_preregistration.py",
    "scripts/run_reasoning_anchor_inference.py",
    "scripts/analyze_reasoning_anchor.py",
    "scripts/figures_reasoning_anchor.py",
    "scripts/verify_reasoning_anchor_integrity.py",
)
