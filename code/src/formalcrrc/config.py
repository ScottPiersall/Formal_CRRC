"""Single source of truth for every preregistered FormalCRRC Day-1 constant.

Nothing in this module may be changed after the preregistration is frozen except
through a numbered, checksummed amendment (see ``docs/PREREGISTRATION_DAY1.md``).
Every constant here is serialised verbatim into ``artifacts/day1/preregistration.json``.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------
# Global determinism
# --------------------------------------------------------------------------

#: Master seed for every deterministic randomised construction in Day 1.
SEED: Final[int] = 42

#: Salt mixed into the hash that fixes the frozen inference order.
INFERENCE_ORDER_SALT: Final[str] = "formalcrrc-day1-inference-order-v1"

#: Salt mixed into the hash that fixes the order in which judge models are run.
MODEL_ORDER_SALT: Final[str] = "formalcrrc-day1-model-order-v1"

# --------------------------------------------------------------------------
# Design dimensions
# --------------------------------------------------------------------------

#: Largest strictness index; the ladder is s = 0 .. SCALE_MAX.
SCALE_MAX: Final[int] = 8

#: Number of ordered strictness levels evaluated for every artifact.
N_STRICTNESS: Final[int] = SCALE_MAX + 1  # 9

#: Number of latent levels per predicate family (m, v or e in 0..8).
N_LATENT_LEVELS: Final[int] = SCALE_MAX + 1  # 9

#: Independently instantiated artifacts per (family, latent level) cell.
N_INSTANCES: Final[int] = 12

#: Number of items in the Required / Prohibited list of families A and B.
LIST_SIZE: Final[int] = 8

#: Markers emitted by every family-A / family-B candidate response.
#: Held constant across latent levels so that response length carries no signal.
MARKERS_PER_RESPONSE: Final[int] = 14

#: Inclusive range from which family-C target values are drawn.
NUMERIC_TARGET_MIN: Final[int] = 20
NUMERIC_TARGET_MAX: Final[int] = 200

#: Marker token format: MK- followed by MARKER_CODE_LEN uppercase letters.
MARKER_PREFIX: Final[str] = "MK-"
MARKER_CODE_LEN: Final[int] = 4
MARKER_ALPHABET: Final[str] = "ABCDEFGHJKLMNPQRSTVWXYZ"

# --------------------------------------------------------------------------
# Predicate families
# --------------------------------------------------------------------------

FAMILY_COVERAGE: Final[str] = "coverage"
FAMILY_MAX_VIOLATION: Final[str] = "max_violation"
FAMILY_NUMERIC_TOLERANCE: Final[str] = "numeric_tolerance"

FAMILIES: Final[tuple[str, ...]] = (
    FAMILY_COVERAGE,
    FAMILY_MAX_VIOLATION,
    FAMILY_NUMERIC_TOLERANCE,
)

FAMILY_CODES: Final[dict[str, str]] = {
    FAMILY_COVERAGE: "A",
    FAMILY_MAX_VIOLATION: "B",
    FAMILY_NUMERIC_TOLERANCE: "C",
}

#: Derived design totals, asserted by the test suite.
N_ARTIFACTS: Final[int] = len(FAMILIES) * N_LATENT_LEVELS * N_INSTANCES  # 324
N_ROWS: Final[int] = N_ARTIFACTS * N_STRICTNESS  # 2916

# --------------------------------------------------------------------------
# Judge prompt
# --------------------------------------------------------------------------

#: Token returned when the criterion is judged met.
LABEL_MET: Final[str] = "A"
#: Token returned when the criterion is judged not met.
LABEL_NOT_MET: Final[str] = "B"

#: The one and only judge prompt. A single user message; identical semantics for
#: every model; the rubric threshold is the sole study-variable text.
PROMPT_TEMPLATE: Final[str] = (
    "You are evaluating exactly one criterion for one candidate response.\n"
    "\n"
    "Use only the stated criterion. Do not reward or penalize the response for "
    "properties outside that criterion.\n"
    "\n"
    "Criterion:\n"
    "{rubric_text}\n"
    "\n"
    "Candidate response:\n"
    "{candidate_response}\n"
    "\n"
    "Return A if the criterion is met.\n"
    "Return B if the criterion is not met."
)

# --------------------------------------------------------------------------
# Judge models (preregistered panel)
# --------------------------------------------------------------------------

MODEL_IDS: Final[tuple[str, ...]] = (
    "Qwen/Qwen2.5-14B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
)

MODEL_SLUGS: Final[dict[str, str]] = {
    "Qwen/Qwen2.5-14B-Instruct": "qwen2_5_14b_instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3_1_8b_instruct",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral_7b_instruct_v0_3",
    "google/gemma-2-9b-it": "gemma_2_9b_it",
}

MODEL_SHORT_NAMES: Final[dict[str, str]] = {
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "google/gemma-2-9b-it": "Gemma-2-9B",
}

#: Weight dtype. No quantisation, no fine-tuning, no adapters for Day 1.
MODEL_DTYPE: Final[str] = "bfloat16"

#: Forward passes are run one prompt at a time so that no result depends on
#: padding or batch composition.
INFERENCE_BATCH_SIZE: Final[int] = 1

# --------------------------------------------------------------------------
# Probability extraction
# --------------------------------------------------------------------------

SCORING_PRIMARY: Final[str] = "single_token_next_logit"
SCORING_FALLBACK: Final[str] = "sequence_loglikelihood"

#: How the answer-label token ids are determined. Encoding the bare label string
#: is not sufficient: a SentencePiece tokenizer returns a word-boundary-marked
#: token for a string encoded in isolation, which is a different id from the one
#: that actually continues a prompt, and scoring it would read the wrong logit.
SCORING_LABEL_RESOLUTION: Final[str] = "prompt_continuation_difference"

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

#: Practical adjacent-monotonicity tolerance for the primary MVR.
MVR_TOLERANCE: Final[float] = 0.05

#: Secondary zero-tolerance MVR.
MVR_TOLERANCE_STRICT: Final[float] = 0.0

#: Probability at or above which the judge is treated as saying "met".
DECISION_THRESHOLD: Final[float] = 0.5

#: Sentinel first-fail index used when a curve never falls below the boundary.
NO_CROSSING_INDEX: Final[int] = N_STRICTNESS  # 9

# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

BOOTSTRAP_REPLICATES: Final[int] = 5000
BOOTSTRAP_SEED: Final[int] = 42
BOOTSTRAP_CI: Final[float] = 0.95
BOOTSTRAP_UNIT: Final[str] = "artifact_id"
BOOTSTRAP_METHOD: Final[str] = "percentile"
BOOTSTRAP_STRATIFY_BY: Final[str] = "family"

# --------------------------------------------------------------------------
# Paths (repository-relative)
# --------------------------------------------------------------------------

ARTIFACT_DIR: Final[str] = "artifacts/day1"
FIGURE_DIR: Final[str] = "figures/day1"

DATASET_PATH: Final[str] = f"{ARTIFACT_DIR}/formal_dataset.parquet"
DATASET_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/dataset_manifest.json"
PROMPT_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/prompt_manifest.parquet"
PREREG_JSON_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration.json"
PREREG_SHA_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration.sha256"
MODEL_PROVENANCE_PATH: Final[str] = f"{ARTIFACT_DIR}/model_provenance.json"
RAW_SCORES_DIR: Final[str] = f"{ARTIFACT_DIR}/raw_scores"
METRICS_PATH: Final[str] = f"{ARTIFACT_DIR}/metrics_by_artifact.parquet"
BOOTSTRAP_PATH: Final[str] = f"{ARTIFACT_DIR}/bootstrap_results.parquet"
SUMMARY_PATH: Final[str] = f"{ARTIFACT_DIR}/summary.json"
INTEGRITY_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/integrity_manifest.json"
FINAL_CONSOLE_PATH: Final[str] = f"{ARTIFACT_DIR}/final_console_block.txt"

ANALYSIS_CODE_PATHS: Final[tuple[str, ...]] = (
    "src/formalcrrc/config.py",
    "src/formalcrrc/predicates.py",
    "src/formalcrrc/dataset.py",
    "src/formalcrrc/prompts.py",
    "src/formalcrrc/scoring.py",
    "src/formalcrrc/metrics.py",
    "src/formalcrrc/bootstrap.py",
    "src/formalcrrc/provenance.py",
    "scripts/generate_day1_dataset.py",
    "scripts/run_day1_inference.py",
    "scripts/analyze_day1.py",
    "scripts/verify_day1_integrity.py",
)
