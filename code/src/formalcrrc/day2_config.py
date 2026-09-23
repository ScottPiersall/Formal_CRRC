"""Preregistered constants for FormalCRRC Day 2.

Kept in its own module so that Day 2 adds no constant to the Day-1
configuration. Everything here is serialised verbatim into
``artifacts/day2/preregistration.json`` and may not change after the freeze
except through a numbered, checksummed amendment.

Day 2 reuses the Day-1 predicate families, strictness ladder, prompt template,
answer labels, decision threshold, model panel and scoring rules unchanged; the
Day-1 constants remain authoritative for all of those.
"""

from __future__ import annotations

from typing import Final

from formalcrrc.config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED  # noqa: F401

# --------------------------------------------------------------------------
# Partitions
# --------------------------------------------------------------------------

PARTITION_CALIBRATION: Final[str] = "CAL"
PARTITION_TEST: Final[str] = "TST"
PARTITIONS: Final[tuple[str, ...]] = (PARTITION_CALIBRATION, PARTITION_TEST)

#: Master seeds for the two new, disjoint partitions.
SEED_CALIBRATION: Final[int] = 43
SEED_TEST: Final[int] = 44

SEEDS: Final[dict[str, int]] = {
    PARTITION_CALIBRATION: SEED_CALIBRATION,
    PARTITION_TEST: SEED_TEST,
}

#: Order in which partitions are generated. Family-C targets already used by an
#: earlier partition are excluded from later ones, so the union is disjoint.
GENERATION_ORDER: Final[tuple[str, ...]] = (
    PARTITION_CALIBRATION,
    PARTITION_TEST,
)

# --------------------------------------------------------------------------
# Inference ordering
# --------------------------------------------------------------------------

CALIBRATION_ORDER_SALT: Final[str] = "formalcrrc-day2-calibration-order-v1"
TEST_ORDER_SALT: Final[str] = "formalcrrc-day2-test-order-v1"
MODEL_ORDER_SALT: Final[str] = "formalcrrc-day2-model-order-v1"

ORDER_SALTS: Final[dict[str, str]] = {
    PARTITION_CALIBRATION: CALIBRATION_ORDER_SALT,
    PARTITION_TEST: TEST_ORDER_SALT,
}

# --------------------------------------------------------------------------
# Calibration model
# --------------------------------------------------------------------------

#: Primary calibration: one additive intercept per judge, slope fixed at 1.
CALIBRATION_PRIMARY: Final[str] = "global_intercept"

#: Secondary calibration: one additive intercept per judge and predicate family.
CALIBRATION_SECONDARY: Final[str] = "family_intercept"

#: Fixed slope. Day 2 never fits a slope, temperature or nonlinear map.
CALIBRATION_SLOPE: Final[float] = 1.0

#: Number of free parameters in the primary model, per judge.
CALIBRATION_N_PARAMETERS_PRIMARY: Final[int] = 1

#: Number of free parameters in the secondary model, per judge.
CALIBRATION_N_PARAMETERS_SECONDARY: Final[int] = 3

CALIBRATION_OBJECTIVE: Final[str] = (
    "argmin_alpha of the calibration-set logistic log-loss; equivalently the "
    "root of sum_i [sigma(M_i + alpha) - Y_i] = 0"
)

CALIBRATION_SOLVER: Final[str] = "CLUSTER_with_bisection_safeguard"

# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

BOOTSTRAP_STAGES: Final[str] = (
    "two independent stages per replicate: resample calibration artifacts within "
    "family and refit the intercept on that sample; independently resample test "
    "artifacts within family and apply the refitted intercept"
)

BOOTSTRAP_CALIBRATION_STREAM: Final[int] = 0
BOOTSTRAP_TEST_STREAM: Final[int] = 1

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ARTIFACT_DIR: Final[str] = "artifacts/day2"
FIGURE_DIR: Final[str] = "figures/day2"

DAY1_BASELINE_PATH: Final[str] = f"{ARTIFACT_DIR}/day1_baseline_manifest.json"
DAY1_AUDIT_FINDINGS_PATH: Final[str] = f"{ARTIFACT_DIR}/day1_audit_findings.json"

CALIBRATION_DATASET_PATH: Final[str] = f"{ARTIFACT_DIR}/calibration_dataset.parquet"
TEST_DATASET_PATH: Final[str] = f"{ARTIFACT_DIR}/test_dataset.parquet"
CALIBRATION_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/calibration_manifest.json"
TEST_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/test_manifest.json"
CALIBRATION_PROMPTS_PATH: Final[str] = f"{ARTIFACT_DIR}/calibration_prompts.parquet"
TEST_PROMPTS_PATH: Final[str] = f"{ARTIFACT_DIR}/test_prompts.parquet"

PREREG_JSON_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration.json"
PREREG_SHA_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration.sha256"
MODEL_PROVENANCE_PATH: Final[str] = f"{ARTIFACT_DIR}/model_provenance.json"

RAW_SCORES_DIR: Final[str] = f"{ARTIFACT_DIR}/raw_scores"
RAW_SCORES_CALIBRATION_DIR: Final[str] = f"{RAW_SCORES_DIR}/calibration"
RAW_SCORES_TEST_DIR: Final[str] = f"{RAW_SCORES_DIR}/test"

CALIBRATION_PARAMETERS_PATH: Final[str] = f"{ARTIFACT_DIR}/calibration_parameters.json"
METRICS_PATH: Final[str] = f"{ARTIFACT_DIR}/metrics_by_artifact.parquet"
BOOTSTRAP_PATH: Final[str] = f"{ARTIFACT_DIR}/bootstrap_results.parquet"
SUMMARY_PATH: Final[str] = f"{ARTIFACT_DIR}/summary.json"
INTEGRITY_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/integrity_manifest.json"
FINAL_CONSOLE_PATH: Final[str] = f"{ARTIFACT_DIR}/final_console_block.txt"

RAW_SCORES_SUBDIR: Final[dict[str, str]] = {
    PARTITION_CALIBRATION: RAW_SCORES_CALIBRATION_DIR,
    PARTITION_TEST: RAW_SCORES_TEST_DIR,
}

DATASET_PATHS: Final[dict[str, str]] = {
    PARTITION_CALIBRATION: CALIBRATION_DATASET_PATH,
    PARTITION_TEST: TEST_DATASET_PATH,
}

MANIFEST_PATHS: Final[dict[str, str]] = {
    PARTITION_CALIBRATION: CALIBRATION_MANIFEST_PATH,
    PARTITION_TEST: TEST_MANIFEST_PATH,
}

PROMPT_PATHS: Final[dict[str, str]] = {
    PARTITION_CALIBRATION: CALIBRATION_PROMPTS_PATH,
    PARTITION_TEST: TEST_PROMPTS_PATH,
}

#: Source files whose content defines Day-2 outcomes. The Day-2 preregistration
#: and the Day-2 integrity manifest hash exactly this list, so the two digests
#: are directly comparable -- the mismatch Day 1 exhibited cannot recur.
ANALYSIS_CODE_PATHS: Final[tuple[str, ...]] = (
    "src/formalcrrc/config.py",
    "src/formalcrrc/predicates.py",
    "src/formalcrrc/dataset.py",
    "src/formalcrrc/prompts.py",
    "src/formalcrrc/scoring.py",
    "src/formalcrrc/provenance.py",
    "src/formalcrrc/day1_audit.py",
    "src/formalcrrc/day2.py",
    "src/formalcrrc/day2_bootstrap.py",
    "src/formalcrrc/day2_config.py",
    "scripts/generate_day2_datasets.py",
    "scripts/run_day2_inference.py",
    "scripts/fit_day2_bias.py",
    "scripts/analyze_day2.py",
    "scripts/figures_day2.py",
    "scripts/verify_day2_integrity.py",
)

# --------------------------------------------------------------------------
# Day-1 source extensions
# --------------------------------------------------------------------------

#: Day-1 source files extended by Day 2 in a backward-compatible way. Day-1
#: outputs are unaffected: ``dataset.build_dataset()`` with no arguments still
#: reproduces the frozen Day-1 content hash, which the test suite asserts.
DAY1_SOURCE_EXTENSIONS: Final[dict[str, str]] = {
    "src/formalcrrc/dataset.py": (
        "added optional seed, partition, forbidden_pairs and "
        "enforce_unique_targets parameters, all defaulting to the Day-1 "
        "behaviour; no Day-1 code path altered"
    ),
}

#: The Day-1 dataset content hash that must keep reproducing.
DAY1_DATASET_CONTENT_SHA256: Final[str] = (
    "4462bedbf5f086daa8bdfa077ad82e1ea7b79b4db59d906fafe6c814cfc85a2e"
)
