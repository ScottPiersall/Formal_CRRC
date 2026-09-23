"""Preregistered constants for the FormalCRRC Label-Swap Robustness Study.

Kept in its own module so this study adds no constant to the Day-1, Day-2 or
Day-3 configurations, and touches no file any prior preregistration pinned by
hash. Everything here is serialised verbatim into
``artifacts/labelswap/preregistration.json`` and may not change after the freeze
except through a numbered, checksummed amendment.

The study reuses the frozen Day-3 TEST partition, the Day-3 model panel at the
Day-3 revisions, the Day-3 scoring procedure, the Day-3 crossing convention and
the Day-3 reachability rule **unchanged**. Exactly one conceptual factor moves:
which response token carries which semantic class.
"""

from __future__ import annotations

from typing import Final

from formalcrrc.config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED  # noqa: F401

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

STUDY_NAME: Final[str] = "FormalCRRC Label-Swap Counterbalance Robustness Study"

#: Namespace prefix for this study's own outputs. The artifact ids themselves
#: stay exactly as Day 3 wrote them, so pairing is by identity, not by mapping.
CONDITION_TAG: Final[str] = "SWAP"

#: The condition label written into every swapped row.
CONDITION_ORIGINAL: Final[str] = "original"
CONDITION_SWAPPED: Final[str] = "swapped"

# --------------------------------------------------------------------------
# The intervention
# --------------------------------------------------------------------------

#: The single conceptual factor this study moves.
INTERVENTION: Final[str] = (
    "the semantic mapping of 'criterion met' and 'criterion not met' to the "
    "response tokens A and B"
)

#: Canonical margins. Positive means 'criterion met' in BOTH conditions.
CANONICAL_MARGIN_ORIGINAL: Final[str] = "M_O(s) = S_A - S_B"
CANONICAL_MARGIN_SWAPPED: Final[str] = "M_S(s) = S_B - S_A"

#: Number of byte positions the swapped prompt template may differ by: the two
#: label characters in the final response-instruction lines, and nothing else.
PERMITTED_TEMPLATE_DIFF_POSITIONS: Final[int] = 2

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
    "introduced. Tie frequencies are recorded descriptively and the rule is not "
    "changed after observing them."
)

# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------

#: The frozen partition reused wholesale. No new dataset is generated.
SOURCE_PARTITION: Final[str] = "DAY3_TEST"
SOURCE_SEED: Final[int] = 45

N_ARTIFACTS: Final[int] = 324
N_STRICTNESS_LEVELS: Final[int] = 9
N_ROWS_PER_MODEL: Final[int] = 2916
N_MODELS: Final[int] = 4
N_NEW_EVALUATIONS: Final[int] = 11664

#: Curves whose boundary is informative for reachability: j* in 1..8.
N_NONTRIVIAL_PER_MODEL: Final[int] = 288

# --------------------------------------------------------------------------
# Prespecified practical robustness bands
# --------------------------------------------------------------------------
#
# Interpretive thresholds fixed before inference. They are NOT claims about
# universal operational significance, and they are not changed after seeing
# outcomes.

DELTA_TCE_BAND: Final[tuple[float, float]] = (-0.5, 0.5)
DELTA_TRR_BAND: Final[tuple[float, float]] = (-0.10, 0.10)
AGREEMENT_LOWER_BOUND: Final[float] = 0.90

ROBUSTNESS_CRITERIA: Final[dict[str, str]] = {
    "delta_tce": "CI95(delta TCE) contained in [-0.5, +0.5]",
    "delta_trr": "CI95(delta TRR) contained in [-0.10, +0.10]",
    "agreement": "lower 95% bound of semantic decision agreement >= 0.90",
}

# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

#: Paired, artifact-clustered. The nine thresholds of a curve move together and
#: the original/swapped pair is resampled jointly.
BOOTSTRAP_SCHEME: Final[str] = (
    "paired artifact-clustered resampling with replacement; all nine strictness "
    "levels of a curve stay together; the original and swapped members of a pair "
    "are drawn jointly; family strata preserved for family-stratified analyses"
)
BOOTSTRAP_CI: Final[str] = "percentile, 95%"

# --------------------------------------------------------------------------
# Paths -- all additive
# --------------------------------------------------------------------------

ARTIFACT_DIR: Final[str] = "artifacts/labelswap"
FIGURE_DIR: Final[str] = "figures/labelswap"

PREREG_MD_PATH: Final[str] = "docs/PREREGISTRATION_LABEL_SWAP.md"
PREREG_JSON_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration.json"
PREREG_SHA_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration.sha256"
PREREG_FROZEN_COPY_PATH: Final[str] = f"{ARTIFACT_DIR}/preregistration_frozen.md"

FROZEN_FINGERPRINT_PATH: Final[str] = f"{ARTIFACT_DIR}/frozen_record_fingerprint.json"
PROMPT_DIFF_PATH: Final[str] = f"{ARTIFACT_DIR}/prompt_diff.json"
PROMPT_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/prompt_manifest.parquet"
MODEL_PROVENANCE_PATH: Final[str] = f"{ARTIFACT_DIR}/model_provenance.json"
READINESS_PATH: Final[str] = f"{ARTIFACT_DIR}/readiness.json"

RAW_SCORES_DIR: Final[str] = f"{ARTIFACT_DIR}/raw_scores"
PAIRED_ROWS_PATH: Final[str] = f"{ARTIFACT_DIR}/paired_rows.parquet"
METRICS_PATH: Final[str] = f"{ARTIFACT_DIR}/metrics_by_artifact.parquet"
BOOTSTRAP_PATH: Final[str] = f"{ARTIFACT_DIR}/bootstrap_results.parquet"
TABLES_DIR: Final[str] = f"{ARTIFACT_DIR}/tables"
SUMMARY_PATH: Final[str] = f"{ARTIFACT_DIR}/summary.json"
INTEGRITY_MANIFEST_PATH: Final[str] = f"{ARTIFACT_DIR}/integrity_manifest.json"
FINAL_CONSOLE_PATH: Final[str] = f"{ARTIFACT_DIR}/final_console_block.txt"

# --------------------------------------------------------------------------
# Frozen record that must remain byte-identical
# --------------------------------------------------------------------------

PROTECTED_TREES: Final[tuple[str, ...]] = (
    "artifacts/day1",
    "artifacts/day2",
    "artifacts/day3",
    "figures/day1",
    "figures/day2",
    "figures/day3",
    "artifacts/poststudy_baselines",
    "artifacts/poststudy_theory",
)

PROTECTED_DOCS: Final[tuple[str, ...]] = (
    "docs/PREREGISTRATION_DAY1.md",
    "docs/PREREGISTRATION_DAY2.md",
    "docs/PREREGISTRATION_DAY3.md",
    "docs/RESULTS_DAY1.md",
    "docs/RESULTS_DAY2.md",
    "docs/RESULTS_DAY3.md",
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
    "artifacts/day3/model_provenance.json",
    "artifacts/day3/metrics_by_artifact.parquet",
    "artifacts/day3/raw_scores/qwen2_5_14b_instruct.parquet",
    "artifacts/day3/raw_scores/llama3_1_8b_instruct.parquet",
    "artifacts/day3/raw_scores/mistral_7b_instruct_v0_3.parquet",
    "artifacts/day3/raw_scores/gemma_2_9b_it.parquet",
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
    "src/formalcrrc/labelswap.py",
    "src/formalcrrc/labelswap_bootstrap.py",
    "src/formalcrrc/labelswap_config.py",
    "scripts/freeze_labelswap_preregistration.py",
    "scripts/verify_labelswap_ready.py",
    "scripts/run_labelswap_inference.py",
    "scripts/analyze_labelswap.py",
    "scripts/figures_labelswap.py",
    "scripts/verify_labelswap_integrity.py",
)
