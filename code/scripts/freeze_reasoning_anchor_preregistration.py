#!/usr/bin/env python
"""Freeze the reasoning-anchor preregistration and fingerprint the frozen record.

Fails closed. Nothing here may run after any FormalCRRC reasoning-anchor
evaluation exists: the script refuses to freeze when anchor rows are already
present, so the study cannot be represented as preregistered after the fact.
Synthetic pre-freeze probes are not study inference and do not block the freeze.

Writes, all additive:

* ``artifacts/reasoning_anchor/preregistration.json``       -- machine form
* ``artifacts/reasoning_anchor/preregistration.sha256``
* ``artifacts/reasoning_anchor/preregistration_frozen.md``  -- immutable copy
* ``artifacts/reasoning_anchor/frozen_record_fingerprint.json``
* ``artifacts/reasoning_anchor/model_provenance.json``
* ``artifacts/reasoning_anchor/generation_config.json``
* ``artifacts/reasoning_anchor/readiness.json``

``docs/PREREGISTRATION_REASONING_ANCHOR.md`` is the governing draft. This script
fills only repository-verifiable placeholders -- the model and tokenizer
revision, the resolved separator, the freeze timestamp and the machine
document's SHA-256 -- and never edits its protocol text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import dataset as ds  # noqa: E402
from formalcrrc import provenance  # noqa: E402
from formalcrrc import reasoning_anchor as ra  # noqa: E402
from formalcrrc import reasoning_anchor_config as rc  # noqa: E402
from formalcrrc import day3_config as d3c  # noqa: E402
from formalcrrc.config import PROMPT_TEMPLATE  # noqa: E402

ROOT = provenance.repo_root()


# --------------------------------------------------------------------------
# Hard stop
# --------------------------------------------------------------------------


def refuse_if_inference_exists() -> None:
    """A preregistration frozen after inference is not a preregistration."""
    candidates = [
        ROOT / rc.RAW_ROWS_PATH,
        ROOT / rc.REASONING_TRACES_PATH,
        ROOT / rc.SUMMARY_PATH,
    ]
    existing = [p for p in candidates if p.exists()]
    if existing:
        print("REFUSING TO FREEZE: reasoning-anchor inference already exists:")
        for path in existing:
            print(f"  {path.relative_to(ROOT)}")
        print(
            "The preregistration would be contaminated. Report the situation "
            "rather than freezing."
        )
        raise SystemExit(2)


# --------------------------------------------------------------------------
# Section 9 -- dataset validation, using no model output
# --------------------------------------------------------------------------


def _canonical_z(family: str, latent_level: int) -> int:
    """Canonical compliance level, per Section 5 of the preregistration."""
    if family == "coverage":
        return int(latent_level)
    if family in ("max_violation", "numeric_tolerance"):
        return 8 - int(latent_level)
    raise ValueError(f"unknown family {family!r}")


def validate_dataset() -> dict:
    """Verify the frozen Day-3 TEST partition, from primitive fields only."""
    frame = pd.read_parquet(ROOT / d3c.DATASET_PATH)
    manifest = json.loads(
        (ROOT / d3c.DATASET_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    per_artifact = frame.drop_duplicates("artifact_id")

    crossing_counts = per_artifact["true_first_fail_index"].value_counts()
    family_counts = per_artifact["family"].value_counts()
    level_counts = frame.groupby("artifact_id")["strictness_index"].nunique()

    truth_ok = all(
        int(_canonical_z(row.family, row.latent_level) >= int(row.strictness_index))
        == int(row.formal_truth)
        for row in frame.itertuples(index=False)
    )
    crossing_ok = all(
        _canonical_z(row.family, row.latent_level) + 1 == int(row.true_first_fail_index)
        for row in per_artifact.itertuples(index=False)
    )

    checks = {
        "row_count": len(frame) == rc.N_ROWS,
        "artifact_count": int(per_artifact.shape[0]) == rc.N_ARTIFACTS,
        "nine_levels_each": set(level_counts.unique()) == {rc.N_STRICTNESS_LEVELS},
        "one_hundred_eight_per_family": set(family_counts.unique()) == {108},
        "thirty_six_at_each_crossing": all(
            int(crossing_counts.get(j, 0)) == 36 for j in range(1, 10)
        ),
        "nontrivial_curve_count": int(
            per_artifact["true_first_fail_index"].between(1, 8).sum()
        )
        == rc.N_NONTRIVIAL,
        "seed_is_45": int(manifest["seed"]) == rc.SOURCE_SEED,
        "partition_is_day3_test": manifest["partition"] == rc.SOURCE_PARTITION,
        "content_hash_matches_manifest": (
            ds.dataset_content_hash(frame)
            == manifest["hashes"]["dataset_content_sha256"]
        ),
        "formal_truth_reconstructs": truth_ok,
        "true_crossing_reconstructs": crossing_ok,
    }
    return {
        "checked_at": provenance.utc_now(),
        "dataset_path": d3c.DATASET_PATH,
        "dataset_file_sha256": provenance.sha256_file(ROOT / d3c.DATASET_PATH),
        "dataset_content_sha256": ds.dataset_content_hash(frame),
        "n_rows": int(len(frame)),
        "n_artifacts": int(per_artifact.shape[0]),
        "n_nontrivial_curves": int(
            per_artifact["true_first_fail_index"].between(1, 8).sum()
        ),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def load_preflight() -> dict:
    path = ROOT / rc.PREFLIGHT_PATH
    if not path.is_file():
        print(f"REFUSING TO FREEZE: {rc.PREFLIGHT_PATH} is missing.")
        print("Run scripts/preflight_reasoning_anchor.py on the cluster first.")
        raise SystemExit(2)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        print(f"REFUSING TO FREEZE: preflight status is {report.get('status')!r}.")
        for problem in report.get("problems", []):
            print(f"  {problem}")
        raise SystemExit(2)
    return report


# --------------------------------------------------------------------------
# Document assembly
# --------------------------------------------------------------------------


def build_preregistration(preflight: dict, dataset: dict, fingerprint: dict) -> dict:
    """The machine form of the frozen protocol, from verbatim constants."""
    return {
        "study": rc.STUDY_NAME,
        "study_type": (
            "post-study preregistered external-validity / reasoning-capable "
            "anchor experiment"
        ),
        "anchor_description": rc.ANCHOR_DESCRIPTION,
        "amendments": 0,
        "frozen_at": provenance.utc_now(),
        # The document cites this JSON's digest; this JSON deliberately does not
        # cite the document's. One direction only -- recording both would be
        # circular, because stamping the digest into the document changes the
        # document, which would change a hash recorded here.
        "document": {"path": rc.PREREG_MD_PATH},
        "model": {
            "model_id": rc.MODEL_ID,
            "slug": rc.MODEL_SLUG,
            "short_name": rc.MODEL_SHORT_NAME,
            "revision": preflight["requested_revision"],
            "tokenizer_revision": preflight["requested_revision"],
            "dtype": rc.MODEL_DTYPE,
            "quantization": rc.QUANTIZATION,
            "config_sha256": preflight["config_sha256"],
            "chat_template_sha256": preflight["chat_template_sha256"],
            "config_torch_dtype": preflight["config_torch_dtype"],
            "quantization_config_present": preflight["quantization_config_present"],
        },
        "protocol": {
            "name": rc.PROTOCOL,
            "rationale": rc.SCORING_RATIONALE,
            "stage_r": {
                "temperature": rc.TEMPERATURE,
                "top_p": rc.TOP_P,
                "top_k": rc.TOP_K,
                "min_p": rc.MIN_P,
                "samples_per_row": rc.REASONING_SAMPLES_PER_ROW,
                "reasoning_token_budget": rc.REASONING_TOKEN_BUDGET,
                "experiment_seed": rc.EXPERIMENT_SEED,
                "seed_rule": rc.SEED_RULE,
                "seed_modulus": rc.SEED_MODULUS,
                "think_end_text": rc.THINK_END_TEXT,
                "think_end_token_id": preflight["think_end_token_id"],
                "generation_prompt_opens_thinking": preflight["checks"][
                    "generation_prompt_opens_thinking"
                ],
            },
            "stage_d": {
                "post_think_separator": rc.POST_THINK_SEPARATOR,
                "post_think_separator_bytes": rc.POST_THINK_SEPARATOR_BYTES,
                "post_think_separator_hex": preflight["post_think_separator_bytes"],
                "post_think_separator_token_ids": preflight[
                    "post_think_separator_token_ids"
                ],
                "post_think_separator_source": rc.POST_THINK_SEPARATOR_SOURCE,
                "candidate_met": rc.CANDIDATE_MET,
                "candidate_not_met": rc.CANDIDATE_NOT_MET,
                "candidate_tokenization_rule": rc.CANDIDATE_TOKENIZATION,
                "scoring_method": rc.SCORING_METHOD,
                "canonical_margin": rc.CANONICAL_MARGIN,
                "candidates": preflight["candidates"],
            },
            "crossing_convention": rc.CROSSING_CONVENTION,
            "decision_convention": rc.DECISION_CONVENTION,
            "reachability_condition": rc.REACHABILITY_CONDITION,
            "tie_policy": rc.TIE_POLICY,
        },
        "dataset": {
            "partition": rc.SOURCE_PARTITION,
            "seed": rc.SOURCE_SEED,
            "n_artifacts": rc.N_ARTIFACTS,
            "n_strictness_levels": rc.N_STRICTNESS_LEVELS,
            "n_rows": rc.N_ROWS,
            "n_nontrivial_curves": rc.N_NONTRIVIAL,
            "prompt_template_sha256": provenance.sha256_text(PROMPT_TEMPLATE),
            "validation": dataset,
        },
        "completeness": {
            "truncation_warning_fraction": rc.TRUNCATION_WARNING_FRACTION,
            "truncation_warning_flag": rc.TRUNCATION_WARNING_FLAG,
            "curve_exclusion_rule": rc.CURVE_EXCLUSION_RULE,
        },
        "baselines": {
            "fixed_center_tce": rc.BASELINE_FIXED_CENTER_TCE,
            "random_crossing_tce": rc.BASELINE_RANDOM_CROSSING_TCE,
            "always_met_tce": rc.BASELINE_ALWAYS_MET_TCE,
            "always_not_met_tce": rc.BASELINE_ALWAYS_NOT_MET_TCE,
            "fixed_center_accuracy": rc.BASELINE_FIXED_CENTER_ACC,
            "random_crossing_accuracy": rc.BASELINE_RANDOM_CROSSING_ACC,
            "majority_accuracy": rc.BASELINE_MAJORITY_ACC,
        },
        "day3_comparison": {
            "tce": rc.DAY3_TCE,
            "unreachable_pct": rc.DAY3_UNREACHABLE_PCT,
            "unreachable_range": list(rc.DAY3_UNREACHABLE_RANGE),
            "status": rc.COMPARISON_STATUS,
        },
        "statistics": {
            "bootstrap_replicates": rc.BOOTSTRAP_REPLICATES,
            "bootstrap_seed": rc.BOOTSTRAP_SEED,
            "bootstrap_scheme": rc.BOOTSTRAP_SCHEME,
            "bootstrap_ci": rc.BOOTSTRAP_CI_DESCRIPTION,
        },
        "frozen_record": {
            "n_files": fingerprint["n_files"],
            "combined_sha256": fingerprint["combined_sha256"],
        },
        "analysis_code": provenance.source_manifest(rc.ANALYSIS_CODE_PATHS, ROOT),
        "git": provenance.git_state(ROOT),
    }


#: Placeholder for the machine form's digest, stamped in after the JSON exists.
SHA_PLACEHOLDER = (
    "**Preregistration SHA-256:** [FILL BEFORE FIRST FORMALCRRC INFERENCE]"
)


def fill_document(prereg: dict, preflight: dict) -> str:
    """Fill the draft's bracketed fields. Protocol text is never edited.

    The SHA-256 line is left alone here: it names the digest of the machine
    form, which cannot be computed until this document is final.
    """
    text = (ROOT / rc.PREREG_MD_PATH).read_text(encoding="utf-8")
    separator_ids = preflight["post_think_separator_token_ids"]
    replacements = {
        "**Preregistration freeze timestamp:** [FILL BEFORE FIRST FORMALCRRC INFERENCE]": (
            f"**Preregistration freeze timestamp:** `{prereg['frozen_at']}`"
        ),
        "**Exact model revision:** [RESOLVE AND PIN BEFORE FREEZE]": (
            f"**Exact model revision:** `{prereg['model']['revision']}`"
        ),
        "**Exact tokenizer revision:** [RESOLVE AND PIN BEFORE FREEZE]": (
            f"**Exact tokenizer revision:** "
            f"`{prereg['model']['tokenizer_revision']}`"
        ),
        "`POST_THINK_SEPARATOR = [RESOLVE BEFORE FREEZE]`": (
            f"`POST_THINK_SEPARATOR = \"\\n\\n\"` "
            f"(bytes `{rc.POST_THINK_SEPARATOR_BYTES}`, "
            f"token ids `{separator_ids}`)"
        ),
        "**NOT FROZEN UNTIL ALL BRACKETED MODEL/SCORING DETAILS, UTC TIMESTAMP, AND SHA-256 ARE FILLED.**": (
            f"**FROZEN {prereg['frozen_at']} — amendments 0.**"
        ),
        "**NO FORMALCRRC REASONING-ANCHOR INFERENCE MAY BEGIN BEFORE FREEZE.**": (
            "**Frozen before the first FormalCRRC reasoning-anchor evaluation. "
            "Only synthetic non-study prompts were used before this point.**"
        ),
    }
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
        else:
            print(f"  note: placeholder not found (already filled?): {old[:60]}...")
    return text


def main() -> int:
    print("=" * 74)
    print("REASONING-ANCHOR PREREGISTRATION FREEZE")
    print("=" * 74)

    refuse_if_inference_exists()
    print("hard stop        PASS  no anchor inference exists")

    preflight = load_preflight()
    print(f"preflight        PASS  {preflight['status']} on {preflight.get('slurm')}")

    dataset = validate_dataset()
    print(f"dataset          {dataset['status']}")
    for name, ok in dataset["checks"].items():
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    if dataset["status"] != "PASS":
        print("REFUSING TO FREEZE: dataset validation failed.")
        return 2

    fingerprint = ra.build_frozen_fingerprint(ROOT)
    print(
        f"frozen record    {fingerprint['n_files']} files  "
        f"{fingerprint['combined_sha256'][:16]}..."
    )

    prereg = build_preregistration(preflight, dataset, fingerprint)

    # Order matters, and it is one-directional. The document is filled and
    # written first; the machine form is then written and hashed; finally that
    # digest is stamped into the document. The machine form never records the
    # document's hash, so stamping cannot invalidate anything already written.
    document = fill_document(prereg, preflight)
    (ROOT / rc.PREREG_MD_PATH).write_text(document, encoding="utf-8", newline="\n")

    digest = provenance.write_json(ROOT / rc.PREREG_JSON_PATH, prereg)
    provenance.write_checksum_file(
        ROOT / rc.PREREG_JSON_PATH, ROOT / rc.PREREG_SHA_PATH
    )

    if SHA_PLACEHOLDER not in document:
        raise SystemExit(
            "REFUSING TO FREEZE: the document has no SHA-256 placeholder to "
            "stamp; it may already have been frozen."
        )
    document = document.replace(
        SHA_PLACEHOLDER, f"**Preregistration SHA-256:** `{digest}`"
    )
    (ROOT / rc.PREREG_MD_PATH).write_text(document, encoding="utf-8", newline="\n")

    frozen_copy = ROOT / rc.PREREG_FROZEN_COPY_PATH
    frozen_copy.parent.mkdir(parents=True, exist_ok=True)
    frozen_copy.write_text(document, encoding="utf-8", newline="\n")
    document_sha = provenance.sha256_file(ROOT / rc.PREREG_MD_PATH)

    provenance.write_json(ROOT / rc.FROZEN_FINGERPRINT_PATH, fingerprint)
    provenance.write_json(
        ROOT / rc.MODEL_PROVENANCE_PATH,
        {"model": prereg["model"], "preflight": preflight},
    )
    provenance.write_json(
        ROOT / rc.GENERATION_CONFIG_PATH,
        {
            "stage_r": prereg["protocol"]["stage_r"],
            "stage_d": prereg["protocol"]["stage_d"],
        },
    )
    provenance.write_json(
        ROOT / rc.READINESS_PATH,
        {
            "frozen_at": prereg["frozen_at"],
            "preregistration_sha256": digest,
            "dataset": dataset,
            "frozen_record_combined_sha256": fingerprint["combined_sha256"],
            "amendments": 0,
        },
    )

    print()
    print(f"frozen_at              {prereg['frozen_at']}")
    print(f"preregistration.json   {digest}")
    print(f"document sha256        {document_sha}")
    print(f"amendments             0")
    print("FREEZE COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
