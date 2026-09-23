#!/usr/bin/env python
"""Freeze the reason-then-score ablation preregistration.

Fails closed. Nothing here may run after any Qwen2.5 reason-then-score study row
exists: the script refuses to freeze when ablation rows are present, so the
experiment cannot be represented as preregistered after the fact. Synthetic
pre-freeze probes are not study inference and do not block the freeze.

Writes, all additive, under ``artifacts/reason_then_score_ablation/``.

``docs/PREREGISTRATION_REASON_THEN_SCORE_ABLATION.md`` is the governing draft.
This script fills only repository-verifiable placeholders -- the model and
tokenizer revision recovered from frozen Day-3 provenance, the freeze timestamp,
and the machine form's SHA-256 -- and never edits its protocol text.

The document cites the machine form's digest; the machine form deliberately does
not cite the document's. One direction only: stamping a digest into the document
changes the document, so recording both would be circular.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import dataset as ds  # noqa: E402
from formalcrrc import provenance  # noqa: E402
from formalcrrc import rts_ablation as ra  # noqa: E402
from formalcrrc import rts_ablation_config as rc  # noqa: E402
from formalcrrc import day3_config as d3c  # noqa: E402
from formalcrrc.config import PROMPT_TEMPLATE  # noqa: E402

ROOT = provenance.repo_root()

SHA_PLACEHOLDER = "**Preregistration SHA-256:** [RESOLVE BEFORE FIRST STUDY INFERENCE]"


def refuse_if_inference_exists() -> None:
    """A preregistration frozen after inference is not a preregistration."""
    candidates = [
        ROOT / rc.RAW_ROWS_PATH,
        ROOT / rc.SUMMARY_PATH,
        ROOT / rc.PAIRED_CURVES_PATH,
    ]
    existing = [p for p in candidates if p.exists()]
    if existing:
        print("REFUSING TO FREEZE: reason-then-score inference already exists:")
        for path in existing:
            print(f"  {path.relative_to(ROOT)}")
        raise SystemExit(2)


def _canonical_z(family: str, latent_level: int) -> int:
    if family == "coverage":
        return int(latent_level)
    if family in ("max_violation", "numeric_tolerance"):
        return 8 - int(latent_level)
    raise ValueError(f"unknown family {family!r}")


def validate_dataset() -> dict:
    """Phase 5: verify the frozen partition from primitive fields only."""
    frame = pd.read_parquet(ROOT / d3c.DATASET_PATH)
    manifest = json.loads(
        (ROOT / d3c.DATASET_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    per = frame.drop_duplicates("artifact_id")
    counts = per["true_first_fail_index"].value_counts()

    checks = {
        "row_count": len(frame) == rc.N_ROWS,
        "artifact_count": int(per.shape[0]) == rc.N_ARTIFACTS,
        "nine_levels_each": set(
            frame.groupby("artifact_id")["strictness_index"].nunique().unique()
        )
        == {rc.N_STRICTNESS_LEVELS},
        "one_hundred_eight_per_family": set(per["family"].value_counts().unique())
        == {108},
        "thirty_six_at_each_crossing": all(
            int(counts.get(j, 0)) == 36 for j in range(1, 10)
        ),
        "nontrivial_curve_count": int(
            per["true_first_fail_index"].between(1, 8).sum()
        )
        == rc.N_NONTRIVIAL,
        "seed_is_45": int(manifest["seed"]) == rc.SOURCE_SEED,
        "content_hash_matches_manifest": (
            ds.dataset_content_hash(frame)
            == manifest["hashes"]["dataset_content_sha256"]
        ),
        "formal_truth_reconstructs": all(
            int(_canonical_z(r.family, r.latent_level) >= int(r.strictness_index))
            == int(r.formal_truth)
            for r in frame.itertuples(index=False)
        ),
        "true_crossing_reconstructs": all(
            _canonical_z(r.family, r.latent_level) + 1 == int(r.true_first_fail_index)
            for r in per.itertuples(index=False)
        ),
    }
    return {
        "checked_at": provenance.utc_now(),
        "dataset_path": d3c.DATASET_PATH,
        "dataset_file_sha256": provenance.sha256_file(ROOT / d3c.DATASET_PATH),
        "dataset_content_sha256": ds.dataset_content_hash(frame),
        "n_rows": int(len(frame)),
        "n_artifacts": int(per.shape[0]),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def recover_and_verify_comparator() -> dict:
    """Phase 4: recompute Condition O and check it against the frozen table."""
    raw = pd.read_parquet(
        ROOT / "artifacts/day3/raw_scores/qwen2_5_14b_instruct.parquet"
    )
    data = pd.read_parquet(ROOT / d3c.DATASET_PATH)
    recovered = ra.recover_comparator(raw, data)
    frozen = pd.read_parquet(ROOT / d3c.METRICS_PATH)
    frozen = frozen[frozen["model_id"] == rc.MODEL_ID]
    verdict = ra.verify_comparator(recovered, frozen)
    verdict["expected"] = {
        "tce": rc.COMPARATOR_TCE,
        "trr": rc.COMPARATOR_TRR,
        "unreachable_pct": rc.COMPARATOR_UNREACHABLE_PCT,
    }
    verdict["source"] = rc.COMPARATOR_SOURCE
    verdict["raw_scores_sha256"] = provenance.sha256_file(
        ROOT / "artifacts/day3/raw_scores/qwen2_5_14b_instruct.parquet"
    )
    return verdict


def load_preflight() -> dict:
    path = ROOT / rc.PREFLIGHT_PATH
    if not path.is_file():
        print(f"REFUSING TO FREEZE: {rc.PREFLIGHT_PATH} is missing.")
        raise SystemExit(2)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        print(f"REFUSING TO FREEZE: preflight status is {report.get('status')!r}.")
        for problem in report.get("problems", []):
            print(f"  {problem}")
        raise SystemExit(2)
    return report


def build_preregistration(
    preflight: dict, dataset: dict, comparator: dict, diff: dict, fingerprint: dict
) -> dict:
    return {
        "study": rc.STUDY_NAME,
        "study_type": "post-study preregistered within-model protocol ablation",
        "amendments": 0,
        "frozen_at": provenance.utc_now(),
        "document": {"path": rc.PREREG_MD_PATH},
        "design": {
            "intervention": rc.INTERVENTION,
            "intervention_is_composite": rc.INTERVENTION_IS_COMPOSITE,
            "protocol": rc.PROTOCOL,
            "protocol_rationale": rc.PROTOCOL_RATIONALE,
            "condition_original": rc.CONDITION_ORIGINAL,
            "condition_reasoning": rc.CONDITION_REASONING,
        },
        "model": {
            "model_id": rc.MODEL_ID,
            "slug": rc.MODEL_SLUG,
            "short_name": rc.MODEL_SHORT_NAME,
            "revision": rc.DAY3_REVISION,
            "tokenizer_revision": rc.DAY3_TOKENIZER_REVISION,
            "revision_source": "artifacts/day3/model_provenance.json (frozen)",
            "dtype": rc.MODEL_DTYPE,
            "quantization": rc.QUANTIZATION,
            "chat_template_sha256": preflight["chat_template_sha256"],
            "chat_template_matches_day3": preflight["checks"][
                "chat_template_matches_day3"
            ],
            "config_sha256": preflight["config_sha256"],
        },
        "prompt": {
            "day3_template_sha256": provenance.sha256_text(PROMPT_TEMPLATE),
            "rts_template_sha256": diff["rts_template_sha256"],
            "added_suffix_sha256": diff["added_suffix_sha256"],
            "added_bytes": diff["added_bytes"],
            "construction": rc.PROMPT_CONSTRUCTION,
            "is_pure_suffix_extension": diff["is_pure_suffix_extension"],
            "removed_lines": diff["removed_lines"],
            "scaffold": rc.REASONING_SCAFFOLD,
        },
        "stage_r": {
            "temperature": rc.TEMPERATURE,
            "top_p": rc.TOP_P,
            "top_k": rc.TOP_K,
            "min_p": rc.MIN_P,
            "min_p_status": preflight.get("min_p_status"),
            "samples_per_row": rc.REASONING_SAMPLES_PER_ROW,
            "reasoning_token_budget": rc.REASONING_TOKEN_BUDGET,
            "experiment_seed": rc.EXPERIMENT_SEED,
            "seed_rule": rc.SEED_RULE,
            "seed_modulus": rc.SEED_MODULUS,
            "generation_parameter_source": rc.GENERATION_PARAMETER_SOURCE,
            "marker": rc.MARKER,
            "marker_stop_rule": rc.MARKER_STOP_RULE,
            "marker_tokenization": preflight["marker"],
        },
        "stage_d": {
            "candidate_met": rc.CANDIDATE_MET,
            "candidate_not_met": rc.CANDIDATE_NOT_MET,
            "candidate_tokenization_rule": rc.CANDIDATE_TOKENIZATION,
            "candidates": preflight["candidates"],
            "scoring_method": rc.SCORING_METHOD,
            "canonical_margin": rc.CANONICAL_MARGIN,
            "no_eos_probability": rc.NO_EOS_PROBABILITY,
            "no_length_normalization": rc.NO_LENGTH_NORMALIZATION,
        },
        "conventions": {
            "crossing": rc.CROSSING_CONVENTION,
            "decision": rc.DECISION_CONVENTION,
            "reachability": rc.REACHABILITY_CONDITION,
            "tie_policy": rc.TIE_POLICY,
            "failure_rule": rc.FAILURE_RULE,
        },
        "dataset": {
            "partition": rc.SOURCE_PARTITION,
            "seed": rc.SOURCE_SEED,
            "n_artifacts": rc.N_ARTIFACTS,
            "n_rows": rc.N_ROWS,
            "n_nontrivial_curves": rc.N_NONTRIVIAL,
            "validation": dataset,
        },
        "comparator": comparator,
        "completeness": {
            "truncation_warning_fraction": rc.TRUNCATION_WARNING_FRACTION,
            "truncation_warning_flag": rc.TRUNCATION_WARNING_FLAG,
            "curve_exclusion_rule": rc.CURVE_EXCLUSION_RULE,
            "premature_label_flag": rc.PREMATURE_LABEL_FLAG,
            "premature_label_pattern": ra.PREMATURE_LABEL_PATTERN,
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
        "anchor_context": {
            "model": rc.ANCHOR_MODEL,
            "tce": rc.ANCHOR_TCE,
            "trr": rc.ANCHOR_TRR,
            "unreachable_pct": rc.ANCHOR_UNREACHABLE_PCT,
            "status": rc.ANCHOR_COMPARISON_STATUS,
        },
        "statistics": {
            "bootstrap_replicates": rc.BOOTSTRAP_REPLICATES,
            "bootstrap_seed": rc.BOOTSTRAP_SEED,
            "bootstrap_scheme": rc.BOOTSTRAP_SCHEME,
            "bootstrap_ci": rc.BOOTSTRAP_CI_DESCRIPTION,
            "zero_count_rule": rc.ZERO_COUNT_RULE,
            "clopper_pearson_level": rc.CLOPPER_PEARSON_LEVEL,
        },
        "frozen_record": {
            "n_files": fingerprint["n_files"],
            "combined_sha256": fingerprint["combined_sha256"],
        },
        "analysis_code": provenance.source_manifest(rc.ANALYSIS_CODE_PATHS, ROOT),
        "git": provenance.git_state(ROOT),
    }


def fill_document(prereg: dict) -> str:
    """Fill the draft's bracketed fields. Protocol text is never edited."""
    text = (ROOT / rc.PREREG_MD_PATH).read_text(encoding="utf-8")
    replacements = {
        "**Preregistration freeze timestamp:** [RESOLVE BEFORE FIRST STUDY INFERENCE]": (
            f"**Preregistration freeze timestamp:** `{prereg['frozen_at']}`"
        ),
        "**Exact model revision:** [READ FROM FROZEN DAY-3 PROVENANCE]": (
            f"**Exact model revision:** `{prereg['model']['revision']}`"
        ),
        "**Exact tokenizer revision:** [READ FROM FROZEN DAY-3 PROVENANCE]": (
            f"**Exact tokenizer revision:** "
            f"`{prereg['model']['tokenizer_revision']}`"
        ),
        "**NOT FROZEN UNTIL MODEL/TOKENIZER REVISIONS, MARKER TOKENIZATION, CANDIDATE TOKENIZATION, GENERATION CONFIGURATION, UTC TIMESTAMP, AND SHA-256 ARE RECORDED.**": (
            f"**FROZEN {prereg['frozen_at']} — amendments 0.**"
        ),
        "**NO FORMALCRRC REASON-THEN-SCORE STUDY ROW MAY BE RUN BEFORE FREEZE.**": (
            "**Frozen before the first FormalCRRC reason-then-score study row. "
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
    print("REASON-THEN-SCORE ABLATION PREREGISTRATION FREEZE")
    print("=" * 74)

    refuse_if_inference_exists()
    print("hard stop        PASS  no ablation inference exists")

    preflight = load_preflight()
    print(f"preflight        PASS  on {preflight.get('slurm')}")

    dataset = validate_dataset()
    print(f"dataset          {dataset['status']}")
    for name, ok in dataset["checks"].items():
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    if dataset["status"] != "PASS":
        return 2

    comparator = recover_and_verify_comparator()
    print(f"comparator       {comparator['status']}")
    print(
        f"    TCE {comparator['mean_tce']:.4f}  TRR {comparator['trr']:.4f}  "
        f"unreachable {comparator['unreachable_pct']:.2f}%"
    )
    for name, ok in comparator["checks"].items():
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    if comparator["status"] != "PASS":
        print("REFUSING TO FREEZE: comparator recomputation disagrees with Day 3.")
        return 2

    diff = ra.prompt_diff()
    print(f"prompt diff      {'PASS' if all(diff['checks'].values()) else 'FAIL'}")
    print(f"    added {diff['added_bytes']} bytes, removed {len(diff['removed_lines'])} lines")
    if not all(diff["checks"].values()):
        print("REFUSING TO FREEZE: the prompt intervention is not a pure addition.")
        return 2

    data = pd.read_parquet(ROOT / d3c.DATASET_PATH)
    bodies = ra.bodies_identical(data)
    print(f"prompt bodies    {bodies['status']} ({bodies['n_rows_checked']} rows)")
    if bodies["status"] != "PASS":
        return 2

    fingerprint = ra.build_frozen_fingerprint(ROOT)
    print(
        f"frozen record    {fingerprint['n_files']} files  "
        f"{fingerprint['combined_sha256'][:16]}..."
    )

    prereg = build_preregistration(preflight, dataset, comparator, diff, fingerprint)

    document = fill_document(prereg)
    (ROOT / rc.PREREG_MD_PATH).write_text(document, encoding="utf-8", newline="\n")

    digest = provenance.write_json(ROOT / rc.PREREG_JSON_PATH, prereg)
    provenance.write_checksum_file(
        ROOT / rc.PREREG_JSON_PATH, ROOT / rc.PREREG_SHA_PATH
    )

    if SHA_PLACEHOLDER not in document:
        raise SystemExit(
            "REFUSING TO FREEZE: no SHA-256 placeholder to stamp; already frozen?"
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
    provenance.write_json(ROOT / rc.PROMPT_DIFF_PATH, diff)
    provenance.write_json(ROOT / rc.COMPARATOR_PATH, comparator)
    provenance.write_json(
        ROOT / rc.MODEL_PROVENANCE_PATH,
        {"model": prereg["model"], "preflight": preflight},
    )
    provenance.write_json(
        ROOT / rc.GENERATION_CONFIG_PATH,
        {"stage_r": prereg["stage_r"], "stage_d": prereg["stage_d"]},
    )
    provenance.write_json(
        ROOT / rc.READINESS_PATH,
        {
            "frozen_at": prereg["frozen_at"],
            "preregistration_sha256": digest,
            "dataset": dataset,
            "comparator": {k: comparator[k] for k in ("mean_tce", "trr", "status")},
            "frozen_record_combined_sha256": fingerprint["combined_sha256"],
            "amendments": 0,
        },
    )

    print()
    print(f"frozen_at              {prereg['frozen_at']}")
    print(f"preregistration.json   {digest}")
    print(f"document sha256        {document_sha}")
    print("amendments             0")
    print("FREEZE COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
