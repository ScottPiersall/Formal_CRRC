#!/usr/bin/env python
"""Freeze the Label-Swap preregistration and fingerprint the frozen record.

Fails closed. Nothing here may run after any swapped-label inference exists: the
script refuses to freeze if swapped raw scores are already present, so a study
cannot be represented as preregistered after the fact.

Writes, all additive:

* ``artifacts/labelswap/preregistration.json``  -- machine form, verbatim constants
* ``artifacts/labelswap/preregistration.sha256``
* ``artifacts/labelswap/preregistration_frozen.md`` -- immutable copy of the document
* ``artifacts/labelswap/frozen_record_fingerprint.json`` -- Phase-2 protection
* ``artifacts/labelswap/prompt_diff.json``      -- the exact intervention
* ``artifacts/labelswap/prompt_manifest.parquet``

The markdown document at ``docs/PREREGISTRATION_LABEL_SWAP.md`` is the governing
draft. This script fills only repository-verifiable placeholders -- the four
model revisions, the dataset hashes, the two prompt-template hashes, the freeze
timestamp and the document's own SHA-256 -- and never edits its protocol text.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import labelswap, prompts, provenance  # noqa: E402
from formalcrrc import labelswap_config as lsc  # noqa: E402
from formalcrrc import day3_config as d3c  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    MODEL_DTYPE,
    MODEL_IDS,
    MODEL_SLUGS,
    N_ROWS,
    PROMPT_TEMPLATE,
)

ROOT = provenance.repo_root()


def refuse_if_inference_exists() -> None:
    """A preregistration frozen after inference is not a preregistration."""
    raw_dir = ROOT / lsc.RAW_SCORES_DIR
    existing = sorted(raw_dir.glob("*.parquet")) if raw_dir.is_dir() else []
    if existing:
        print("REFUSING TO FREEZE: swapped-label inference already exists:", flush=True)
        for path in existing:
            print(f"  {path.relative_to(ROOT)}", flush=True)
        print(
            "This study cannot be represented as preregistered. Report the "
            "situation instead of freezing.",
            flush=True,
        )
        raise SystemExit(2)


def fingerprint_frozen_record() -> dict:
    """Phase 2: hash every protected file before anything is built."""
    files: dict[str, str] = {}
    for tree in lsc.PROTECTED_TREES:
        directory = ROOT / tree
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = provenance.sha256_file(path)
    for relative in lsc.PROTECTED_DOCS:
        if (ROOT / relative).is_file():
            files[relative] = provenance.sha256_file(ROOT / relative)

    prior = json.loads((ROOT / d3c.PRIOR_MANIFEST_PATH).read_text(encoding="utf-8"))
    mismatched = [
        relative
        for relative, digest in prior["files"].items()
        if provenance.sha256_file(ROOT / relative) != digest
    ]
    return {
        "purpose": "files that must remain byte-identical across the label-swap study",
        "created_at": provenance.utc_now(),
        "n_files": len(files),
        "combined_sha256": provenance.sha256_text(
            "\n".join(f"{k}:{v}" for k, v in sorted(files.items()))
        ),
        "recorded_day3_prior_manifest_verifies": not mismatched,
        "mismatched_against_recorded_manifest": mismatched,
        "trees": list(lsc.PROTECTED_TREES),
        "documents": list(lsc.PROTECTED_DOCS),
        "files": files,
    }


def day3_facts() -> dict:
    """Everything recovered from the frozen Day-3 record, verified live."""
    manifest = json.loads(
        (ROOT / d3c.DATASET_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    provenance_doc = json.loads(
        (ROOT / d3c.MODEL_PROVENANCE_PATH).read_text(encoding="utf-8")
    )

    models = {}
    for model_id in MODEL_IDS:
        record = provenance_doc["models"][model_id]
        models[model_id] = {
            "model_id": model_id,
            "slug": MODEL_SLUGS[model_id],
            "revision": record["revision"],
            "tokenizer_revision": record["tokenizer_revision"],
            "dtype": record["dtype"],
            "chat_template_sha256": record["chat_template_sha256"],
            "transformers_version": record["transformers_version"],
            "torch_version": record["torch_version"],
            "day3_label_tokenization": record["label_tokenization"],
            "day3_raw_scores_sha256": provenance.sha256_file(
                ROOT / f"artifacts/day3/raw_scores/{MODEL_SLUGS[model_id]}.parquet"
            ),
        }

    live_dataset = provenance.sha256_file(ROOT / d3c.DATASET_PATH)
    live_prompts = provenance.sha256_file(ROOT / d3c.PROMPT_MANIFEST_PATH)
    return {
        "partition": d3c.PARTITION_NAME,
        "seed": d3c.SEED,
        "dataset_path": d3c.DATASET_PATH,
        "dataset_file_sha256": live_dataset,
        "dataset_content_sha256": manifest["hashes"]["dataset_content_sha256"],
        "prompt_manifest_file_sha256": live_prompts,
        "dataset_hash_matches_manifest": live_dataset
        == manifest["hashes"]["dataset_file_sha256"],
        "prompt_manifest_hash_matches_manifest": live_prompts
        == manifest["hashes"]["prompt_manifest_file_sha256"],
        "prompt_template_sha256": prompts.prompt_template_sha256(),
        "prompt_template_matches_manifest": prompts.prompt_template_sha256()
        == manifest["hashes"]["prompt_template_sha256"],
        "models": models,
    }


def fill_document(text: str, frozen_at: str, facts: dict, diff: dict) -> str:
    """Substitute only the repository-derived placeholders. Protocol text untouched."""
    filled = text.replace(
        "**Preregistration freeze timestamp:** [FILL AUTOMATICALLY BEFORE FIRST INFERENCE]",
        f"**Preregistration freeze timestamp:** {frozen_at}",
    )

    revision_block = "\n".join(
        f"{i}. `{mid}` — revision `{facts['models'][mid]['revision']}`, "
        f"tokenizer revision `{facts['models'][mid]['tokenizer_revision']}`"
        for i, mid in enumerate(MODEL_IDS, start=1)
    )
    original_list = "\n".join(f"{i}. {mid}" for i, mid in enumerate(MODEL_IDS, start=1))
    if original_list in filled:
        filled = filled.replace(original_list, revision_block, 1)

    appendix = f"""
---

## Appendix A — Repository-derived values recorded at freeze

Filled automatically by `scripts/freeze_labelswap_preregistration.py`. No
protocol text above was altered.

**Freeze timestamp (UTC):** `{frozen_at}`

### A.1 Dataset (frozen Day-3 TEST, reused read-only)

| | |
|---|---|
| Path | `{facts["dataset_path"]}` |
| Seed | {facts["seed"]} |
| Dataset file SHA-256 | `{facts["dataset_file_sha256"]}` |
| Dataset content SHA-256 | `{facts["dataset_content_sha256"]}` |
| Day-3 prompt manifest SHA-256 | `{facts["prompt_manifest_file_sha256"]}` |
| Hashes match Day-3 manifest | {facts["dataset_hash_matches_manifest"]} / {facts["prompt_manifest_hash_matches_manifest"]} |

### A.2 Model panel — exact Day-3 revisions

| Model | Revision | Tokenizer revision | dtype | Chat template SHA-256 |
|---|---|---|---|---|
""" + "\n".join(
        "| `{m}` | `{r}` | `{t}` | {d} | `{c}` |".format(
            m=mid,
            r=facts["models"][mid]["revision"],
            t=facts["models"][mid]["tokenizer_revision"],
            d=facts["models"][mid]["dtype"],
            c=facts["models"][mid]["chat_template_sha256"][:16] + "…",
        )
        for mid in MODEL_IDS
    ) + f"""

Inference configuration: `{MODEL_DTYPE}`, transformers
`{facts["models"][MODEL_IDS[0]]["transformers_version"]}`, torch
`{facts["models"][MODEL_IDS[0]]["torch_version"]}`, scoring rule
`single_token_next_logit` via prompt-continuation label resolution — all
inherited from Day 3 unchanged.

### A.3 Prompt templates

| | |
|---|---|
| Original template SHA-256 | `{diff["original_sha256"]}` |
| Swapped template SHA-256 | `{diff["swapped_sha256"]}` |
| Equal length | {diff["same_length"]} |
| Differing byte positions | {diff["n_differing_positions"]} |
| All differences are label characters | {diff["all_differences_are_label_characters"]} |
| Non-label text identical | {diff["non_label_text_identical"]} |
| Verdict | **{diff["status"]}** |

Differing positions: {", ".join(f"index {d['index']} ({d['original']}→{d['swapped']})" for d in diff["differing_positions"])}

Original response instruction:

```text
{diff["original_response_instruction"]}
```

Swapped response instruction:

```text
{diff["swapped_response_instruction"]}
```

### A.4 Statistics

| | |
|---|---|
| Bootstrap replicates | {BOOTSTRAP_REPLICATES} |
| Bootstrap seed | {BOOTSTRAP_SEED} |
| Scheme | {lsc.BOOTSTRAP_SCHEME} |
| Interval | {lsc.BOOTSTRAP_CI} |

### A.5 Amendments

**0** at freeze.
"""
    filled = filled.replace(
        "**PREREGISTRATION STATUS: NOT FROZEN UNTIL HASH AND TIMESTAMP ARE RECORDED.**",
        "**PREREGISTRATION STATUS: FROZEN.** See Appendix A for the recorded "
        "hashes and timestamp.",
    )
    return filled + appendix


def main() -> int:
    refuse_if_inference_exists()
    out_dir = ROOT / lsc.ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    draft_path = ROOT / lsc.PREREG_MD_PATH
    if not draft_path.is_file():
        print(f"FAIL: draft preregistration missing at {lsc.PREREG_MD_PATH}")
        return 2
    draft = draft_path.read_text(encoding="utf-8")

    # ---- consistency of the draft against the implementation ----------------
    diff = labelswap.prompt_diff()
    facts = day3_facts()
    consistency = {
        "draft_quotes_the_real_response_instruction": (
            "Return A if the criterion is met." in draft
            and "Return B if the criterion is not met." in draft
            and PROMPT_TEMPLATE.endswith(labelswap.ORIGINAL_RESPONSE_INSTRUCTION)
        ),
        "draft_swapped_margin_is_sb_minus_sa": "S_B-S_A" in draft.replace(" ", ""),
        "draft_seed_matches_day3": f"Seed: {d3c.SEED}" in draft,
        "draft_artifact_count_matches": "324" in draft,
        "draft_nontrivial_count_matches": "288" in draft,
        "prompt_diff_passes": diff["status"] == "PASS",
        "dataset_hashes_verify": facts["dataset_hash_matches_manifest"]
        and facts["prompt_manifest_hash_matches_manifest"],
        "prompt_template_matches_day3": facts["prompt_template_matches_manifest"],
        "all_four_revisions_recovered": all(
            facts["models"][m]["revision"] for m in MODEL_IDS
        ),
    }
    if not all(consistency.values()):
        print("FAIL: draft preregistration is inconsistent with the implementation")
        for key, value in consistency.items():
            print(f"  {key}: {value}")
        return 2

    # ---- Phase 2 fingerprint, taken before anything is written --------------
    fingerprint = fingerprint_frozen_record()
    if not fingerprint["recorded_day3_prior_manifest_verifies"]:
        print("FAIL: recorded Day-3 prior manifest does not verify")
        print(fingerprint["mismatched_against_recorded_manifest"][:10])
        return 2

    # ---- prompt manifest ----------------------------------------------------
    dataset = pd.read_parquet(ROOT / d3c.DATASET_PATH)
    manifest = labelswap.build_swapped_prompt_manifest(dataset)
    equivalence = labelswap.verify_row_equivalence(dataset, manifest)
    if equivalence["status"] != "PASS":
        print("FAIL: per-row prompt equivalence check failed")
        print(json.dumps(equivalence, indent=2)[:2000])
        return 2

    frozen_at = provenance.utc_now()
    document = fill_document(draft, frozen_at, facts, diff)

    prereg = {
        "study": lsc.STUDY_NAME,
        "study_type": "post-study preregistered robustness experiment",
        "preregistered": True,
        "frozen_at": frozen_at,
        "amendments": 0,
        "relationship_to_prior_work": (
            "Reuses the frozen Day-3 TEST partition and Day-3 model panel. Not a "
            "Day-3 amendment, not a replacement for Day-3 TRR, not part of the "
            "original three preregistrations."
        ),
        "intervention": {
            "factor": lsc.INTERVENTION,
            "canonical_margin_original": lsc.CANONICAL_MARGIN_ORIGINAL,
            "canonical_margin_swapped": lsc.CANONICAL_MARGIN_SWAPPED,
            "prompt_diff": diff,
        },
        "conventions": {
            "crossing": lsc.CROSSING_CONVENTION,
            "decision": lsc.DECISION_CONVENTION,
            "reachability": lsc.REACHABILITY_CONDITION,
            "ties": lsc.TIE_POLICY,
        },
        "design": {
            "source_partition": lsc.SOURCE_PARTITION,
            "source_seed": lsc.SOURCE_SEED,
            "n_artifacts": lsc.N_ARTIFACTS,
            "n_strictness_levels": lsc.N_STRICTNESS_LEVELS,
            "n_rows_per_model": lsc.N_ROWS_PER_MODEL,
            "n_models": lsc.N_MODELS,
            "n_new_evaluations": lsc.N_NEW_EVALUATIONS,
            "n_nontrivial_per_model": lsc.N_NONTRIVIAL_PER_MODEL,
        },
        "robustness_bands": {
            "delta_tce": list(lsc.DELTA_TCE_BAND),
            "delta_trr": list(lsc.DELTA_TRR_BAND),
            "agreement_lower_bound": lsc.AGREEMENT_LOWER_BOUND,
            "criteria": lsc.ROBUSTNESS_CRITERIA,
        },
        "statistics": {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "scheme": lsc.BOOTSTRAP_SCHEME,
            "interval": lsc.BOOTSTRAP_CI,
        },
        "day3_facts": facts,
        "draft_consistency_checks": consistency,
        "row_equivalence": equivalence,
        "frozen_record_fingerprint": {
            "n_files": fingerprint["n_files"],
            "combined_sha256": fingerprint["combined_sha256"],
        },
        "analysis_code": provenance.source_manifest(
            list(lsc.ANALYSIS_CODE_PATHS), ROOT
        ),
        "document_sha256": provenance.sha256_text(document),
        "git": provenance.git_state(ROOT),
        "environment": provenance.environment_snapshot(include_gpu=False),
    }

    (ROOT / lsc.PREREG_FROZEN_COPY_PATH).write_text(document, encoding="utf-8")
    draft_path.write_text(document, encoding="utf-8")
    provenance.write_json(ROOT / lsc.PREREG_JSON_PATH, prereg)
    provenance.write_checksum_file(
        ROOT / lsc.PREREG_JSON_PATH, ROOT / lsc.PREREG_SHA_PATH
    )
    provenance.write_json(ROOT / lsc.FROZEN_FINGERPRINT_PATH, fingerprint)
    provenance.write_json(ROOT / lsc.PROMPT_DIFF_PATH, {**diff, "row_equivalence": equivalence})
    manifest.to_parquet(ROOT / lsc.PROMPT_MANIFEST_PATH, index=False)

    prereg_sha = (ROOT / lsc.PREREG_SHA_PATH).read_text(encoding="utf-8").split()[0]

    print("=" * 74)
    print("LABEL-SWAP PREREGISTRATION FROZEN")
    print("=" * 74)
    print(f"frozen_at            {frozen_at}")
    print(f"preregistration sha  {prereg_sha}")
    print(f"document sha256      {prereg['document_sha256']}")
    print(f"amendments           0")
    print()
    print(f"prompt diff          {diff['n_differing_positions']} positions, {diff['status']}")
    for entry in diff["differing_positions"]:
        print(f"    index {entry['index']}: {entry['original']} -> {entry['swapped']}")
    print(f"original template    {diff['original_sha256']}")
    print(f"swapped  template    {diff['swapped_sha256']}")
    print()
    print(f"row equivalence      {equivalence['n_bodies_identical']}/{equivalence['n_rows']} {equivalence['status']}")
    print(f"prompt manifest      {len(manifest)} rows")
    print()
    print(f"frozen record        {fingerprint['n_files']} files, "
          f"combined {fingerprint['combined_sha256']}")
    print()
    print("model panel (exact Day-3 revisions):")
    for model_id in MODEL_IDS:
        print(f"    {model_id:<40} {facts['models'][model_id]['revision']}")
    print()
    print(f"wrote {lsc.PREREG_JSON_PATH}")
    print(f"wrote {lsc.PREREG_SHA_PATH}")
    print(f"wrote {lsc.PREREG_FROZEN_COPY_PATH}")
    print(f"wrote {lsc.FROZEN_FINGERPRINT_PATH}")
    print(f"wrote {lsc.PROMPT_DIFF_PATH}")
    print(f"wrote {lsc.PROMPT_MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
