#!/usr/bin/env python
"""Score the frozen Day-3 partition under the SWAPPED label mapping, one judge.

Raw scores only. No metric, ranking, agreement rate or reachability figure is
computed or printed here: the analysis stage cannot begin until every intended
run is complete, so the console carries only progress, failures, timing and
memory.

Fails closed unless the label-swap preregistration is frozen and verified, the
frozen scientific record is byte-identical to its recorded fingerprint, the
prompt diff passes, and the model's tokenisation is the exact Day-3 tokenisation
with the two semantic roles exchanged.

Because the swapped tokenisation declares ``label_met = B``, the score this
script records as ``raw_score_met`` is ``S_B`` and the recorded ``margin`` is
``S_B - S_A``: the canonical swapped margin, positive for "criterion met". The
raw ``logit_a`` and ``logit_b`` are stored separately so nothing downstream has
to trust that convention.

Writes ``artifacts/labelswap/raw_scores/<slug>.parquet`` plus a run record.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import labelswap, prompts, provenance, scoring  # noqa: E402
from formalcrrc import labelswap_config as lsc  # noqa: E402
from formalcrrc import day3_config as d3c  # noqa: E402
from formalcrrc.config import MODEL_DTYPE, MODEL_IDS, MODEL_SLUGS  # noqa: E402

RAW_SCORE_COLUMNS = [
    "prompt_id",
    "artifact_id",
    "family",
    "latent_level",
    "strictness_index",
    "order_index",
    "condition",
    "partition",
    "model_id",
    "revision",
    "logit_a",
    "logit_b",
    "raw_score_met",
    "raw_score_not_met",
    "margin",
    "p_met",
    "semantic_decision",
    "formal_truth",
    "true_first_fail_index",
    "scoring_method",
    "n_prompt_tokens",
    "user_message_sha256",
    "rendered_prompt_sha256",
    "inference_status",
]


def preflight(root: Path, model_id: str) -> dict:
    """Every precondition, checked before a single prompt is scored."""
    problems: list[str] = []

    prereg_path = root / lsc.PREREG_JSON_PATH
    sha_path = root / lsc.PREREG_SHA_PATH
    if not (prereg_path.is_file() and sha_path.is_file()):
        problems.append("label-swap preregistration is not frozen")
    elif not provenance.verify_checksum_file(prereg_path, sha_path):
        problems.append("label-swap preregistration checksum does not match")

    diff_path = root / lsc.PROMPT_DIFF_PATH
    if not diff_path.is_file():
        problems.append("prompt diff record missing")
    else:
        recorded = json.loads(diff_path.read_text(encoding="utf-8"))
        live = labelswap.prompt_diff()
        if recorded["status"] != "PASS":
            problems.append("recorded prompt diff did not pass")
        if recorded["swapped_sha256"] != live["swapped_sha256"]:
            problems.append("swapped prompt template changed since the freeze")
        if recorded["original_sha256"] != live["original_sha256"]:
            problems.append("original prompt template changed since the freeze")

    fingerprint_path = root / lsc.FROZEN_FINGERPRINT_PATH
    if not fingerprint_path.is_file():
        problems.append("frozen-record fingerprint missing")
    else:
        fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        changed = [
            relative
            for relative, digest in fingerprint["files"].items()
            if provenance.sha256_file(root / relative) != digest
        ]
        if changed:
            problems.append(f"frozen scientific files changed: {changed[:5]}")

    dataset_manifest = json.loads(
        (root / d3c.DATASET_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    if provenance.sha256_file(root / d3c.DATASET_PATH) != (
        dataset_manifest["hashes"]["dataset_file_sha256"]
    ):
        problems.append("Day-3 dataset hash does not match its manifest")

    day3_provenance = json.loads(
        (root / d3c.MODEL_PROVENANCE_PATH).read_text(encoding="utf-8")
    )
    record = day3_provenance.get("models", {}).get(model_id, {})
    if not record:
        problems.append(f"no Day-3 provenance record for {model_id}")
    elif record.get("status") != "AVAILABLE":
        problems.append(f"{model_id} is not AVAILABLE in the Day-3 record")

    if problems:
        for problem in problems:
            print(f"PREFLIGHT FAIL: {problem}", flush=True)
        raise SystemExit(2)
    return record


def load_model(model_id: str, revision: str, token: str | None, cache_dir: str | None):
    """Load tokenizer and model at the pinned Day-3 revision, in bf16, on the GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, token=token, cache_dir=cache_dir
    )
    dtype = getattr(torch, MODEL_DTYPE)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=dtype, token=token, cache_dir=cache_dir
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            token=token,
            cache_dir=cache_dir,
        )
    return tokenizer, model.to("cuda").eval()


def _gpu_allocated_gb() -> float:
    try:
        import torch

        return torch.cuda.memory_allocated() / (1024**3)
    except Exception:  # pragma: no cover - operational reporting only
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--model", required=True, choices=list(MODEL_IDS))
    parser.add_argument("--hf-token-file", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    args = parser.parse_args()

    root = Path(args.root)
    model_id = args.model
    slug = MODEL_SLUGS[model_id]

    record = preflight(root, model_id)
    revision = record["revision"]
    day3_tokenization = record["label_tokenization"]

    manifest = pd.read_parquet(root / lsc.PROMPT_MANIFEST_PATH).sort_values(
        "order_index"
    )
    if len(manifest) != lsc.N_ROWS_PER_MODEL:
        print(f"PREFLIGHT FAIL: manifest has {len(manifest)} rows", flush=True)
        return 2

    dataset = pd.read_parquet(root / d3c.DATASET_PATH).set_index("prompt_id")

    token = None
    if args.hf_token_file and Path(args.hf_token_file).is_file():
        token = Path(args.hf_token_file).read_text(encoding="utf-8").strip() or None

    started = time.time()
    print(f"model={model_id}", flush=True)
    print(f"revision={revision}", flush=True)
    print(f"condition={lsc.CONDITION_SWAPPED}", flush=True)
    print(f"partition={lsc.SOURCE_PARTITION}", flush=True)
    print(f"prompts={len(manifest)}", flush=True)
    print("loading model ...", flush=True)

    tokenizer, model = load_model(model_id, revision, token, args.cache_dir)

    reference = scoring.render_chat_prompt(
        tokenizer, labelswap.swapped_label_reference_message()
    )
    tokenization = labelswap.resolve_swapped_label_tokenization(tokenizer, reference)

    mirror = labelswap.tokenization_is_day3_mirror(tokenization, day3_tokenization)
    if mirror["status"] != "PASS":
        print(
            "PREFLIGHT FAIL: swapped tokenisation is not the Day-3 tokenisation "
            "with roles exchanged\n"
            f"  day3:     {day3_tokenization}\n"
            f"  swapped:  {tokenization.to_dict()}\n"
            f"  checks:   {mirror['checks']}",
            flush=True,
        )
        return 2

    chat_hash = provenance.sha256_text(tokenizer.chat_template or "")
    if chat_hash != record["chat_template_sha256"]:
        print("PREFLIGHT FAIL: chat template hash differs from Day 3", flush=True)
        return 2

    print(f"label mapping: met -> {tokenization.label_met} "
          f"{tokenization.met_token_ids}, not_met -> {tokenization.label_not_met} "
          f"{tokenization.not_met_token_ids}", flush=True)
    print(f"canonical margin: {lsc.CANONICAL_MARGIN_SWAPPED}", flush=True)

    out_dir = root / lsc.RAW_SCORES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.parquet"
    partial_path = out_dir / f"{slug}.partial.parquet"

    rows: list[dict] = []
    failures: list[dict] = []
    load_seconds = time.time() - started
    print(f"loaded in {load_seconds:.1f}s; scoring ...", flush=True)

    scoring_started = time.time()
    for position, entry in enumerate(manifest.itertuples(index=False), start=1):
        rendered = scoring.render_chat_prompt(tokenizer, entry.user_message)
        try:
            result = scoring.score_prompt(model, tokenizer, rendered, tokenization)
        except Exception as error:  # noqa: BLE001 - recorded, never hidden
            failures.append(
                {
                    "prompt_id": entry.prompt_id,
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                }
            )
            print(f"FAILURE at {entry.prompt_id}: {type(error).__name__}", flush=True)
            continue

        # raw_score_met is S_B and raw_score_not_met is S_A, because the
        # tokenisation declares label_met = B. Store the letters explicitly too.
        score_met = result["raw_score_met"]
        score_not_met = result["raw_score_not_met"]
        margin = score_met - score_not_met
        source = dataset.loc[entry.prompt_id]

        rows.append(
            {
                "prompt_id": entry.prompt_id,
                "artifact_id": entry.artifact_id,
                "family": entry.family,
                "latent_level": int(entry.latent_level),
                "strictness_index": int(entry.strictness_index),
                "order_index": int(entry.order_index),
                "condition": lsc.CONDITION_SWAPPED,
                "partition": lsc.SOURCE_PARTITION,
                "model_id": model_id,
                "revision": revision,
                "logit_a": score_not_met,
                "logit_b": score_met,
                "raw_score_met": score_met,
                "raw_score_not_met": score_not_met,
                "margin": margin,
                "p_met": result["p_met"],
                "semantic_decision": int(margin >= 0.0),
                "formal_truth": int(source["formal_truth"]),
                "true_first_fail_index": int(source["true_first_fail_index"]),
                "scoring_method": result["scoring_method"],
                "n_prompt_tokens": result["n_prompt_tokens"],
                "user_message_sha256": entry.user_message_sha256,
                "rendered_prompt_sha256": result["rendered_prompt_sha256"],
                "inference_status": "OK",
            }
        )

        if position % args.checkpoint_every == 0:
            pd.DataFrame(rows, columns=RAW_SCORE_COLUMNS).to_parquet(
                partial_path, index=False
            )
            elapsed = time.time() - scoring_started
            rate = position / elapsed
            remaining = (len(manifest) - position) / rate if rate else float("nan")
            print(
                f"progress {position}/{len(manifest)} elapsed={elapsed:.0f}s "
                f"rate={rate:.1f}/s eta={remaining:.0f}s "
                f"gpu_alloc={_gpu_allocated_gb():.1f}GiB",
                flush=True,
            )

    frame = pd.DataFrame(rows, columns=RAW_SCORE_COLUMNS)
    frame.to_parquet(out_path, index=False)
    if partial_path.exists():
        partial_path.unlink()

    total_seconds = time.time() - started
    provenance.write_json(
        out_dir / f"{slug}_run.json",
        {
            "experiment": lsc.STUDY_NAME,
            "condition": lsc.CONDITION_SWAPPED,
            "partition": lsc.SOURCE_PARTITION,
            "model_id": model_id,
            "slug": slug,
            "revision": revision,
            "dtype": MODEL_DTYPE,
            "scoring_method": tokenization.scoring_method,
            "label_tokenization": tokenization.to_dict(),
            "day3_label_tokenization": day3_tokenization,
            "tokenization_mirror_check": mirror,
            "chat_template_sha256": chat_hash,
            "original_prompt_template_sha256": prompts.prompt_template_sha256(),
            "swapped_prompt_template_sha256": (
                labelswap.swapped_prompt_template_sha256()
            ),
            "canonical_margin": lsc.CANONICAL_MARGIN_SWAPPED,
            "dataset_file_sha256": provenance.sha256_file(root / d3c.DATASET_PATH),
            "preregistration_sha256": (
                (root / lsc.PREREG_SHA_PATH).read_text(encoding="utf-8").split()[0]
            ),
            "prompts_requested": int(len(manifest)),
            "prompts_scored": int(len(frame)),
            "failures": failures,
            "started_at_unix": started,
            "finished_at": provenance.utc_now(),
            "load_seconds": round(load_seconds, 2),
            "total_seconds": round(total_seconds, 2),
            "raw_scores_sha256": provenance.sha256_file(out_path),
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
            },
            "environment": provenance.environment_snapshot(include_gpu=True),
            "source_manifest": provenance.source_manifest(
                [
                    "src/formalcrrc/config.py",
                    "src/formalcrrc/prompts.py",
                    "src/formalcrrc/scoring.py",
                    "src/formalcrrc/labelswap.py",
                    "src/formalcrrc/labelswap_config.py",
                    "scripts/run_labelswap_inference.py",
                ],
                root=root,
            ),
        },
    )

    print(f"scored {len(frame)}/{len(manifest)} prompts", flush=True)
    print(f"failures: {len(failures)}", flush=True)
    print(f"total time: {total_seconds:.0f}s", flush=True)
    print(f"wrote {lsc.RAW_SCORES_DIR}/{slug}.parquet", flush=True)
    return 0 if len(frame) == len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
