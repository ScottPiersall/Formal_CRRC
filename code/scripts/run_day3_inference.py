#!/usr/bin/env python
"""Score the Day-3 evaluation partition with one judge (Phase F).

Raw scores only. No reachable set is computed, no oracle is fitted, no metric is
produced and nothing scientific is printed: the oracle stage cannot begin until
every intended raw run is complete, so the console carries only progress,
failures, timing and memory.

Fails closed unless the Day-3 preregistration is frozen and verified, the
dataset matches its manifest, the prompt template matches the prior days, and
the model's scoring conditions match the recorded provenance exactly.

Writes ``artifacts/day3/raw_scores/<slug>.parquet`` plus a run record.
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

from formalcrrc import day3_config as d3c  # noqa: E402
from formalcrrc import prompts, provenance, scoring  # noqa: E402
from formalcrrc.config import MODEL_DTYPE, MODEL_IDS, MODEL_SLUGS, N_ROWS  # noqa: E402

RAW_SCORE_COLUMNS = [
    "prompt_id",
    "artifact_id",
    "family",
    "latent_level",
    "strictness_index",
    "order_index",
    "partition",
    "model_id",
    "revision",
    "raw_score_met",
    "raw_score_not_met",
    "margin",
    "p_met",
    "scoring_method",
    "n_prompt_tokens",
    "rendered_prompt_sha256",
]


def preflight(root: Path, model_id: str) -> dict:
    """Every provenance precondition, checked before a single prompt is scored."""
    problems: list[str] = []

    prereg = root / d3c.PREREG_JSON_PATH
    sha_path = root / d3c.PREREG_SHA_PATH
    if not (prereg.is_file() and sha_path.is_file()):
        problems.append("Day-3 preregistration is not frozen")
    elif not provenance.verify_checksum_file(prereg, sha_path):
        problems.append("Day-3 preregistration checksum does not match")

    manifest_path = root / d3c.DATASET_MANIFEST_PATH
    if not manifest_path.is_file():
        problems.append("Day-3 dataset manifest is missing")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_path = root / d3c.DATASET_PATH
        prompt_path = root / d3c.PROMPT_MANIFEST_PATH
        if provenance.sha256_file(dataset_path) != manifest["hashes"][
            "dataset_file_sha256"
        ]:
            problems.append("dataset hash does not match its manifest")
        if provenance.sha256_file(prompt_path) != manifest["hashes"][
            "prompt_manifest_file_sha256"
        ]:
            problems.append("prompt manifest hash does not match its manifest")
        if manifest["hashes"]["prompt_template_sha256"] != (
            prompts.prompt_template_sha256()
        ):
            problems.append("prompt template hash has changed since generation")

    provenance_path = root / d3c.MODEL_PROVENANCE_PATH
    record: dict = {}
    if not provenance_path.is_file():
        problems.append("Day-3 model provenance missing; run audit_day3_models.py")
    else:
        document = json.loads(provenance_path.read_text(encoding="utf-8"))
        record = document.get("models", {}).get(model_id, {})
        if not record:
            problems.append(f"no Day-3 provenance record for {model_id}")
        elif record.get("status") != "AVAILABLE":
            problems.append(
                f"{model_id} is not AVAILABLE (status={record.get('status')})"
            )
        elif not all(record.get("continuity", {}).values()):
            problems.append(f"{model_id} continuity with prior days not established")

    if problems:
        for problem in problems:
            print(f"PREFLIGHT FAIL: {problem}", flush=True)
        raise SystemExit(2)
    return record


def load_model(model_id: str, revision: str, token: str | None, cache_dir: str | None):
    """Load tokenizer and model at the pinned revision, in bf16, on the GPU."""
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
    parser.add_argument("--limit", type=int, default=None, help="debug only")
    args = parser.parse_args()

    root = Path(args.root)
    model_id = args.model
    slug = MODEL_SLUGS[model_id]

    record = preflight(root, model_id)
    revision = record["revision"]
    expected_tokenization = record["label_tokenization"]

    manifest = pd.read_parquet(root / d3c.PROMPT_MANIFEST_PATH).sort_values(
        "order_index"
    )
    if len(manifest) != N_ROWS:
        print(f"PREFLIGHT FAIL: prompt manifest has {len(manifest)} rows", flush=True)
        return 2
    if args.limit:
        manifest = manifest.head(args.limit)

    token = None
    if args.hf_token_file and Path(args.hf_token_file).is_file():
        token = Path(args.hf_token_file).read_text(encoding="utf-8").strip() or None

    started = time.time()
    print(f"model={model_id}", flush=True)
    print(f"revision={revision}", flush=True)
    print(f"partition={d3c.PARTITION_NAME}", flush=True)
    print(f"prompts={len(manifest)}", flush=True)
    print("loading model ...", flush=True)

    tokenizer, model = load_model(model_id, revision, token, args.cache_dir)
    label_reference = scoring.render_chat_prompt(
        tokenizer, prompts.label_reference_message()
    )
    tokenization = scoring.resolve_label_tokenization(tokenizer, label_reference)

    if tokenization.to_dict() != expected_tokenization:
        print(
            "PREFLIGHT FAIL: label tokenisation differs from the Day-3 audit\n"
            f"  recorded: {expected_tokenization}\n"
            f"  observed: {tokenization.to_dict()}",
            flush=True,
        )
        return 2
    chat_hash = provenance.sha256_text(tokenizer.chat_template or "")
    if chat_hash != record["chat_template_sha256"]:
        print("PREFLIGHT FAIL: chat template hash differs from the audit", flush=True)
        return 2

    out_dir = root / d3c.RAW_SCORES_DIR
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

        rows.append(
            {
                "prompt_id": entry.prompt_id,
                "artifact_id": entry.artifact_id,
                "family": entry.family,
                "latent_level": int(entry.latent_level),
                "strictness_index": int(entry.strictness_index),
                "order_index": int(entry.order_index),
                "partition": d3c.PARTITION_TAG,
                "model_id": model_id,
                "revision": revision,
                "raw_score_met": result["raw_score_met"],
                "raw_score_not_met": result["raw_score_not_met"],
                "margin": result["raw_score_met"] - result["raw_score_not_met"],
                "p_met": result["p_met"],
                "scoring_method": result["scoring_method"],
                "n_prompt_tokens": result["n_prompt_tokens"],
                "rendered_prompt_sha256": result["rendered_prompt_sha256"],
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
            "experiment": "FormalCRRC Day 3",
            "partition": d3c.PARTITION_NAME,
            "model_id": model_id,
            "slug": slug,
            "revision": revision,
            "dtype": MODEL_DTYPE,
            "scoring_method": tokenization.scoring_method,
            "label_tokenization": tokenization.to_dict(),
            "chat_template_sha256": chat_hash,
            "prompt_template_sha256": prompts.prompt_template_sha256(),
            "dataset_file_sha256": provenance.sha256_file(root / d3c.DATASET_PATH),
            "preregistration_sha256": (
                (root / d3c.PREREG_SHA_PATH).read_text(encoding="utf-8").split()[0]
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
                    "src/formalcrrc/day3_config.py",
                    "scripts/run_day3_inference.py",
                ],
                root=root,
            ),
        },
    )

    print(f"scored {len(frame)}/{len(manifest)} prompts", flush=True)
    print(f"failures: {len(failures)}", flush=True)
    print(f"total time: {total_seconds:.0f}s", flush=True)
    print(f"wrote {d3c.RAW_SCORES_DIR}/{slug}.parquet", flush=True)
    return 0 if len(frame) == len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
