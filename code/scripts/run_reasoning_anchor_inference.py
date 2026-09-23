#!/usr/bin/env python
"""Run the frozen reason-then-score protocol over the Day-3 TEST partition.

Two stages per row, exactly as frozen. Stage R generates a reasoning trace under
the model author's sampling family with a deterministic per-row seed; Stage D
rebuilds the standardised decision context from that exact trace and scores the
two candidate answers as continuations of it.

Console output here is operational only. No metric, ranking or fitted parameter
is computed in this script -- the analysis runs separately, after every intended
row has been attempted.

The run is checkpointed in shards so that a cluster failure or a wall-clock
timeout can be resumed without regenerating work that already succeeded. A
resumed row keeps its original seed, prompt and configuration, so a retry is
byte-for-byte the same experiment; retries are logged.

Refuses to start unless the preregistration is frozen and its checksum verifies.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import provenance  # noqa: E402
from formalcrrc import reasoning_anchor as ra  # noqa: E402
from formalcrrc import reasoning_anchor_config as rc  # noqa: E402
from formalcrrc import day3_config as d3c  # noqa: E402

SHARD_DIR_NAME = "_shards"


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------


def require_frozen(root: pathlib.Path) -> dict:
    """Refuse to run unless a verified frozen preregistration exists."""
    json_path = root / rc.PREREG_JSON_PATH
    sha_path = root / rc.PREREG_SHA_PATH
    if not (json_path.is_file() and sha_path.is_file()):
        raise SystemExit(
            "REFUSING TO RUN: the preregistration is not frozen. "
            "Run scripts/freeze_reasoning_anchor_preregistration.py first."
        )
    if not provenance.verify_checksum_file(json_path, sha_path):
        raise SystemExit(
            "REFUSING TO RUN: the frozen preregistration does not match its "
            "recorded checksum."
        )
    prereg = json.loads(json_path.read_text(encoding="utf-8"))
    if int(prereg.get("amendments", -1)) != 0:
        raise SystemExit(
            f"REFUSING TO RUN: amendments = {prereg.get('amendments')!r}, expected 0."
        )
    return prereg


def load_prompts(root: pathlib.Path) -> pd.DataFrame:
    """The frozen Day-3 user messages, in the frozen inference order."""
    manifest = pd.read_parquet(root / d3c.PROMPT_MANIFEST_PATH)
    dataset = pd.read_parquet(root / d3c.DATASET_PATH)
    keep = [
        "prompt_id",
        "artifact_id",
        "family",
        "latent_level",
        "strictness_index",
        "formal_truth",
        "true_first_fail_index",
    ]
    merged = manifest.merge(dataset[keep], on=
        ["prompt_id", "artifact_id", "family", "latent_level", "strictness_index"],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != rc.N_ROWS or merged["formal_truth"].isna().any():
        raise SystemExit(
            f"REFUSING TO RUN: prompt/dataset join produced {len(merged)} usable "
            f"rows, expected {rc.N_ROWS}."
        )
    return merged.sort_values("order_index").reset_index(drop=True)


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------


def shard_dir(root: pathlib.Path) -> pathlib.Path:
    path = root / rc.ARTIFACT_DIR / SHARD_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_completed(root: pathlib.Path) -> tuple[list[dict], set[str]]:
    """Rows already produced by an earlier attempt of this same run."""
    rows: list[dict] = []
    for path in sorted(shard_dir(root).glob("shard_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows, {r["prompt_id"] for r in rows}


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--hf-token-file", default=os.environ.get("HF_TOKEN_FILE"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=96)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="operational cap for a smoke run; never used for the study")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    prereg = require_frozen(root)
    model_block = prereg["model"]
    stage_r = prereg["protocol"]["stage_r"]
    stage_d = prereg["protocol"]["stage_d"]
    revision = model_block["revision"]
    think_end_id = int(stage_r["think_end_token_id"])
    budget = int(stage_r["reasoning_token_budget"])

    print(f"preregistration frozen at {prereg['frozen_at']}", flush=True)
    print(f"model {rc.MODEL_ID} @ {revision}", flush=True)

    prompts_frame = load_prompts(root)
    if args.limit:
        prompts_frame = prompts_frame.head(args.limit)
    done_rows, done_ids = load_completed(root)
    todo = prompts_frame[~prompts_frame["prompt_id"].isin(done_ids)]
    print(
        f"rows: {len(prompts_frame)} total, {len(done_ids)} already complete, "
        f"{len(todo)} to run",
        flush=True,
    )
    if todo.empty:
        print("nothing to do; all rows already present", flush=True)

    token = None
    if args.hf_token_file and pathlib.Path(args.hf_token_file).is_file():
        token = pathlib.Path(args.hf_token_file).read_text().strip()

    from huggingface_hub import snapshot_download

    local_path = snapshot_download(
        rc.MODEL_ID, revision=revision, token=token,
        cache_dir=args.cache_dir, max_workers=4,
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(local_path)
    chat_sha = provenance.sha256_text(tok.chat_template)
    if chat_sha != model_block["chat_template_sha256"]:
        raise SystemExit(
            "REFUSING TO RUN: the chat template does not match the frozen "
            f"record ({chat_sha} != {model_block['chat_template_sha256']})."
        )
    print("chat template matches the frozen record", flush=True)

    separator_ids = list(stage_d["post_think_separator_token_ids"])
    cand_a = list(stage_d["candidates"][rc.CANDIDATE_MET]["contextual_token_ids"])
    cand_b = list(stage_d["candidates"][rc.CANDIDATE_NOT_MET]["contextual_token_ids"])

    retries: list[dict[str, Any]] = []
    if not todo.empty:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        t0 = time.time()
        llm = LLM(
            model=local_path, tokenizer=local_path, dtype="bfloat16",
            quantization=None,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=rc.EXPERIMENT_SEED, enforce_eager=False,
        )
        load_seconds = round(time.time() - t0, 2)
        print(f"model loaded in {load_seconds}s", flush=True)

        records = todo.to_dict("records")
        shard_index = len(list(shard_dir(root).glob("shard_*.json")))

        for start in range(0, len(records), args.chunk_size):
            chunk = records[start : start + args.chunk_size]
            chunk_t0 = time.time()

            # ---------- Stage R ----------
            prompt_ids_by_row: list[list[int]] = []
            sampling: list[Any] = []
            for record in chunk:
                rendered = tok.apply_chat_template(
                    [{"role": "user", "content": record["user_message"]}],
                    tokenize=False, add_generation_prompt=True,
                )
                ids = tok.encode(rendered, add_special_tokens=False)
                prompt_ids_by_row.append(ids)
                seed = ra.row_seed(
                    record["artifact_id"], int(record["strictness_index"])
                )
                record["_seed"] = seed
                record["_rendered_sha256"] = provenance.sha256_text(rendered)
                sampling.append(
                    SamplingParams(
                        temperature=rc.TEMPERATURE, top_p=rc.TOP_P, top_k=rc.TOP_K,
                        min_p=rc.MIN_P, max_tokens=budget, seed=seed,
                        stop_token_ids=[think_end_id],
                    )
                )

            gen_outputs = llm.generate(
                [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids_by_row],
                sampling,
            )

            traces = []
            for record, output in zip(chunk, gen_outputs, strict=True):
                completion = output.outputs[0]
                raw = list(completion.token_ids)
                if completion.finish_reason == "stop" and (
                    not raw or raw[-1] != think_end_id
                ):
                    raw = raw + [think_end_id]
                traces.append(ra.split_reasoning(raw, think_end_id, budget))

            # ---------- Stage D ----------
            score_prompts: list[Any] = []
            score_index: list[tuple[int, str, int]] = []
            contexts: list[tuple[int, ...]] = []
            for position, (record, trace) in enumerate(zip(chunk, traces, strict=True)):
                if not trace.think_end_reached:
                    contexts.append(())
                    continue
                ctx = ra.build_decision_context(
                    prompt_ids_by_row[position], trace.token_ids, separator_ids
                )
                contexts.append(ctx)
                for label, cand in (
                    (rc.CANDIDATE_MET, cand_a),
                    (rc.CANDIDATE_NOT_MET, cand_b),
                ):
                    score_prompts.append(
                        TokensPrompt(prompt_token_ids=list(ctx) + list(cand))
                    )
                    score_index.append((position, label, len(ctx)))

            logprobs: dict[tuple[int, str], float] = {}
            if score_prompts:
                score_outputs = llm.generate(
                    score_prompts,
                    SamplingParams(
                        max_tokens=1, temperature=0.0, prompt_logprobs=0
                    ),
                )
                for (position, label, ctx_len), output in zip(
                    score_index, score_outputs, strict=True
                ):
                    table = [
                        None
                        if entry is None
                        else {int(k): float(v.logprob) for k, v in entry.items()}
                        for entry in output.prompt_logprobs
                    ]
                    cand = cand_a if label == rc.CANDIDATE_MET else cand_b
                    logprobs[(position, label)] = ra.sequence_logprob(
                        table, ctx_len, cand
                    )

            # ---------- assemble ----------
            shard_rows: list[dict[str, Any]] = []
            for position, (record, trace) in enumerate(zip(chunk, traces, strict=True)):
                row: dict[str, Any] = {
                    "experiment": rc.STUDY_NAME,
                    "condition_tag": rc.CONDITION_TAG,
                    "model_id": rc.MODEL_ID,
                    "model_revision": revision,
                    "tokenizer_revision": model_block["tokenizer_revision"],
                    "prompt_id": record["prompt_id"],
                    "artifact_id": record["artifact_id"],
                    "family": record["family"],
                    "latent_level": int(record["latent_level"]),
                    "strictness_index": int(record["strictness_index"]),
                    "formal_truth": int(record["formal_truth"]),
                    "true_first_fail_index": int(record["true_first_fail_index"]),
                    "user_message_sha256": record["user_message_sha256"],
                    "formatted_prompt_sha256": record["_rendered_sha256"],
                    "n_prompt_tokens": len(prompt_ids_by_row[position]),
                    "generation_seed": int(record["_seed"]),
                    "reasoning_token_ids": list(trace.token_ids),
                    "decoded_reasoning": tok.decode(trace.token_ids),
                    "reasoning_token_count": trace.n_tokens,
                    "think_end_reached": bool(trace.think_end_reached),
                    "reasoning_truncated": bool(trace.truncated),
                    "stop_reason": trace.stop_reason,
                    "candidate_a_token_ids": list(cand_a),
                    "candidate_b_token_ids": list(cand_b),
                    "status": "ok" if trace.think_end_reached else "truncated",
                    "error": None,
                }
                if trace.think_end_reached:
                    l_a = logprobs[(position, rc.CANDIDATE_MET)]
                    l_b = logprobs[(position, rc.CANDIDATE_NOT_MET)]
                    margin = ra.canonical_margin(l_a, l_b)
                    row.update(
                        {
                            "logprob_a": l_a,
                            "logprob_b": l_b,
                            "margin": margin,
                            "semantic_decision": ra.semantic_decision(margin),
                        }
                    )
                else:
                    row.update(
                        {
                            "logprob_a": None,
                            "logprob_b": None,
                            "margin": None,
                            "semantic_decision": None,
                        }
                    )
                shard_rows.append(row)

            elapsed = time.time() - chunk_t0
            tokens = sum(r["reasoning_token_count"] for r in shard_rows)
            shard_path = shard_dir(root) / f"shard_{shard_index:05d}.json"
            shard_path.write_text(
                json.dumps(shard_rows, ensure_ascii=False), encoding="utf-8"
            )
            shard_index += 1
            done = start + len(chunk)
            print(
                f"[{done}/{len(records)}] chunk in {elapsed:.1f}s "
                f"({tokens} reasoning tokens, {tokens / max(elapsed, 1e-9):.0f} tok/s) "
                f"-> {shard_path.name}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Consolidate
    # ------------------------------------------------------------------
    all_rows, _ = load_completed(root)
    frame = pd.DataFrame(all_rows)
    if frame.empty:
        raise SystemExit("no rows produced")
    frame = frame.drop_duplicates("prompt_id", keep="last")
    frame = frame.sort_values(["artifact_id", "strictness_index"]).reset_index(drop=True)

    out_path = root / rc.RAW_ROWS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)

    attempted = int(len(frame))
    truncated = int(frame["reasoning_truncated"].sum())
    scored = int(frame["margin"].notna().sum())
    run_record = {
        "experiment": rc.STUDY_NAME,
        "model_id": rc.MODEL_ID,
        "revision": revision,
        "tokenizer_revision": model_block["tokenizer_revision"],
        "dtype": rc.MODEL_DTYPE,
        "quantization": rc.QUANTIZATION,
        "protocol": rc.PROTOCOL,
        "preregistration_sha256": provenance.sha256_file(root / rc.PREREG_JSON_PATH),
        "rows_requested": int(len(prompts_frame)),
        "rows_attempted": attempted,
        "rows_scored": scored,
        "rows_truncated": truncated,
        "truncation_rate": truncated / attempted if attempted else None,
        "raw_rows_sha256": provenance.sha256_file(out_path),
        "retries": retries,
        "finished_at": provenance.utc_now(),
        "environment": provenance.environment_snapshot(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
        "source_manifest": provenance.source_manifest(rc.ANALYSIS_CODE_PATHS, root),
    }
    provenance.write_json(root / rc.ARTIFACT_DIR / "run.json", run_record)
    provenance.write_json(root / rc.RETRY_LOG_PATH, {"retries": retries})

    print()
    print(f"rows attempted   {attempted}")
    print(f"rows scored      {scored}")
    print(f"rows truncated   {truncated}")
    print(f"wrote            {out_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
