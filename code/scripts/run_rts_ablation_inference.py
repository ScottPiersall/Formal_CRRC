#!/usr/bin/env python
"""Run the frozen elicited reason-then-score protocol over the Day-3 partition.

Two stages per row. Stage R generates under a deterministic per-row seed and
stops on the complete ``FINAL:`` marker, so the model never samples the A/B
label that the primary margin is meant to measure. Stage D rebuilds the decision
context from that exact trace and scores `` A`` and `` B`` as continuations.

The Stage-D context is assembled from the exact **text** through the marker and
tokenised once. A stop-string implementation can return a final token whose text
runs past the marker; cutting on text keeps the decision boundary exactly where
the protocol says it is. The generated token ids are retained alongside, so the
raw record shows both.

Console output is operational only. No metric is computed here.

Checkpointed in shards so a timeout or node failure resumes rather than
restarts; a resumed row keeps its original seed, prompt and configuration.
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
from formalcrrc import rts_ablation as ra  # noqa: E402
from formalcrrc import rts_ablation_config as rc  # noqa: E402
from formalcrrc import day3_config as d3c  # noqa: E402

SHARD_DIR_NAME = "_shards"


def require_frozen(root: pathlib.Path) -> dict:
    json_path = root / rc.PREREG_JSON_PATH
    sha_path = root / rc.PREREG_SHA_PATH
    if not (json_path.is_file() and sha_path.is_file()):
        raise SystemExit(
            "REFUSING TO RUN: the preregistration is not frozen. Run "
            "scripts/freeze_rts_ablation_preregistration.py first."
        )
    if not provenance.verify_checksum_file(json_path, sha_path):
        raise SystemExit(
            "REFUSING TO RUN: the frozen preregistration does not match its "
            "recorded checksum."
        )
    prereg = json.loads(json_path.read_text(encoding="utf-8"))
    if int(prereg.get("amendments", -1)) != 0:
        raise SystemExit(f"REFUSING TO RUN: amendments = {prereg.get('amendments')!r}")
    return prereg


def load_rows(root: pathlib.Path) -> pd.DataFrame:
    """Frozen Day-3 rows, in the frozen inference order, with RTS prompts built."""
    manifest = pd.read_parquet(root / d3c.PROMPT_MANIFEST_PATH)
    dataset = pd.read_parquet(root / d3c.DATASET_PATH)
    keep = [
        "prompt_id", "artifact_id", "family", "latent_level", "strictness_index",
        "formal_truth", "true_first_fail_index", "rubric_text", "candidate_response",
    ]
    merged = manifest[["order_index", "prompt_id", "user_message_sha256"]].merge(
        dataset[keep], on="prompt_id", how="left", validate="one_to_one"
    )
    if len(merged) != rc.N_ROWS or merged["formal_truth"].isna().any():
        raise SystemExit(
            f"REFUSING TO RUN: join produced {len(merged)} usable rows, "
            f"expected {rc.N_ROWS}."
        )
    merged["rts_message"] = [
        ra.build_rts_message(r, c)
        for r, c in zip(merged["rubric_text"], merged["candidate_response"], strict=True)
    ]
    return merged.sort_values("order_index").reset_index(drop=True)


def shard_dir(root: pathlib.Path) -> pathlib.Path:
    path = root / rc.ARTIFACT_DIR / SHARD_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_completed(root: pathlib.Path) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    for path in sorted(shard_dir(root).glob("shard_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows, {r["prompt_id"] for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--hf-token-file", default=os.environ.get("HF_TOKEN_FILE"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=10240)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    prereg = require_frozen(root)
    model_block, stage_r, stage_d = prereg["model"], prereg["stage_r"], prereg["stage_d"]
    revision = model_block["revision"]
    budget = int(stage_r["reasoning_token_budget"])

    print(f"preregistration frozen at {prereg['frozen_at']}", flush=True)
    print(f"model {rc.MODEL_ID} @ {revision} (frozen Day-3 revision)", flush=True)

    rows_frame = load_rows(root)
    if args.limit:
        rows_frame = rows_frame.head(args.limit)
    _, done_ids = load_completed(root)
    todo = rows_frame[~rows_frame["prompt_id"].isin(done_ids)]
    print(
        f"rows: {len(rows_frame)} total, {len(done_ids)} complete, {len(todo)} to run",
        flush=True,
    )

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
            f"REFUSING TO RUN: chat template {chat_sha} does not match the "
            f"frozen record {model_block['chat_template_sha256']}."
        )
    if chat_sha != rc.DAY3_CHAT_TEMPLATE_SHA256:
        raise SystemExit("REFUSING TO RUN: chat template differs from Day 3.")
    print("chat template matches Day 3 exactly", flush=True)

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
            seed=rc.EXPERIMENT_SEED, enforce_eager=False,
        )
        print(f"model loaded in {round(time.time() - t0, 2)}s", flush=True)

        records = todo.to_dict("records")
        shard_index = len(list(shard_dir(root).glob("shard_*.json")))

        for start in range(0, len(records), args.chunk_size):
            chunk = records[start : start + args.chunk_size]
            chunk_t0 = time.time()

            # ---------- Stage R ----------
            prompt_texts: list[str] = []
            sampling: list[Any] = []
            for record in chunk:
                rendered = tok.apply_chat_template(
                    [{"role": "user", "content": record["rts_message"]}],
                    tokenize=False, add_generation_prompt=True,
                )
                prompt_texts.append(rendered)
                seed = ra.row_seed(
                    record["artifact_id"], int(record["strictness_index"])
                )
                record["_seed"] = seed
                record["_rendered_sha256"] = provenance.sha256_text(rendered)
                record["_rts_message_sha256"] = provenance.sha256_text(
                    record["rts_message"]
                )
                sampling.append(
                    SamplingParams(
                        temperature=rc.TEMPERATURE, top_p=rc.TOP_P, top_k=rc.TOP_K,
                        min_p=rc.MIN_P, max_tokens=budget, seed=seed,
                        stop=[rc.MARKER], include_stop_str_in_output=True,
                    )
                )

            prompt_ids = [tok.encode(t, add_special_tokens=False) for t in prompt_texts]
            gen_outputs = llm.generate(
                [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids], sampling
            )

            splits = []
            for output in gen_outputs:
                completion = output.outputs[0]
                splits.append(
                    ra.split_at_marker(
                        completion.text,
                        budget_exhausted=completion.finish_reason == "length",
                    )
                )

            # ---------- Stage D ----------
            score_prompts: list[Any] = []
            score_index: list[tuple[int, str, int, list[int]]] = []
            contexts: list[list[int]] = []
            for position, (record, split) in enumerate(zip(chunk, splits, strict=True)):
                if not split.marker_reached:
                    contexts.append([])
                    continue
                context_text = prompt_texts[position] + split.text_through_marker
                context_ids = tok.encode(context_text, add_special_tokens=False)
                contexts.append(context_ids)
                for label in (rc.CANDIDATE_MET, rc.CANDIDATE_NOT_MET):
                    cand = list(ra.continuation_ids(tok, context_text, label))
                    score_prompts.append(
                        TokensPrompt(prompt_token_ids=context_ids + cand)
                    )
                    score_index.append((position, label, len(context_ids), cand))

            logprobs: dict[tuple[int, str], float] = {}
            cand_ids_by_row: dict[tuple[int, str], list[int]] = {}
            if score_prompts:
                score_outputs = llm.generate(
                    score_prompts,
                    SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0),
                )
                for (position, label, ctx_len, cand), output in zip(
                    score_index, score_outputs, strict=True
                ):
                    table = [
                        None
                        if entry is None
                        else {int(k): float(v.logprob) for k, v in entry.items()}
                        for entry in output.prompt_logprobs
                    ]
                    logprobs[(position, label)] = ra.sequence_logprob(
                        table, ctx_len, cand
                    )
                    cand_ids_by_row[(position, label)] = cand

            # ---------- assemble ----------
            shard_rows: list[dict[str, Any]] = []
            for position, (record, split) in enumerate(zip(chunk, splits, strict=True)):
                completion = gen_outputs[position].outputs[0]
                row: dict[str, Any] = {
                    "experiment": rc.STUDY_NAME,
                    "condition": rc.CONDITION_REASONING,
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
                    "day3_user_message_sha256": record["user_message_sha256"],
                    "rts_message_sha256": record["_rts_message_sha256"],
                    "formatted_prompt_sha256": record["_rendered_sha256"],
                    "n_prompt_tokens": len(prompt_ids[position]),
                    "generation_seed": int(record["_seed"]),
                    "generated_token_ids": list(completion.token_ids),
                    "generated_token_count": len(completion.token_ids),
                    "reasoning_text": split.reasoning_text,
                    "text_through_marker": split.text_through_marker,
                    "reasoning_char_count": len(split.reasoning_text),
                    "vllm_finish_reason": completion.finish_reason,
                    "n_context_tokens": len(contexts[position]),
                    **split.to_dict(),
                    "status": "ok" if split.marker_reached else "truncated",
                    "error": None,
                }
                if split.marker_reached:
                    l_a = logprobs[(position, rc.CANDIDATE_MET)]
                    l_b = logprobs[(position, rc.CANDIDATE_NOT_MET)]
                    margin = ra.canonical_margin(l_a, l_b)
                    row.update(
                        {
                            "candidate_a_token_ids": cand_ids_by_row[
                                (position, rc.CANDIDATE_MET)
                            ],
                            "candidate_b_token_ids": cand_ids_by_row[
                                (position, rc.CANDIDATE_NOT_MET)
                            ],
                            "logprob_a": l_a,
                            "logprob_b": l_b,
                            "margin": margin,
                            "semantic_decision": ra.semantic_decision(margin),
                        }
                    )
                else:
                    row.update(
                        {
                            "candidate_a_token_ids": None,
                            "candidate_b_token_ids": None,
                            "logprob_a": None,
                            "logprob_b": None,
                            "margin": None,
                            "semantic_decision": None,
                        }
                    )
                shard_rows.append(row)

            elapsed = time.time() - chunk_t0
            tokens = sum(r["generated_token_count"] for r in shard_rows)
            markers = sum(1 for r in shard_rows if r["marker_reached"])
            shard_path = shard_dir(root) / f"shard_{shard_index:05d}.json"
            shard_path.write_text(
                json.dumps(shard_rows, ensure_ascii=False), encoding="utf-8"
            )
            shard_index += 1
            print(
                f"[{start + len(chunk)}/{len(records)}] {elapsed:.1f}s "
                f"({tokens} tok, {tokens / max(elapsed, 1e-9):.0f} tok/s, "
                f"marker {markers}/{len(chunk)}) -> {shard_path.name}",
                flush=True,
            )

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
    reached = int(frame["marker_reached"].sum())
    truncated = int(frame["reasoning_truncated"].sum())
    premature = int(frame["premature_label_observed"].sum())
    scored = int(frame["margin"].notna().sum())
    provenance.write_json(
        root / rc.ARTIFACT_DIR / "run.json",
        {
            "experiment": rc.STUDY_NAME,
            "condition": rc.CONDITION_REASONING,
            "model_id": rc.MODEL_ID,
            "revision": revision,
            "tokenizer_revision": model_block["tokenizer_revision"],
            "dtype": rc.MODEL_DTYPE,
            "quantization": rc.QUANTIZATION,
            "protocol": rc.PROTOCOL,
            "marker": rc.MARKER,
            "preregistration_sha256": provenance.sha256_file(
                root / rc.PREREG_JSON_PATH
            ),
            "rows_requested": int(len(rows_frame)),
            "rows_attempted": attempted,
            "rows_marker_reached": reached,
            "rows_scored": scored,
            "rows_truncated": truncated,
            "rows_premature_label": premature,
            "truncation_rate": truncated / attempted if attempted else None,
            "premature_label_rate": premature / attempted if attempted else None,
            "raw_rows_sha256": provenance.sha256_file(out_path),
            "retries": retries,
            "finished_at": provenance.utc_now(),
            "environment": provenance.environment_snapshot(),
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
            },
            "source_manifest": provenance.source_manifest(
                rc.ANALYSIS_CODE_PATHS, root
            ),
        },
    )
    provenance.write_json(root / rc.RETRY_LOG_PATH, {"retries": retries})

    print()
    print(f"rows attempted      {attempted}")
    print(f"marker reached      {reached}")
    print(f"rows scored         {scored}")
    print(f"rows truncated      {truncated}")
    print(f"premature label     {premature}")
    print(f"wrote               {out_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
