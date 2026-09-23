#!/usr/bin/env python
"""Validate the anchor model's mechanics before the preregistration is frozen.

Everything this script needs to establish is a property of the model, its
tokenizer and its chat template -- never of FormalCRRC data. It therefore runs
on synthetic prompts that have nothing to do with the study, and it refuses to
touch the Day-3 partition at all. Running it cannot contaminate the
preregistration.

What must be settled here, because the frozen protocol depends on all of it:

* the exact model and tokenizer revision actually loaded;
* that the official BF16 weights load on the documented cluster hardware, with
  no quantisation anywhere in the path;
* the token id of the native end-of-thinking delimiter;
* that the generation prompt really does open a thinking block, which is what
  makes immediate next-token scoring inappropriate;
* the token ids of the frozen post-thinking separator in context;
* the contextual tokenisation of candidates A and B, which may be more than one
  token and must never be encoded in isolation;
* that Stage D scoring runs end to end and produces finite log-likelihoods.

The separator is read out of the model's own chat template. It is fixed here,
before any study inference, and is not revisited afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import provenance  # noqa: E402
from formalcrrc import reasoning_anchor as ra  # noqa: E402
from formalcrrc import reasoning_anchor_config as rc  # noqa: E402

#: Deliberately unrelated to FormalCRRC: no rubric, no criterion, no candidate
#: response, no strictness ladder. Used only to observe model mechanics.
SYNTHETIC_PROMPTS: tuple[str, ...] = (
    "Is 17 a prime number? Return A if yes. Return B if no.",
    "Does the word 'strawberry' contain three letter r characters? "
    "Return A if it does. Return B if it does not.",
    "Is the Baltic Sea larger in surface area than the Caspian Sea? "
    "Return A if it is. Return B if it is not.",
)


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--hf-token-file", default=os.environ.get("HF_TOKEN_FILE"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--preflight-budget",
        type=int,
        default=2048,
        help="reasoning cap for the synthetic probes only; the study budget is "
        "the preregistered one and is not set here",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    report: dict[str, Any] = {
        "study": rc.STUDY_NAME,
        "purpose": "pre-freeze model mechanics validation on synthetic prompts",
        "uses_formalcrrc_data": False,
        "started_at": provenance.utc_now(),
        "model_id": rc.MODEL_ID,
        "requested_revision": args.revision,
        "checks": {},
        "problems": [],
    }

    def fail(msg: str) -> None:
        report["problems"].append(msg)
        print(f"PROBLEM: {msg}", flush=True)

    token = None
    if args.hf_token_file and pathlib.Path(args.hf_token_file).is_file():
        token = pathlib.Path(args.hf_token_file).read_text().strip()

    # ------------------------------------------------------------------
    # 1. Materialise the pinned revision
    # ------------------------------------------------------------------
    from huggingface_hub import snapshot_download

    print("resolving snapshot ...", flush=True)
    t0 = time.time()
    local_path = snapshot_download(
        rc.MODEL_ID,
        revision=args.revision,
        token=token,
        cache_dir=args.cache_dir,
        max_workers=4,
    )
    report["snapshot_path"] = local_path
    report["snapshot_seconds"] = round(time.time() - t0, 2)
    print(f"snapshot at {local_path} in {report['snapshot_seconds']}s", flush=True)

    config = json.loads((pathlib.Path(local_path) / "config.json").read_text())
    report["config_sha256"] = provenance.sha256_file(
        pathlib.Path(local_path) / "config.json"
    )
    report["config_torch_dtype"] = config.get("torch_dtype")
    report["config_model_type"] = config.get("model_type")
    report["quantization_config_present"] = "quantization_config" in config
    report["checks"]["dtype_is_bfloat16"] = config.get("torch_dtype") == "bfloat16"
    report["checks"]["no_quantization_config"] = "quantization_config" not in config
    if not report["checks"]["dtype_is_bfloat16"]:
        fail(f"config torch_dtype is {config.get('torch_dtype')!r}, expected bfloat16")
    if report["quantization_config_present"]:
        fail("config.json carries a quantization_config; official BF16 required")

    present = sorted(p.name for p in pathlib.Path(local_path).glob("*"))
    banned = [
        name
        for name in present
        for marker in rc.FORBIDDEN_QUANTIZATION
        if marker in name.lower()
    ]
    report["checks"]["no_quantized_files"] = not banned
    if banned:
        fail(f"quantised artefacts present in snapshot: {banned}")

    # ------------------------------------------------------------------
    # 2. Tokenizer, chat template, delimiter, separator, candidates
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(local_path)
    chat_template = getattr(tok, "chat_template", None)
    report["chat_template_sha256"] = (
        _sha_text(chat_template) if isinstance(chat_template, str) else None
    )

    think_ids = tok.encode(rc.THINK_END_TEXT, add_special_tokens=False)
    report["think_end_text"] = rc.THINK_END_TEXT
    report["think_end_token_ids"] = list(think_ids)
    report["checks"]["think_end_single_token"] = len(think_ids) == 1
    if len(think_ids) != 1:
        fail(f"{rc.THINK_END_TEXT!r} is not a single token: {think_ids}")
    think_end_id = int(think_ids[0])
    report["think_end_token_id"] = think_end_id

    rendered = tok.apply_chat_template(
        [{"role": "user", "content": SYNTHETIC_PROMPTS[0]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    report["generation_prompt_suffix"] = rendered[-40:]
    opens_thinking = rendered.rstrip().endswith("<think>")
    report["checks"]["generation_prompt_opens_thinking"] = opens_thinking
    if not opens_thinking:
        fail(
            "the generation prompt does not open a thinking block; the "
            "reason-then-score rationale must be re-derived before freeze"
        )

    # Separator token ids, resolved as a continuation of a realistic context.
    ctx_text = rendered + "some reasoning here" + rc.THINK_END_TEXT
    base_ids = tok.encode(ctx_text, add_special_tokens=False)
    with_sep = tok.encode(ctx_text + rc.POST_THINK_SEPARATOR, add_special_tokens=False)
    if with_sep[: len(base_ids)] != base_ids:
        fail("appending the separator perturbed the preceding tokenisation")
        sep_ids: list[int] = []
    else:
        sep_ids = list(with_sep[len(base_ids) :])
    report["post_think_separator"] = rc.POST_THINK_SEPARATOR
    report["post_think_separator_bytes"] = rc.POST_THINK_SEPARATOR.encode().hex()
    report["post_think_separator_token_ids"] = sep_ids
    report["checks"]["separator_resolves"] = bool(sep_ids)

    # Candidate tokenisation, contextual and never isolated.
    decision_ctx_text = ctx_text + rc.POST_THINK_SEPARATOR
    cand: dict[str, dict[str, Any]] = {}
    for label in (rc.CANDIDATE_MET, rc.CANDIDATE_NOT_MET):
        full = tok.encode(decision_ctx_text + label, add_special_tokens=False)
        base = tok.encode(decision_ctx_text, add_special_tokens=False)
        if full[: len(base)] != base:
            fail(f"candidate {label!r} perturbed the decision-context tokenisation")
            continuation = []
        else:
            continuation = list(full[len(base) :])
        cand[label] = {
            "contextual_token_ids": continuation,
            "n_tokens": len(continuation),
            "isolated_token_ids": list(tok.encode(label, add_special_tokens=False)),
        }
        cand[label]["differs_from_isolated"] = (
            continuation != cand[label]["isolated_token_ids"]
        )
    report["candidates"] = cand
    report["checks"]["candidates_resolve"] = all(
        c["n_tokens"] >= 1 for c in cand.values()
    )

    # ------------------------------------------------------------------
    # 3. Load BF16 and exercise both stages on synthetic prompts
    # ------------------------------------------------------------------
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    print("loading model (bfloat16) ...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=local_path,
        tokenizer=local_path,
        dtype="bfloat16",
        quantization=None,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=rc.EXPERIMENT_SEED,
        enforce_eager=False,
    )
    report["model_load_seconds"] = round(time.time() - t0, 2)
    report["checks"]["bf16_load_ok"] = True
    print(f"loaded in {report['model_load_seconds']}s", flush=True)

    probes: list[dict[str, Any]] = []
    for index, text in enumerate(SYNTHETIC_PROMPTS):
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        seed = ra.row_seed(f"SYNTHETIC-{index}", 0)

        gen = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            SamplingParams(
                temperature=rc.TEMPERATURE,
                top_p=rc.TOP_P,
                top_k=rc.TOP_K,
                min_p=rc.MIN_P,
                max_tokens=args.preflight_budget,
                seed=seed,
                stop_token_ids=[think_end_id],
            ),
        )[0].outputs[0]
        raw_ids = list(gen.token_ids)
        # vLLM may drop the stop token from the output; restore it so the trace
        # ends exactly where the native delimiter does.
        if gen.finish_reason == "stop" and (not raw_ids or raw_ids[-1] != think_end_id):
            raw_ids = raw_ids + [think_end_id]
        trace = ra.split_reasoning(raw_ids, think_end_id, args.preflight_budget)

        probe: dict[str, Any] = {
            "index": index,
            "seed": seed,
            "vllm_finish_reason": gen.finish_reason,
            "n_reasoning_tokens": trace.n_tokens,
            "think_end_reached": trace.think_end_reached,
            "truncated": trace.truncated,
            "reasoning_preview": tok.decode(trace.token_ids)[:300],
        }

        if trace.think_end_reached:
            ctx = ra.build_decision_context(prompt_ids, trace.token_ids, sep_ids)
            scores: dict[str, float] = {}
            for label in (rc.CANDIDATE_MET, rc.CANDIDATE_NOT_MET):
                cand_ids = cand[label]["contextual_token_ids"]
                full_ids = list(ctx) + list(cand_ids)
                out = llm.generate(
                    [TokensPrompt(prompt_token_ids=full_ids)],
                    SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0),
                )[0]
                table = [
                    None
                    if entry is None
                    else {int(k): float(v.logprob) for k, v in entry.items()}
                    for entry in out.prompt_logprobs
                ]
                scores[label] = ra.sequence_logprob(table, len(ctx), cand_ids)
            margin = ra.canonical_margin(
                scores[rc.CANDIDATE_MET], scores[rc.CANDIDATE_NOT_MET]
            )
            probe.update(
                {
                    "logprob_A": scores[rc.CANDIDATE_MET],
                    "logprob_B": scores[rc.CANDIDATE_NOT_MET],
                    "margin": margin,
                    "decision": ra.semantic_decision(margin),
                }
            )
        probes.append(probe)
        print(json.dumps({k: v for k, v in probe.items()}, default=str)[:400], flush=True)

    report["synthetic_probes"] = probes
    report["checks"]["all_probes_reached_think_end"] = all(
        p["think_end_reached"] for p in probes
    )
    report["checks"]["stage_d_scored"] = all(
        "margin" in p and abs(p["margin"]) < float("inf") for p in probes
    )
    if not report["checks"]["all_probes_reached_think_end"]:
        fail("a synthetic probe did not reach the end-of-thinking delimiter")

    # ------------------------------------------------------------------
    # 4. Verdict
    # ------------------------------------------------------------------
    report["environment"] = provenance.environment_snapshot()
    report["slurm"] = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    report["finished_at"] = provenance.utc_now()
    report["status"] = (
        "PASS" if all(report["checks"].values()) and not report["problems"] else "FAIL"
    )

    out_path = pathlib.Path(args.out) if args.out else root / rc.PREFLIGHT_PATH
    provenance.write_json(out_path, report)
    print(f"\nwrote {out_path}")
    print(f"PREFLIGHT STATUS: {report['status']}")
    for name, ok in report["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
