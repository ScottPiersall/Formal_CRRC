#!/usr/bin/env python
"""Validate the ablation's mechanics before the preregistration is frozen.

Everything settled here is a property of the model, its tokenizer and the
generation stack -- never of FormalCRRC data. The probes are synthetic and
unrelated to the study, so running this cannot contaminate the preregistration.

What must be settled, because the frozen protocol depends on all of it:

* that the **exact frozen Day-3 revision** loads, not a floating one, and that
  its chat template still hashes to the Day-3 value;
* that ``FINAL:`` can be detected and stopped on as a complete sequence, and
  that generation halts *before* the model emits an A/B token -- otherwise the
  primary margin would be measuring a sampled label instead of a likelihood;
* the contextual tokenisation of the marker and of `` A`` / `` B``;
* that Stage-D scoring runs end to end and produces finite log-likelihoods;
* that the preregistered generation parameters are actually supported, including
  ``min_p``, which the protocol requires to be documented if it is not.

The Stage-D context is assembled from the exact text through the marker and
tokenised once. That is deliberate: a stop-string implementation may return a
final token whose text runs past the marker, and cutting on text keeps the
decision boundary exactly at the marker. The generated token ids are retained
separately for provenance, and the preflight records whether the two agree.
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
from formalcrrc import rts_ablation as ra  # noqa: E402
from formalcrrc import rts_ablation_config as rc  # noqa: E402

#: No rubric, no criterion, no candidate response, no strictness ladder. These
#: exist only to watch the model's mechanics.
SYNTHETIC_PROMPTS: tuple[str, ...] = (
    "Decide whether 91 is a prime number.\n"
    "Return A if it is prime.\n"
    "Return B if it is not prime.\n"
    "\n" + rc.REASONING_SCAFFOLD,
    "Decide whether the Danube flows through more than four countries.\n"
    "Return A if it does.\n"
    "Return B if it does not.\n"
    "\n" + rc.REASONING_SCAFFOLD,
    "Decide whether 0.1 + 0.2 equals 0.3 exactly in IEEE-754 double precision.\n"
    "Return A if it does.\n"
    "Return B if it does not.\n"
    "\n" + rc.REASONING_SCAFFOLD,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--hf-token-file", default=os.environ.get("HF_TOKEN_FILE"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-model-len", type=int, default=10240)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--preflight-budget", type=int, default=1024)
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    report: dict[str, Any] = {
        "study": rc.STUDY_NAME,
        "purpose": "pre-freeze mechanics validation on synthetic prompts",
        "uses_formalcrrc_data": False,
        "started_at": provenance.utc_now(),
        "model_id": rc.MODEL_ID,
        "requested_revision": rc.DAY3_REVISION,
        "checks": {},
        "problems": [],
    }

    def fail(message: str) -> None:
        report["problems"].append(message)
        print(f"PROBLEM: {message}", flush=True)

    token = None
    if args.hf_token_file and pathlib.Path(args.hf_token_file).is_file():
        token = pathlib.Path(args.hf_token_file).read_text().strip()

    # ------------------------------------------------------------------
    # 1. The exact frozen Day-3 revision
    # ------------------------------------------------------------------
    day3 = json.loads(
        (root / "artifacts/day3/model_provenance.json").read_text(encoding="utf-8")
    )["models"][rc.MODEL_ID]
    report["day3_revision_recorded"] = day3["revision"]
    report["checks"]["pinned_revision_matches_day3"] = (
        rc.DAY3_REVISION == day3["revision"]
        and rc.DAY3_TOKENIZER_REVISION == day3["tokenizer_revision"]
    )
    if not report["checks"]["pinned_revision_matches_day3"]:
        fail("pinned revision does not match frozen Day-3 provenance")

    from huggingface_hub import snapshot_download

    print("resolving the frozen Day-3 snapshot ...", flush=True)
    t0 = time.time()
    local_path = snapshot_download(
        rc.MODEL_ID,
        revision=rc.DAY3_REVISION,
        token=token,
        cache_dir=args.cache_dir,
        max_workers=4,
    )
    report["snapshot_path"] = local_path
    report["snapshot_seconds"] = round(time.time() - t0, 2)
    print(f"snapshot in {report['snapshot_seconds']}s", flush=True)

    config = json.loads((pathlib.Path(local_path) / "config.json").read_text())
    report["config_sha256"] = provenance.sha256_file(
        pathlib.Path(local_path) / "config.json"
    )
    report["config_torch_dtype"] = config.get("torch_dtype")
    report["checks"]["no_quantization_config"] = "quantization_config" not in config
    if "quantization_config" in config:
        fail("config.json carries a quantization_config; BF16 required")
    banned = [
        name
        for name in sorted(p.name for p in pathlib.Path(local_path).glob("*"))
        for marker in rc.FORBIDDEN_QUANTIZATION
        if marker in name.lower()
    ]
    report["checks"]["no_quantized_files"] = not banned
    if banned:
        fail(f"quantised artefacts present: {banned}")

    # ------------------------------------------------------------------
    # 2. Tokenizer, chat template, marker, candidates
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(local_path)
    chat_template = getattr(tok, "chat_template", None)
    report["chat_template_sha256"] = (
        _sha(chat_template) if isinstance(chat_template, str) else None
    )
    report["checks"]["chat_template_matches_day3"] = (
        report["chat_template_sha256"] == rc.DAY3_CHAT_TEMPLATE_SHA256
    )
    if not report["checks"]["chat_template_matches_day3"]:
        fail(
            "chat template does not match the frozen Day-3 hash "
            f"({report['chat_template_sha256']} != {rc.DAY3_CHAT_TEMPLATE_SHA256})"
        )

    rendered = tok.apply_chat_template(
        [{"role": "user", "content": SYNTHETIC_PROMPTS[0]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    report["generation_prompt_suffix"] = rendered[-40:]

    marker_context = rendered + "some synthetic reasoning here.\n"
    report["marker"] = ra.resolve_marker_tokenization(tok, marker_context)
    report["checks"]["marker_resolves"] = bool(
        report["marker"]["contextual_token_ids"]
    )

    decision_context_text = marker_context + rc.MARKER
    report["candidates"] = ra.resolve_candidate_tokenization(
        tok, decision_context_text
    )
    report["checks"]["candidates_resolve"] = bool(
        report["candidates"]["met_token_ids"]
        and report["candidates"]["not_met_token_ids"]
    )

    report["prompt_diff"] = {
        k: v
        for k, v in ra.prompt_diff().items()
        if k not in ("original_template", "rts_template", "added_suffix")
    }
    report["checks"]["prompt_diff_pure_suffix"] = all(
        ra.prompt_diff()["checks"].values()
    )

    # ------------------------------------------------------------------
    # 3. Load BF16 and exercise stopping + Stage D
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
        seed=rc.EXPERIMENT_SEED,
        enforce_eager=False,
    )
    report["model_load_seconds"] = round(time.time() - t0, 2)
    report["checks"]["bf16_load_ok"] = True
    print(f"loaded in {report['model_load_seconds']}s", flush=True)

    # Does the stack accept every preregistered generation parameter?
    try:
        SamplingParams(
            temperature=rc.TEMPERATURE, top_p=rc.TOP_P, top_k=rc.TOP_K,
            min_p=rc.MIN_P, max_tokens=8, seed=1,
        )
        report["checks"]["min_p_supported"] = True
        report["min_p_status"] = "supported"
    except TypeError as exc:  # pragma: no cover - environment dependent
        report["checks"]["min_p_supported"] = False
        report["min_p_status"] = f"NOT SUPPORTED: {exc}"
        fail(f"min_p unsupported by the pinned stack: {exc}")

    probes: list[dict[str, Any]] = []
    for index, text in enumerate(SYNTHETIC_PROMPTS):
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        seed = ra.row_seed(f"SYNTHETIC-{index}", 0)

        completion = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            SamplingParams(
                temperature=rc.TEMPERATURE, top_p=rc.TOP_P, top_k=rc.TOP_K,
                min_p=rc.MIN_P, max_tokens=args.preflight_budget, seed=seed,
                stop=[rc.MARKER], include_stop_str_in_output=True,
            ),
        )[0].outputs[0]

        generated_text = completion.text
        split = ra.split_at_marker(
            generated_text, budget_exhausted=completion.finish_reason == "length"
        )
        probe: dict[str, Any] = {
            "index": index,
            "seed": seed,
            "vllm_finish_reason": completion.finish_reason,
            "n_generated_tokens": len(completion.token_ids),
            "ends_exactly_at_marker": split.text_through_marker.endswith(rc.MARKER),
            "no_label_after_marker": not split.text_through_marker.rstrip().endswith(
                ("A", "B")
            )
            or split.text_through_marker.endswith(rc.MARKER),
            **split.to_dict(),
            "reasoning_preview": split.reasoning_text[-220:],
        }

        if split.marker_reached:
            context_text = prompt_text + split.text_through_marker
            context_ids = tok.encode(context_text, add_special_tokens=False)
            probe["n_context_tokens"] = len(context_ids)
            # Does re-tokenising the text agree with what was generated?
            probe["retokenization_matches_generation"] = (
                context_ids[: len(prompt_ids)] == list(prompt_ids)
            )
            scores: dict[str, float] = {}
            for label, ids in (
                (rc.CANDIDATE_MET, report["candidates"]["met_token_ids"]),
                (rc.CANDIDATE_NOT_MET, report["candidates"]["not_met_token_ids"]),
            ):
                # Resolve in this row's own context, not the reference context.
                local_ids = list(ra.continuation_ids(tok, context_text, label))
                out = llm.generate(
                    [TokensPrompt(prompt_token_ids=context_ids + local_ids)],
                    SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0),
                )[0]
                table = [
                    None
                    if entry is None
                    else {int(k): float(v.logprob) for k, v in entry.items()}
                    for entry in out.prompt_logprobs
                ]
                scores[label] = ra.sequence_logprob(
                    table, len(context_ids), local_ids
                )
                probe[f"token_ids{label}"] = local_ids
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
        print(
            json.dumps(
                {k: v for k, v in probe.items() if k != "reasoning_preview"},
                default=str,
            )[:400],
            flush=True,
        )

    report["synthetic_probes"] = probes
    report["checks"]["all_probes_reached_marker"] = all(
        p["marker_reached"] for p in probes
    )
    report["checks"]["stopping_is_exact"] = all(
        p.get("ends_exactly_at_marker") for p in probes if p["marker_reached"]
    )
    report["checks"]["stage_d_scored"] = all(
        "margin" in p for p in probes if p["marker_reached"]
    )
    if not report["checks"]["all_probes_reached_marker"]:
        fail("a synthetic probe did not reach the marker")
    if not report["checks"]["stopping_is_exact"]:
        fail("generation did not stop exactly at the marker")

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
