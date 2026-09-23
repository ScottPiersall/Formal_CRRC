#!/usr/bin/env python
"""CPU tokenizer checks, GPU smoke, and protected paired inference, in that order.

Models are always loaded offline at the frozen Day-3 commit, unquantized bf16.
No Day-3 main function is called. Existing complete/partial runs are immutable;
retry with a fresh debug ID or explicitly archive an incomplete run first.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from formalcrrc import label_swap as ls, provenance, scoring, prompts
from formalcrrc.config import MODEL_IDS, MODEL_SLUGS


def load_historical(root, manifest, model_id, record):
    """Keep exact raw references and explicit mismatches; no silent inner join."""
    slug = MODEL_SLUGS[model_id]
    path = root / ls.d3c.RAW_SCORES_DIR / f"{slug}.parquet"
    run = ls.read_json(path.with_name(f"{slug}_run.json"))
    raw = pd.read_parquet(path)
    issues = []
    expected = {"model_id": model_id, "revision": record["revision"], "dtype": "bfloat16",
        "label_tokenization": record["label_tokenization"], "scoring_method": record["label_tokenization"]["scoring_method"],
        "chat_template_sha256": record["chat_template_sha256"],
        "dataset_file_sha256": provenance.sha256_file(root / ls.d3c.DATASET_PATH),
        "prompt_template_sha256": prompts.prompt_template_sha256(),
        "preregistration_sha256": (root / ls.d3c.PREREG_SHA_PATH).read_text().split()[0],
        "raw_scores_sha256": provenance.sha256_file(path), "prompts_scored": 2916,
        "prompts_requested": 2916, "failures": []}
    issues += [f"historical run mismatch: {key}" for key, value in expected.items() if run.get(key) != value]
    if raw.prompt_id.duplicated().any() or len(raw) != 2916:
        raise ValueError("historical raw file has duplicate/missing rows")
    if set(raw.prompt_id) != set(pd.read_parquet(root / ls.d3c.PROMPT_MANIFEST_PATH).prompt_id):
        raise ValueError("historical full prompt set mismatch")
    if set(raw.model_id) != {model_id}:
        raise ValueError("historical model_id mismatch")
    if set(raw.revision) != {record["revision"]}:
        issues.append("historical row revisions differ")
    if set(raw.scoring_method) != {expected["scoring_method"]}:
        issues.append("historical row scoring methods differ")
    selected = raw[raw.prompt_id.isin(manifest.original_prompt_id)].set_index("prompt_id")
    if set(selected.index) != set(manifest.original_prompt_id):
        raise ValueError("historical subsample missing rows")
    rows = []
    for entry in manifest.to_dict("records"):
        old = selected.loc[entry["original_prompt_id"]]
        for field in ls.META[:4]:
            if old[field] != entry[field]:
                raise ValueError(f"historical row metadata mismatch: {field}")
        if old.order_index != entry["historical_order_index"]:
            raise ValueError("historical order mismatch")
        if old.margin != old.raw_score_met - old.raw_score_not_met:
            raise ValueError("historical stored margin differs from semantic raw scores")
        # New manifest carries unchanged truth; historical raw did not store truth.
        result = {field: old[field] for field in ls.SCORE_FIELDS}
        rows.append({**entry, "model_id": model_id, "revision": old.revision,
                     **ls.semantic_scores(result, "original"), "score_source": "historical_reference"})
    frame = pd.DataFrame(rows)
    ls.validate_scores(frame, manifest, model_id, "original")
    return frame, run, issues


def environment_differences(historical, current):
    """Strict predeclared reuse gate. Different environments still permit fresh arms."""
    differences = []
    for key in ("python_version", "machine"):
        if historical.get(key) != current.get(key):
            differences.append(f"environment.{key}")
    for key in ("torch", "transformers", "tokenizers", "numpy"):
        if historical.get("packages", {}).get(key) != current.get("packages", {}).get(key):
            differences.append(f"packages.{key}")
    h, c = historical.get("gpu", {}), current.get("gpu", {})
    for key in ("torch_version", "torch_cuda_version", "cudnn_version"):
        if h.get(key) != c.get(key):
            differences.append(f"gpu.{key}")
    for key in ("name", "capability"):
        if not c.get("devices") or not h.get("devices") or h["devices"][0].get(key) != c["devices"][0].get(key):
            differences.append(f"gpu.device.{key}")
    return differences


def replay_comparison(current, historical):
    h = historical.set_index("original_prompt_id")
    rows = []
    for entry in current.to_dict("records"):
        old = h.loc[entry["original_prompt_id"]]
        row = {"original_prompt_id": entry["original_prompt_id"]}
        for field in ("raw_score_met", "raw_score_not_met", "margin", "p_met"):
            row[f"historical_{field}"] = float(old[field])
            row[f"current_{field}"] = float(entry[field])
            row[f"delta_{field}"] = float(entry[field] - old[field])
        row["decision_changed"] = bool((entry["margin"] >= 0) != (old.margin >= 0))
        row["rendered_hash_equal"] = entry["rendered_prompt_sha256"] == old.rendered_prompt_sha256
        rows.append(row)
    exact = all(not row["decision_changed"] and row["rendered_hash_equal"] and
                all(row[f"delta_{f}"] == 0 for f in ("raw_score_met", "raw_score_not_met", "margin", "p_met")) for row in rows)
    return rows, exact


def runtime_fingerprint(environment, settings):
    stable = {"python_version": environment["python_version"], "machine": environment["machine"],
              "packages": environment["packages"], "gpu": environment.get("gpu"), "settings": settings}
    return provenance.sha256_json(stable)


def score_rows(model, tokenizer, tokenization, manifest, model_id, revision, directory, name):
    rows, failures = [], []
    for position, entry in enumerate(manifest.to_dict("records"), 1):
        try:
            rendered = scoring.render_chat_prompt(tokenizer, entry["user_message"])
            result = scoring.score_prompt(model, tokenizer, rendered, tokenization)
            rows.append({**entry, "model_id": model_id, "revision": revision,
                         **ls.semantic_scores(result, entry["condition"]), "score_source": "fresh_inference"})
        except Exception as error:
            failures.append({"condition": entry["condition"], "original_prompt_id": entry["original_prompt_id"],
                             "error_type": type(error).__name__, "error": str(error)[:1000]})
        if position % 100 == 0:
            pd.DataFrame(rows).to_parquet(directory / f"{name}.partial.parquet", index=False)
            ls.write_json(directory / "failures.json", failures)
            print(f"{name} {position}/{len(manifest)}; failures={len(failures)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(directory / f"{name}.partial.parquet", index=False)
    if failures:
        ls.write_json(directory / "failures.json", failures)
        raise RuntimeError(f"incomplete {name}: {len(failures)} scoring failures")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=provenance.repo_root())
    parser.add_argument("--experiment-id", default=ls.DEFAULT_EXPERIMENT)
    parser.add_argument("--model", required=True, choices=MODEL_IDS)
    parser.add_argument("--stage", choices=("tokenize", "smoke", "run"), required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--hf-token-file", type=Path)
    parser.add_argument("--allow-tokenizer-download", action="store_true", help="tokenizer files only; model weights always offline")
    parser.add_argument("--debug-id", default="gpu_smoke", help="smoke output ID (also required by --stage run)")
    args = parser.parse_args()
    out, config, frames = ls.load_frozen(args.root, args.experiment_id)
    record = config["models"][args.model]
    slug = MODEL_SLUGS[args.model]
    if args.stage == "tokenize":
        directory = out / "tokenizer_checks" / slug
        directory.mkdir(parents=True, exist_ok=False)
    else:
        directory = ls.reserve_run(out, slug, debug_id=args.debug_id if args.stage == "smoke" else None)
    started = time.time()
    metadata = {"model_id": args.model, "stage": args.stage, "revision": record["revision"],
                "tokenizer_revision": record["tokenizer_revision"], "status": "incomplete",
                "frozen_sha256": provenance.sha256_file(out / "FROZEN.json"), "started_at": provenance.utc_now(),
                "slurm": {"job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME"),
                          "partition": os.environ.get("SLURM_JOB_PARTITION")}}
    ls.write_json(directory / "run.json", metadata)
    if not (directory / "INCOMPLETE.json").exists():
        ls.write_json(directory / "INCOMPLETE.json", {"status": "incomplete", "started_at": metadata["started_at"]}, exclusive=True)
    try:
        from transformers import AutoTokenizer
        if not re.fullmatch(r"[0-9a-f]{40}", record["revision"]) or record["tokenizer_revision"] != record["revision"]:
            raise ValueError("expected pinned Day-3 model and tokenizer commits")
        token = args.hf_token_file.read_text().strip() if args.hf_token_file else None
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=record["tokenizer_revision"],
            cache_dir=args.cache_dir, token=token, local_files_only=not args.allow_tokenizer_download)
        tokens, token_check = ls.check_tokenizer(tokenizer, frames, record)
        metadata["tokenizer_check"] = token_check
        if args.stage == "tokenize":
            metadata["environment"] = provenance.environment_snapshot(False)
            metadata["status"] = "complete"
            ls.write_json(directory / "run.json", metadata)
            ls.write_json(directory / "COMPLETE.json", {"status": "complete", "run_sha256": provenance.sha256_file(directory / "run.json")}, exclusive=True)
            (directory / "INCOMPLETE.json").unlink()
            print(f"tokenizer verified: {slug}: 729 prompts x 2 conditions", flush=True)
            return 0
        cpu_dir = out / "tokenizer_checks" / slug
        cpu_complete = ls.read_json(cpu_dir / "COMPLETE.json")
        if provenance.sha256_file(cpu_dir / "run.json") != cpu_complete["run_sha256"]:
            raise ValueError("CPU tokenizer completion checksum mismatch")
        if ls.read_json(cpu_dir / "run.json")["tokenizer_check"] != token_check:
            raise ValueError("tokenizer differs from completed CPU check")
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("a CUDA GPU supporting bf16 is required")
        environment = provenance.environment_snapshot(True)
        model_kwargs = dict(revision=record["revision"], token=token, cache_dir=args.cache_dir, local_files_only=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, **model_kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, **model_kwargs)
        model = model.to("cuda").eval()
        if model.dtype != torch.bfloat16 or model.training or getattr(model, "is_quantized", False):
            raise ValueError("model must be unquantized bf16 in eval mode")
        settings = {**config["inference"], "revision": record["revision"],
                    "attention_implementation": getattr(model.config, "_attn_implementation", None)}
        fingerprint = runtime_fingerprint(environment, settings)
        metadata.update(environment=environment, settings=settings, runtime_fingerprint=fingerprint,
                        source_manifest=provenance.source_manifest([
                            "src/formalcrrc/label_swap.py", "src/formalcrrc/scoring.py",
                            "scripts/run_label_swap_inference.py"], root=args.root))
        if args.stage == "run":
            smoke_path = out / "debug" / args.debug_id / slug
            smoke_complete = ls.read_json(smoke_path / "SMOKE_COMPLETE.json")
            ls.assert_hashes(smoke_path, smoke_complete["files"])
            smoke = ls.read_json(smoke_path / "run.json")
            if smoke["runtime_fingerprint"] != fingerprint or smoke["frozen_sha256"] != metadata["frozen_sha256"]:
                raise ValueError("completed GPU smoke uses a different runtime or selection")
            if smoke["source_manifest"] != metadata["source_manifest"]:
                raise ValueError("inference code changed after GPU smoke")
            smoke_decision = ls.read_json(smoke_path / "historical_reuse_decision.json")
        historical, historical_run, issues = load_historical(args.root, frames["original"], args.model, record)
        historical.to_parquet(directory / "historical_reference.parquet", index=False)
        expected_rendered = {r["original_prompt_id"]: r["rendered_prompt_sha256"]
                             for r in token_check["conditions"]["original"]["prompts"]}
        if any(row.rendered_prompt_sha256 != expected_rendered[row.original_prompt_id]
               for row in historical.itertuples()):
            issues.append("selected historical rendered prompt hash differs")
        issues += environment_differences(historical_run["environment"], environment)
        probes = frames["original"][frames["original"].original_prompt_id.isin(config["replay_prompt_ids"])]
        replay = score_rows(model, tokenizer, tokens["original"], probes, args.model, record["revision"], directory, "replay_original")
        comparison, exact = replay_comparison(replay, historical)
        if not exact:
            issues.append("fixed replay not exactly equal (zero-tolerance conservative gate)")
        if args.stage == "run" and smoke_decision["reuse_historical_original"] != (not issues):
            raise ValueError("original replay reuse decision changed since pre-swap smoke; audit required")
        # Must be persisted BEFORE any swapped inference, including smoke.
        decision = {"reuse_historical_original": not issues, "reasons_to_rerun_original": issues,
                    "decided_before_swapped_scoring": True, "replay_prompt_count": len(replay),
                    "replay": comparison, "decided_at": provenance.utc_now()}
        if args.stage == "run":
            decision["pre_swap_decision_sha256"] = provenance.sha256_file(smoke_path / "historical_reuse_decision.json")
            decision["note"] = "Formal run confirms and retains the decision made before GPU smoke produced any swapped score."
        ls.write_json(directory / "historical_reuse_decision.json", decision, exclusive=True)
        if args.stage == "smoke":
            swapped_probes = frames["swapped"][frames["swapped"].original_prompt_id.isin(config["replay_prompt_ids"][:3])]
            smoke = score_rows(model, tokenizer, tokens["swapped"], swapped_probes, args.model, record["revision"], directory, "smoke_swapped")
            if not np.isfinite(smoke[["raw_score_met", "raw_score_not_met", "margin", "p_met"]].to_numpy(float)).all():
                raise ValueError("non-finite GPU smoke scores")
        else:
            if decision["reuse_historical_original"]:
                original = historical.copy()
                original["score_source"] = "historical_reused_after_replay"
            else:
                original = score_rows(model, tokenizer, tokens["original"], frames["original"], args.model, record["revision"], directory, "original")
            ls.validate_scores(original, frames["original"], args.model, "original")
            swapped = score_rows(model, tokenizer, tokens["swapped"], frames["swapped"], args.model, record["revision"], directory, "swapped")
            ls.paired_frames(original, swapped, frames, args.model)
            original.to_parquet(directory / "original.parquet", index=False)
            swapped.to_parquet(directory / "swapped.parquet", index=False)
        metadata["historical_input_profile"] = ls.verify_input_profile(args.root, out, config)
        metadata.update(status="complete" if args.stage == "run" else "smoke_complete", finished_at=provenance.utc_now(), elapsed_seconds=time.time() - started,
                        original_source="historical" if not issues else "fresh", failures=[])
        ls.write_json(directory / "run.json", metadata)
        files = [str(p.relative_to(directory)).replace("\\", "/") for p in directory.iterdir() if p.is_file() and p.name != "INCOMPLETE.json"]
        ls.write_json(directory / ("COMPLETE.json" if args.stage == "run" else "SMOKE_COMPLETE.json"),
                      {"status": metadata["status"], "files": ls.file_hashes(directory, files)}, exclusive=True)
        (directory / "INCOMPLETE.json").unlink()
        print(f"{metadata['status']}: {slug}", flush=True)
        return 0
    except Exception as error:
        metadata.update(status="incomplete", finished_at=provenance.utc_now(), error_type=type(error).__name__,
                        error=str(error)[:1500], traceback=traceback.format_exc())
        ls.write_json(directory / "run.json", metadata)
        ls.write_json(directory / "INCOMPLETE.json", {"status": "incomplete", "error_type": type(error).__name__, "error": str(error)[:1500]})
        print(f"INCOMPLETE {slug}/{args.stage}: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
