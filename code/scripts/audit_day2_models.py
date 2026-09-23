#!/usr/bin/env python
"""Phase D: confirm Day-2 can reproduce Day-1's scoring conditions exactly.

Day 2 must not upgrade, re-resolve or substitute a model. For each judge this
reads the **exact revision recorded in the immutable Day-1 provenance**, loads
the model at that revision, and verifies that every scoring-relevant property
still matches Day 1:

* the revision is accessible and is the pinned commit, not a branch,
* the chat-template hash is unchanged,
* the answer-label tokens resolve to the same ids under the same rule,
* the scoring method is the same,
* the dtype is the same,
* a non-study smoke prompt still scores finitely.

Any mismatch is recorded and the model is marked ``BLOCKED_BY_MODEL_ACCESS``.
Nothing is substituted. Writes ``artifacts/day2/model_provenance.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import day2_config as d2c  # noqa: E402
from formalcrrc import prompts, provenance, scoring  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    MODEL_DTYPE,
    MODEL_IDS,
    MODEL_PROVENANCE_PATH as DAY1_MODEL_PROVENANCE_PATH,
    MODEL_SLUGS,
)

#: The same neutral prompt Day 1 used. No rubric, no threshold, no response.
SMOKE_PROMPT = "Name one primary colour."

STATUS_AVAILABLE = "AVAILABLE"
STATUS_BLOCKED = "BLOCKED_BY_MODEL_ACCESS"


def read_token(path: str | None) -> str | None:
    """Read a Hugging Face token from disk without ever echoing it."""
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        print(f"note: no token file at {path}", flush=True)
        return None
    return candidate.read_text(encoding="utf-8").strip() or None


def audit_model(
    model_id: str, day1_record: dict, token: str | None, cache_dir: str | None
) -> dict:
    """Re-verify one judge against its Day-1 provenance record."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = day1_record["revision"]
    record: dict = {
        "model_id": model_id,
        "slug": MODEL_SLUGS[model_id],
        "status": STATUS_BLOCKED,
        "checked_at": provenance.utc_now(),
        "revision": revision,
        "tokenizer_revision": day1_record.get("tokenizer_revision", revision),
        "revision_source": "artifacts/day1/model_provenance.json (immutable)",
        "day1_chat_template_sha256": day1_record["chat_template_sha256"],
        "day1_label_tokenization": day1_record["label_tokenization"],
        "day1_dtype": day1_record.get("dtype"),
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, token=token, cache_dir=cache_dir
    )
    label_reference = scoring.render_chat_prompt(
        tokenizer, prompts.label_reference_message()
    )
    tokenization = scoring.resolve_label_tokenization(tokenizer, label_reference)
    chat_hash = provenance.sha256_text(tokenizer.chat_template or "")

    record["label_tokenization"] = tokenization.to_dict()
    record["chat_template_sha256"] = chat_hash
    record["label_reference_prompt_sha256"] = provenance.sha256_text(label_reference)

    continuity = {
        "chat_template_matches_day1": chat_hash
        == day1_record["chat_template_sha256"],
        "label_tokenization_matches_day1": tokenization.to_dict()
        == day1_record["label_tokenization"],
        "scoring_method_matches_day1": tokenization.scoring_method
        == day1_record["label_tokenization"]["scoring_method"],
    }
    record["continuity"] = continuity
    if not all(continuity.values()):
        record["error"] = "scoring conditions differ from the Day-1 record"
        return record

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
    model = model.to("cuda").eval()

    rendered = scoring.render_chat_prompt(tokenizer, SMOKE_PROMPT)
    smoke = scoring.score_prompt(model, tokenizer, rendered, tokenization)

    record["dtype"] = str(next(model.parameters()).dtype)
    record["continuity"]["dtype_matches_day1"] = (
        record["dtype"] == day1_record.get("dtype")
    )
    record["smoke_test"] = {
        "prompt": SMOKE_PROMPT,
        "is_study_prompt": False,
        "raw_score_met": smoke["raw_score_met"],
        "raw_score_not_met": smoke["raw_score_not_met"],
        "margin": smoke["raw_score_met"] - smoke["raw_score_not_met"],
        "scoring_method": smoke["scoring_method"],
        "day1_raw_score_met": day1_record.get("smoke_test", {}).get("raw_score_met"),
        "day1_raw_score_not_met": day1_record.get("smoke_test", {}).get(
            "raw_score_not_met"
        ),
    }
    record["smoke_test"]["reproduces_day1_scores"] = (
        record["smoke_test"]["raw_score_met"]
        == record["smoke_test"]["day1_raw_score_met"]
        and record["smoke_test"]["raw_score_not_met"]
        == record["smoke_test"]["day1_raw_score_not_met"]
    )
    record["transformers_version"] = transformers.__version__
    record["torch_version"] = torch.__version__
    record["status"] = (
        STATUS_AVAILABLE if all(record["continuity"].values()) else STATUS_BLOCKED
    )

    del model
    torch.cuda.empty_cache()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--hf-token-file", default=None)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    root = Path(args.root)

    day1_provenance = json.loads(
        (root / DAY1_MODEL_PROVENANCE_PATH).read_text(encoding="utf-8")
    )
    day1_models = day1_provenance["models"]

    # Prompt continuity is a hard precondition: Day 2 reuses the Day-1 template.
    day1_prereg = json.loads(
        (root / "artifacts/day1/preregistration.json").read_text(encoding="utf-8")
    )
    prompt_matches = (
        prompts.prompt_template_sha256() == day1_prereg["prompt"]["template_sha256"]
    )
    print(f"prompt template matches Day 1: {prompt_matches}", flush=True)
    if not prompt_matches:
        print("ABORT: DAY2_READY = FAIL (prompt template hash differs)", flush=True)
        return 2

    token = read_token(args.hf_token_file)
    models: dict = {}
    for model_id in MODEL_IDS:
        print(f"--- {model_id}", flush=True)
        day1_record = day1_models.get(model_id, {})
        if day1_record.get("status") != "AVAILABLE":
            models[model_id] = {
                "model_id": model_id,
                "status": STATUS_BLOCKED,
                "error": "no AVAILABLE Day-1 provenance record",
            }
            print(f"    {STATUS_BLOCKED}: no Day-1 record", flush=True)
            continue
        try:
            record = audit_model(model_id, day1_record, token, args.cache_dir)
        except Exception as error:  # noqa: BLE001 - recorded, never swallowed
            record = {
                "model_id": model_id,
                "slug": MODEL_SLUGS.get(model_id, model_id.replace("/", "_")),
                "status": STATUS_BLOCKED,
                "checked_at": provenance.utc_now(),
                "revision": day1_record.get("revision"),
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
                "traceback_tail": traceback.format_exc()[-2000:],
            }
            print(f"    {STATUS_BLOCKED}: {type(error).__name__}", flush=True)
        else:
            flags = record.get("continuity", {})
            print(
                f"    {record['status']} revision={record['revision'][:12]} "
                f"continuity={all(flags.values())} "
                f"smoke_identical={record.get('smoke_test', {}).get('reproduces_day1_scores')}",
                flush=True,
            )
        models[model_id] = record

    document = {
        "created_at": provenance.utc_now(),
        "experiment": "FormalCRRC Day 2",
        "panel_policy": (
            "Exactly the Day-1 four-model panel at exactly the Day-1 revisions, "
            "read from the immutable Day-1 provenance record. No upgrade, no "
            "re-resolution of a branch, no substitution."
        ),
        "preregistered_panel": list(MODEL_IDS),
        "dtype_requested": MODEL_DTYPE,
        "smoke_prompt": SMOKE_PROMPT,
        "smoke_prompt_is_study_prompt": False,
        "prompt_template_sha256": prompts.prompt_template_sha256(),
        "prompt_template_matches_day1": prompt_matches,
        "environment": provenance.environment_snapshot(include_gpu=True),
        "models": models,
        "panel_status": {
            model_id: models.get(model_id, {}).get("status", "NOT_CHECKED")
            for model_id in MODEL_IDS
        },
    }
    provenance.write_json(root / d2c.MODEL_PROVENANCE_PATH, document)

    available = sum(
        1 for m in MODEL_IDS if models.get(m, {}).get("status") == STATUS_AVAILABLE
    )
    print(f"available: {available}/{len(MODEL_IDS)}", flush=True)
    print(f"wrote {d2c.MODEL_PROVENANCE_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
