#!/usr/bin/env python
"""Verify that the scored label token is the one the model must actually emit.

Part of the Phase-C tokenizer inspection, run before any study prompt is scored
and before the preregistration is frozen. Loading a wrong token id into the
logit lookup is precisely the class of implementation defect the protocol names,
and it is invisible in the results, so it is checked directly.

For each judge and each answer label, this compares

* ``tokenizer.encode(label, add_special_tokens=False)`` -- the naive encoding, and
* the continuation ids implied by ``encode(rendered_prompt + label)`` minus
  ``encode(rendered_prompt)`` -- what the model would really have to emit.

A mismatch (typically a SentencePiece word-boundary marker) means the naive
encoding must not be used as the logit index. Stability is checked across
several rendered study prompts, since the continuation could in principle depend
on the prompt's final characters.

Tokenizers only; no model weights are loaded, and no probability is computed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import prompts as prompts_module  # noqa: E402
from formalcrrc import provenance, scoring  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    LABEL_MET,
    LABEL_NOT_MET,
    MODEL_IDS,
    MODEL_PROVENANCE_PATH,
    PROMPT_MANIFEST_PATH,
)

#: How many real study prompts to check the continuation against.
N_PROBES = 25


def check_model(model_id: str, revision: str, token: str | None, probes: list[str]):
    """Resolve the label tokens and confirm they hold across real study prompts."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, token=token
    )
    reference = scoring.render_chat_prompt(
        tokenizer, prompts_module.label_reference_message()
    )
    resolved = scoring.resolve_label_tokenization(tokenizer, reference)

    report: dict = {
        "model_id": model_id,
        "revision": revision,
        "reference_prompt_sha256": provenance.sha256_text(reference),
        "resolved_continuation_ids": {
            LABEL_MET: list(resolved.met_token_ids),
            LABEL_NOT_MET: list(resolved.not_met_token_ids),
        },
        "resolved_continuation_tokens": {
            LABEL_MET: tokenizer.convert_ids_to_tokens(list(resolved.met_token_ids)),
            LABEL_NOT_MET: tokenizer.convert_ids_to_tokens(
                list(resolved.not_met_token_ids)
            ),
        },
        "naive_encoding_ids": {
            LABEL_MET: list(resolved.naive_met_token_ids),
            LABEL_NOT_MET: list(resolved.naive_not_met_token_ids),
        },
        "naive_encoding_tokens": {
            LABEL_MET: tokenizer.convert_ids_to_tokens(
                list(resolved.naive_met_token_ids)
            ),
            LABEL_NOT_MET: tokenizer.convert_ids_to_tokens(
                list(resolved.naive_not_met_token_ids)
            ),
        },
        "differs_from_naive_encoding": resolved.differs_from_naive,
        "single_token": resolved.single_token,
        "scoring_method": resolved.scoring_method,
        "probes": [],
    }

    expected = {
        LABEL_MET: list(resolved.met_token_ids),
        LABEL_NOT_MET: list(resolved.not_met_token_ids),
    }
    stable = True
    for probe in probes:
        rendered = scoring.render_chat_prompt(tokenizer, probe)
        found = {
            label: list(scoring.label_continuation_ids(tokenizer, rendered, label))
            for label in (LABEL_MET, LABEL_NOT_MET)
        }
        matches = found == expected
        stable = stable and matches
        report["probes"].append(
            {
                "rendered_sha256": provenance.sha256_text(rendered),
                "continuation_ids": found,
                "matches_reference": matches,
            }
        )

    report["stable_across_study_prompts"] = stable
    report["verdict"] = "OK" if stable else "UNSTABLE"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--hf-token-file", default=None)
    args = parser.parse_args()
    root = Path(args.root)

    token = None
    if args.hf_token_file and Path(args.hf_token_file).is_file():
        token = Path(args.hf_token_file).read_text(encoding="utf-8").strip() or None

    manifest = pd.read_parquet(root / PROMPT_MANIFEST_PATH)
    probes = manifest["user_message"].head(N_PROBES).tolist()

    provenance_doc = json.loads(
        (root / MODEL_PROVENANCE_PATH).read_text(encoding="utf-8")
    )

    reports = []
    ok = True
    for model_id in MODEL_IDS:
        record = provenance_doc["models"].get(model_id, {})
        if record.get("status") != "AVAILABLE":
            print(f"{model_id}: skipped ({record.get('status')})", flush=True)
            continue
        report = check_model(model_id, record["revision"], token, probes)
        reports.append(report)
        ok = ok and report["verdict"] == "OK"
        print(f"=== {model_id}", flush=True)
        print(
            f"    scored (continuation) A={report['resolved_continuation_ids']['A']} "
            f"{report['resolved_continuation_tokens']['A']}  "
            f"B={report['resolved_continuation_ids']['B']} "
            f"{report['resolved_continuation_tokens']['B']}",
            flush=True,
        )
        print(
            f"    naive encoding        A={report['naive_encoding_ids']['A']} "
            f"{report['naive_encoding_tokens']['A']}  "
            f"B={report['naive_encoding_ids']['B']} "
            f"{report['naive_encoding_tokens']['B']}",
            flush=True,
        )
        print(
            f"    differs_from_naive={report['differs_from_naive_encoding']} "
            f"single_token={report['single_token']} "
            f"stable_across_{len(probes)}_study_prompts="
            f"{report['stable_across_study_prompts']} "
            f"verdict={report['verdict']}",
            flush=True,
        )

    out = root / "artifacts/day1/label_token_check.json"
    provenance.write_json(
        out,
        {
            "created_at": provenance.utc_now(),
            "n_probe_prompts": len(probes),
            "probe_source": "first rows of the frozen prompt manifest",
            "models": reports,
            "all_ok": ok,
        },
    )
    print(f"\nwrote artifacts/day1/label_token_check.json  all_ok={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
