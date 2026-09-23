#!/usr/bin/env python
"""Compute the Day-3 oracle bounds (Phase H, first stage).

Runs only after every intended raw run is complete. For each judge it:

1. builds the reachable crossing set of every artifact by **both** the
   prefix-record-low theorem and explicit intercept enumeration, and refuses to
   continue if they disagree anywhere;
2. computes the artifact-level oracle floor ``OTCE``;
3. enumerates the exact truth-optimised global and family intercepts;
4. asserts the nesting inequality ``TCE_G >= TCE_F >= TCE_A``.

Writes ``reachable_crossings.parquet``, ``oracle_global.json``,
``oracle_family.json`` and ``oracle_artifact.parquet``.

The oracles use formal truth deliberately. They are diagnostic bounds, never
deployable corrections.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import day3, day3_config as d3c, provenance  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    FAMILIES,
    MODEL_IDS,
    MODEL_SHORT_NAMES,
    MODEL_SLUGS,
    N_ROWS,
)


def load_scores(root: Path) -> tuple[dict, dict]:
    """Load each judge's raw scores joined to formal truth."""
    dataset = pd.read_parquet(root / d3c.DATASET_PATH)
    truth = dataset[["prompt_id", "formal_truth", "true_first_fail_index"]]

    scores: dict[str, pd.DataFrame] = {}
    status: dict[str, dict] = {}
    for model_id in MODEL_IDS:
        slug = MODEL_SLUGS[model_id]
        path = root / d3c.RAW_SCORES_DIR / f"{slug}.parquet"
        record: dict = {"model_id": model_id, "slug": slug}
        if not path.is_file():
            record.update(status="BLOCKED", reason="raw scores absent")
            status[model_id] = record
            continue
        frame = pd.read_parquet(path)
        problems: list[str] = []
        if len(frame) != N_ROWS:
            problems.append(f"{len(frame)} rows, expected {N_ROWS}")
        if not frame["prompt_id"].is_unique:
            problems.append("duplicate prompt_id")
        if set(frame["prompt_id"]) != set(truth["prompt_id"]):
            problems.append("prompt_id set does not match the frozen dataset")
        margins = frame["margin"].to_numpy(dtype=float)
        if not np.all(np.isfinite(margins)):
            problems.append("non-finite margins")
        recomputed = (
            frame["raw_score_met"].to_numpy(float)
            - frame["raw_score_not_met"].to_numpy(float)
        )
        if not np.array_equal(margins, recomputed):
            problems.append("margin does not equal S_A - S_B")

        record.update(
            rows=int(len(frame)),
            sha256=provenance.sha256_file(path),
            revisions=sorted(set(frame["revision"])),
        )
        if problems:
            record.update(status="BLOCKED", reason="; ".join(problems))
        else:
            record.update(status="COMPLETE", reason=None)
            scores[model_id] = frame.merge(
                truth, on="prompt_id", how="inner", validate="1:1"
            )
        status[model_id] = record
    return scores, status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = Path(args.root)

    prereg = root / d3c.PREREG_JSON_PATH
    sha_path = root / d3c.PREREG_SHA_PATH
    if not (prereg.is_file() and sha_path.is_file()):
        print("ABORT: Day-3 preregistration is not frozen", flush=True)
        return 2
    if not provenance.verify_checksum_file(prereg, sha_path):
        print("ABORT: Day-3 preregistration checksum does not match", flush=True)
        return 2

    scores, status = load_scores(root)
    for model_id, record in status.items():
        detail = f" ({record['reason']})" if record.get("reason") else ""
        print(f"{MODEL_SHORT_NAMES[model_id]:<14} {record['status']}{detail}", flush=True)
    if not scores:
        print("ABORT: no judge produced a complete run", flush=True)
        return 2

    # ---- reachability, cross-checked ---------------------------------------
    print("", flush=True)
    print("cross-checking the reachability theorem against enumeration ...", flush=True)
    tables = []
    disagreements = 0
    for model_id, frame in scores.items():
        table = day3.reachability_table(frame)
        bad = int((~table["methods_agree"]).sum())
        bad += int((table["otce_theorem"] != table["otce_enumeration"]).sum())
        disagreements += bad
        print(
            f"  {MODEL_SHORT_NAMES[model_id]:<14} {len(table)} artifacts, "
            f"{bad} disagreements",
            flush=True,
        )
        tables.append(table)
    reachable = pd.concat(tables, ignore_index=True)
    reachable.to_parquet(root / d3c.REACHABLE_CROSSINGS_PATH, index=False)

    if disagreements:
        print("", flush=True)
        print("ORACLE_IMPLEMENTATION = FAIL", flush=True)
        print("DAY3_ANALYSIS          = BLOCKED", flush=True)
        return 2
    print("ORACLE_IMPLEMENTATION = PASS", flush=True)

    # ---- oracle bounds -----------------------------------------------------
    global_doc: dict = {}
    family_doc: dict = {}
    artifact_rows = []
    nesting: dict = {}

    print("", flush=True)
    print("enumerating exact oracle intercepts ...", flush=True)
    for model_id, frame in scores.items():
        ids, block, boundaries, families = day3.stack_margins(frame)
        families_arr = np.asarray(families)

        raw_mean = float(
            np.mean(np.abs(day3.crossing_index(block) - boundaries))
        )
        global_oracle = day3.oracle_group_intercept(block, boundaries)

        family_oracles: dict[str, dict] = {}
        weighted, total = 0.0, 0
        for family in FAMILIES:
            rows = np.flatnonzero(families_arr == family)
            oracle = day3.oracle_group_intercept(block[rows], boundaries[rows])
            family_oracles[family] = oracle.to_dict()
            weighted += oracle.mean_tce * rows.size
            total += rows.size
        family_mean = weighted / total

        otce = np.array(
            [
                day3.oracle_translation_tce(block[i], int(boundaries[i]))
                for i in range(block.shape[0])
            ]
        )
        artifact_mean = float(otce.mean())

        report = day3.assert_oracle_nesting(
            raw_mean, global_oracle.mean_tce, family_mean, artifact_mean
        )
        nesting[model_id] = {
            "raw": raw_mean,
            "oracle_global": global_oracle.mean_tce,
            "oracle_family": family_mean,
            "oracle_artifact": artifact_mean,
            **report,
        }

        global_doc[model_id] = {
            "short_name": MODEL_SHORT_NAMES[model_id],
            **global_oracle.to_dict(),
            "raw_mean_tce": raw_mean,
        }
        family_doc[model_id] = {
            "short_name": MODEL_SHORT_NAMES[model_id],
            "weighted_mean_tce": family_mean,
            "by_family": family_oracles,
        }
        for position, artifact_id in enumerate(ids):
            artifact_rows.append(
                {
                    "model_id": model_id,
                    "artifact_id": artifact_id,
                    "family": families[position],
                    "true_first_fail_index": int(boundaries[position]),
                    "otce": int(otce[position]),
                    "nearest_reachable": day3.nearest_reachable_crossing(
                        block[position], int(boundaries[position])
                    ),
                }
            )

        print(
            f"  {MODEL_SHORT_NAMES[model_id]:<14} "
            f"alpha_G={global_oracle.alpha:+9.4f}  "
            f"nesting={report['status']}",
            flush=True,
        )

    pd.DataFrame(artifact_rows).to_parquet(
        root / d3c.ORACLE_ARTIFACT_PATH, index=False
    )

    stamp = {
        "created_at": provenance.utc_now(),
        "experiment": "FormalCRRC Day 3",
        "oracle_uses_formal_truth": True,
        "deployable": False,
        "note": (
            "Diagnostic bounds on what increasingly flexible location-only "
            "explanations can accomplish. Not evaluator corrections."
        ),
        "preregistration_sha256": sha_path.read_text(encoding="utf-8").split()[0],
        "enumeration": d3c.ORACLE_ENUMERATION,
        "tie_break": list(d3c.ORACLE_TIE_BREAK),
    }
    provenance.write_json(
        root / d3c.ORACLE_GLOBAL_PATH, {**stamp, "class": "global", "models": global_doc}
    )
    provenance.write_json(
        root / d3c.ORACLE_FAMILY_PATH, {**stamp, "class": "family", "models": family_doc}
    )

    all_nested = all(block["status"] == "PASS" for block in nesting.values())
    print("", flush=True)
    print(f"ORACLE_NESTING = {'PASS' if all_nested else 'FAIL'}", flush=True)
    if not all_nested:
        for model_id, block in nesting.items():
            if block["status"] != "PASS":
                print(f"  {MODEL_SHORT_NAMES[model_id]}: {block['checks']}", flush=True)
        return 2

    for path in (
        d3c.REACHABLE_CROSSINGS_PATH,
        d3c.ORACLE_GLOBAL_PATH,
        d3c.ORACLE_FAMILY_PATH,
        d3c.ORACLE_ARTIFACT_PATH,
    ):
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
