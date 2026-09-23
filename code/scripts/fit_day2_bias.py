#!/usr/bin/env python
"""Fit the Day-2 calibration intercepts (Phase G).

Runs only after every intended raw run is complete. Fits, per judge:

* one global intercept ``alpha_J``, and
* one intercept per predicate family ``alpha_{J,F}`` (secondary),

**on the CALIBRATION partition alone**. The TEST partition is not read here, and
neither is any Day-1 outcome. The fitted values are written to
``artifacts/day2/calibration_parameters.json`` and frozen; the held-out analysis
applies them without refitting.

Exactly one free parameter per fitted unit. No slope, no temperature, no
isotonic map, no per-artifact or per-threshold adjustment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import day2, day2_config as d2c, provenance  # noqa: E402
from formalcrrc.config import FAMILIES, MODEL_IDS, MODEL_SHORT_NAMES, MODEL_SLUGS, N_ROWS  # noqa: E402


def load_partition_scores(
    root: Path, partition: str, model_id: str, dataset: pd.DataFrame
) -> pd.DataFrame | None:
    """Load one judge's raw scores for a partition, joined to formal truth."""
    slug = MODEL_SLUGS[model_id]
    path = root / d2c.RAW_SCORES_SUBDIR[partition] / f"{slug}.parquet"
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    truth = dataset[
        ["prompt_id", "formal_truth", "true_first_fail_index"]
    ]
    merged = frame.merge(truth, on="prompt_id", how="inner", validate="1:1")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="fit for whichever judges are complete instead of refusing",
    )
    args = parser.parse_args()
    root = Path(args.root)

    prereg = root / d2c.PREREG_JSON_PATH
    sha_path = root / d2c.PREREG_SHA_PATH
    if not (prereg.is_file() and sha_path.is_file()):
        print("ABORT: Day-2 preregistration is not frozen", flush=True)
        return 2
    if not provenance.verify_checksum_file(prereg, sha_path):
        print("ABORT: Day-2 preregistration checksum does not match", flush=True)
        return 2

    calibration_dataset = pd.read_parquet(root / d2c.CALIBRATION_DATASET_PATH)

    parameters: dict[str, dict] = {}
    blocked: list[str] = []

    for model_id in MODEL_IDS:
        scores = load_partition_scores(
            root, d2c.PARTITION_CALIBRATION, model_id, calibration_dataset
        )
        if scores is None or len(scores) != N_ROWS:
            blocked.append(model_id)
            parameters[model_id] = {
                "model_id": model_id,
                "status": "BLOCKED",
                "reason": (
                    "calibration raw scores absent"
                    if scores is None
                    else f"expected {N_ROWS} rows, found {len(scores)}"
                ),
            }
            print(f"{MODEL_SHORT_NAMES[model_id]:<14} BLOCKED", flush=True)
            continue

        margins = scores["margin"].to_numpy(dtype=float)
        truths = scores["formal_truth"].to_numpy(dtype=int)
        if not np.all(np.isfinite(margins)):
            blocked.append(model_id)
            parameters[model_id] = {
                "model_id": model_id,
                "status": "BLOCKED",
                "reason": "non-finite margins in the calibration partition",
            }
            continue

        global_fit = day2.fit_intercept(margins, truths)
        family_fits = day2.fit_family_intercepts(scores)

        parameters[model_id] = {
            "model_id": model_id,
            "short_name": MODEL_SHORT_NAMES[model_id],
            "status": "FITTED",
            "revision": sorted(set(scores["revision"]))[0],
            "scoring_method": sorted(set(scores["scoring_method"]))[0],
            "fitted_on": {
                "partition": d2c.PARTITION_CALIBRATION,
                "n_rows": int(len(scores)),
                "n_artifacts": int(scores["artifact_id"].nunique()),
                "positive_label_rate": float(truths.mean()),
                "dataset_content_sha256": json.loads(
                    (root / d2c.CALIBRATION_MANIFEST_PATH).read_text(encoding="utf-8")
                )["hashes"]["dataset_content_sha256"],
            },
            "global": global_fit.to_dict(),
            "family": {
                family: fit.to_dict() for family, fit in family_fits.items()
            },
            "margin_summary": {
                "mean": float(margins.mean()),
                "median": float(np.median(margins)),
                "sd": float(margins.std(ddof=1)),
                "p05": float(np.quantile(margins, 0.05)),
                "p95": float(np.quantile(margins, 0.95)),
            },
        }
        print(
            f"{MODEL_SHORT_NAMES[model_id]:<14} FITTED  "
            f"alpha={global_fit.alpha:+.4f} "
            f"converged={global_fit.converged} iters={global_fit.iterations}",
            flush=True,
        )
        for family in FAMILIES:
            fit = family_fits[family]
            print(
                f"    {family:<20} alpha={fit.alpha:+.4f} "
                f"converged={fit.converged}",
                flush=True,
            )

    if blocked and not args.allow_partial:
        print("", flush=True)
        print(
            "ABORT: calibration incomplete for "
            f"{[MODEL_SHORT_NAMES[m] for m in blocked]}; "
            "rerun with --allow-partial only if those judges are formally blocked",
            flush=True,
        )
        return 2

    document = {
        "created_at": provenance.utc_now(),
        "experiment": "FormalCRRC Day 2",
        "stage": "calibration fitting (Phase G)",
        "model": {
            "primary": d2c.CALIBRATION_PRIMARY,
            "primary_equation": "M(s) + alpha_J",
            "secondary": d2c.CALIBRATION_SECONDARY,
            "secondary_equation": "M(s) + alpha_{J,F}",
            "slope": d2c.CALIBRATION_SLOPE,
            "objective": d2c.CALIBRATION_OBJECTIVE,
            "solver": d2c.CALIBRATION_SOLVER,
        },
        "fitted_on": "CALIBRATION partition only",
        "test_partition_read": False,
        "day1_outcomes_read": False,
        "preregistration_sha256": sha_path.read_text(encoding="utf-8").split()[0],
        "parameters": parameters,
        "blocked": blocked,
        "environment": provenance.environment_snapshot(include_gpu=False),
        "source_manifest": provenance.source_manifest(
            [
                "src/formalcrrc/day2.py",
                "src/formalcrrc/day2_config.py",
                "scripts/fit_day2_bias.py",
            ],
            root=root,
        ),
    }
    provenance.write_json(root / d2c.CALIBRATION_PARAMETERS_PATH, document)
    print("", flush=True)
    print(f"wrote {d2c.CALIBRATION_PARAMETERS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
