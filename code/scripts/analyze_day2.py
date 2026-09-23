#!/usr/bin/env python
"""Run the preregistered Day-2 held-out analysis exactly once (Phase H).

Applies the frozen calibration parameters -- fitted on CALIBRATION alone -- to
the held-out TEST partition, computes the preregistered metrics, runs the
two-stage bootstrap that refits the intercept inside every replicate, and
renders the six predeclared figures.

Nothing is refitted here. Nothing is recalibrated, smoothed, monotonised or
filtered. Judges without complete runs on both partitions are reported as
BLOCKED rather than analysed as if complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import figures_day2  # noqa: E402  (sibling module in scripts/)
from formalcrrc import day2, day2_bootstrap, day2_config as d2c, provenance  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FAMILIES,
    MODEL_IDS,
    MODEL_SHORT_NAMES,
    MODEL_SLUGS,
    N_ARTIFACTS,
    N_ROWS,
    N_STRICTNESS,
)

STATUS_COMPLETE = "COMPLETE"
STATUS_BLOCKED = "BLOCKED"


def load_scores(root: Path) -> tuple[dict, dict, dict]:
    """Load both partitions for every judge, joined to formal truth."""
    datasets = {
        partition: pd.read_parquet(root / path)
        for partition, path in d2c.DATASET_PATHS.items()
    }
    truth = {
        partition: frame[
            ["prompt_id", "formal_truth", "true_first_fail_index"]
        ]
        for partition, frame in datasets.items()
    }

    calibration: dict[str, pd.DataFrame] = {}
    test: dict[str, pd.DataFrame] = {}
    status: dict[str, dict] = {}

    for model_id in MODEL_IDS:
        slug = MODEL_SLUGS[model_id]
        record: dict = {"model_id": model_id, "slug": slug, "partitions": {}}
        loaded: dict[str, pd.DataFrame] = {}
        problems: list[str] = []

        for partition in d2c.DATASET_PATHS:
            path = root / d2c.RAW_SCORES_SUBDIR[partition] / f"{slug}.parquet"
            if not path.is_file():
                problems.append(f"{partition}: raw scores absent")
                continue
            frame = pd.read_parquet(path)
            if len(frame) != N_ROWS:
                problems.append(f"{partition}: {len(frame)} rows, expected {N_ROWS}")
            if not frame["prompt_id"].is_unique:
                problems.append(f"{partition}: duplicate prompt_id")
            if set(frame["prompt_id"]) != set(truth[partition]["prompt_id"]):
                problems.append(f"{partition}: prompt_id set does not match")
            margins = frame["margin"].to_numpy(dtype=float)
            if not np.all(np.isfinite(margins)):
                problems.append(f"{partition}: non-finite margins")
            record["partitions"][partition] = {
                "rows": int(len(frame)),
                "sha256": provenance.sha256_file(path),
                "revisions": sorted(set(frame["revision"])),
                "scoring_methods": sorted(set(frame["scoring_method"])),
            }
            loaded[partition] = frame.merge(
                truth[partition], on="prompt_id", how="inner", validate="1:1"
            )

        if problems:
            record.update(status=STATUS_BLOCKED, reason="; ".join(problems))
        else:
            record.update(status=STATUS_COMPLETE, reason=None)
            calibration[model_id] = loaded[d2c.PARTITION_CALIBRATION]
            test[model_id] = loaded[d2c.PARTITION_TEST]
        status[model_id] = record

    return calibration, test, status


def tce_summary(values: np.ndarray) -> dict:
    """Distributional summary of a threshold-crossing-error vector."""
    values = np.asarray(values, dtype=float)
    keys, counts = np.unique(values.astype(int), return_counts=True)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "exact_boundary_rate": float(np.mean(values == 0)),
        "within_one_step_rate": float(np.mean(values <= 1)),
        "max": int(values.max()),
        "distribution": {str(int(k)): int(v) for k, v in zip(keys, counts)},
    }


def build_summary(
    metrics: dict[str, pd.DataFrame],
    test_scores: dict[str, pd.DataFrame],
    parameters: dict,
    bootstrap_rows: list[dict],
    comparison_rows: list[dict],
    invariance: dict[str, dict],
    status: dict[str, dict],
    root: Path,
) -> dict:
    per_model: dict[str, dict] = {}

    for model_id, frame in metrics.items():
        block = parameters["parameters"][model_id]
        alphas_family = {f: block["family"][f]["alpha"] for f in FAMILIES}

        family_block: dict[str, dict] = {}
        for family in FAMILIES:
            subset = frame[frame["family"] == family]
            family_block[family] = {
                "n_artifacts": int(len(subset)),
                "alpha_family": alphas_family[family],
                "tce_raw": tce_summary(subset["tce_raw"].to_numpy()),
                "tce_global": tce_summary(subset["tce_global"].to_numpy()),
                "tce_family": tce_summary(subset["tce_family"].to_numpy()),
                "delta_tce_global": float(subset["delta_tce_global"].mean()),
                "delta_tce_family": float(subset["delta_tce_family"].mean()),
                "mmvr": float(subset["mmvr"].mean()),
                "mmvm": float(subset["mmvm"].mean()),
                "span": float(subset["span"].mean()),
                "accuracy_raw": float(subset["accuracy_raw"].mean()),
                "accuracy_global": float(subset["accuracy_global"].mean()),
                "accuracy_family": float(subset["accuracy_family"].mean()),
                "brier_raw": float(subset["brier_raw"].mean()),
                "brier_global": float(subset["brier_global"].mean()),
                "brier_family": float(subset["brier_family"].mean()),
            }

        spearman = frame["margin_spearman"].to_numpy(dtype=float)
        finite = spearman[np.isfinite(spearman)]

        per_model[model_id] = {
            "short_name": MODEL_SHORT_NAMES[model_id],
            "status": STATUS_COMPLETE,
            "revision": block["revision"],
            "scoring_method": block["scoring_method"],
            "alpha_global": block["global"]["alpha"],
            "alpha_global_converged": block["global"]["converged"],
            "alpha_family": alphas_family,
            "n_artifacts": int(len(frame)),
            "overall": {
                "tce_raw": tce_summary(frame["tce_raw"].to_numpy()),
                "tce_global": tce_summary(frame["tce_global"].to_numpy()),
                "tce_family": tce_summary(frame["tce_family"].to_numpy()),
                "delta_tce_global": float(frame["delta_tce_global"].mean()),
                "delta_tce_family": float(frame["delta_tce_family"].mean()),
                "mmvr": float(frame["mmvr"].mean()),
                "mmvm": float(frame["mmvm"].mean()),
                "span": float(frame["span"].mean()),
                "margin_spearman_mean": (
                    float(finite.mean()) if finite.size else float("nan")
                ),
                "margin_spearman_undefined": int(spearman.size - finite.size),
                "accuracy_raw": float(frame["accuracy_raw"].mean()),
                "accuracy_global": float(frame["accuracy_global"].mean()),
                "accuracy_family": float(frame["accuracy_family"].mean()),
                "brier_raw": float(frame["brier_raw"].mean()),
                "brier_global": float(frame["brier_global"].mean()),
                "brier_family": float(frame["brier_family"].mean()),
                "never_crosses_raw": int(frame["never_crosses_raw"].sum()),
                "never_crosses_global": int(frame["never_crosses_global"].sum()),
                "always_below_raw": int(frame["always_below_raw"].sum()),
                "always_below_global": int(frame["always_below_global"].sum()),
            },
            "taxonomy": {
                name: int(frame[name].sum()) for name in day2.TAXONOMY_COLUMNS
            },
            "by_family": family_block,
            "shape_invariance": invariance.get(model_id, {}),
        }

    boundary = (
        pd.concat(
            [
                day2.boundary_aligned_curves(
                    test_scores[m], parameters["parameters"][m]["global"]["alpha"]
                )
                for m in metrics
            ],
            ignore_index=True,
        )
        if metrics
        else pd.DataFrame()
    )

    complete = [m for m, r in status.items() if r["status"] == STATUS_COMPLETE]
    return {
        "created_at": provenance.utc_now(),
        "experiment": "FormalCRRC Day 2",
        "analysis_run": "single preregistered run",
        "design": {
            "n_artifacts_per_partition": N_ARTIFACTS,
            "n_rows_per_partition": N_ROWS,
            "n_strictness_levels": N_STRICTNESS,
            "families": list(FAMILIES),
            "calibration_seed": d2c.SEED_CALIBRATION,
            "test_seed": d2c.SEED_TEST,
        },
        "panel": {
            model_id: {
                "status": record["status"],
                "reason": record.get("reason"),
                "partitions": record.get("partitions"),
            }
            for model_id, record in status.items()
        },
        "panel_completeness": (
            "COMPLETE"
            if len(complete) == len(MODEL_IDS)
            else ("PARTIAL" if complete else "BLOCKED")
        ),
        "models": per_model,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "stages": d2c.BOOTSTRAP_STAGES,
            "results": bootstrap_rows,
        },
        "paired_comparisons": comparison_rows,
        "boundary_aligned_curve": json.loads(boundary.to_json(orient="records")),
        "preregistration_sha256": (
            (root / d2c.PREREG_SHA_PATH).read_text(encoding="utf-8").split()[0]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
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

    parameters_path = root / d2c.CALIBRATION_PARAMETERS_PATH
    if not parameters_path.is_file():
        print("ABORT: calibration parameters absent; run fit_day2_bias.py", flush=True)
        return 2
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))

    calibration_scores, test_scores, status = load_scores(root)
    for model_id, record in status.items():
        detail = f" ({record['reason']})" if record.get("reason") else ""
        print(
            f"{MODEL_SHORT_NAMES[model_id]:<14} {record['status']}{detail}", flush=True
        )
    if not test_scores:
        print("ABORT: no judge produced complete runs on both partitions", flush=True)
        return 2

    # ---- per-artifact held-out metrics -------------------------------------
    metrics: dict[str, pd.DataFrame] = {}
    invariance: dict[str, dict] = {}
    for model_id, frame in test_scores.items():
        block = parameters["parameters"][model_id]
        alpha_global = block["global"]["alpha"]
        alphas_family = {f: block["family"][f]["alpha"] for f in FAMILIES}
        metrics[model_id] = day2.compute_artifact_metrics(
            frame, alpha_global, alphas_family
        )
        invariance[model_id] = day2.shape_invariance_report(
            frame, [alpha_global, *alphas_family.values(), -5.0, 5.0]
        )

    combined = pd.concat(metrics.values(), ignore_index=True)
    combined.to_parquet(root / d2c.METRICS_PATH, index=False)

    # ---- two-stage bootstrap ----------------------------------------------
    reference = next(iter(metrics.values())).sort_values("artifact_id")
    calibration_artifacts = (
        next(iter(calibration_scores.values()))
        .drop_duplicates("artifact_id")[["artifact_id", "family"]]
        .sort_values("artifact_id")
        .reset_index(drop=True)
    )
    test_artifacts = reference[["artifact_id", "family"]].reset_index(drop=True)

    plan = day2_bootstrap.build_two_stage_plan(
        calibration_artifacts, test_artifacts, replicates=args.replicates
    )
    print("", flush=True)
    print(
        f"bootstrap: {plan.replicates} replicates, seed {plan.seed}, "
        f"refitting alpha inside each replicate ...",
        flush=True,
    )

    samples: dict[str, dict[str, np.ndarray]] = {}
    for model_id in metrics:
        samples[model_id] = day2_bootstrap.bootstrap_judge(
            calibration_scores[model_id], test_scores[model_id], plan
        )
        print(f"  {MODEL_SHORT_NAMES[model_id]:<14} done", flush=True)

    bootstrap_rows: list[dict] = []
    for model_id, frame in metrics.items():
        block = parameters["parameters"][model_id]
        points = {
            "alpha_global": block["global"]["alpha"],
            "tce_raw": float(frame["tce_raw"].mean()),
            "tce_global": float(frame["tce_global"].mean()),
            "tce_family": float(frame["tce_family"].mean()),
            "delta_tce_global": float(frame["delta_tce_global"].mean()),
            "delta_tce_family": float(frame["delta_tce_family"].mean()),
            "exact_rate_raw": float((frame["tce_raw"] == 0).mean()),
            "exact_rate_global": float((frame["tce_global"] == 0).mean()),
            "within_one_raw": float((frame["tce_raw"] <= 1).mean()),
            "within_one_global": float((frame["tce_global"] <= 1).mean()),
        }
        for family in FAMILIES:
            subset = frame[frame["family"] == family]
            points[f"alpha_{family}"] = block["family"][family]["alpha"]
            points[f"tce_raw_{family}"] = float(subset["tce_raw"].mean())
            points[f"tce_global_{family}"] = float(subset["tce_global"].mean())
            points[f"tce_family_{family}"] = float(subset["tce_family"].mean())
            points[f"delta_tce_global_{family}"] = float(
                subset["delta_tce_global"].mean()
            )
        for quantity, point in points.items():
            bootstrap_rows.append(
                {
                    "model_id": model_id,
                    "quantity": quantity,
                    **day2_bootstrap.summarise(samples[model_id][quantity], point),
                }
            )

    comparison_rows: list[dict] = []
    model_list = list(metrics)
    for i, model_a in enumerate(model_list):
        for model_b in model_list[i + 1 :]:
            for quantity in ("delta_tce_global", "tce_raw", "tce_global", "alpha_global"):
                point = float(
                    np.mean(samples[model_a][quantity])
                    - np.mean(samples[model_b][quantity])
                )
                comparison_rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "quantity": quantity,
                        **day2_bootstrap.paired_difference(
                            samples[model_a][quantity],
                            samples[model_b][quantity],
                            point,
                        ),
                    }
                )

    pd.DataFrame(bootstrap_rows).to_parquet(root / d2c.BOOTSTRAP_PATH, index=False)

    summary = build_summary(
        metrics,
        test_scores,
        parameters,
        bootstrap_rows,
        comparison_rows,
        invariance,
        status,
        root,
    )
    provenance.write_json(root / d2c.SUMMARY_PATH, summary)

    if not args.skip_figures:
        figures_day2.render_all(root, test_scores, metrics, parameters, samples)

    print("", flush=True)
    print(f"wrote {d2c.METRICS_PATH}", flush=True)
    print(f"wrote {d2c.BOOTSTRAP_PATH}", flush=True)
    print(f"wrote {d2c.SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
