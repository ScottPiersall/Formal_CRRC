#!/usr/bin/env python
"""Run the preregistered Day-1 analysis exactly once (Phase G).

Consumes only frozen inputs -- the dataset, the preregistration, and whatever
raw score files exist -- and produces:

* ``artifacts/day1/metrics_by_artifact.parquet``
* ``artifacts/day1/bootstrap_results.parquet``
* ``artifacts/day1/summary.json``
* ``figures/day1/figure1..figure5``

Nothing is calibrated, smoothed, monotonised or filtered. Judges whose raw
scores are absent or incomplete are reported as BLOCKED rather than analysed as
if complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import figures_day1  # noqa: E402  (sibling module in scripts/)
from formalcrrc import bootstrap, metrics, provenance  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    BOOTSTRAP_PATH,
    DATASET_PATH,
    DECISION_THRESHOLD,
    FAMILIES,
    METRICS_PATH,
    MODEL_IDS,
    MODEL_SHORT_NAMES,
    MODEL_SLUGS,
    MVR_TOLERANCE,
    N_ARTIFACTS,
    N_ROWS,
    N_STRICTNESS,
    PREREG_JSON_PATH,
    PREREG_SHA_PATH,
    RAW_SCORES_DIR,
    SUMMARY_PATH,
)

STATUS_COMPLETE = "COMPLETE"
STATUS_BLOCKED = "BLOCKED"


def load_raw_scores(root: Path, dataset: pd.DataFrame) -> tuple[dict, dict]:
    """Load and validate each judge's raw scores; report per-judge status."""
    truth = dataset[
        [
            "prompt_id",
            "artifact_id",
            "family",
            "latent_level",
            "strictness_index",
            "formal_truth",
            "true_first_fail_index",
        ]
    ]
    scores: dict[str, pd.DataFrame] = {}
    status: dict[str, dict] = {}

    for model_id in MODEL_IDS:
        slug = MODEL_SLUGS[model_id]
        path = root / RAW_SCORES_DIR / f"{slug}.parquet"
        record: dict = {"model_id": model_id, "slug": slug, "path": str(path.name)}
        if not path.is_file():
            record.update(
                status=STATUS_BLOCKED,
                reason="raw scores absent",
                rows=0,
            )
            status[model_id] = record
            continue

        frame = pd.read_parquet(path)
        problems: list[str] = []
        if len(frame) != N_ROWS:
            problems.append(f"expected {N_ROWS} rows, found {len(frame)}")
        if not frame["prompt_id"].is_unique:
            problems.append("duplicate prompt_id values")
        if set(frame["prompt_id"]) != set(truth["prompt_id"]):
            problems.append("prompt_id set does not match the frozen dataset")
        if not np.all(np.isfinite(frame["p_met"].to_numpy(dtype=float))):
            problems.append("non-finite p_met values")
        outside = frame[(frame["p_met"] < 0.0) | (frame["p_met"] > 1.0)]
        if len(outside):
            problems.append(f"{len(outside)} probabilities outside [0, 1]")

        record.update(
            rows=int(len(frame)),
            revisions=sorted(set(frame["revision"])),
            scoring_methods=sorted(set(frame["scoring_method"])),
            raw_scores_sha256=provenance.sha256_file(path),
        )
        if problems:
            record.update(status=STATUS_BLOCKED, reason="; ".join(problems))
            status[model_id] = record
            continue

        merged = frame.merge(truth, on="prompt_id", how="inner", validate="1:1")
        merged = merged.drop(
            columns=[c for c in merged.columns if c.endswith("_y")], errors="ignore"
        )
        merged.columns = [c[:-2] if c.endswith("_x") else c for c in merged.columns]
        record.update(status=STATUS_COMPLETE, reason=None)
        status[model_id] = record
        scores[model_id] = merged

    return scores, status


def summarise_tce(values: np.ndarray) -> dict:
    """Distributional summary of the threshold crossing error."""
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "exact_boundary_rate": float(np.mean(values == 0)),
        "within_one_step_rate": float(np.mean(values <= 1)),
        "max": int(values.max()),
        "distribution": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(values.astype(int), return_counts=True))
        },
    }


def build_summary(
    scores: dict[str, pd.DataFrame],
    artifact_metrics: dict[str, pd.DataFrame],
    plan: bootstrap.BootstrapPlan,
    bootstrap_table: pd.DataFrame,
    comparisons: pd.DataFrame,
    status: dict[str, dict],
    root: Path,
) -> dict:
    """Assemble every reported number into one machine-readable document."""
    per_model: dict[str, dict] = {}
    for model_id, frame in artifact_metrics.items():
        raw = scores[model_id]
        tce = frame["tce"].to_numpy(dtype=float)
        spearman = frame["spearman"].to_numpy(dtype=float)
        finite = spearman[np.isfinite(spearman)]

        family_block: dict[str, dict] = {}
        for family in FAMILIES:
            subset = frame[frame["family"] == family]
            family_block[family] = {
                "n_artifacts": int(len(subset)),
                "mvr": float(subset["mvr"].mean()),
                "mvr_strict": float(subset["mvr_strict"].mean()),
                "mvm": float(subset["mvm"].mean()),
                "tce": summarise_tce(subset["tce"].to_numpy(dtype=float)),
                "accuracy": float(subset["accuracy"].mean()),
                "brier": float(subset["brier"].mean()),
                "spearman_mean": float(
                    np.nanmean(subset["spearman"].to_numpy(dtype=float))
                ),
                "predicted_never_crosses": int(
                    (subset["predicted_first_fail_index"] == N_STRICTNESS).sum()
                ),
            }

        per_model[model_id] = {
            "short_name": MODEL_SHORT_NAMES[model_id],
            "status": STATUS_COMPLETE,
            "revision": sorted(set(raw["revision"]))[0],
            "scoring_method": sorted(set(raw["scoring_method"]))[0],
            "n_artifacts": int(len(frame)),
            "n_cells": int(len(raw)),
            "overall": {
                "mvr": float(frame["mvr"].mean()),
                "mvr_strict": float(frame["mvr_strict"].mean()),
                "mvm": float(frame["mvm"].mean()),
                "tce": summarise_tce(tce),
                "accuracy": float(frame["accuracy"].mean()),
                "brier": float(frame["brier"].mean()),
                "spearman_mean": float(finite.mean()) if finite.size else float("nan"),
                "spearman_undefined_artifacts": int(spearman.size - finite.size),
                "artifacts_with_any_violation": int((frame["mvr"] > 0).sum()),
                "artifacts_perfectly_monotone": int((frame["mvr_strict"] == 0).sum()),
                "predicted_never_crosses": int(
                    (frame["predicted_first_fail_index"] == N_STRICTNESS).sum()
                ),
                "predicted_always_below": int(
                    (frame["predicted_first_fail_index"] == 0).sum()
                ),
                "mean_p_met": float(raw["p_met"].mean()),
                "p_met_p05": float(raw["p_met"].quantile(0.05)),
                "p_met_p95": float(raw["p_met"].quantile(0.95)),
            },
            "by_family": family_block,
        }

    boundary = (
        pd.concat(
            [metrics.boundary_aligned_curve(scores[m]) for m in scores], ignore_index=True
        )
        if scores
        else pd.DataFrame()
    )

    return {
        "created_at": provenance.utc_now(),
        "analysis_run": "single preregistered run",
        "decision_threshold": DECISION_THRESHOLD,
        "mvr_tolerance": MVR_TOLERANCE,
        "design": {
            "n_artifacts": N_ARTIFACTS,
            "n_strictness_levels": N_STRICTNESS,
            "n_rubric_response_pairs": N_ROWS,
            "families": list(FAMILIES),
        },
        "panel": {
            model_id: {
                "status": record["status"],
                "reason": record.get("reason"),
                "rows": record.get("rows"),
            }
            for model_id, record in status.items()
        },
        "panel_completeness": (
            "COMPLETE"
            if all(r["status"] == STATUS_COMPLETE for r in status.values())
            else (
                "PARTIAL"
                if any(r["status"] == STATUS_COMPLETE for r in status.values())
                else "BLOCKED"
            )
        ),
        "models": per_model,
        "bootstrap": {
            "replicates": plan.replicates,
            "seed": plan.seed,
            "unit": "artifact_id",
            "stratified_by": "family",
            "n_units": len(plan.artifact_ids),
            "intervals": json.loads(bootstrap_table.to_json(orient="records")),
        },
        "paired_comparisons": json.loads(comparisons.to_json(orient="records")),
        "boundary_aligned_curve": json.loads(boundary.to_json(orient="records")),
        "preregistration_sha256": (
            (root / PREREG_SHA_PATH).read_text(encoding="utf-8").split()[0]
            if (root / PREREG_SHA_PATH).is_file()
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)

    prereg = root / PREREG_JSON_PATH
    sha_path = root / PREREG_SHA_PATH
    if not (prereg.is_file() and sha_path.is_file()):
        print("ABORT: preregistration is not frozen", flush=True)
        return 2
    if not provenance.verify_checksum_file(prereg, sha_path):
        print("ABORT: preregistration checksum does not match", flush=True)
        return 2

    dataset = pd.read_parquet(root / DATASET_PATH)
    scores, status = load_raw_scores(root, dataset)

    for model_id, record in status.items():
        detail = f" ({record['reason']})" if record.get("reason") else ""
        print(f"{MODEL_SHORT_NAMES[model_id]:<14} {record['status']}{detail}",
              flush=True)

    if not scores:
        print("ABORT: no judge produced a complete run", flush=True)
        return 2

    artifact_metrics = {
        model_id: metrics.compute_artifact_metrics(frame)
        for model_id, frame in scores.items()
    }

    reference = next(iter(artifact_metrics.values())).sort_values("artifact_id")
    plan = bootstrap.build_plan(
        reference["artifact_id"].tolist(), reference["family"].tolist()
    )

    table = bootstrap.bootstrap_table(artifact_metrics, plan)
    comparisons = (
        bootstrap.paired_comparison_table(artifact_metrics, plan)
        if len(artifact_metrics) > 1
        else pd.DataFrame(
            columns=["model_a", "model_b", "scope", "metric", "point", "ci_low",
                     "ci_high", "excludes_zero", "n_units"]
        )
    )

    combined = pd.concat(artifact_metrics.values(), ignore_index=True)
    combined.to_parquet(root / METRICS_PATH, index=False)
    table.to_parquet(root / BOOTSTRAP_PATH, index=False)

    summary = build_summary(
        scores, artifact_metrics, plan, table, comparisons, status, root
    )
    provenance.write_json(root / SUMMARY_PATH, summary)

    if not args.skip_figures:
        figures_day1.render_all(root, scores, artifact_metrics, table)

    print("", flush=True)
    print(f"wrote {METRICS_PATH}", flush=True)
    print(f"wrote {BOOTSTRAP_PATH}", flush=True)
    print(f"wrote {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
