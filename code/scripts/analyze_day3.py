#!/usr/bin/env python
"""Run the preregistered Day-3 analysis exactly once (Phase H, second stage).

Consumes the frozen dataset, the raw scores and the oracle outputs, computes the
preregistered metrics, runs the artifact-clustered bootstrap that refits the
group oracles inside every replicate, and renders the six predeclared figures.

Nothing is refitted beyond the preregistered oracle classes. No stronger
transformation is tried, no curve is smoothed, no artifact is filtered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import figures_day3  # noqa: E402  (sibling module in scripts/)
from formalcrrc import day3, day3_bootstrap, day3_config as d3c, provenance  # noqa: E402
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_day3_oracle import load_scores  # noqa: E402


def distribution_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    keys, counts = np.unique(values.astype(int), return_counts=True)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "exact_rate": float(np.mean(values == 0)),
        "within_one_rate": float(np.mean(values <= 1)),
        "above_one_rate": float(np.mean(values > 1)),
        "max": int(values.max()),
        "distribution": {str(int(k)): int(v) for k, v in zip(keys, counts)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    root = Path(args.root)

    prereg = root / d3c.PREREG_JSON_PATH
    sha_path = root / d3c.PREREG_SHA_PATH
    if not provenance.verify_checksum_file(prereg, sha_path):
        print("ABORT: Day-3 preregistration checksum does not match", flush=True)
        return 2
    for required in (d3c.ORACLE_GLOBAL_PATH, d3c.ORACLE_FAMILY_PATH):
        if not (root / required).is_file():
            print(f"ABORT: {required} missing; run compute_day3_oracle.py", flush=True)
            return 2

    oracle_global = json.loads(
        (root / d3c.ORACLE_GLOBAL_PATH).read_text(encoding="utf-8")
    )["models"]
    oracle_family = json.loads(
        (root / d3c.ORACLE_FAMILY_PATH).read_text(encoding="utf-8")
    )["models"]

    scores, status = load_scores(root)
    for model_id, record in status.items():
        detail = f" ({record['reason']})" if record.get("reason") else ""
        print(f"{MODEL_SHORT_NAMES[model_id]:<14} {record['status']}{detail}", flush=True)
    if not scores:
        print("ABORT: no judge produced a complete run", flush=True)
        return 2

    # ---- per-artifact metrics ---------------------------------------------
    metrics: dict[str, pd.DataFrame] = {}
    stacks: dict[str, tuple] = {}
    associations: list[dict] = []
    for model_id, frame in scores.items():
        alphas_family = {
            family: oracle_family[model_id]["by_family"][family]["alpha"]
            for family in FAMILIES
        }
        metrics[model_id] = day3.compute_artifact_metrics(
            frame, oracle_global[model_id]["alpha"], alphas_family
        )
        stacks[model_id] = day3.stack_margins(frame)
        associations.extend(day3.shape_metric_associations(metrics[model_id]))

    combined = pd.concat(metrics.values(), ignore_index=True)
    combined.to_parquet(root / d3c.METRICS_PATH, index=False)

    # The guarantee OTCE <= TCE_raw must hold on every artifact.
    violations = int((combined["otce"] > combined["tce_raw"]).sum())
    if violations:
        print(f"ABORT: OTCE exceeded raw TCE on {violations} artifacts", flush=True)
        return 2
    print("", flush=True)
    print(f"OTCE <= TCE_raw on all {len(combined)} artifact-judge rows: PASS", flush=True)

    # ---- bootstrap ---------------------------------------------------------
    reference = next(iter(metrics.values())).sort_values("artifact_id")
    plan = day3_bootstrap.build_plan(
        reference["artifact_id"].tolist(),
        reference["family"].tolist(),
        replicates=args.replicates,
    )
    print(
        f"bootstrap: {plan.replicates} replicates, seed {plan.seed}, "
        "refitting the group oracles inside each replicate ...",
        flush=True,
    )
    counts = day3_bootstrap.multiplicity_matrix(plan)
    counts_family = {
        family: day3_bootstrap.multiplicity_matrix(plan, family)
        for family in FAMILIES
    }

    samples: dict[str, dict[str, np.ndarray]] = {}
    for model_id in metrics:
        ids, block, boundaries, _ = stacks[model_id]
        order = {a: i for i, a in enumerate(ids)}
        rows = [order[a] for a in plan.artifact_ids]
        samples[model_id] = day3_bootstrap.bootstrap_judge(
            metrics[model_id],
            block[rows],
            boundaries[rows],
            plan,
            counts,
            counts_family,
        )
        print(f"  {MODEL_SHORT_NAMES[model_id]:<14} done", flush=True)

    bootstrap_rows: list[dict] = []
    for model_id, frame in metrics.items():
        nontrivial = frame[frame["nontrivial_boundary"]]
        points = {
            "tce_raw": float(frame["tce_raw"].mean()),
            "otce": float(frame["otce"].mean()),
            "delta_tce_artifact": float(frame["delta_tce_artifact"].mean()),
            "tce_oracle_global": float(frame["tce_oracle_global"].mean()),
            "tce_oracle_family": float(frame["tce_oracle_family"].mean()),
            "reachable_count": float(frame["reachable_count"].mean()),
            "prefix_record_rate": float(frame["prefix_record_rate"].mean()),
            "mmvr": float(frame["mmvr"].mean()),
            "exact_rate_raw": float((frame["tce_raw"] == 0).mean()),
            "within_one_raw": float((frame["tce_raw"] <= 1).mean()),
            "exact_rate_otce": float((frame["otce"] == 0).mean()),
            "within_one_otce": float((frame["otce"] <= 1).mean()),
            "slr1": float((frame["otce"] > 0).mean()),
            "slr2": float((frame["otce"] > 1).mean()),
            "trr_overall": float(frame["translation_reachable"].mean()),
            "trr_nontrivial": float(nontrivial["translation_reachable"].mean()),
            "alpha_oracle_global": oracle_global[model_id]["alpha"],
        }
        for family in FAMILIES:
            points[f"alpha_oracle_family_{family}"] = oracle_family[model_id][
                "by_family"
            ][family]["alpha"]
            points[f"tce_oracle_family_{family}"] = oracle_family[model_id][
                "by_family"
            ][family]["mean_tce"]
        for quantity, point in points.items():
            if quantity not in samples[model_id]:
                continue
            bootstrap_rows.append(
                {
                    "model_id": model_id,
                    "quantity": quantity,
                    **day3_bootstrap.summarise(samples[model_id][quantity], point),
                }
            )

    comparison_rows: list[dict] = []
    model_list = list(metrics)
    for i, model_a in enumerate(model_list):
        for model_b in model_list[i + 1 :]:
            for quantity in ("otce", "tce_raw", "trr_nontrivial", "reachable_count"):
                point = float(
                    np.nanmean(samples[model_a][quantity])
                    - np.nanmean(samples[model_b][quantity])
                )
                comparison_rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "quantity": quantity,
                        **day3_bootstrap.paired_difference(
                            samples[model_a][quantity],
                            samples[model_b][quantity],
                            point,
                        ),
                    }
                )

    pd.DataFrame(bootstrap_rows).to_parquet(root / d3c.BOOTSTRAP_PATH, index=False)

    # ---- summary -----------------------------------------------------------
    per_model: dict[str, dict] = {}
    for model_id, frame in metrics.items():
        nontrivial = frame[frame["nontrivial_boundary"]]
        by_family: dict[str, dict] = {}
        for family in FAMILIES:
            subset = frame[frame["family"] == family]
            sub_nontrivial = subset[subset["nontrivial_boundary"]]
            by_family[family] = {
                "n_artifacts": int(len(subset)),
                "tce_raw": distribution_summary(subset["tce_raw"].to_numpy()),
                "otce": distribution_summary(subset["otce"].to_numpy()),
                "tce_oracle_global": float(subset["tce_oracle_global"].mean()),
                "tce_oracle_family": float(subset["tce_oracle_family"].mean()),
                "alpha_oracle_family": oracle_family[model_id]["by_family"][family][
                    "alpha"
                ],
                "trr_overall": float(subset["translation_reachable"].mean()),
                "trr_nontrivial": float(
                    sub_nontrivial["translation_reachable"].mean()
                ),
                "reachable_count": float(subset["reachable_count"].mean()),
                "prefix_record_rate": float(subset["prefix_record_rate"].mean()),
                "mmvr": float(subset["mmvr"].mean()),
                "mmvm": float(subset["mmvm"].mean()),
                "span": float(subset["span"].mean()),
                "slr1": float((subset["otce"] > 0).mean()),
                "slr2": float((subset["otce"] > 1).mean()),
            }

        reachability_by_boundary = {
            str(j): float(
                frame[frame["true_first_fail_index"] == j][
                    "translation_reachable"
                ].mean()
            )
            for j in range(1, N_STRICTNESS + 1)
            if (frame["true_first_fail_index"] == j).any()
        }

        per_model[model_id] = {
            "short_name": MODEL_SHORT_NAMES[model_id],
            "status": "COMPLETE",
            "revision": status[model_id]["revisions"][0],
            "n_artifacts": int(len(frame)),
            "overall": {
                "tce_raw": distribution_summary(frame["tce_raw"].to_numpy()),
                "otce": distribution_summary(frame["otce"].to_numpy()),
                "delta_tce_artifact": float(frame["delta_tce_artifact"].mean()),
                "trr_overall": float(frame["translation_reachable"].mean()),
                "trr_nontrivial": float(nontrivial["translation_reachable"].mean()),
                "n_nontrivial": int(len(nontrivial)),
                "slr1": float((frame["otce"] > 0).mean()),
                "slr2": float((frame["otce"] > 1).mean()),
                "reachable_count": float(frame["reachable_count"].mean()),
                "reachable_count_median": float(frame["reachable_count"].median()),
                "reachable_count_distribution": {
                    str(int(k)): int(v)
                    for k, v in zip(
                        *np.unique(
                            frame["reachable_count"].to_numpy(), return_counts=True
                        )
                    )
                },
                "prefix_record_rate": float(frame["prefix_record_rate"].mean()),
                "mmvr": float(frame["mmvr"].mean()),
                "mmvm": float(frame["mmvm"].mean()),
                "span": float(frame["span"].mean()),
                "margin_spearman_mean": float(
                    np.nanmean(frame["margin_spearman"].to_numpy(dtype=float))
                ),
                "adjacent_equal_margins": int(frame["adjacent_equal_margins"].sum()),
                "prefix_tie_levels": int(frame["prefix_tie_levels"].sum()),
                "target_blocked_by_equality": int(
                    frame["target_blocked_by_equality"].sum()
                ),
            },
            "oracle_hierarchy": {
                "raw": float(frame["tce_raw"].mean()),
                "oracle_global": float(frame["tce_oracle_global"].mean()),
                "oracle_family": float(frame["tce_oracle_family"].mean()),
                "oracle_artifact": float(frame["otce"].mean()),
                "alpha_global": oracle_global[model_id]["alpha"],
                "alpha_family": {
                    family: oracle_family[model_id]["by_family"][family]["alpha"]
                    for family in FAMILIES
                },
                "nesting": day3.assert_oracle_nesting(
                    float(frame["tce_raw"].mean()),
                    float(frame["tce_oracle_global"].mean()),
                    float(frame["tce_oracle_family"].mean()),
                    float(frame["otce"].mean()),
                ),
            },
            "taxonomy": {
                name: int(frame[name].sum()) for name in day3.TAXONOMY_COLUMNS
            },
            "reachability_by_true_boundary": reachability_by_boundary,
            "by_family": by_family,
        }

    summary = {
        "created_at": provenance.utc_now(),
        "experiment": "FormalCRRC Day 3",
        "analysis_run": "single preregistered run",
        "design": {
            "n_artifacts": N_ARTIFACTS,
            "n_rows": N_ROWS,
            "n_strictness_levels": N_STRICTNESS,
            "families": list(FAMILIES),
            "seed": d3c.SEED,
            "partition": d3c.PARTITION_NAME,
        },
        "panel": {
            model_id: {"status": record["status"], "reason": record.get("reason")}
            for model_id, record in status.items()
        },
        "panel_completeness": (
            "COMPLETE"
            if len(metrics) == len(MODEL_IDS)
            else ("PARTIAL" if metrics else "BLOCKED")
        ),
        "oracle_guarantee_holds": True,
        "models": per_model,
        "shape_metric_associations": associations,
        "bootstrap": {
            "replicates": plan.replicates,
            "seed": BOOTSTRAP_SEED,
            "unit": "artifact_id",
            "stratified_by": "family",
            "oracle_refit_inside_replicate": True,
            "results": bootstrap_rows,
        },
        "paired_comparisons": comparison_rows,
        "preregistration_sha256": sha_path.read_text(encoding="utf-8").split()[0],
    }
    provenance.write_json(root / d3c.SUMMARY_PATH, summary)

    if not args.skip_figures:
        figures_day3.render_all(root, metrics, stacks, oracle_global, samples)

    print("", flush=True)
    print(f"wrote {d3c.METRICS_PATH}", flush=True)
    print(f"wrote {d3c.BOOTSTRAP_PATH}", flush=True)
    print(f"wrote {d3c.SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
