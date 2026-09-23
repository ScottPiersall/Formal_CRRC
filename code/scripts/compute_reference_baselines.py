#!/usr/bin/env python
"""Post-study analytic reference baselines for TCE and pointwise accuracy.

**This is not an experiment.** It runs no model inference, reads every frozen
artifact read-only, and writes only under ``artifacts/poststudy_baselines/``. It
was added after Days 1-3 were complete and preregisters nothing.

What it does:

1. Derives four artifact-blind reference strategies in exact rational
   arithmetic from the design alone.
2. Verifies the design assumption they rest on -- an exactly uniform first-fail
   boundary -- against all four frozen partitions.
3. Confirms the step identity ``row errors = |j_hat - j*|`` exhaustively.
4. Re-evaluates the baselines empirically on the frozen labels and requires
   exact agreement with the closed forms.
5. Places every frozen judge number against those baselines.

Run: ``python scripts/compute_reference_baselines.py``
"""

from __future__ import annotations

import csv
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import baselines, predicates, provenance  # noqa: E402
from formalcrrc.config import N_STRICTNESS, NO_CROSSING_INDEX  # noqa: E402

ROOT = provenance.repo_root()
OUT_DIR = ROOT / "artifacts" / "poststudy_baselines"

#: The four frozen evaluation partitions, in study order. Read-only.
PARTITIONS: tuple[tuple[str, str, str], ...] = (
    ("day1", "artifacts/day1/formal_dataset.parquet", "Day-1 main partition (seed 42)"),
    (
        "day2_calibration",
        "artifacts/day2/calibration_dataset.parquet",
        "Day-2 CALIBRATION partition (seed 43)",
    ),
    (
        "day2_test",
        "artifacts/day2/test_dataset.parquet",
        "Day-2 TEST partition (seed 44)",
    ),
    (
        "day3_test",
        "artifacts/day3/test_dataset.parquet",
        "Day-3 TEST partition (seed 45)",
    ),
)

#: Frozen per-judge results to place against the baselines.
#: ``(label, artifacts path, tce column, accuracy column or None)``.
JUDGE_VIEWS: tuple[tuple[str, str, str, str | None], ...] = (
    ("day1_raw", "artifacts/day1/metrics_by_artifact.parquet", "tce", "accuracy"),
    (
        "day2_test_raw",
        "artifacts/day2/metrics_by_artifact.parquet",
        "tce_raw",
        "accuracy_raw",
    ),
    (
        "day2_test_global_calibrated",
        "artifacts/day2/metrics_by_artifact.parquet",
        "tce_global",
        "accuracy_global",
    ),
    (
        "day2_test_family_calibrated",
        "artifacts/day2/metrics_by_artifact.parquet",
        "tce_family",
        "accuracy_family",
    ),
    ("day3_test_raw", "artifacts/day3/metrics_by_artifact.parquet", "tce_raw", None),
    (
        "day3_oracle_global",
        "artifacts/day3/metrics_by_artifact.parquet",
        "tce_oracle_global",
        None,
    ),
    (
        "day3_oracle_family",
        "artifacts/day3/metrics_by_artifact.parquet",
        "tce_oracle_family",
        None,
    ),
    ("day3_oracle_artifact", "artifacts/day3/metrics_by_artifact.parquet", "otce", None),
)


def fraction_str(value: Fraction) -> str:
    """Canonical rendering: ``Fraction(4)`` prints as ``4``, not ``4/1``."""
    return str(value)


# --------------------------------------------------------------------------
# 1. Analytic derivation
# --------------------------------------------------------------------------


def design_boundary_counts() -> dict[int, int]:
    """The boundary distribution the design guarantees: uniform over 1..9."""
    return {j: 1 for j in baselines.BOUNDARY_SUPPORT}


def analytic_section() -> dict[str, object]:
    counts = design_boundary_counts()
    table = baselines.analytic_table(counts)

    per_crossing = {
        j: fraction_str(baselines.constant_crossing_tce(j, counts))
        for j in baselines.PREDICTION_SUPPORT
    }
    best_j, best_tce = baselines.minimum_tce_constant_crossing(counts)
    tied = [
        j
        for j in baselines.PREDICTION_SUPPORT
        if baselines.constant_crossing_tce(j, counts) == best_tce
    ]
    majority_lab, majority_share = baselines.majority_label(counts)

    return {
        "boundary_support": list(baselines.BOUNDARY_SUPPORT),
        "boundary_distribution": "uniform by construction",
        "n_thresholds": N_STRICTNESS,
        "baselines": table,
        "constant_crossing_tce_by_j_hat": per_crossing,
        "optimal_constant_crossing": {
            "j_hat": best_j,
            "tce_exact": fraction_str(best_tce),
            "tce": float(best_tce),
            "unique": len(tied) == 1,
            "tied_at": tied,
        },
        "majority_row_label": {
            "label": "met" if majority_lab == 1 else "not_met",
            "share_exact": fraction_str(majority_share),
            "share": float(majority_share),
        },
    }


# --------------------------------------------------------------------------
# 2. Design verification against the frozen partitions
# --------------------------------------------------------------------------


def partition_block(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-artifact ``(ids, j*, truth block)`` sorted by artifact id."""
    wide = (
        frame.pivot(
            index="artifact_id", columns="strictness_index", values="formal_truth"
        )
        .sort_index()
        .astype(int)
    )
    if list(wide.columns) != list(range(N_STRICTNESS)):
        raise AssertionError(f"unexpected strictness grid: {list(wide.columns)}")
    grouped = frame.groupby("artifact_id")["true_first_fail_index"]
    if int(grouped.nunique().max()) != 1:
        raise AssertionError("true_first_fail_index is not constant within an artifact")
    boundaries = grouped.first().sort_index()
    if not boundaries.index.equals(wide.index):
        raise AssertionError("artifact index mismatch between labels and boundaries")
    return (
        np.asarray(wide.index),
        boundaries.to_numpy().astype(int),
        wide.to_numpy(),
    )


def recompute_labels(frame: pd.DataFrame) -> dict[str, object]:
    """Rebuild ``formal_truth`` and ``j*`` from the frozen design fields.

    Uses only ``family``, ``latent_level`` and ``strictness_index`` through the
    preregistered predicate definitions. Read-only: nothing is written back.
    """
    recomputed_truth = np.array(
        [
            predicates.formal_truth(row.family, row.latent_level, row.strictness_index)
            for row in frame.itertuples()
        ],
        dtype=int,
    )
    recomputed_boundary = np.array(
        [
            predicates.true_first_fail_index(row.family, row.latent_level)
            for row in frame.itertuples()
        ],
        dtype=int,
    )
    truth_ok = bool(
        np.array_equal(recomputed_truth, frame["formal_truth"].to_numpy().astype(int))
    )
    boundary_ok = bool(
        np.array_equal(
            recomputed_boundary, frame["true_first_fail_index"].to_numpy().astype(int)
        )
    )
    return {
        "rows_recomputed": int(len(frame)),
        "formal_truth_matches": truth_ok,
        "true_first_fail_index_matches": boundary_ok,
        "independent_recomputation_agrees": truth_ok and boundary_ok,
    }


def verify_partition(key: str, relative: str, description: str) -> dict[str, object]:
    frame = pd.read_parquet(ROOT / relative)
    recomputation = recompute_labels(frame)
    ids, j_star, truth = partition_block(frame)

    # The formal labels must themselves be a step at j*: this is what makes the
    # step identity applicable to the ground truth side.
    expected = np.arange(N_STRICTNESS)[None, :] < j_star[:, None]
    labels_are_steps = bool(np.array_equal(truth.astype(bool), expected))

    counts = {int(j): int(c) for j, c in zip(*np.unique(j_star, return_counts=True))}
    per_cell = len(ids) // len(baselines.BOUNDARY_SUPPORT)
    uniform = counts == {j: per_cell for j in baselines.BOUNDARY_SUPPORT}

    empirical: dict[str, object] = {}
    for baseline in baselines.BASELINES:
        if baseline.j_hat is None:
            result = baselines.empirical_random_crossing(truth, j_star)
        else:
            result = baselines.empirical_baseline(baseline.j_hat, truth, j_star)
        analytic_tce = baseline.tce(counts)
        analytic_acc = baselines.accuracy_from_tce(analytic_tce)
        empirical[baseline.key] = {
            **result,
            "tce_analytic": float(analytic_tce),
            "tce_analytic_exact": fraction_str(analytic_tce),
            "accuracy_analytic": float(analytic_acc),
            "accuracy_analytic_exact": fraction_str(analytic_acc),
            "tce_matches_analytic": abs(result["mean_tce"] - float(analytic_tce)) < 1e-12,
            "accuracy_matches_analytic": abs(
                result["accuracy"] - float(analytic_acc)
            )
            < 1e-12,
        }

    met_rate = Fraction(int(truth.sum()), int(truth.size))
    return {
        "key": key,
        "path": relative,
        "description": description,
        "n_artifacts": int(len(ids)),
        "n_rows": int(truth.size),
        "boundary_counts": counts,
        "boundary_uniform": uniform,
        "expected_per_boundary_value": per_cell,
        "formal_labels_are_steps": labels_are_steps,
        "label_recomputation": recomputation,
        "met_rate_exact": fraction_str(met_rate),
        "met_rate": float(met_rate),
        "baselines": empirical,
        "all_baselines_match_analytic": all(
            v["tce_matches_analytic"] and v["accuracy_matches_analytic"]
            for v in empirical.values()
        ),
        "status": "PASS"
        if (
            uniform
            and labels_are_steps
            and recomputation["independent_recomputation_agrees"]
            and all(
                v["tce_matches_analytic"] and v["accuracy_matches_analytic"]
                for v in empirical.values()
            )
        )
        else "FAIL",
    }


# --------------------------------------------------------------------------
# 3. The step identity
# --------------------------------------------------------------------------


def identity_section() -> dict[str, object]:
    """Exhaustive check of ``row errors = |j_hat - j*|`` over the full grid."""
    failures = []
    for j_hat in baselines.PREDICTION_SUPPORT:
        for j_star in baselines.PREDICTION_SUPPORT:
            observed = baselines.row_errors(j_hat, j_star)
            expected = baselines.crossing_error(j_hat, j_star)
            if observed != expected:
                failures.append({"j_hat": j_hat, "j_star": j_star})
    n_pairs = len(baselines.PREDICTION_SUPPORT) ** 2
    return {
        "statement": (
            "For a step predictor with crossing j_hat against step labels with "
            "boundary j*, the number of disagreeing thresholds equals "
            "|j_hat - j*|; hence accuracy = 1 - TCE / 9."
        ),
        "scope": "step predictors only; judges need not be step predictors",
        "pairs_checked": n_pairs,
        "failures": failures,
        "holds": not failures,
    }


# --------------------------------------------------------------------------
# 4. Placing the frozen judge results
# --------------------------------------------------------------------------


def judge_section(analytic: dict[str, object]) -> dict[str, object]:
    by_key = {row["key"]: row for row in analytic["baselines"]}
    fixed_center_tce = by_key["fixed_center"]["tce"]
    random_tce = by_key["uniform_random_crossing"]["tce"]

    rows: list[dict[str, object]] = []
    for label, relative, tce_col, acc_col in JUDGE_VIEWS:
        frame = pd.read_parquet(ROOT / relative)
        for model_id, group in frame.groupby("model_id", sort=True):
            tce = float(group[tce_col].mean())
            entry: dict[str, object] = {
                "view": label,
                "model_id": model_id,
                "tce": tce,
                "tce_minus_fixed_center": tce - fixed_center_tce,
                "beats_fixed_center": tce < fixed_center_tce,
                "beats_uniform_random": tce < random_tce,
            }
            if acc_col is not None:
                accuracy = float(group[acc_col].mean())
                entry["accuracy"] = accuracy
                entry["accuracy_minus_majority_class"] = (
                    accuracy - by_key["always_met"]["accuracy"]
                )
                entry["beats_majority_class"] = (
                    accuracy > by_key["always_met"]["accuracy"]
                )
                # A judge that were a step predictor would satisfy this exactly.
                implied = 1.0 - tce / N_STRICTNESS
                entry["accuracy_implied_by_step_identity"] = implied
                entry["step_identity_gap"] = accuracy - implied
                entry["is_step_predictor"] = abs(accuracy - implied) < 1e-9
            rows.append(entry)
    return {
        "reference_tce_fixed_center": fixed_center_tce,
        "reference_tce_uniform_random": random_tce,
        "reference_accuracy_majority_class": by_key["always_met"]["accuracy"],
        "rows": rows,
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def write_csv(path: Path, partitions: list[dict[str, object]]) -> None:
    fields = [
        "partition",
        "n_artifacts",
        "n_rows",
        "baseline_key",
        "j_hat",
        "tce_analytic_exact",
        "tce_analytic",
        "tce_empirical",
        "accuracy_analytic_exact",
        "accuracy_analytic",
        "accuracy_empirical",
        "exact_agreement",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for part in partitions:
            for key, result in part["baselines"].items():
                writer.writerow(
                    {
                        "partition": part["key"],
                        "n_artifacts": part["n_artifacts"],
                        "n_rows": part["n_rows"],
                        "baseline_key": key,
                        "j_hat": "" if result["j_hat"] is None else result["j_hat"],
                        "tce_analytic_exact": result["tce_analytic_exact"],
                        "tce_analytic": f"{result['tce_analytic']:.6f}",
                        "tce_empirical": f"{result['mean_tce']:.6f}",
                        "accuracy_analytic_exact": result["accuracy_analytic_exact"],
                        "accuracy_analytic": f"{result['accuracy_analytic']:.6f}",
                        "accuracy_empirical": f"{result['accuracy']:.6f}",
                        "exact_agreement": result["tce_matches_analytic"]
                        and result["accuracy_matches_analytic"],
                    }
                )


#: Frozen scientific record: the files that may never change. Engineering
#: documentation (README, DEVLOG) is deliberately excluded -- it is not part of
#: the scientific record and is not pinned by any integrity manifest.
FROZEN_TREES = (
    "artifacts/day1",
    "artifacts/day2",
    "artifacts/day3",
    "figures/day1",
    "figures/day2",
    "figures/day3",
)
FROZEN_DOCS = (
    "docs/PREREGISTRATION_DAY1.md",
    "docs/PREREGISTRATION_DAY2.md",
    "docs/PREREGISTRATION_DAY3.md",
    "docs/RESULTS_DAY1.md",
    "docs/RESULTS_DAY2.md",
    "docs/RESULTS_DAY3.md",
    "docs/DAY1_ERRATA.md",
    "docs/DAY1_PROVENANCE_AUDIT.md",
    "docs/DAY1_INTEGRITY_ADDENDUM.md",
)


def frozen_record_fingerprint() -> dict[str, object]:
    """Hash the frozen scientific record, and re-verify the recorded manifests.

    Read-only. The manifests are consulted, never rewritten.
    """
    files: dict[str, str] = {}
    for tree in FROZEN_TREES:
        for path in sorted((ROOT / tree).rglob("*")):
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = provenance.sha256_file(path)
    for relative in FROZEN_DOCS:
        if (ROOT / relative).is_file():
            files[relative] = provenance.sha256_file(ROOT / relative)

    prior = json.loads(
        (ROOT / "artifacts/day3/prior_experiments_manifest.json").read_text("utf-8")
    )
    mismatched = [
        relative
        for relative, digest in prior["files"].items()
        if provenance.sha256_file(ROOT / relative) != digest
    ]
    return {
        "n_files": len(files),
        "combined_sha256": provenance.sha256_text(
            "\n".join(f"{k}:{v}" for k, v in sorted(files.items()))
        ),
        "files": files,
        "recorded_prior_manifest_verifies": not mismatched,
        "mismatched_against_recorded_manifest": mismatched,
    }


def final_console_block(
    analytic: dict[str, object], partitions: list[dict[str, object]], immutable: bool
) -> str:
    by_key = {row["key"]: row for row in analytic["baselines"]}
    labels = {
        "day1": "Day 1:",
        "day2_calibration": "Day-2 CAL:",
        "day2_test": "Day-2 TEST:",
        "day3_test": "Day-3 TEST:",
    }
    lines = [
        "FORMALCRRC POST-STUDY BASELINE AUDIT",
        "",
        "PRIOR EXPERIMENT IMMUTABILITY: " + ("PASS" if immutable else "FAIL"),
        "",
        "ANALYTIC BASELINES:",
        "  fixed-center TCE:              %.6f" % by_key["fixed_center"]["tce"],
        "  fixed-center accuracy:         %.4f%%"
        % (100 * by_key["fixed_center"]["accuracy"]),
        "  random-crossing expected TCE:  %.6f"
        % by_key["uniform_random_crossing"]["tce"],
        "  random-crossing exp. accuracy: %.4f%%"
        % (100 * by_key["uniform_random_crossing"]["accuracy"]),
        "  always-met TCE:                %.6f" % by_key["always_met"]["tce"],
        "  majority-class accuracy:       %.4f%%"
        % (100 * by_key["always_met"]["accuracy"]),
        "  always-not-met TCE:            %.6f" % by_key["always_not_met"]["tce"],
        "",
        "DATASET CROSS-CHECK:",
    ]
    for part in partitions:
        lines.append("  %-30s%s" % (labels[part["key"]], part["status"]))
    lines += [
        "",
        "No model inference performed.",
        "No frozen scientific outcome modified.",
        "Baselines labeled post-study analytic references.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    analytic = analytic_section()
    identity = identity_section()
    partitions = [verify_partition(*spec) for spec in PARTITIONS]
    judges = judge_section(analytic)

    checks = {
        "step_identity_holds": identity["holds"],
        "all_partitions_uniform": all(p["boundary_uniform"] for p in partitions),
        "all_labels_are_steps": all(p["formal_labels_are_steps"] for p in partitions),
        "empirical_matches_analytic": all(
            p["all_baselines_match_analytic"] for p in partitions
        ),
        "labels_independently_recomputed": all(
            p["label_recomputation"]["independent_recomputation_agrees"]
            for p in partitions
        ),
        "optimal_constant_crossing_is_unique": analytic["optimal_constant_crossing"][
            "unique"
        ],
    }
    frozen = frozen_record_fingerprint()
    checks["frozen_record_manifest_verifies"] = frozen[
        "recorded_prior_manifest_verifies"
    ]
    ok = all(checks.values())

    summary = {
        "title": "FormalCRRC post-study analytic reference baselines",
        "disclaimer": (
            "These reference baselines were derived and added after completion of "
            "the three preregistered FormalCRRC studies. They do not modify any "
            "frozen experimental outcome and require no additional model inference."
        ),
        "preregistered": False,
        "model_inference_performed": False,
        "randomness_used": False,
        "bootstrap_used": False,
        "generated_utc": provenance.utc_now(),
        "checks": checks,
        "status": "PASS" if ok else "FAIL",
        "analytic": analytic,
        "step_identity": identity,
        "partitions": partitions,
        "judge_comparison": judges,
        "frozen_scientific_record": {
            k: v for k, v in frozen.items() if k != "files"
        },
        "git": provenance.git_state(ROOT),
        "environment": provenance.environment_snapshot(include_gpu=False),
    }

    summary_path = OUT_DIR / "baseline_summary.json"
    csv_path = OUT_DIR / "baseline_by_partition.csv"
    block_path = OUT_DIR / "final_console_block.txt"
    block = final_console_block(
        analytic, partitions, bool(checks["frozen_record_manifest_verifies"])
    )
    provenance.write_json(summary_path, summary)
    write_csv(csv_path, partitions)
    block_path.write_text(block, encoding="utf-8")

    manifest = {
        "note": (
            "Covers only the additive post-study baseline outputs. Prior integrity "
            "manifests are untouched and are not restated here."
        ),
        "generated_utc": provenance.utc_now(),
        "files": provenance.hash_paths(
            {
                "baseline_summary": "artifacts/poststudy_baselines/baseline_summary.json",
                "baseline_by_partition": "artifacts/poststudy_baselines/baseline_by_partition.csv",
                "final_console_block": "artifacts/poststudy_baselines/final_console_block.txt",
            },
            ROOT,
        ),
        "source_manifest": provenance.source_manifest(
            ["src/formalcrrc/baselines.py", "scripts/compute_reference_baselines.py"],
            ROOT,
        ),
        "inputs_read_only": [relative for _, relative, _ in PARTITIONS]
        + sorted({relative for _, relative, _, _ in JUDGE_VIEWS}),
        "git": provenance.git_state(ROOT),
    }
    provenance.write_json(OUT_DIR / "integrity_manifest.json", manifest)

    # ----------------------------------------------------------------- report
    print("=" * 74)
    print("FormalCRRC post-study analytic reference baselines")
    print("=" * 74)
    print("Post-study, not preregistered. No model inference. No frozen output altered.")
    print()
    print(f"{'baseline':<26}{'j_hat':>7}{'TCE (exact)':>14}{'TCE':>10}{'accuracy':>11}")
    print("-" * 74)
    for row in analytic["baselines"]:
        j_hat = "random" if row["j_hat"] is None else str(row["j_hat"])
        print(
            "%-26s%7s%14s%10.6f%10.4f%%"
            % (row["name"], j_hat, row["tce_exact"], row["tce"], 100 * row["accuracy"])
        )
    print()
    best = analytic["optimal_constant_crossing"]
    print(
        f"Optimal constant crossing: j_hat = {best['j_hat']} "
        f"(TCE {best['tce_exact']}), unique = {best['unique']}"
    )
    print()

    print("Partition verification")
    print("-" * 74)
    for part in partitions:
        print(
            "%-18s artifacts=%d rows=%d uniform=%s relabel=%s met_rate=%s exact=%s  %s"
            % (
                part["key"],
                part["n_artifacts"],
                part["n_rows"],
                part["boundary_uniform"],
                part["label_recomputation"]["independent_recomputation_agrees"],
                part["met_rate_exact"],
                part["all_baselines_match_analytic"],
                part["status"],
            )
        )
    print()

    print("Frozen judge results against the fixed-center baseline (TCE 20/9)")
    print("-" * 74)
    for row in judges["rows"]:
        flag = "BELOW" if row["beats_fixed_center"] else "ABOVE"
        print(
            "%-28s %-26s TCE=%.4f  %s baseline by %+.4f"
            % (
                row["view"],
                row["model_id"],
                row["tce"],
                flag,
                row["tce_minus_fixed_center"],
            )
        )
    print()
    print(f"Frozen scientific record: {frozen['n_files']} files, "
          f"combined {frozen['combined_sha256'][:8]}...  "
          f"recorded manifest verifies = {frozen['recorded_prior_manifest_verifies']}")
    print()
    print(block)
    print(f"Checks: {json.dumps(checks)}")
    print(f"STATUS: {summary['status']}")
    print(f"Wrote {block_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {(OUT_DIR / 'integrity_manifest.json').relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
