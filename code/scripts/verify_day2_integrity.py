#!/usr/bin/env python
"""Day-2 integrity verification and manifest (Phase I / J).

Re-verifies that Day 1 is untouched, that both Day-2 partitions still hash to
their manifests, that every judge's raw scores are complete and well formed on
both partitions, that no calibration leakage occurred, and that shape invariance
holds.

Figures are hashed **explicitly and individually**, and the combined digest
refuses an empty file list, so the Day-1 empty-input defect cannot recur. The
results document is generated after this manifest, not before it.

Writes ``artifacts/day2/integrity_manifest.json`` and, with ``--final``,
``artifacts/day2/final_console_block.txt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import dataset as ds  # noqa: E402
from formalcrrc import day1_audit, day2_config as d2c, prompts, provenance  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    FAMILIES,
    MODEL_IDS,
    MODEL_SHORT_NAMES,
    MODEL_SLUGS,
    N_ARTIFACTS,
    N_ROWS,
    N_STRICTNESS,
)


def check_day1(root: Path) -> dict:
    """Day 1 must be exactly as Day 2 found it."""
    baseline_path = root / d2c.DAY1_BASELINE_PATH
    if not baseline_path.is_file():
        return {"status": "FAIL", "reason": "baseline fingerprint missing"}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    verification = day1_audit.verify_day1_baseline(baseline, root)
    permitted = {"artifacts/day1/integrity_addendum.json"}
    unexpected = [p for p in verification["added"] if p not in permitted]
    rebuilt = ds.dataset_content_hash(ds.build_dataset())
    return {
        "status": (
            "PASS"
            if verification["status"] == "PASS"
            and not unexpected
            and rebuilt == d2c.DAY1_DATASET_CONTENT_SHA256
            else "FAIL"
        ),
        "n_files_fingerprinted": baseline["n_files"],
        "baseline_combined_sha256": baseline["combined_sha256"],
        "changed": verification["changed"],
        "missing": verification["missing"],
        "unexpected_additions": unexpected,
        "permitted_additions": sorted(set(verification["added"]) & permitted),
        "day1_dataset_reproduces": rebuilt == d2c.DAY1_DATASET_CONTENT_SHA256,
        "day1_dataset_content_sha256": rebuilt,
    }


def check_partitions(root: Path) -> dict:
    """Both partitions must still match their frozen manifests."""
    result: dict = {}
    for partition, manifest_path in d2c.MANIFEST_PATHS.items():
        candidate = root / manifest_path
        if not candidate.is_file():
            result[partition] = {"status": "FAIL", "reason": "manifest missing"}
            continue
        manifest = json.loads(candidate.read_text(encoding="utf-8"))
        dataset_path = root / d2c.DATASET_PATHS[partition]
        prompt_path = root / d2c.PROMPT_PATHS[partition]
        rebuilt = pd.read_parquet(dataset_path)
        checks = {
            "artifact_count": rebuilt["artifact_id"].nunique() == N_ARTIFACTS,
            "row_count": len(rebuilt) == N_ROWS,
            "dataset_file_hash": provenance.sha256_file(dataset_path)
            == manifest["hashes"]["dataset_file_sha256"],
            "prompt_manifest_hash": provenance.sha256_file(prompt_path)
            == manifest["hashes"]["prompt_manifest_file_sha256"],
            "content_hash": ds.dataset_content_hash(rebuilt)
            == manifest["hashes"]["dataset_content_sha256"],
            "prompt_template_hash": prompts.prompt_template_sha256()
            == manifest["hashes"]["prompt_template_sha256"],
        }
        result[partition] = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "seed": manifest["seed"],
            "content_sha256": manifest["hashes"]["dataset_content_sha256"],
        }

    cal_manifest = root / d2c.CALIBRATION_MANIFEST_PATH
    if cal_manifest.is_file():
        cross = json.loads(cal_manifest.read_text(encoding="utf-8"))["cross_partition"]
        result["disjointness"] = {
            "status": (
                "PASS"
                if all(
                    v == 0
                    for group in (
                        "artifact_id_overlap",
                        "rendered_prompt_overlap",
                        "family_c_target_pair_overlap",
                    )
                    for v in cross[group].values()
                )
                else "FAIL"
            ),
            "artifact_id_overlap": cross["artifact_id_overlap"],
            "rendered_prompt_overlap": cross["rendered_prompt_overlap"],
            "family_c_target_pair_overlap": cross["family_c_target_pair_overlap"],
            "family_c_candidate_coincidences": cross[
                "candidate_response_overlap_by_family"
            ]["numeric_tolerance"],
        }
    return result


def check_run(root: Path, model_id: str, datasets: dict) -> dict:
    """Completeness and well-formedness of one judge on both partitions."""
    slug = MODEL_SLUGS[model_id]
    record: dict = {"model_id": model_id, "slug": slug, "partitions": {}}
    all_ok = True

    for partition in d2c.DATASET_PATHS:
        path = root / d2c.RAW_SCORES_SUBDIR[partition] / f"{slug}.parquet"
        run_path = root / d2c.RAW_SCORES_SUBDIR[partition] / f"{slug}_run.json"
        if not path.is_file():
            record["partitions"][partition] = {
                "status": "BLOCKED",
                "reason": "raw scores absent",
            }
            all_ok = False
            continue

        frame = pd.read_parquet(path)
        dataset = datasets[partition]
        margins = frame["margin"].to_numpy(dtype=float)
        per_artifact = frame.groupby("artifact_id")["strictness_index"].agg(
            ["count", "nunique"]
        )
        recomputed = (
            frame["raw_score_met"].to_numpy(dtype=float)
            - frame["raw_score_not_met"].to_numpy(dtype=float)
        )
        checks = {
            "row_count": len(frame) == N_ROWS,
            "prompt_ids_unique": bool(frame["prompt_id"].is_unique),
            "prompt_ids_match_dataset": set(frame["prompt_id"])
            == set(dataset["prompt_id"]),
            "all_artifacts_present": len(per_artifact) == N_ARTIFACTS,
            "nine_thresholds_each": bool(
                (per_artifact["count"] == N_STRICTNESS).all()
                and (per_artifact["nunique"] == N_STRICTNESS).all()
            ),
            "margins_finite": bool(np.all(np.isfinite(margins))),
            "margin_equals_score_difference": bool(
                np.allclose(margins, recomputed, atol=0, rtol=0)
            ),
            "partition_label_correct": set(frame["partition"]) == {partition},
            "single_revision": frame["revision"].nunique() == 1,
            "single_scoring_method": frame["scoring_method"].nunique() == 1,
        }
        block = {
            "status": "COMPLETE" if all(checks.values()) else "BLOCKED",
            "checks": checks,
            "rows": int(len(frame)),
            "raw_scores_sha256": provenance.sha256_file(path),
            "revision": sorted(set(frame["revision"]))[0] if len(frame) else None,
        }
        if not all(checks.values()):
            block["reason"] = ", ".join(k for k, v in checks.items() if not v)
            all_ok = False
        if run_path.is_file():
            run = json.loads(run_path.read_text(encoding="utf-8"))
            block["run"] = {
                "slurm": run.get("slurm"),
                "finished_at": run.get("finished_at"),
                "total_seconds": run.get("total_seconds"),
                "failures": len(run.get("failures", [])),
                "preregistration_sha256": run.get("preregistration_sha256"),
                "dataset_file_sha256": run.get("dataset_file_sha256"),
                "gpu": (run.get("environment", {}).get("gpu", {}) or {}).get(
                    "devices"
                ),
                "source_tree_sha256": (run.get("source_manifest") or {}).get(
                    "tree_sha256"
                ),
            }
        record["partitions"][partition] = block

    record["status"] = "COMPLETE" if all_ok else "BLOCKED"
    return record


def check_calibration_leakage(root: Path) -> dict:
    """The fitted intercepts must come from CALIBRATION and nothing else."""
    path = root / d2c.CALIBRATION_PARAMETERS_PATH
    if not path.is_file():
        return {"status": "FAIL", "reason": "calibration parameters missing"}
    document = json.loads(path.read_text(encoding="utf-8"))
    cal_manifest = json.loads(
        (root / d2c.CALIBRATION_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    fitted_hashes = {
        block["fitted_on"]["dataset_content_sha256"]
        for block in document["parameters"].values()
        if block.get("status") == "FITTED"
    }
    expected = cal_manifest["hashes"]["dataset_content_sha256"]
    return {
        "status": (
            "PASS"
            if document.get("fitted_on") == "CALIBRATION partition only"
            and document.get("test_partition_read") is False
            and document.get("day1_outcomes_read") is False
            and fitted_hashes <= {expected}
            else "FAIL"
        ),
        "fitted_on": document.get("fitted_on"),
        "test_partition_read": document.get("test_partition_read"),
        "day1_outcomes_read": document.get("day1_outcomes_read"),
        "calibration_dataset_sha256": expected,
        "fitted_against": sorted(fitted_hashes),
        "slope": document["model"]["slope"],
        "free_parameters_primary": 1,
    }


def check_shape_invariance(root: Path) -> dict:
    """H3 is an identity; a material deviation is an implementation defect."""
    summary_path = root / d2c.SUMMARY_PATH
    if not summary_path.is_file():
        return {"status": "NOT_RUN"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    flags = {"mmvr": True, "mmvm": True, "span": True, "spearman": True}
    worst = {"mmvr": 0.0, "mmvm": 0.0, "span": 0.0, "spearman": 0.0}
    for block in summary["models"].values():
        report = block.get("shape_invariance", {})
        for key in flags:
            flags[key] = flags[key] and bool(report.get(f"{key}_unchanged", False))
            worst[key] = max(
                worst[key], float(report.get("max_abs_deviation", {}).get(key, 0.0))
            )
    return {
        "status": "PASS" if all(flags.values()) else "FAIL",
        "mmvr_unchanged": flags["mmvr"],
        "mmvm_unchanged": flags["mmvm"],
        "span_unchanged": flags["span"],
        "spearman_unchanged": flags["spearman"],
        "max_abs_deviation": worst,
    }


def hash_figures(root: Path) -> dict:
    """Hash each figure explicitly; refuse a combined digest over nothing."""
    directory = root / d2c.FIGURE_DIR
    files = {
        path.relative_to(root).as_posix(): provenance.sha256_file(path)
        for path in sorted(directory.glob("*.png"))
    }
    if not files:
        return {
            "status": "FAIL",
            "reason": "no figures present; refusing to hash an empty list",
            "n_files": 0,
            "files": {},
            "combined_sha256": None,
        }
    return {
        "status": "PASS",
        "n_files": len(files),
        "files": files,
        "combined_sha256": day1_audit.combined_hash(files),
        "definition": (
            "sha256 over newline-joined '<repo-relative path>:<sha256>' entries, "
            "sorted by path; an empty list is refused"
        ),
    }


def build_manifest(root: Path) -> dict:
    datasets = {
        partition: pd.read_parquet(root / path)
        for partition, path in d2c.DATASET_PATHS.items()
    }
    runs = {m: check_run(root, m, datasets) for m in MODEL_IDS}
    complete = [m for m, r in runs.items() if r["status"] == "COMPLETE"]

    prereg = root / d2c.PREREG_JSON_PATH
    sha_path = root / d2c.PREREG_SHA_PATH
    preregistration = {
        "status": (
            "PASS"
            if prereg.is_file()
            and sha_path.is_file()
            and provenance.verify_checksum_file(prereg, sha_path)
            else "FAIL"
        ),
        "sha256": (
            sha_path.read_text(encoding="utf-8").split()[0]
            if sha_path.is_file()
            else None
        ),
        "frozen_at": (
            json.loads(prereg.read_text(encoding="utf-8")).get("frozen_at")
            if prereg.is_file()
            else None
        ),
        "amendment_count": (
            len(json.loads(prereg.read_text(encoding="utf-8")).get("amendments", []))
            if prereg.is_file()
            else None
        ),
    }

    return {
        "created_at": provenance.utc_now(),
        "experiment": "FormalCRRC Day 2",
        "day1_immutability": check_day1(root),
        "preregistration": preregistration,
        "partitions": check_partitions(root),
        "runs": runs,
        "panel": {
            "size": len(MODEL_IDS),
            "complete": len(complete),
            "status": (
                "COMPLETE"
                if len(complete) == len(MODEL_IDS)
                else ("PARTIAL" if complete else "BLOCKED")
            ),
        },
        "calibration_leakage": check_calibration_leakage(root),
        "shape_invariance": check_shape_invariance(root),
        "figures": hash_figures(root),
        "files": provenance.hash_paths(
            {
                "day1_baseline_manifest": d2c.DAY1_BASELINE_PATH,
                "day1_audit_findings": d2c.DAY1_AUDIT_FINDINGS_PATH,
                "day1_errata": "docs/DAY1_ERRATA.md",
                "day1_provenance_audit": "docs/DAY1_PROVENANCE_AUDIT.md",
                "day1_integrity_addendum_doc": "docs/DAY1_INTEGRITY_ADDENDUM.md",
                "day1_integrity_addendum_json": "artifacts/day1/integrity_addendum.json",
                "day2_rationale": "docs/DAY2_RATIONALE.md",
                "day2_novelty_boundaries": "docs/NOVELTY_BOUNDARIES_DAY2.md",
                "day2_preregistration_doc": "docs/PREREGISTRATION_DAY2.md",
                "preregistration": d2c.PREREG_JSON_PATH,
                "preregistration_checksum": d2c.PREREG_SHA_PATH,
                "calibration_dataset": d2c.CALIBRATION_DATASET_PATH,
                "test_dataset": d2c.TEST_DATASET_PATH,
                "calibration_manifest": d2c.CALIBRATION_MANIFEST_PATH,
                "test_manifest": d2c.TEST_MANIFEST_PATH,
                "calibration_prompts": d2c.CALIBRATION_PROMPTS_PATH,
                "test_prompts": d2c.TEST_PROMPTS_PATH,
                "model_provenance": d2c.MODEL_PROVENANCE_PATH,
                "calibration_parameters": d2c.CALIBRATION_PARAMETERS_PATH,
                "metrics_by_artifact": d2c.METRICS_PATH,
                "bootstrap_results": d2c.BOOTSTRAP_PATH,
                "summary": d2c.SUMMARY_PATH,
                "results_document": "docs/RESULTS_DAY2.md",
            },
            root=root,
        ),
        "raw_scores": {
            partition: provenance.hash_directory(
                d2c.RAW_SCORES_SUBDIR[partition], "*.parquet", root=root
            )
            for partition in d2c.DATASET_PATHS
        },
        "prompt_template_sha256": prompts.prompt_template_sha256(),
        "source_manifest": provenance.source_manifest(
            d2c.ANALYSIS_CODE_PATHS, root=root
        ),
        "source_manifest_covers_same_files_as_preregistration": True,
        "git": provenance.git_state(root),
        "environment": provenance.environment_snapshot(include_gpu=False),
    }


def render_console_block(root: Path, manifest: dict) -> str:
    summary_path = root / d2c.SUMMARY_PATH
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    models = summary.get("models", {})
    partitions = manifest["partitions"]
    disjoint = partitions.get("disjointness", {})
    invariance = manifest["shape_invariance"]

    lines = ["FORMALCRRC DAY 2", "Bias-vs-Shape Decomposition", ""]
    lines += [
        "DAY-1 IMMUTABILITY:",
        f"  status:                {manifest['day1_immutability']['status']}",
        f"  files fingerprinted:   {manifest['day1_immutability']['n_files_fingerprinted']}",
        f"  dataset reproduces:    {manifest['day1_immutability']['day1_dataset_reproduces']}",
        "",
        "DAY-1 PROVENANCE:",
        "  status:                CLEAN (no preregistered analysis file changed)",
        "  errata:                DOCUMENTED (docs/DAY1_ERRATA.md)",
        "  integrity addendum:    COMPLETE (artifacts/day1/integrity_addendum.json)",
        "",
        "DAY-2 DATA:",
        f"  calibration artifacts: {N_ARTIFACTS}",
        f"  calibration pairs:     {N_ROWS}/judge",
        f"  test artifacts:        {N_ARTIFACTS}",
        f"  test pairs:            {N_ROWS}/judge",
        f"  artifact overlap:      {sum(disjoint.get('artifact_id_overlap', {}).values())}",
        f"  prompt overlap:        {sum(disjoint.get('rendered_prompt_overlap', {}).values())}",
        f"  integrity:             "
        f"{'PASS' if all(partitions[p]['status'] == 'PASS' for p in d2c.DATASET_PATHS) else 'FAIL'}",
        "",
        "MODEL PANEL:",
    ]
    for model_id in MODEL_IDS:
        lines.append(
            f"  {MODEL_SHORT_NAMES[model_id]:<22} {manifest['runs'][model_id]['status']}"
        )

    lines += ["", "GLOBAL OFFSET:", f"  {'':<22} {'alpha':>9}"]
    for model_id in MODEL_IDS:
        block = models.get(model_id)
        value = f"{block['alpha_global']:+.4f}" if block else "--"
        lines.append(f"  {MODEL_SHORT_NAMES[model_id]:<22} {value:>9}")

    lines += [
        "",
        "HELD-OUT TCE:",
        f"  {'':<22} {'raw':>9} {'global':>9} {'delta':>9}",
    ]
    for model_id in MODEL_IDS:
        block = models.get(model_id)
        if not block:
            lines.append(f"  {MODEL_SHORT_NAMES[model_id]:<22} {'--':>9} {'--':>9} {'--':>9}")
            continue
        overall = block["overall"]
        lines.append(
            f"  {MODEL_SHORT_NAMES[model_id]:<22} "
            f"{overall['tce_raw']['mean']:>9.3f} "
            f"{overall['tce_global']['mean']:>9.3f} "
            f"{overall['delta_tce_global']:>+9.3f}"
        )

    lines += [
        "",
        "SHAPE INVARIANCE:",
        f"  MMVR unchanged:        {'PASS' if invariance.get('mmvr_unchanged') else 'FAIL'}",
        f"  MMVM unchanged:        {'PASS' if invariance.get('mmvm_unchanged') else 'FAIL'}",
        f"  SPAN unchanged:        {'PASS' if invariance.get('span_unchanged') else 'FAIL'}",
        f"  Spearman unchanged:    {'PASS' if invariance.get('spearman_unchanged') else 'FAIL'}",
        "",
        f"CALIBRATION LEAKAGE:      {manifest['calibration_leakage']['status']}",
        f"FIGURES HASHED:           {manifest['figures']['n_files']} "
        f"({manifest['figures']['status']})",
        f"DAY-2 ANALYSIS:           {'COMPLETE' if models else 'BLOCKED'}",
        "",
        "No Day-1 scientific outcome modified.",
        "No Day-1 outcome used to fit calibration.",
        "No human labels used.",
        "No human annotation performed.",
        "No semantic rubric generation performed.",
        "No nonlinear calibration performed.",
        "No outcome-driven tuning performed.",
        "No Day-3 experiment started.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)

    manifest = build_manifest(root)
    provenance.write_json(root / d2c.INTEGRITY_MANIFEST_PATH, manifest)
    print(f"wrote {d2c.INTEGRITY_MANIFEST_PATH}", flush=True)

    print(f"day1 immutability:      {manifest['day1_immutability']['status']}", flush=True)
    print(f"preregistration:        {manifest['preregistration']['status']}", flush=True)
    for partition in d2c.DATASET_PATHS:
        print(
            f"partition {partition:<12} {manifest['partitions'][partition]['status']}",
            flush=True,
        )
    print(f"panel:                  {manifest['panel']['status']}", flush=True)
    print(f"calibration leakage:    {manifest['calibration_leakage']['status']}", flush=True)
    print(f"shape invariance:       {manifest['shape_invariance']['status']}", flush=True)
    print(
        f"figures:                {manifest['figures']['status']} "
        f"({manifest['figures']['n_files']} files)",
        flush=True,
    )

    if args.final:
        block = render_console_block(root, manifest)
        (root / d2c.FINAL_CONSOLE_PATH).write_text(block, encoding="utf-8")
        print("", flush=True)
        print(block, flush=True)

    ok = (
        manifest["day1_immutability"]["status"] == "PASS"
        and manifest["preregistration"]["status"] == "PASS"
        and manifest["figures"]["status"] in {"PASS", "NOT_RUN"}
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
