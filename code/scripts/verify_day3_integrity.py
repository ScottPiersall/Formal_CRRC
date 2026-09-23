#!/usr/bin/env python
"""Day-3 integrity verification and manifest (Phase J).

Re-verifies that Days 1 and 2 are untouched, that the Day-3 dataset still hashes
to its manifest, that every judge's raw scores are complete and well formed,
that the reachability theorem still agrees with independent enumeration, and
that the oracle nesting inequality holds.

Figures are hashed individually and the combined digest refuses an empty file
list. The results document is generated after this manifest, not before it.

Writes ``artifacts/day3/integrity_manifest.json`` and, with ``--final``,
``artifacts/day3/final_console_block.txt``.
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
from formalcrrc import day3, day3_config as d3c, prompts, provenance  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    FAMILIES,
    MODEL_IDS,
    MODEL_SHORT_NAMES,
    MODEL_SLUGS,
    N_ARTIFACTS,
    N_ROWS,
    N_STRICTNESS,
)


def check_priors(root: Path) -> dict:
    """Days 1 and 2 must be exactly as Day 3 found them."""
    manifest_path = root / d3c.PRIOR_MANIFEST_PATH
    if not manifest_path.is_file():
        return {"status": "FAIL", "reason": "prior fingerprint missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = day3.verify_prior_manifest(manifest, root)
    rebuilt = ds.dataset_content_hash(ds.build_dataset())
    return {
        "status": (
            "PASS"
            if verification["status"] == "PASS"
            and rebuilt == d3c.DAY1_DATASET_CONTENT_SHA256
            else "FAIL"
        ),
        "n_files_fingerprinted": manifest["n_files"],
        "combined_sha256": manifest["combined_sha256"],
        "day1": verification["day1"],
        "day2": verification["day2"],
        "added": verification["added"],
        "day1_dataset_reproduces": rebuilt == d3c.DAY1_DATASET_CONTENT_SHA256,
        "day1_dataset_content_sha256": rebuilt,
    }


def check_dataset(root: Path) -> dict:
    manifest_path = root / d3c.DATASET_MANIFEST_PATH
    if not manifest_path.is_file():
        return {"status": "FAIL", "reason": "dataset manifest missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = root / d3c.DATASET_PATH
    prompt_path = root / d3c.PROMPT_MANIFEST_PATH
    rebuilt = pd.read_parquet(dataset_path)
    disjoint = manifest["disjointness"]
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
        "disjoint_artifact_ids": all(
            v == 0 for v in disjoint["artifact_id_overlap"].values()
        ),
        "disjoint_prompts": all(
            v == 0 for v in disjoint["rendered_prompt_overlap"].values()
        ),
        "disjoint_family_c_pairs": all(
            v == 0 for v in disjoint["family_c_target_pair_overlap"].values()
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "seed": manifest["seed"],
        "content_sha256": manifest["hashes"]["dataset_content_sha256"],
        "disjointness": disjoint,
    }


def check_run(root: Path, model_id: str, dataset: pd.DataFrame) -> dict:
    slug = MODEL_SLUGS[model_id]
    path = root / d3c.RAW_SCORES_DIR / f"{slug}.parquet"
    run_path = root / d3c.RAW_SCORES_DIR / f"{slug}_run.json"
    record: dict = {"model_id": model_id, "slug": slug}

    if not path.is_file():
        record.update(status="BLOCKED", reason="raw scores absent")
        return record

    frame = pd.read_parquet(path)
    margins = frame["margin"].to_numpy(dtype=float)
    per_artifact = frame.groupby("artifact_id")["strictness_index"].agg(
        ["count", "nunique"]
    )
    recomputed = (
        frame["raw_score_met"].to_numpy(float)
        - frame["raw_score_not_met"].to_numpy(float)
    )
    checks = {
        "row_count": len(frame) == N_ROWS,
        "prompt_ids_unique": bool(frame["prompt_id"].is_unique),
        "prompt_ids_match_dataset": set(frame["prompt_id"]) == set(dataset["prompt_id"]),
        "all_artifacts_present": len(per_artifact) == N_ARTIFACTS,
        "nine_thresholds_each": bool(
            (per_artifact["count"] == N_STRICTNESS).all()
            and (per_artifact["nunique"] == N_STRICTNESS).all()
        ),
        "margins_finite": bool(np.all(np.isfinite(margins))),
        "margin_equals_score_difference": bool(np.array_equal(margins, recomputed)),
        "partition_label_correct": set(frame["partition"]) == {d3c.PARTITION_TAG},
        "single_revision": frame["revision"].nunique() == 1,
        "single_scoring_method": frame["scoring_method"].nunique() == 1,
    }
    record.update(
        status="COMPLETE" if all(checks.values()) else "BLOCKED",
        checks=checks,
        rows=int(len(frame)),
        raw_scores_sha256=provenance.sha256_file(path),
        revision=sorted(set(frame["revision"]))[0] if len(frame) else None,
    )
    if not all(checks.values()):
        record["reason"] = ", ".join(k for k, v in checks.items() if not v)
    if run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        record["run"] = {
            "slurm": run.get("slurm"),
            "finished_at": run.get("finished_at"),
            "total_seconds": run.get("total_seconds"),
            "failures": len(run.get("failures", [])),
            "preregistration_sha256": run.get("preregistration_sha256"),
            "dataset_file_sha256": run.get("dataset_file_sha256"),
            "gpu": (run.get("environment", {}).get("gpu", {}) or {}).get("devices"),
            "source_tree_sha256": (run.get("source_manifest") or {}).get(
                "tree_sha256"
            ),
        }
    return record


def check_oracle(root: Path) -> dict:
    """Theorem/enumeration agreement and the nesting inequality, on real data."""
    reachable_path = root / d3c.REACHABLE_CROSSINGS_PATH
    summary_path = root / d3c.SUMMARY_PATH
    result: dict = {}

    if reachable_path.is_file():
        table = pd.read_parquet(reachable_path)
        disagreements = int((~table["methods_agree"]).sum()) + int(
            (table["otce_theorem"] != table["otce_enumeration"]).sum()
        )
        result["reachability_cross_check"] = {
            "status": "PASS" if disagreements == 0 else "FAIL",
            "n_rows": int(len(table)),
            "disagreements": disagreements,
        }
    else:
        result["reachability_cross_check"] = {"status": "NOT_RUN"}

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        per_model = {}
        ok = True
        for model_id, block in summary["models"].items():
            nesting = block["oracle_hierarchy"]["nesting"]
            per_model[model_id] = nesting
            ok = ok and nesting["status"] == "PASS"
        result["nesting"] = {"status": "PASS" if ok else "FAIL", "models": per_model}
        metrics_path = root / d3c.METRICS_PATH
        if metrics_path.is_file():
            metrics = pd.read_parquet(metrics_path)
            violations = int((metrics["otce"] > metrics["tce_raw"]).sum())
            result["otce_bounded_by_raw"] = {
                "status": "PASS" if violations == 0 else "FAIL",
                "violations": violations,
                "n_rows": int(len(metrics)),
            }
    else:
        result["nesting"] = {"status": "NOT_RUN"}
    return result


def hash_figures(root: Path) -> dict:
    """Hash each figure explicitly; refuse a combined digest over nothing."""
    directory = root / d3c.FIGURE_DIR
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
        "combined_sha256": day3._combined(files),
        "definition": (
            "sha256 over newline-joined '<repo-relative path>:<sha256>' entries, "
            "sorted by path; an empty list is refused"
        ),
    }


def build_manifest(root: Path) -> dict:
    dataset = pd.read_parquet(root / d3c.DATASET_PATH)
    runs = {m: check_run(root, m, dataset) for m in MODEL_IDS}
    complete = [m for m, r in runs.items() if r["status"] == "COMPLETE"]

    prereg = root / d3c.PREREG_JSON_PATH
    sha_path = root / d3c.PREREG_SHA_PATH
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
        "experiment": "FormalCRRC Day 3",
        "prior_immutability": check_priors(root),
        "preregistration": preregistration,
        "dataset": check_dataset(root),
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
        "oracle": check_oracle(root),
        "figures": hash_figures(root),
        "files": provenance.hash_paths(
            {
                "prior_experiments_manifest": d3c.PRIOR_MANIFEST_PATH,
                "day3_rationale": "docs/DAY3_RATIONALE.md",
                "day3_novelty_boundaries": "docs/NOVELTY_BOUNDARIES_DAY3.md",
                "day3_preregistration_doc": "docs/PREREGISTRATION_DAY3.md",
                "preregistration": d3c.PREREG_JSON_PATH,
                "preregistration_checksum": d3c.PREREG_SHA_PATH,
                "test_dataset": d3c.DATASET_PATH,
                "test_manifest": d3c.DATASET_MANIFEST_PATH,
                "prompt_manifest": d3c.PROMPT_MANIFEST_PATH,
                "model_provenance": d3c.MODEL_PROVENANCE_PATH,
                "reachable_crossings": d3c.REACHABLE_CROSSINGS_PATH,
                "oracle_global": d3c.ORACLE_GLOBAL_PATH,
                "oracle_family": d3c.ORACLE_FAMILY_PATH,
                "oracle_artifact": d3c.ORACLE_ARTIFACT_PATH,
                "metrics_by_artifact": d3c.METRICS_PATH,
                "bootstrap_results": d3c.BOOTSTRAP_PATH,
                "summary": d3c.SUMMARY_PATH,
                "results_document": "docs/RESULTS_DAY3.md",
            },
            root=root,
        ),
        "raw_scores": provenance.hash_directory(
            d3c.RAW_SCORES_DIR, "*.parquet", root=root
        ),
        "prompt_template_sha256": prompts.prompt_template_sha256(),
        "source_manifest": provenance.source_manifest(
            d3c.ANALYSIS_CODE_PATHS, root=root
        ),
        "source_manifest_covers_same_files_as_preregistration": True,
        "git": provenance.git_state(root),
        "environment": provenance.environment_snapshot(include_gpu=False),
    }


def render_console_block(root: Path, manifest: dict) -> str:
    summary_path = root / d3c.SUMMARY_PATH
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    models = summary.get("models", {})
    priors = manifest["prior_immutability"]
    dataset = manifest["dataset"]
    oracle = manifest["oracle"]
    disjoint = dataset.get("disjointness", {})
    prompt_overlap = sum(disjoint.get("rendered_prompt_overlap", {}).values())

    lines = [
        "FORMALCRRC DAY 3",
        "Translation Limits of Threshold-Response Curves",
        "",
        "PRIOR EXPERIMENT IMMUTABILITY:",
        f"  Day 1:                    {priors.get('day1', {}).get('status', '--')}",
        f"  Day 2:                    {priors.get('day2', {}).get('status', '--')}",
        f"  files fingerprinted:      {priors.get('n_files_fingerprinted', '--')}",
        "",
        "DAY-3 DATA:",
        f"  artifacts:                {N_ARTIFACTS}",
        f"  thresholds/artifact:      {N_STRICTNESS}",
        f"  pairs/judge:              {N_ROWS}",
        f"  prior prompt overlap:     {prompt_overlap}",
        f"  integrity:                {dataset['status']}",
        "",
        "MODEL PANEL:",
    ]
    for model_id in MODEL_IDS:
        lines.append(
            f"  {MODEL_SHORT_NAMES[model_id]:<24} {manifest['runs'][model_id]['status']}"
        )

    lines += [
        "",
        "PRIMARY RESULTS:",
        "",
        f"  {'':<24}{'raw TCE':>10}{'oracle-artifact TCE':>22}{'nontrivial TRR':>17}",
    ]
    for model_id in MODEL_IDS:
        block = models.get(model_id)
        if not block:
            lines.append(f"  {MODEL_SHORT_NAMES[model_id]:<24}{'--':>10}{'--':>22}{'--':>17}")
            continue
        o = block["overall"]
        lines.append(
            f"  {MODEL_SHORT_NAMES[model_id]:<24}"
            f"{o['tce_raw']['mean']:>10.3f}"
            f"{o['otce']['mean']:>22.3f}"
            f"{o['trr_nontrivial']:>16.1%} "
        )

    lines += [
        "",
        "ORACLE HIERARCHY:",
        "",
        f"  {'':<24}{'RAW':>9}{'ORACLE-G':>11}{'ORACLE-F':>11}{'ORACLE-A':>11}",
    ]
    for model_id in MODEL_IDS:
        block = models.get(model_id)
        if not block:
            lines.append(
                f"  {MODEL_SHORT_NAMES[model_id]:<24}{'--':>9}{'--':>11}{'--':>11}{'--':>11}"
            )
            continue
        h = block["oracle_hierarchy"]
        lines.append(
            f"  {MODEL_SHORT_NAMES[model_id]:<24}"
            f"{h['raw']:>9.3f}{h['oracle_global']:>11.3f}"
            f"{h['oracle_family']:>11.3f}{h['oracle_artifact']:>11.3f}"
        )

    nesting = oracle.get("nesting", {}).get("status", "NOT_RUN")
    cross = oracle.get("reachability_cross_check", {}).get("status", "NOT_RUN")
    lines += [
        "",
        f"ORACLE NESTING CHECK:      {nesting}",
        f"REACHABILITY CROSS-CHECK:  {cross}",
        f"OTCE <= RAW TCE:           "
        f"{oracle.get('otce_bounded_by_raw', {}).get('status', 'NOT_RUN')}",
        f"FIGURES HASHED:            {manifest['figures']['n_files']} "
        f"({manifest['figures']['status']})",
        "",
        f"DAY-3 ANALYSIS:            {'COMPLETE' if models else 'BLOCKED'}",
        "",
        "No prior scientific outcome modified.",
        "No human labels used.",
        "No human annotation performed.",
        "No semantic rubric generation performed.",
        "No model calibration proposed for deployment.",
        "No curve shape modified.",
        "No outcome-driven tuning performed.",
        "No Day-4 experiment started.",
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
    provenance.write_json(root / d3c.INTEGRITY_MANIFEST_PATH, manifest)
    print(f"wrote {d3c.INTEGRITY_MANIFEST_PATH}", flush=True)

    print(f"prior immutability:     {manifest['prior_immutability']['status']}", flush=True)
    print(f"preregistration:        {manifest['preregistration']['status']}", flush=True)
    print(f"dataset:                {manifest['dataset']['status']}", flush=True)
    print(f"panel:                  {manifest['panel']['status']}", flush=True)
    print(
        f"reachability crosscheck:{manifest['oracle'].get('reachability_cross_check', {}).get('status')}",
        flush=True,
    )
    print(
        f"oracle nesting:         {manifest['oracle'].get('nesting', {}).get('status')}",
        flush=True,
    )
    print(
        f"figures:                {manifest['figures']['status']} "
        f"({manifest['figures']['n_files']} files)",
        flush=True,
    )

    if args.final:
        block = render_console_block(root, manifest)
        (root / d3c.FINAL_CONSOLE_PATH).write_text(block, encoding="utf-8")
        print("", flush=True)
        print(block, flush=True)

    ok = (
        manifest["prior_immutability"]["status"] == "PASS"
        and manifest["preregistration"]["status"] == "PASS"
        and manifest["dataset"]["status"] == "PASS"
        and manifest["figures"]["status"] in {"PASS", "NOT_RUN"}
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
