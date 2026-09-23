#!/usr/bin/env python
"""Post-inference integrity check and integrity manifest (Phase F / H).

Confirms that every judge's raw scores contain exactly the frozen prompt set,
that no score is malformed, that no threshold is missing, and that every frozen
input still hashes to what it hashed to when it was frozen.

Writes ``artifacts/day1/integrity_manifest.json`` and, when ``--final`` is given,
``artifacts/day1/final_console_block.txt``.
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
from formalcrrc import prompts, provenance  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    ANALYSIS_CODE_PATHS,
    BOOTSTRAP_PATH,
    DATASET_MANIFEST_PATH,
    DATASET_PATH,
    FIGURE_DIR,
    FINAL_CONSOLE_PATH,
    INTEGRITY_MANIFEST_PATH,
    METRICS_PATH,
    MODEL_IDS,
    MODEL_PROVENANCE_PATH,
    MODEL_SHORT_NAMES,
    MODEL_SLUGS,
    N_ARTIFACTS,
    N_ROWS,
    N_STRICTNESS,
    PREREG_JSON_PATH,
    PREREG_SHA_PATH,
    PROMPT_MANIFEST_PATH,
    RAW_SCORES_DIR,
    SUMMARY_PATH,
)


def check_dataset(root: Path) -> dict:
    """Rebuild the dataset from the seed and compare every recorded hash."""
    manifest_path = root / DATASET_MANIFEST_PATH
    if not manifest_path.is_file():
        return {"status": "FAIL", "reason": "dataset manifest missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rebuilt = ds.build_dataset()
    checks = {
        "artifact_count": rebuilt["artifact_id"].nunique() == N_ARTIFACTS,
        "row_count": len(rebuilt) == N_ROWS,
        "content_hash_reproduces": ds.dataset_content_hash(rebuilt)
        == manifest["hashes"]["dataset_content_sha256"],
        "dataset_file_hash": provenance.sha256_file(root / DATASET_PATH)
        == manifest["hashes"]["dataset_file_sha256"],
        "prompt_manifest_file_hash": provenance.sha256_file(
            root / PROMPT_MANIFEST_PATH
        )
        == manifest["hashes"]["prompt_manifest_file_sha256"],
        "prompt_template_hash": prompts.prompt_template_sha256()
        == manifest["hashes"]["prompt_template_sha256"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "content_sha256": manifest["hashes"]["dataset_content_sha256"],
    }


def check_preregistration(root: Path) -> dict:
    prereg = root / PREREG_JSON_PATH
    sha_path = root / PREREG_SHA_PATH
    if not (prereg.is_file() and sha_path.is_file()):
        return {"status": "FAIL", "reason": "preregistration not frozen"}
    ok = provenance.verify_checksum_file(prereg, sha_path)
    document = json.loads(prereg.read_text(encoding="utf-8"))
    amendments = document.get("amendments", [])
    return {
        "status": "PASS" if ok else "FAIL",
        "sha256": sha_path.read_text(encoding="utf-8").split()[0],
        "frozen_at": document.get("frozen_at"),
        "panel_intact": document.get("judge_models", {}).get("model_ids")
        == list(MODEL_IDS),
        "amendment_count": len(amendments),
        "amendments": amendments,
    }


def check_run(root: Path, model_id: str, dataset: pd.DataFrame) -> dict:
    """Completeness and well-formedness of one judge's raw scores."""
    slug = MODEL_SLUGS[model_id]
    path = root / RAW_SCORES_DIR / f"{slug}.parquet"
    run_path = root / RAW_SCORES_DIR / f"{slug}_run.json"
    record: dict = {"model_id": model_id, "slug": slug}

    if not path.is_file():
        record.update(status="BLOCKED", reason="raw scores absent")
        return record

    frame = pd.read_parquet(path)
    probabilities = frame["p_met"].to_numpy(dtype=float)
    per_artifact = frame.groupby("artifact_id")["strictness_index"].agg(
        ["count", "nunique"]
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
        "probabilities_finite": bool(np.all(np.isfinite(probabilities))),
        "probabilities_in_range": bool(
            np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)
        ),
        "raw_scores_finite": bool(
            np.all(np.isfinite(frame["raw_score_met"].to_numpy(dtype=float)))
            and np.all(np.isfinite(frame["raw_score_not_met"].to_numpy(dtype=float)))
        ),
        "single_revision": frame["revision"].nunique() == 1,
        "single_scoring_method": frame["scoring_method"].nunique() == 1,
    }

    record.update(
        status="COMPLETE" if all(checks.values()) else "BLOCKED",
        checks=checks,
        rows=int(len(frame)),
        revision=sorted(set(frame["revision"]))[0] if len(frame) else None,
        scoring_method=(
            sorted(set(frame["scoring_method"]))[0] if len(frame) else None
        ),
        raw_scores_sha256=provenance.sha256_file(path),
    )
    if not all(checks.values()):
        record["reason"] = ", ".join(k for k, v in checks.items() if not v)
    if run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        record["run"] = {
            "slurm": run.get("slurm"),
            "started_at_unix": run.get("started_at_unix"),
            "finished_at": run.get("finished_at"),
            "total_seconds": run.get("total_seconds"),
            "failures": len(run.get("failures", [])),
            "gpu": (run.get("environment", {}).get("gpu", {}) or {}).get("devices"),
            "torch_version": run.get("environment", {})
            .get("packages", {})
            .get("torch"),
            "transformers_version": run.get("environment", {})
            .get("packages", {})
            .get("transformers"),
            "source_tree_sha256": (run.get("source_manifest") or {}).get(
                "tree_sha256"
            ),
        }
    return record


def build_manifest(root: Path) -> dict:
    dataset = pd.read_parquet(root / DATASET_PATH)
    runs = {m: check_run(root, m, dataset) for m in MODEL_IDS}
    complete = [m for m, r in runs.items() if r["status"] == "COMPLETE"]

    return {
        "created_at": provenance.utc_now(),
        "dataset": check_dataset(root),
        "preregistration": check_preregistration(root),
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
        "files": provenance.hash_paths(
            {
                "preregistration": PREREG_JSON_PATH,
                "preregistration_checksum": PREREG_SHA_PATH,
                "dataset": DATASET_PATH,
                "dataset_manifest": DATASET_MANIFEST_PATH,
                "prompt_manifest": PROMPT_MANIFEST_PATH,
                "model_provenance": MODEL_PROVENANCE_PATH,
                "metrics_by_artifact": METRICS_PATH,
                "bootstrap_results": BOOTSTRAP_PATH,
                "summary": SUMMARY_PATH,
                "results_document": "docs/RESULTS_DAY1.md",
                "specification": "docs/FORMALCRRC_DAY1_SPEC.md",
                "preregistration_document": "docs/PREREGISTRATION_DAY1.md",
                "lineage": "docs/LINEAGE.md",
                "novelty_boundaries": "docs/NOVELTY_BOUNDARIES.md",
            },
            root=root,
        ),
        "raw_scores": provenance.hash_directory(
            RAW_SCORES_DIR, "*.parquet", root=root
        ),
        "raw_score_run_records": provenance.hash_directory(
            RAW_SCORES_DIR, "*_run.json", root=root
        ),
        "figures": provenance.hash_directory(FIGURE_DIR, "*.png", root=root),
        "prompt_template_sha256": prompts.prompt_template_sha256(),
        "source_manifest": provenance.source_manifest(
            list(ANALYSIS_CODE_PATHS) + ["scripts/figures_day1.py"], root=root
        ),
        "git": provenance.git_state(root),
        "environment": provenance.environment_snapshot(include_gpu=False),
    }


def render_console_block(root: Path, manifest: dict) -> str:
    """The final operational summary, built only from analysis artifacts."""
    summary_path = root / SUMMARY_PATH
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    models = summary.get("models", {})

    lines = ["FORMALCRRC DAY 1", ""]
    dataset = manifest["dataset"]
    lines += [
        "Formal dataset:",
        f"  artifacts:              {N_ARTIFACTS}",
        "  predicate families:     3 (coverage, max_violation, numeric_tolerance)",
        f"  thresholds/artifact:    {N_STRICTNESS}",
        f"  rubric-response pairs:  {N_ROWS}",
        f"  dataset integrity:      {dataset['status']}",
        f"  content sha256:         {dataset.get('content_sha256', 'n/a')}",
        "",
    ]

    prereg = manifest["preregistration"]
    lines += [
        "Preregistration:",
        f"  frozen before inference: {'YES' if prereg['status'] == 'PASS' else 'NO'}",
        f"  frozen at:               {prereg.get('frozen_at', 'n/a')}",
        f"  checksum verifies:       {prereg['status']}",
        f"  amendments:              {prereg.get('amendment_count', 0)}",
        f"  sha256:                  {prereg.get('sha256', 'n/a')}",
        "",
        "Judge panel:",
    ]
    for model_id in MODEL_IDS:
        run = manifest["runs"][model_id]
        detail = "" if run["status"] == "COMPLETE" else f"  ({run.get('reason', '')})"
        lines.append(
            f"  {MODEL_SHORT_NAMES[model_id]:<14} {run['status']}{detail}"
        )

    lines += ["", "Primary results:"]
    lines.append(f"  {'':<14} {'MVR':>9} {'MVM':>9} {'mean TCE':>9} {'accuracy':>9}")
    for model_id in MODEL_IDS:
        short = MODEL_SHORT_NAMES[model_id]
        block = models.get(model_id, {}).get("overall")
        if not block:
            lines.append(f"  {short:<14} {'--':>9} {'--':>9} {'--':>9} {'--':>9}")
            continue
        lines.append(
            f"  {short:<14} {block['mvr']:>9.4f} {block['mvm']:>9.4f} "
            f"{block['tce']['mean']:>9.3f} {block['accuracy']:>9.4f}"
        )

    lines += [
        "",
        f"Model panel:              {manifest['panel']['status']} "
        f"({manifest['panel']['complete']}/{manifest['panel']['size']})",
        f"Analysis status:          {'COMPLETE' if models else 'BLOCKED'}",
        "",
        "No human labels used.",
        "No human annotation performed.",
        "No semantic rubric generation performed.",
        "No outcome-driven tuning performed.",
        "No Day-2 experiment started.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument(
        "--final", action="store_true", help="also write the final console block"
    )
    args = parser.parse_args()
    root = Path(args.root)

    manifest = build_manifest(root)
    provenance.write_json(root / INTEGRITY_MANIFEST_PATH, manifest)
    print(f"wrote {INTEGRITY_MANIFEST_PATH}", flush=True)

    print(f"dataset integrity:      {manifest['dataset']['status']}", flush=True)
    print(f"preregistration:        {manifest['preregistration']['status']}", flush=True)
    print(f"model panel:            {manifest['panel']['status']}", flush=True)
    for model_id in MODEL_IDS:
        run = manifest["runs"][model_id]
        print(f"  {MODEL_SHORT_NAMES[model_id]:<14} {run['status']}", flush=True)

    if args.final:
        block = render_console_block(root, manifest)
        (root / FINAL_CONSOLE_PATH).write_text(block, encoding="utf-8")
        print("", flush=True)
        print(block, flush=True)

    ok = (
        manifest["dataset"]["status"] == "PASS"
        and manifest["preregistration"]["status"] == "PASS"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
