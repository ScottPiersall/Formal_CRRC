#!/usr/bin/env python
"""Phase-14 integrity audit for the label-swap study.

Recomputes the pre-study fingerprint, proves every protected file is still
byte-identical, confirms the Day-3 raw scores were read and never written, and
records the hashes of everything this study produced.

Fails loudly. If any frozen scientific file changed, the verdict is FAIL and the
recorded manifests are **not** updated to accommodate it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import provenance  # noqa: E402
from formalcrrc import labelswap_config as lsc  # noqa: E402
from formalcrrc.config import MODEL_IDS, MODEL_SLUGS  # noqa: E402

ROOT = provenance.repo_root()


def recheck_fingerprint() -> dict:
    """Every protected file, against the fingerprint taken before the study."""
    path = ROOT / lsc.FROZEN_FINGERPRINT_PATH
    if not path.is_file():
        return {"status": "FAIL", "reason": "pre-study fingerprint missing"}
    recorded = json.loads(path.read_text(encoding="utf-8"))

    changed, missing = [], []
    for relative, digest in recorded["files"].items():
        candidate = ROOT / relative
        if not candidate.is_file():
            missing.append(relative)
        elif provenance.sha256_file(candidate) != digest:
            changed.append(relative)

    combined = provenance.sha256_text(
        "\n".join(
            f"{relative}:{provenance.sha256_file(ROOT / relative)}"
            for relative in sorted(recorded["files"])
            if (ROOT / relative).is_file()
        )
    )
    return {
        "n_files": recorded["n_files"],
        "recorded_combined_sha256": recorded["combined_sha256"],
        "recomputed_combined_sha256": combined,
        "combined_matches": combined == recorded["combined_sha256"],
        "changed": changed,
        "missing": missing,
        "status": "PASS" if not changed and not missing else "FAIL",
    }


def day3_read_only() -> dict:
    """The Day-3 inputs this study consumed are unchanged."""
    fingerprint = json.loads(
        (ROOT / lsc.FROZEN_FINGERPRINT_PATH).read_text(encoding="utf-8")
    )
    rows = []
    for relative in lsc.DAY3_READ_ONLY:
        recorded = fingerprint["files"].get(relative)
        live = (
            provenance.sha256_file(ROOT / relative)
            if (ROOT / relative).is_file()
            else None
        )
        rows.append(
            {
                "path": relative,
                "recorded_sha256": recorded,
                "live_sha256": live,
                "unchanged": recorded is not None and recorded == live,
            }
        )
    return {
        "inputs": rows,
        "status": "PASS" if all(r["unchanged"] for r in rows) else "FAIL",
    }


def new_files() -> dict:
    """Everything this study wrote, hashed, and checked to be additive."""
    produced: dict[str, str] = {}
    for tree in (lsc.ARTIFACT_DIR, lsc.FIGURE_DIR):
        directory = ROOT / tree
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    produced[path.relative_to(ROOT).as_posix()] = (
                        provenance.sha256_file(path)
                    )

    forbidden = [
        relative
        for relative in produced
        if not relative.startswith(("artifacts/labelswap/", "figures/labelswap/"))
    ]
    return {
        "n_files": len(produced),
        "files": produced,
        "combined_sha256": provenance.sha256_text(
            "\n".join(f"{k}:{v}" for k, v in sorted(produced.items()))
        ),
        "outside_additive_locations": forbidden,
        "status": "PASS" if not forbidden else "FAIL",
    }


def raw_data_hashes() -> dict:
    """Per-model swapped raw scores and their run records."""
    rows = []
    for model_id in MODEL_IDS:
        slug = MODEL_SLUGS[model_id]
        scores = ROOT / lsc.RAW_SCORES_DIR / f"{slug}.parquet"
        run = ROOT / lsc.RAW_SCORES_DIR / f"{slug}_run.json"
        record = json.loads(run.read_text(encoding="utf-8")) if run.is_file() else {}
        rows.append(
            {
                "model_id": model_id,
                "slug": slug,
                "raw_scores_present": scores.is_file(),
                "raw_scores_sha256": (
                    provenance.sha256_file(scores) if scores.is_file() else None
                ),
                "run_record_sha256": (
                    provenance.sha256_file(run) if run.is_file() else None
                ),
                "prompts_scored": record.get("prompts_scored"),
                "prompts_requested": record.get("prompts_requested"),
                "failures": len(record.get("failures", [])),
                "revision": record.get("revision"),
                "slurm": record.get("slurm"),
                "complete": record.get("prompts_scored")
                == record.get("prompts_requested")
                and record.get("prompts_requested") == lsc.N_ROWS_PER_MODEL,
            }
        )
    return {
        "models": rows,
        "panel_complete": all(r["complete"] for r in rows),
        "status": "PASS" if all(r["complete"] for r in rows) else "FAIL",
    }


def test_status() -> dict:
    """Run the suite and record the count. Reported whatever it says."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        tail = [line for line in result.stdout.strip().splitlines() if line.strip()]
        return {
            "returncode": result.returncode,
            "summary": tail[-1] if tail else "",
            "status": "PASS" if result.returncode == 0 else "FAIL",
        }
    except Exception as error:  # noqa: BLE001 - reported, never hidden
        return {"status": "ERROR", "error": str(error)[:300]}


def main() -> int:
    fingerprint = recheck_fingerprint()
    day3 = day3_read_only()
    produced = new_files()
    raw = raw_data_hashes()
    tests = test_status()

    checks = {
        "frozen_record_byte_identical": fingerprint["status"] == "PASS",
        "combined_fingerprint_matches": fingerprint.get("combined_matches", False),
        "day3_inputs_read_only": day3["status"] == "PASS",
        "new_files_additive_only": produced["status"] == "PASS",
        "panel_complete": raw["status"] == "PASS",
        "test_suite_passes": tests["status"] == "PASS",
    }
    ok = all(checks.values())

    manifest = {
        "study": lsc.STUDY_NAME,
        "generated_utc": provenance.utc_now(),
        "checks": checks,
        "status": "PASS" if ok else "FAIL",
        "frozen_record": fingerprint,
        "day3_read_only": day3,
        "new_files": produced,
        "raw_data": raw,
        "tests": tests,
        "preregistration_sha256": (
            (ROOT / lsc.PREREG_SHA_PATH).read_text(encoding="utf-8").split()[0]
            if (ROOT / lsc.PREREG_SHA_PATH).is_file()
            else None
        ),
        "analysis_code": provenance.source_manifest(
            list(lsc.ANALYSIS_CODE_PATHS), ROOT
        ),
        "git": provenance.git_state(ROOT),
    }
    provenance.write_json(ROOT / lsc.INTEGRITY_MANIFEST_PATH, manifest)

    print("=" * 74)
    print("LABEL-SWAP INTEGRITY AUDIT")
    print("=" * 74)
    print(f"protected files      {fingerprint['n_files']}")
    print(f"  changed            {fingerprint['changed'] or 'NONE'}")
    print(f"  missing            {fingerprint['missing'] or 'NONE'}")
    print(f"  combined recorded  {fingerprint['recorded_combined_sha256']}")
    print(f"  combined now       {fingerprint['recomputed_combined_sha256']}")
    print(f"  verdict            {fingerprint['status']}")
    print()
    print(f"Day-3 inputs read-only  {day3['status']}")
    print(f"new files additive only {produced['status']} ({produced['n_files']} files)")
    print(f"panel complete          {raw['status']}")
    for row in raw["models"]:
        print(f"    {row['model_id']:<40} {row['prompts_scored']}/"
              f"{row['prompts_requested']} failures={row['failures']}")
    print(f"tests                   {tests['status']}  {tests.get('summary','')}")
    print()
    print(f"Checks: {json.dumps(checks)}")
    print(f"STATUS: {manifest['status']}")
    print(f"wrote {lsc.INTEGRITY_MANIFEST_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
