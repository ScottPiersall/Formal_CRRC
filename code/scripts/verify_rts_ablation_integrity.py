#!/usr/bin/env python
"""Re-verify the frozen scientific record after the reason-then-score ablation.

Checks that every previously protected file is still byte-identical -- including
the Day-3 Qwen2.5 comparator this study reads and the completed reasoning-anchor
study -- that the ablation wrote only into its own additive locations, and that
the frozen preregistration still matches its recorded checksum.

A failing audit never overwrites the manifest. Updating a manifest to accommodate
an unexpected change is how a corrupted record gets laundered into a clean-looking
one, so a FAIL is written to a separate timestamped report and the existing
manifest is left exactly as it was.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import provenance  # noqa: E402
from formalcrrc import rts_ablation as ra  # noqa: E402
from formalcrrc import rts_ablation_config as rc  # noqa: E402

ALLOWED_FILES = {
    rc.PREREG_MD_PATH,
    rc.RESULTS_MD_PATH,
    "src/formalcrrc/rts_ablation.py",
    "src/formalcrrc/rts_ablation_bootstrap.py",
    "src/formalcrrc/rts_ablation_config.py",
    "tests/test_rts_ablation.py",
    "scripts/preflight_rts_ablation.py",
    "scripts/freeze_rts_ablation_preregistration.py",
    "scripts/run_rts_ablation_inference.py",
    "scripts/analyze_rts_ablation.py",
    "scripts/figures_rts_ablation.py",
    "scripts/verify_rts_ablation_integrity.py",
    "scripts/write_results_rts_ablation.py",
    "slurm/preflight_rts_ablation.sbatch",
    "slurm/run_rts_ablation_inference.sbatch",
}


def check_additive(root: pathlib.Path) -> dict[str, Any]:
    new_files: list[str] = []
    for prefix in (rc.ARTIFACT_DIR, rc.FIGURE_DIR):
        directory = root / prefix
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    new_files.append(path.relative_to(root).as_posix())
    return {
        "allowed_prefixes": [rc.ARTIFACT_DIR, rc.FIGURE_DIR],
        "allowed_files": sorted(ALLOWED_FILES),
        "n_new_artifact_files": len(new_files),
        "new_files": new_files,
        "status": "PASS",
    }


def check_read_only(root: pathlib.Path, fingerprint: dict) -> dict[str, Any]:
    """The comparator and anchor inputs this study reads must be unchanged."""
    changed = []
    for relative in rc.READ_ONLY_INPUTS:
        recorded = fingerprint["files"].get(relative)
        path = root / relative
        if recorded is None or not path.is_file():
            changed.append(relative)
        elif provenance.sha256_file(path) != recorded:
            changed.append(relative)
    return {
        "paths": list(rc.READ_ONLY_INPUTS),
        "changed": changed,
        "status": "PASS" if not changed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = pathlib.Path(args.root)

    print("=" * 74)
    print("REASON-THEN-SCORE ABLATION INTEGRITY AUDIT")
    print("=" * 74)

    fingerprint_path = root / rc.FROZEN_FINGERPRINT_PATH
    if not fingerprint_path.is_file():
        print(f"MISSING: {rc.FROZEN_FINGERPRINT_PATH}")
        return 2
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))

    verdict = ra.verify_frozen_fingerprint(fingerprint, root)
    print(f"protected files      {verdict['n_recorded']}")
    print(f"  changed            {verdict['changed'] or 'NONE'}")
    print(f"  missing            {verdict['missing'] or 'NONE'}")
    print(f"  added              {verdict['added'] or 'NONE'}")
    print(f"  combined recorded  {verdict['combined_recorded']}")
    print(f"  combined now       {verdict['combined_now']}")
    print(f"  verdict            {verdict['status']}")

    read_only = check_read_only(root, fingerprint)
    print(f"read-only inputs        {read_only['status']}")

    additive = check_additive(root)
    print(
        f"new files additive only {additive['status']} "
        f"({additive['n_new_artifact_files']} files)"
    )

    prereg_ok = (
        provenance.verify_checksum_file(
            root / rc.PREREG_JSON_PATH, root / rc.PREREG_SHA_PATH
        )
        if (root / rc.PREREG_SHA_PATH).is_file()
        else False
    )
    print(f"preregistration checksum {'PASS' if prereg_ok else 'FAIL'}")

    rows_path = root / rc.RAW_ROWS_PATH
    row_check: dict[str, Any] = {"present": rows_path.is_file()}
    if rows_path.is_file():
        frame = pd.read_parquet(rows_path)
        row_check.update(
            {
                "rows": int(len(frame)),
                "expected": rc.N_ROWS,
                "complete": int(len(frame)) == rc.N_ROWS,
                "raw_rows_sha256": provenance.sha256_file(rows_path),
            }
        )
        print(f"rows present            {row_check['rows']}/{rc.N_ROWS}")

    checks = {
        "frozen_record_byte_identical": verdict["status"] == "PASS",
        "combined_fingerprint_matches": bool(verdict["combined_matches"]),
        "read_only_inputs_unchanged": read_only["status"] == "PASS",
        "new_files_additive_only": additive["status"] == "PASS",
        "preregistration_checksum_verifies": prereg_ok,
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    manifest = {
        "study": rc.STUDY_NAME,
        "audited_at": provenance.utc_now(),
        "frozen_record": verdict,
        "read_only_inputs": read_only,
        "additive": additive,
        "raw_rows": row_check,
        "checks": checks,
        "status": status,
        "analysis_code": provenance.source_manifest(rc.ANALYSIS_CODE_PATHS, root),
    }

    print()
    print("Checks:", json.dumps(checks))
    print(f"STATUS: {status}")

    if status == "PASS":
        provenance.write_json(root / rc.INTEGRITY_MANIFEST_PATH, manifest)
        print(f"wrote {rc.INTEGRITY_MANIFEST_PATH}")
    else:
        stamp = provenance.utc_now().replace(":", "").replace("-", "")
        failure_path = root / rc.ARTIFACT_DIR / f"integrity_failure_{stamp}.json"
        provenance.write_json(failure_path, manifest)
        print(f"wrote {failure_path.relative_to(root)}")
        print(
            "The existing integrity manifest was NOT modified. A manifest is "
            "never updated to accommodate an unexpected change."
        )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
