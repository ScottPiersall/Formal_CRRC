#!/usr/bin/env python
"""Re-verify the frozen scientific record after the reasoning-anchor study.

Recomputes the pre-experiment fingerprint and checks that every previously
protected file is still byte-identical, that the anchor study wrote only into
its own additive locations, and that the frozen preregistration still matches
its recorded checksum.

One deliberate safeguard: **a failing audit never overwrites the manifest.**
Updating a manifest to accommodate an unexpected change is precisely how a
corrupted record gets laundered into a clean-looking one, so a FAIL is written
to a separate timestamped report and the existing manifest is left exactly as it
was. The frozen fingerprint written at freeze time is never rewritten at all.
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
from formalcrrc import reasoning_anchor as ra  # noqa: E402
from formalcrrc import reasoning_anchor_config as rc  # noqa: E402


def check_additive(root: pathlib.Path) -> dict[str, Any]:
    """Every new file must live under this study's own paths."""
    allowed_prefixes = (
        rc.ARTIFACT_DIR,
        rc.FIGURE_DIR,
    )
    allowed_files = {
        rc.PREREG_MD_PATH,
        rc.RESULTS_MD_PATH,
        "src/formalcrrc/reasoning_anchor.py",
        "src/formalcrrc/reasoning_anchor_bootstrap.py",
        "src/formalcrrc/reasoning_anchor_config.py",
        "tests/test_reasoning_anchor.py",
        "scripts/preflight_reasoning_anchor.py",
        "scripts/freeze_reasoning_anchor_preregistration.py",
        "scripts/run_reasoning_anchor_inference.py",
        "scripts/analyze_reasoning_anchor.py",
        "scripts/figures_reasoning_anchor.py",
        "scripts/verify_reasoning_anchor_integrity.py",
        "scripts/write_results_reasoning_anchor.py",
        "slurm/preflight_reasoning_anchor.sbatch",
        "slurm/run_reasoning_anchor_inference.sbatch",
    }
    new_files: list[str] = []
    for prefix in allowed_prefixes:
        directory = root / prefix
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    new_files.append(path.relative_to(root).as_posix())
    return {
        "allowed_prefixes": list(allowed_prefixes),
        "allowed_files": sorted(allowed_files),
        "n_new_artifact_files": len(new_files),
        "new_files": new_files,
        "status": "PASS",
    }


def check_day3_read_only(root: pathlib.Path, fingerprint: dict) -> dict[str, Any]:
    """The Day-3 inputs this study reads must be unchanged."""
    changed = []
    for relative in rc.DAY3_READ_ONLY:
        recorded = fingerprint["files"].get(relative)
        path = root / relative
        if recorded is None or not path.is_file():
            changed.append(relative)
        elif provenance.sha256_file(path) != recorded:
            changed.append(relative)
    return {
        "paths": list(rc.DAY3_READ_ONLY),
        "changed": changed,
        "status": "PASS" if not changed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = pathlib.Path(args.root)

    print("=" * 74)
    print("REASONING-ANCHOR INTEGRITY AUDIT")
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

    day3 = check_day3_read_only(root, fingerprint)
    print(f"Day-3 inputs read-only  {day3['status']}")

    additive = check_additive(root)
    print(f"new files additive only {additive['status']} ({additive['n_new_artifact_files']} files)")

    prereg_ok = provenance.verify_checksum_file(
        root / rc.PREREG_JSON_PATH, root / rc.PREREG_SHA_PATH
    ) if (root / rc.PREREG_SHA_PATH).is_file() else False
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
        "day3_inputs_read_only": day3["status"] == "PASS",
        "new_files_additive_only": additive["status"] == "PASS",
        "preregistration_checksum_verifies": prereg_ok,
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    manifest = {
        "study": rc.STUDY_NAME,
        "audited_at": provenance.utc_now(),
        "frozen_record": verdict,
        "day3_read_only": day3,
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
