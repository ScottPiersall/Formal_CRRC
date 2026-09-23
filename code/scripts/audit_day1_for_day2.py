#!/usr/bin/env python
"""Phase A: fingerprint Day 1 and audit its record before Day 2 begins.

Writes:

* ``artifacts/day2/day1_baseline_manifest.json`` -- immutable Day-1 fingerprint
* ``artifacts/day1/integrity_addendum.json``     -- additive, corrected figure hashes
* ``artifacts/day2/day1_audit_findings.json``    -- machine-readable audit results

Read-only with respect to every existing Day-1 file. The integrity addendum is
the sole addition under ``artifacts/day1``; it is created only if absent and
never overwrites the original manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import day1_audit, provenance  # noqa: E402

BASELINE_PATH = "artifacts/day2/day1_baseline_manifest.json"
ADDENDUM_PATH = "artifacts/day1/integrity_addendum.json"
FINDINGS_PATH = "artifacts/day2/day1_audit_findings.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument(
        "--force-addendum",
        action="store_true",
        help="rewrite the integrity addendum if it already exists",
    )
    args = parser.parse_args()
    root = Path(args.root)

    records = day1_audit.load_day1_records(root)

    # ---- Day-1 fingerprint -------------------------------------------------
    baseline_path = root / BASELINE_PATH
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        print(f"baseline already exists ({baseline['n_files']} files)", flush=True)
    else:
        baseline = day1_audit.build_day1_baseline(root)
        provenance.write_json(baseline_path, baseline)
        print(f"wrote {BASELINE_PATH}  ({baseline['n_files']} files)", flush=True)
    verification = day1_audit.verify_day1_baseline(baseline, root)
    print(f"DAY1_IMMUTABILITY = {verification['status']}", flush=True)

    # ---- Issue A -----------------------------------------------------------
    issue_a = day1_audit.audit_boundary_curve_claim(records["summary"])
    print("", flush=True)
    print("--- Issue A: boundary-aligned monotonicity claim", flush=True)
    print(
        f"    first clause literally true : {issue_a['first_clause_literally_true']}",
        flush=True,
    )
    print(
        f"    second clause literally true: {issue_a['second_clause_literally_true']}",
        flush=True,
    )
    for model in issue_a["models"]:
        print(
            f"    {model['short_name']:<14} increases={model['n_increases']}/"
            f"{model['n_steps']}  net={model['net_change']:+.4f}  "
            f"rho={model['mean_spearman']:+.4f}",
            flush=True,
        )
        for step in model["increases"]:
            print(
                f"        d={step['from_d']:+d} -> {step['to_d']:+d}: "
                f"{step['from_value']:.4f} -> {step['to_value']:.4f} "
                f"(+{step['increase']:.4f}, n={step['n_from']}/{step['n_to']})",
                flush=True,
            )
    print(f"    ERRATA REQUIRED: {issue_a['errata_required']}", flush=True)

    # ---- Issue B -----------------------------------------------------------
    issue_b = day1_audit.audit_source_provenance(
        records["preregistration"], records["integrity"], records["run_records"]
    )
    print("", flush=True)
    print("--- Issue B: post-freeze source provenance", flush=True)
    print(
        f"    preregistered analysis files changed: "
        f"{issue_b['preregistered_files_changed'] or 'NONE'}",
        flush=True,
    )
    print(
        f"    files only in the final manifest    : "
        f"{issue_b['files_only_in_final'] or 'NONE'}",
        flush=True,
    )
    print(
        f"    all inference runs pinned to freeze : "
        f"{issue_b['all_runs_pinned_to_freeze']}",
        flush=True,
    )

    # ---- Issue C -----------------------------------------------------------
    issue_c = day1_audit.audit_figure_integrity(
        records["integrity"], records["results_text"], root
    )
    print("", flush=True)
    print("--- Issue C: figure integrity hash", flush=True)
    print(f"    figures present               : {issue_c['n_figures_present']}", flush=True)
    print(
        f"    manifest combined is empty-in : "
        f"{issue_c['manifest_combined_is_empty_input']}",
        flush=True,
    )
    print(
        f"    results doc quotes empty-input: "
        f"{issue_c['results_document_quotes_empty_input']}",
        flush=True,
    )
    print(f"    defect present                : {issue_c['defect_present']}", flush=True)

    addendum_path = root / ADDENDUM_PATH
    if addendum_path.is_file() and not args.force_addendum:
        print(f"    addendum already present at {ADDENDUM_PATH}", flush=True)
    else:
        provenance.write_json(
            addendum_path,
            {
                "created_at": provenance.utc_now(),
                "document": "FormalCRRC Day-1 integrity addendum",
                "status": "ADDITIVE",
                "relationship_to_original": (
                    "Additive only. artifacts/day1/integrity_manifest.json is "
                    "left exactly as written on 2026-09-07 and is not corrected "
                    "in place. docs/RESULTS_DAY1.md is likewise unchanged. This "
                    "file supplies correctly computed figure hashes and records "
                    "the defect for the historical record."
                ),
                "defect": {
                    "summary": (
                        "docs/RESULTS_DAY1.md reports a combined figure hash equal "
                        "to the SHA-256 of the empty string, because the results "
                        "document was generated from an integrity manifest built "
                        "before the figures existed."
                    ),
                    "empty_input_sha256": issue_c["empty_input_sha256"],
                    "quoted_in_results_document": issue_c[
                        "results_document_quoted_combined_sha256"
                    ],
                    "manifest_was_correct": not issue_c[
                        "manifest_combined_is_empty_input"
                    ],
                    "affects_scientific_outcomes": False,
                },
                "figures": issue_c["figures"],
                "n_figures": issue_c["n_figures_present"],
                "combined_sha256": issue_c["recomputed_combined_sha256"],
                "combined_hash_definition": (
                    "sha256 over newline-joined '<repo-relative path>:<sha256>' "
                    "entries, sorted by path; an empty file list is refused"
                ),
            },
        )
        print(f"    wrote {ADDENDUM_PATH}", flush=True)

    # ---- findings ----------------------------------------------------------
    findings = {
        "created_at": provenance.utc_now(),
        "day1_baseline": {
            "path": BASELINE_PATH,
            "n_files": baseline["n_files"],
            "combined_sha256": baseline["combined_sha256"],
            "verification": verification,
        },
        "issue_a_boundary_curve_claim": issue_a,
        "issue_b_source_provenance": issue_b,
        "issue_c_figure_integrity": issue_c,
    }
    provenance.write_json(root / FINDINGS_PATH, findings)
    print("", flush=True)
    print(f"wrote {FINDINGS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
