#!/usr/bin/env python
"""Fail-closed readiness gate for Day-2 GPU inference.

Day 2 may proceed to study inference only when:

```text
DAY1_IMMUTABILITY       = PASS
DAY1_PROVENANCE         = CLEAN or EXPLAINED_NONMATERIAL
DAY1_ERRATA_STATUS      = DOCUMENTED or NONE_REQUIRED
DAY1_INTEGRITY_ADDENDUM = COMPLETE
```

plus the Day-2 preconditions: frozen partitions, verified disjointness, a frozen
preregistration, a prompt hash identical to Day 1, and models available at the
exact Day-1 revisions with unchanged scoring conditions.

Exits non-zero if anything is missing. Nothing here scores a study prompt, fits a
calibration parameter, or computes an outcome.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import dataset as ds  # noqa: E402
from formalcrrc import day1_audit, day2_config as d2c  # noqa: E402
from formalcrrc import prompts, provenance  # noqa: E402
from formalcrrc.config import MODEL_IDS, N_ARTIFACTS, N_ROWS  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"

DAY1_AUDIT_DOCS = (
    "docs/DAY1_ERRATA.md",
    "docs/DAY1_PROVENANCE_AUDIT.md",
    "docs/DAY1_INTEGRITY_ADDENDUM.md",
    "docs/DAY2_RATIONALE.md",
    "docs/NOVELTY_BOUNDARIES_DAY2.md",
)


class Gate:
    """Accumulates named checks and reports whether all of them passed."""

    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, PASS if ok else FAIL, detail))
        return ok

    def report(self) -> bool:
        width = max(len(name) for name, _, _ in self.results)
        for name, status, detail in self.results:
            suffix = f"  {detail}" if detail else ""
            print(f"{name.ljust(width)}  {status}{suffix}", flush=True)
        return all(status == PASS for _, status, _ in self.results)


def check_day1_immutability(gate: Gate, root: Path) -> str:
    baseline_path = root / d2c.DAY1_BASELINE_PATH
    if not baseline_path.is_file():
        gate.check("day1 baseline fingerprint", False, "run audit_day1_for_day2.py")
        return FAIL
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    gate.check("day1 baseline fingerprint", True, f"{baseline['n_files']} files")

    verification = day1_audit.verify_day1_baseline(baseline, root)
    gate.check(
        "day1 files unchanged",
        not verification["changed"] and not verification["missing"],
        f"changed={verification['changed']} missing={verification['missing']}",
    )
    permitted = {"artifacts/day1/integrity_addendum.json"}
    unexpected = [p for p in verification["added"] if p not in permitted]
    gate.check(
        "day1 gained only the addendum", not unexpected, f"added={unexpected}"
    )
    return verification["status"]


def check_day1_audits(gate: Gate, root: Path) -> tuple[str, str, str]:
    findings_path = root / d2c.DAY1_AUDIT_FINDINGS_PATH
    if not findings_path.is_file():
        gate.check("day1 audit findings", False, "run audit_day1_for_day2.py")
        return FAIL, FAIL, FAIL
    findings = json.loads(findings_path.read_text(encoding="utf-8"))
    gate.check("day1 audit findings", True)

    for relative in DAY1_AUDIT_DOCS:
        gate.check(f"document {relative}", (root / relative).is_file())

    issue_a = findings["issue_a_boundary_curve_claim"]
    errata_present = (root / "docs/DAY1_ERRATA.md").is_file()
    errata_status = (
        "DOCUMENTED"
        if errata_present
        else ("NONE_REQUIRED" if not issue_a["errata_required"] else "MISSING")
    )
    gate.check(
        "day1 errata status",
        errata_status in {"DOCUMENTED", "NONE_REQUIRED"},
        errata_status,
    )

    issue_b = findings["issue_b_source_provenance"]
    clean = (
        not issue_b["preregistered_files_changed"]
        and issue_b["all_runs_pinned_to_freeze"]
    )
    provenance_status = "CLEAN" if clean else "MATERIAL_DEFECT_OR_UNRESOLVED"
    gate.check(
        "day1 provenance",
        clean,
        f"{provenance_status}; changed={issue_b['preregistered_files_changed'] or 'none'}",
    )

    addendum_present = (root / "artifacts/day1/integrity_addendum.json").is_file()
    gate.check("day1 integrity addendum", addendum_present)
    addendum_status = "COMPLETE" if addendum_present else "MISSING"

    return errata_status, provenance_status, addendum_status


def check_day1_behaviour(gate: Gate, root: Path) -> None:
    rebuilt = ds.dataset_content_hash(ds.build_dataset())
    gate.check(
        "day1 dataset still reproduces",
        rebuilt == d2c.DAY1_DATASET_CONTENT_SHA256,
        rebuilt[:16],
    )
    day1_prereg = json.loads(
        (root / "artifacts/day1/preregistration.json").read_text(encoding="utf-8")
    )
    gate.check(
        "prompt template matches day1",
        prompts.prompt_template_sha256() == day1_prereg["prompt"]["template_sha256"],
        prompts.prompt_template_sha256()[:16],
    )


def check_partitions(gate: Gate, root: Path) -> None:
    for partition, manifest_path in d2c.MANIFEST_PATHS.items():
        candidate = root / manifest_path
        if not candidate.is_file():
            gate.check(f"{partition} manifest", False, "run generate_day2_datasets.py")
            continue
        manifest = json.loads(candidate.read_text(encoding="utf-8"))
        gate.check(f"{partition} manifest", True, f"seed={manifest['seed']}")
        gate.check(
            f"{partition} artifact count",
            manifest["dimensions"]["n_artifacts"] == N_ARTIFACTS,
        )
        gate.check(
            f"{partition} row count", manifest["dimensions"]["n_rows"] == N_ROWS
        )
        dataset_path = root / d2c.DATASET_PATHS[partition]
        prompt_path = root / d2c.PROMPT_PATHS[partition]
        gate.check(
            f"{partition} dataset hash",
            dataset_path.is_file()
            and provenance.sha256_file(dataset_path)
            == manifest["hashes"]["dataset_file_sha256"],
        )
        gate.check(
            f"{partition} prompt manifest hash",
            prompt_path.is_file()
            and provenance.sha256_file(prompt_path)
            == manifest["hashes"]["prompt_manifest_file_sha256"],
        )
        gate.check(
            f"{partition} no partition leakage",
            manifest["leakage_checks"]["artifact_id_in_prompt"] == 0
            and manifest["leakage_checks"]["partition_wording_in_prompt"] == 0,
        )

    cal_manifest_path = root / d2c.CALIBRATION_MANIFEST_PATH
    if cal_manifest_path.is_file():
        cross = json.loads(cal_manifest_path.read_text(encoding="utf-8"))[
            "cross_partition"
        ]
        for group in (
            "artifact_id_overlap",
            "rendered_prompt_overlap",
            "family_c_target_pair_overlap",
        ):
            gate.check(
                f"disjoint: {group}",
                all(v == 0 for v in cross[group].values()),
                str(cross[group]),
            )


def check_preregistration(gate: Gate, root: Path) -> dict:
    prereg = root / d2c.PREREG_JSON_PATH
    sha_path = root / d2c.PREREG_SHA_PATH
    if not (prereg.is_file() and sha_path.is_file()):
        gate.check("day2 preregistration frozen", False, "missing json or sha256")
        return {}
    digest = sha_path.read_text(encoding="utf-8").split()[0]
    gate.check("day2 preregistration frozen", True, f"sha256={digest[:16]}")
    gate.check(
        "day2 preregistration checksum",
        provenance.verify_checksum_file(prereg, sha_path),
    )
    document = json.loads(prereg.read_text(encoding="utf-8"))
    gate.check(
        "day2 panel intact",
        document["judge_models"]["model_ids"] == list(MODEL_IDS),
    )
    gate.check(
        "day2 prereg pins the partitions",
        all(
            document["datasets"][p]["content_sha256"]
            == json.loads(
                (root / d2c.MANIFEST_PATHS[p]).read_text(encoding="utf-8")
            )["hashes"]["dataset_content_sha256"]
            for p in d2c.MANIFEST_PATHS
        ),
    )
    gate.check(
        "day2 calibration model is intercept-only",
        document["calibration"]["primary_free_parameters_per_judge"] == 1
        and document["calibration"]["slope"] == 1.0,
    )
    return document


def check_models(gate: Gate, root: Path) -> list[str]:
    path = root / d2c.MODEL_PROVENANCE_PATH
    if not path.is_file():
        gate.check("day2 model provenance", False, "run audit_day2_models.py")
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    gate.check("day2 model provenance", True)
    gate.check(
        "day2 prompt continuity", bool(document.get("prompt_template_matches_day1"))
    )

    day1_models = json.loads(
        (root / "artifacts/day1/model_provenance.json").read_text(encoding="utf-8")
    )["models"]

    available: list[str] = []
    for model_id in MODEL_IDS:
        record = document.get("models", {}).get(model_id, {})
        status = record.get("status", "NOT_CHECKED")
        if status != "AVAILABLE":
            gate.check(f"model {model_id}", False, status)
            continue
        continuity = record.get("continuity", {})
        same_revision = record.get("revision") == day1_models[model_id]["revision"]
        ok = all(continuity.values()) and same_revision
        if gate.check(
            f"model {model_id}",
            ok,
            f"rev={record['revision'][:12]} continuity={all(continuity.values())} "
            f"same_revision_as_day1={same_revision}",
        ):
            available.append(model_id)
    return available


def check_tests(gate: Gate, root: Path, skip: bool) -> None:
    if skip:
        gate.check("test suite", True, "skipped by request")
        return
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    gate.check("test suite", result.returncode == 0, tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--require-full-panel",
        action="store_true",
        help="fail unless all four judges are available at their Day-1 revisions",
    )
    parser.add_argument(
        "--stage",
        choices=["pre-prereg", "pre-inference"],
        default="pre-inference",
        help="pre-prereg skips the frozen-preregistration checks",
    )
    args = parser.parse_args()
    root = Path(args.root)

    gate = Gate()
    immutability = check_day1_immutability(gate, root)
    errata, day1_provenance, addendum = check_day1_audits(gate, root)
    check_day1_behaviour(gate, root)
    check_partitions(gate, root)
    if args.stage == "pre-inference":
        check_preregistration(gate, root)
    available = check_models(gate, root)
    check_tests(gate, root, args.skip_tests)

    print("", flush=True)
    ready = gate.report()

    print("", flush=True)
    print(f"DAY1_IMMUTABILITY       = {immutability}", flush=True)
    print(f"DAY1_PROVENANCE         = {day1_provenance}", flush=True)
    print(f"DAY1_ERRATA_STATUS      = {errata}", flush=True)
    print(f"DAY1_INTEGRITY_ADDENDUM = {addendum}", flush=True)
    print(f"models available        = {len(available)}/{len(MODEL_IDS)}", flush=True)

    if args.require_full_panel and len(available) != len(MODEL_IDS):
        print("DAY2_READY: NO (full panel required)", flush=True)
        return 1
    if not available:
        print("DAY2_READY: NO (no available judge model)", flush=True)
        return 1
    print(f"DAY2_READY: {'YES' if ready else 'NO'}", flush=True)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
