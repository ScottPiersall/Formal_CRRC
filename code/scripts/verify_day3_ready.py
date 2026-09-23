#!/usr/bin/env python
"""Fail-closed readiness gate for Day-3 GPU inference.

Day 3 may proceed to study inference only when:

```text
PRIOR_EXPERIMENT_IMMUTABILITY = PASS
DAY3_DATASET                  = FROZEN and DISJOINT
ORACLE_IMPLEMENTATION         = PASS  (theorem == enumeration)
MODEL_CONTINUITY              = PASS
PROMPT_CONTINUITY             = PASS
DAY3_PREREGISTRATION          = FROZEN
```

Exits non-zero if anything is missing. Nothing here scores a study prompt or
computes an outcome.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import dataset as ds  # noqa: E402
from formalcrrc import day3, day3_config as d3c  # noqa: E402
from formalcrrc import prompts, provenance  # noqa: E402
from formalcrrc.config import MODEL_IDS, N_ARTIFACTS, N_ROWS, N_STRICTNESS  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"

DAY3_DOCS = ("docs/DAY3_RATIONALE.md", "docs/NOVELTY_BOUNDARIES_DAY3.md")


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


def check_priors(gate: Gate, root: Path) -> str:
    manifest_path = root / d3c.PRIOR_MANIFEST_PATH
    if not manifest_path.is_file():
        gate.check("prior fingerprint", False, "run audit_day3_priors.py")
        return FAIL
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate.check(
        "prior fingerprint",
        True,
        f"{manifest['n_files']} files "
        f"({manifest['by_experiment']['day1']['n_files']} Day-1, "
        f"{manifest['by_experiment']['day2']['n_files']} Day-2)",
    )
    verification = day3.verify_prior_manifest(manifest, root)
    gate.check(
        "day1 files unchanged",
        verification["day1"]["status"] == PASS,
        f"changed={verification['day1']['changed']}",
    )
    gate.check(
        "day2 files unchanged",
        verification["day2"]["status"] == PASS,
        f"changed={verification['day2']['changed']}",
    )
    gate.check("no prior files added", not verification["added"], str(verification["added"]))
    gate.check(
        "day1 dataset still reproduces",
        ds.dataset_content_hash(ds.build_dataset())
        == d3c.DAY1_DATASET_CONTENT_SHA256,
    )
    for relative in DAY3_DOCS:
        gate.check(f"document {relative}", (root / relative).is_file())
    return verification["status"]


def check_dataset(gate: Gate, root: Path) -> None:
    manifest_path = root / d3c.DATASET_MANIFEST_PATH
    if not manifest_path.is_file():
        gate.check("day3 dataset", False, "run generate_day3_dataset.py")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate.check("day3 dataset", True, f"seed={manifest['seed']}")
    gate.check("artifact count", manifest["dimensions"]["n_artifacts"] == N_ARTIFACTS)
    gate.check("row count", manifest["dimensions"]["n_rows"] == N_ROWS)

    dataset_path = root / d3c.DATASET_PATH
    prompt_path = root / d3c.PROMPT_MANIFEST_PATH
    gate.check(
        "dataset file hash",
        dataset_path.is_file()
        and provenance.sha256_file(dataset_path)
        == manifest["hashes"]["dataset_file_sha256"],
    )
    gate.check(
        "prompt manifest hash",
        prompt_path.is_file()
        and provenance.sha256_file(prompt_path)
        == manifest["hashes"]["prompt_manifest_file_sha256"],
    )
    gate.check(
        "no partition leakage",
        manifest["leakage_checks"]["artifact_id_in_prompt"] == 0
        and manifest["leakage_checks"]["partition_wording_in_prompt"] == 0,
    )
    disjoint = manifest["disjointness"]
    for group in (
        "artifact_id_overlap",
        "rendered_prompt_overlap",
        "family_c_target_pair_overlap",
    ):
        gate.check(
            f"disjoint: {group}",
            all(v == 0 for v in disjoint[group].values()),
            str(disjoint[group]),
        )


def check_oracle_implementation(gate: Gate) -> str:
    """Theorem and enumeration must agree, checked on random and edge curves."""
    rng = np.random.default_rng(20260908)
    disagreements = 0
    for _ in range(2000):
        curve = rng.normal(0.0, 4.0, N_STRICTNESS)
        if day3.reachable_crossings(curve) != day3.reachable_crossings_by_enumeration(
            curve
        ):
            disagreements += 1
    for curve in (
        np.arange(8, -1, -1.0),
        np.arange(0, 9, 1.0),
        np.ones(N_STRICTNESS),
        np.array([8.0, 9, 6, 7, 5, 6, 7, 3, 4]),
    ):
        if day3.reachable_crossings(curve) != day3.reachable_crossings_by_enumeration(
            curve
        ):
            disagreements += 1
    for _ in range(1000):
        curve = rng.integers(-3, 4, N_STRICTNESS).astype(float)
        if day3.reachable_crossings(curve) != day3.reachable_crossings_by_enumeration(
            curve
        ):
            disagreements += 1

    ok = disagreements == 0
    gate.check(
        "oracle implementation", ok, f"{disagreements} disagreements over 3004 curves"
    )
    return PASS if ok else FAIL


def check_preregistration(gate: Gate, root: Path) -> None:
    prereg = root / d3c.PREREG_JSON_PATH
    sha_path = root / d3c.PREREG_SHA_PATH
    if not (prereg.is_file() and sha_path.is_file()):
        gate.check("day3 preregistration frozen", False, "missing json or sha256")
        return
    digest = sha_path.read_text(encoding="utf-8").split()[0]
    gate.check("day3 preregistration frozen", True, f"sha256={digest[:16]}")
    gate.check(
        "day3 preregistration checksum",
        provenance.verify_checksum_file(prereg, sha_path),
    )
    document = json.loads(prereg.read_text(encoding="utf-8"))
    gate.check(
        "day3 panel intact",
        document["judge_models"]["model_ids"] == list(MODEL_IDS),
    )
    manifest = json.loads(
        (root / d3c.DATASET_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    gate.check(
        "day3 prereg pins the dataset",
        document["dataset"]["content_sha256"]
        == manifest["hashes"]["dataset_content_sha256"],
    )
    gate.check(
        "translation class is additive only",
        document["formal_model"]["translation_class"] == d3c.TRANSLATION_CLASS,
    )


def check_models(gate: Gate, root: Path) -> tuple[list[str], str]:
    path = root / d3c.MODEL_PROVENANCE_PATH
    if not path.is_file():
        gate.check("day3 model provenance", False, "run audit_day3_models.py")
        return [], FAIL
    document = json.loads(path.read_text(encoding="utf-8"))
    gate.check("day3 model provenance", True)
    prompt_ok = bool(document.get("prompt_template_matches_prior"))
    gate.check("prompt continuity", prompt_ok, prompts.prompt_template_sha256()[:16])

    day1_models = json.loads(
        (root / "artifacts/day1/model_provenance.json").read_text(encoding="utf-8")
    )["models"]

    available: list[str] = []
    smoke_identical = True
    for model_id in MODEL_IDS:
        record = document.get("models", {}).get(model_id, {})
        status = record.get("status", "NOT_CHECKED")
        if status != "AVAILABLE":
            gate.check(f"model {model_id}", False, status)
            continue
        continuity = record.get("continuity", {})
        same_revision = record.get("revision") == day1_models[model_id]["revision"]
        reproduces = record.get("smoke_test", {}).get("reproduces_day1_scores")
        smoke_identical = smoke_identical and bool(reproduces)
        if gate.check(
            f"model {model_id}",
            all(continuity.values()) and same_revision,
            f"rev={record['revision'][:12]} continuity={all(continuity.values())} "
            f"smoke_identical={reproduces}",
        ):
            available.append(model_id)
    return available, (PASS if smoke_identical and prompt_ok else FAIL)


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
    parser.add_argument("--require-full-panel", action="store_true")
    parser.add_argument(
        "--stage",
        choices=["pre-prereg", "pre-inference"],
        default="pre-inference",
    )
    args = parser.parse_args()
    root = Path(args.root)

    gate = Gate()
    immutability = check_priors(gate, root)
    check_dataset(gate, root)
    oracle = check_oracle_implementation(gate)
    if args.stage == "pre-inference":
        check_preregistration(gate, root)
    available, continuity = check_models(gate, root)
    check_tests(gate, root, args.skip_tests)

    print("", flush=True)
    ready = gate.report()

    print("", flush=True)
    print(f"PRIOR_EXPERIMENT_IMMUTABILITY = {immutability}", flush=True)
    print(f"ORACLE_IMPLEMENTATION         = {oracle}", flush=True)
    print(f"MODEL_CONTINUITY              = {continuity}", flush=True)
    print(f"models available              = {len(available)}/{len(MODEL_IDS)}", flush=True)

    if args.require_full_panel and len(available) != len(MODEL_IDS):
        print("DAY3_READY: NO (full panel required)", flush=True)
        return 1
    if not available:
        print("DAY3_READY: NO (no available judge model)", flush=True)
        return 1
    print(f"DAY3_READY: {'YES' if ready else 'NO'}", flush=True)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
