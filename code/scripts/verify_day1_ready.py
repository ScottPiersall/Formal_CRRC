#!/usr/bin/env python
"""Fail-closed readiness gate for Day-1 GPU inference (Phase D/E boundary).

Checks, in order:

1. repository state and source manifest,
2. the full test suite,
3. the frozen preregistration and its checksum,
4. the dataset file hash, the prompt manifest hash, and that the dataset
   rebuilds to the same content hash from the seed,
5. model availability, resolved revisions and label tokenisation.

Exits non-zero if any prerequisite is missing. Nothing here scores a study
prompt or computes an outcome.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import dataset as ds  # noqa: E402
from formalcrrc import prompts, provenance  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    ANALYSIS_CODE_PATHS,
    DATASET_MANIFEST_PATH,
    DATASET_PATH,
    MODEL_IDS,
    MODEL_PROVENANCE_PATH,
    N_ARTIFACTS,
    N_ROWS,
    PREREG_JSON_PATH,
    PREREG_SHA_PATH,
    PROMPT_MANIFEST_PATH,
)

PASS = "PASS"
FAIL = "FAIL"


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


def check_repository(gate: Gate, root: Path) -> None:
    manifest = provenance.source_manifest(ANALYSIS_CODE_PATHS, root=root)
    missing = [p for p, h in manifest["files"].items() if h is None]
    gate.check(
        "source files present",
        not missing,
        f"tree={manifest['tree_sha256'][:12]}" if not missing else f"missing={missing}",
    )
    git = provenance.git_state(root)
    gate.check(
        "git state recorded",
        True,
        f"commit={(git.get('commit') or 'unavailable')[:12]} clean={git.get('clean')}",
    )


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


def check_preregistration(gate: Gate, root: Path) -> dict:
    prereg = root / PREREG_JSON_PATH
    sha_path = root / PREREG_SHA_PATH
    if not (prereg.is_file() and sha_path.is_file()):
        gate.check("preregistration frozen", False, "missing json or sha256")
        return {}
    ok = provenance.verify_checksum_file(prereg, sha_path)
    digest = sha_path.read_text(encoding="utf-8").split()[0]
    gate.check("preregistration frozen", True, f"sha256={digest[:12]}")
    gate.check("preregistration checksum", ok)
    document = json.loads(prereg.read_text(encoding="utf-8"))
    gate.check(
        "preregistered panel intact",
        document.get("judge_models", {}).get("model_ids") == list(MODEL_IDS),
    )
    return document


def check_dataset(gate: Gate, root: Path, document: dict) -> None:
    manifest_path = root / DATASET_MANIFEST_PATH
    dataset_path = root / DATASET_PATH
    prompt_path = root / PROMPT_MANIFEST_PATH
    if not (manifest_path.is_file() and dataset_path.is_file()):
        gate.check("dataset present", False, "missing dataset or manifest")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate.check("dataset present", True)
    gate.check(
        "dataset file hash",
        provenance.sha256_file(dataset_path)
        == manifest["hashes"]["dataset_file_sha256"],
    )
    gate.check(
        "prompt manifest hash",
        prompt_path.is_file()
        and provenance.sha256_file(prompt_path)
        == manifest["hashes"]["prompt_manifest_file_sha256"],
    )

    rebuilt = ds.build_dataset()
    gate.check(
        "dataset rebuilds from seed",
        ds.dataset_content_hash(rebuilt)
        == manifest["hashes"]["dataset_content_sha256"],
    )
    gate.check("artifact count", rebuilt["artifact_id"].nunique() == N_ARTIFACTS)
    gate.check("rubric-response pair count", len(rebuilt) == N_ROWS)
    gate.check(
        "prompt template hash",
        prompts.prompt_template_sha256()
        == manifest["hashes"]["prompt_template_sha256"],
    )
    if document:
        gate.check(
            "preregistration pins this dataset",
            document["dataset"]["content_sha256"]
            == manifest["hashes"]["dataset_content_sha256"],
        )


def check_label_tokens(gate: Gate, root: Path) -> None:
    """The scored logit index must be the token the model would actually emit."""
    path = root / "artifacts/day1/label_token_check.json"
    if not path.is_file():
        gate.check(
            "label token check",
            False,
            "run scripts/verify_label_tokens.py",
        )
        return
    document = json.loads(path.read_text(encoding="utf-8"))
    gate.check(
        "label token check",
        bool(document.get("all_ok")),
        f"{len(document.get('models', []))} models verified",
    )
    for report in document.get("models", []):
        gate.check(
            f"label tokens {report['model_id']}",
            report.get("verdict") == "OK",
            f"single_token={report.get('single_token')} "
            f"stable={report.get('stable_across_study_prompts')} "
            f"differs_from_naive={report.get('differs_from_naive_encoding')}",
        )


def check_models(gate: Gate, root: Path) -> list[str]:
    path = root / MODEL_PROVENANCE_PATH
    if not path.is_file():
        gate.check("model provenance present", False, "run audit_environment.py")
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    gate.check("model provenance present", True)

    available: list[str] = []
    for model_id in MODEL_IDS:
        record = document.get("models", {}).get(model_id, {})
        status = record.get("status", "NOT_CHECKED")
        if status != "AVAILABLE":
            gate.check(f"model {model_id}", False, status)
            continue
        problems = provenance.validate_model_provenance(record)
        detail = (
            f"revision={record['revision'][:12]} "
            f"scoring={record['label_tokenization']['scoring_method']}"
        )
        if gate.check(f"model {model_id}", not problems, detail or str(problems)):
            available.append(model_id)
    return available


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--require-full-panel",
        action="store_true",
        help="fail unless all four preregistered judges are available",
    )
    args = parser.parse_args()
    root = Path(args.root)

    gate = Gate()
    check_repository(gate, root)
    check_tests(gate, root, args.skip_tests)
    document = check_preregistration(gate, root)
    check_dataset(gate, root, document)
    available = check_models(gate, root)
    check_label_tokens(gate, root)

    print("", flush=True)
    ready = gate.report()
    print("", flush=True)
    print(f"models available: {len(available)}/{len(MODEL_IDS)}", flush=True)

    if args.require_full_panel and len(available) != len(MODEL_IDS):
        print("READY: NO (full panel required)", flush=True)
        return 1
    if not available:
        print("READY: NO (no available judge model)", flush=True)
        return 1
    print(f"READY: {'YES' if ready else 'NO'}", flush=True)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
