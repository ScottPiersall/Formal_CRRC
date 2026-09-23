"""Read-only audit of the completed Day-1 record.

Day 1 is finished and immutable. Nothing in this module writes to
``artifacts/day1`` except the additive integrity addendum, and nothing rewrites
a Day-1 document. The three audits here answer questions that Day 2 depends on:

* whether the Day-1 prose claim about boundary-aligned curves is literally true,
* whether any post-freeze source change touched a Day-1 scientific outcome,
* whether the Day-1 figure-integrity hash was computed over real input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from formalcrrc import provenance

#: Files whose bytes define the completed Day-1 experiment.
DAY1_IMMUTABLE_TREES: tuple[str, ...] = ("artifacts/day1", "figures/day1")
DAY1_IMMUTABLE_DOCS: tuple[str, ...] = (
    "docs/PREREGISTRATION_DAY1.md",
    "docs/RESULTS_DAY1.md",
    "docs/LINEAGE.md",
    "docs/NOVELTY_BOUNDARIES.md",
    "docs/FORMALCRRC_DAY1_SPEC.md",
)

#: SHA-256 of the empty byte string -- the signature of a hash taken over no input.
EMPTY_SHA256: str = hashlib.sha256(b"").hexdigest()


# --------------------------------------------------------------------------
# Day-1 fingerprint
# --------------------------------------------------------------------------


def _walk(root: Path, relative_dir: str) -> Iterable[Path]:
    base = root / relative_dir
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file())


def build_day1_baseline(root: Path | None = None) -> dict[str, Any]:
    """Fingerprint every immutable Day-1 file.

    The result is the reference against which Day 2 re-verifies, at the end,
    that it changed nothing belonging to Day 1.
    """
    root = root or provenance.repo_root()
    files: dict[str, str] = {}

    for tree in DAY1_IMMUTABLE_TREES:
        for path in _walk(root, tree):
            files[path.relative_to(root).as_posix()] = provenance.sha256_file(path)
    for relative in DAY1_IMMUTABLE_DOCS:
        candidate = root / relative
        if candidate.is_file():
            files[relative] = provenance.sha256_file(candidate)

    return {
        "created_at": provenance.utc_now(),
        "purpose": (
            "Immutable fingerprint of the completed Day-1 experiment. Day-2 code "
            "never writes to these paths; the only permitted addition under "
            "artifacts/day1 is integrity_addendum.json, which is additive and "
            "overwrites nothing."
        ),
        "trees": list(DAY1_IMMUTABLE_TREES),
        "documents": list(DAY1_IMMUTABLE_DOCS),
        "n_files": len(files),
        "files": files,
        "combined_sha256": combined_hash(files),
    }


def combined_hash(files: dict[str, str]) -> str:
    """Deterministic combined digest over an ordered ``(path, sha256)`` listing.

    Refuses an empty listing: a combined hash over no input is the defect this
    module exists partly to catch, and it must never be produced silently.
    """
    if not files:
        raise ValueError("refusing to compute a combined hash over an empty file list")
    payload = "\n".join(f"{name}:{digest}" for name, digest in sorted(files.items()))
    return provenance.sha256_text(payload)


def verify_day1_baseline(
    baseline: dict[str, Any], root: Path | None = None
) -> dict[str, Any]:
    """Re-check a stored fingerprint against the working tree."""
    root = root or provenance.repo_root()
    recorded: dict[str, str] = baseline["files"]

    changed: list[str] = []
    missing: list[str] = []
    for relative, digest in sorted(recorded.items()):
        candidate = root / relative
        if not candidate.is_file():
            missing.append(relative)
        elif provenance.sha256_file(candidate) != digest:
            changed.append(relative)

    current: dict[str, str] = {}
    for tree in DAY1_IMMUTABLE_TREES:
        for path in _walk(root, tree):
            current[path.relative_to(root).as_posix()] = provenance.sha256_file(path)
    for relative in DAY1_IMMUTABLE_DOCS:
        if (root / relative).is_file():
            current[relative] = provenance.sha256_file(root / relative)

    added = sorted(set(current) - set(recorded))
    ok = not changed and not missing

    return {
        "checked_at": provenance.utc_now(),
        "n_recorded": len(recorded),
        "changed": changed,
        "missing": missing,
        "added": added,
        "status": "PASS" if ok else "FAIL",
    }


# --------------------------------------------------------------------------
# Issue A -- boundary-aligned monotonicity claim
# --------------------------------------------------------------------------


def audit_boundary_curve_claim(summary: dict[str, Any]) -> dict[str, Any]:
    """Test the Day-1 prose claim that every aligned curve decreases.

    The claim is about the boundary-aligned mean curves recorded in
    ``summary.json``. Checked literally, step by step.
    """
    curve = pd.DataFrame(summary["boundary_aligned_curve"])
    findings: list[dict[str, Any]] = []

    for model_id, block in summary["models"].items():
        subset = curve[curve["model_id"] == model_id].sort_values("signed_distance")
        distances = subset["signed_distance"].to_numpy()
        values = subset["mean_p_met"].to_numpy(dtype=float)
        steps = np.diff(values)
        increases = [
            {
                "from_d": int(distances[i]),
                "to_d": int(distances[i + 1]),
                "from_value": round(float(values[i]), 6),
                "to_value": round(float(values[i + 1]), 6),
                "increase": round(float(steps[i]), 6),
                "n_from": int(subset["n"].to_numpy()[i]),
                "n_to": int(subset["n"].to_numpy()[i + 1]),
            }
            for i in range(len(steps))
            if steps[i] > 0
        ]
        findings.append(
            {
                "model_id": model_id,
                "short_name": block["short_name"],
                "n_steps": int(len(steps)),
                "n_increases": len(increases),
                "monotone_non_increasing": not increases,
                "increases": increases,
                "first_value": round(float(values[0]), 6),
                "last_value": round(float(values[-1]), 6),
                "net_change": round(float(values[-1] - values[0]), 6),
                "mean_spearman": block["overall"]["spearman_mean"],
            }
        )

    all_monotone = all(f["monotone_non_increasing"] for f in findings)
    return {
        "claim": (
            "every boundary-aligned curve decreases with strictness, and every "
            "judge has a negative mean within-artifact Spearman association"
        ),
        "claim_location": "docs/RESULTS_DAY1.md, section 1 (Executive summary)",
        "first_clause_literally_true": all_monotone,
        "second_clause_literally_true": all(
            f["mean_spearman"] < 0 for f in findings
        ),
        "models": findings,
        "errata_required": not all_monotone,
    }


# --------------------------------------------------------------------------
# Issue B -- post-freeze source provenance
# --------------------------------------------------------------------------

#: How each post-freeze change is classified.
CHANGE_CATEGORIES: tuple[str, ...] = (
    "documentation_only",
    "output_artifact_generation",
    "infrastructure",
    "inference_implementation",
    "metric_implementation",
    "statistical_analysis",
    "figure_generation",
    "other",
)


def audit_source_provenance(
    preregistration: dict[str, Any],
    integrity: dict[str, Any],
    run_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare the source state at freeze with the state at final integrity.

    The decisive question is whether any file that *computes* a Day-1 outcome
    changed after the preregistration was frozen. The preregistration pins the
    analysis code paths by hash, and every inference run record pins the subset
    of files it used, so the question is answerable exactly rather than by
    reading diffs.
    """
    at_freeze: dict[str, str] = preregistration["source_manifest_at_freeze"]["files"]
    at_final: dict[str, str] = integrity["source_manifest"]["files"]

    shared = sorted(set(at_freeze) & set(at_final))
    changed = [f for f in shared if at_freeze[f] != at_final[f]]
    only_final = sorted(set(at_final) - set(at_freeze))
    only_freeze = sorted(set(at_freeze) - set(at_final))

    run_checks: dict[str, Any] = {}
    for slug, record in run_records.items():
        used = (record.get("source_manifest") or {}).get("files", {})
        mismatches = [f for f, h in used.items() if at_freeze.get(f) != h]
        run_checks[slug] = {
            "n_files_pinned": len(used),
            "all_match_freeze": not mismatches,
            "mismatches": mismatches,
        }

    return {
        "tree_sha256_at_freeze": preregistration["source_manifest_at_freeze"][
            "tree_sha256"
        ],
        "tree_sha256_at_final": integrity["source_manifest"]["tree_sha256"],
        "tree_hashes_differ": preregistration["source_manifest_at_freeze"][
            "tree_sha256"
        ]
        != integrity["source_manifest"]["tree_sha256"],
        "n_files_at_freeze": len(at_freeze),
        "n_files_at_final": len(at_final),
        "files_only_in_final": only_final,
        "files_only_in_freeze": only_freeze,
        "preregistered_files_changed": changed,
        "preregistered_files_unchanged": not changed,
        "inference_run_pins": run_checks,
        "all_runs_pinned_to_freeze": all(
            c["all_match_freeze"] for c in run_checks.values()
        ),
        "git_at_freeze": preregistration.get("git_at_freeze"),
        "git_at_final": integrity.get("git"),
    }


# --------------------------------------------------------------------------
# Issue C -- figure integrity hash
# --------------------------------------------------------------------------


def audit_figure_integrity(
    integrity: dict[str, Any], results_text: str, root: Path | None = None
) -> dict[str, Any]:
    """Recompute the Day-1 figure hashes and locate any empty-input digest."""
    root = root or provenance.repo_root()
    figures = {
        path.relative_to(root).as_posix(): provenance.sha256_file(path)
        for path in sorted((root / "figures/day1").glob("*.png"))
    }

    recorded = integrity.get("figures", {})
    recorded_combined = recorded.get("combined_sha256")

    quoted: list[str] = []
    for line in results_text.splitlines():
        if "figures (combined)" in line:
            for token in line.replace("|", " ").split():
                candidate = token.strip("`")
                if len(candidate) == 64 and all(
                    ch in "0123456789abcdef" for ch in candidate
                ):
                    quoted.append(candidate)

    return {
        "n_figures_present": len(figures),
        "figures": figures,
        "recomputed_combined_sha256": combined_hash(figures) if figures else None,
        "manifest_n_files": len(recorded.get("files", {})),
        "manifest_combined_sha256": recorded_combined,
        "manifest_combined_is_empty_input": recorded_combined == EMPTY_SHA256,
        "manifest_matches_recomputation": (
            recorded_combined == combined_hash(figures) if figures else None
        ),
        "results_document_quoted_combined_sha256": quoted,
        "results_document_quotes_empty_input": EMPTY_SHA256 in quoted,
        "empty_input_sha256": EMPTY_SHA256,
        "defect_present": EMPTY_SHA256 in quoted or recorded_combined == EMPTY_SHA256,
    }


def load_day1_records(root: Path | None = None) -> dict[str, Any]:
    """Load the Day-1 JSON records the audits read."""
    root = root or provenance.repo_root()
    day1 = root / "artifacts/day1"
    runs = {
        path.stem.replace("_run", ""): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((day1 / "raw_scores").glob("*_run.json"))
    }
    return {
        "preregistration": json.loads(
            (day1 / "preregistration.json").read_text(encoding="utf-8")
        ),
        "integrity": json.loads(
            (day1 / "integrity_manifest.json").read_text(encoding="utf-8")
        ),
        "summary": json.loads((day1 / "summary.json").read_text(encoding="utf-8")),
        "run_records": runs,
        "results_text": (root / "docs/RESULTS_DAY1.md").read_text(encoding="utf-8"),
    }
