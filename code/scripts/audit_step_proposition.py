#!/usr/bin/env python
"""Post-study theory audit: the scalar latent-estimation / step-function proposition.

**Not an experiment.** No model inference, no new partition, no preregistration.
Every frozen artifact is read-only; output is confined to
``artifacts/poststudy_theory/``.

Five things are verified:

1. All three predicate families canonicalize to a strictly increasing threshold
   ladder, and the canonical form reproduces the implemented labels exactly.
2. The proposition's assumptions are compatible with the Day-3 reachability
   theorem's exact indexing, zero-crossing, strictness and tie conventions.
3. The frozen Day-3 TRR counts, recomputed from raw margins three independent
   ways.
4. The five synthetic cases A-E.
5. That nothing frozen changed.

Run: ``python scripts/audit_step_proposition.py``
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import day3, predicates, provenance  # noqa: E402
from formalcrrc import step_proposition as sp  # noqa: E402
from formalcrrc.config import N_STRICTNESS, NO_CROSSING_INDEX  # noqa: E402
from formalcrrc.day2_bootstrap import crossing_index  # noqa: E402

ROOT = provenance.repo_root()
OUT_DIR = ROOT / "artifacts" / "poststudy_theory"

MODEL_SLUGS = {
    "Qwen/Qwen2.5-14B-Instruct": "qwen2_5_14b_instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3_1_8b_instruct",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral_7b_instruct_v0_3",
    "google/gemma-2-9b-it": "gemma_2_9b_it",
}
SHORT = {
    "Qwen/Qwen2.5-14B-Instruct": "Qwen",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral",
    "google/gemma-2-9b-it": "Gemma",
}

SYNTHETIC = {
    "A_scalar_step": [5.0, 4, 3, 2, 1, -1, -2, -3, -4],
    "B_mislocated_but_coherent": [4.0, 3, 2, 1, 0, -1, -2, -3, -4],
    "C_ordering_failure": [5.0, 4, 2, 3, 1, 0, -1, -2, -3],
    "D_tie": [5.0, 4, 4, 3, 2, 1, 0, -1, -2],
}
TRANSLATIONS = (-100.0, -3.5, -1.0, 0.0, 0.25, 2.0, 17.0, 1e6)


# --------------------------------------------------------------------------
# Independent reimplementations, used to cross-check the repository's code
# --------------------------------------------------------------------------


def reachable_from_theorem(curve: np.ndarray) -> set[int]:
    """The theorem statement, written out directly rather than imported."""
    reachable = {0, NO_CROSSING_INDEX}
    for j in range(1, N_STRICTNESS):
        if curve[j] < min(curve[:j]):
            reachable.add(j)
    return reachable


def reachable_by_scanning(curve: np.ndarray) -> set[int]:
    """No theorem at all: shift the curve and observe where it first goes negative."""
    achieved: set[int] = set()
    breakpoints = sorted({-float(m) for m in curve})
    candidates = [breakpoints[0] - 1.0, breakpoints[-1] + 1.0]
    for index, point in enumerate(breakpoints):
        candidates += [
            point,
            float(np.nextafter(point, -np.inf)),
            float(np.nextafter(point, np.inf)),
        ]
        if index + 1 < len(breakpoints):
            candidates.append((point + breakpoints[index + 1]) / 2.0)
    for alpha in candidates:
        shifted = curve + alpha
        below = shifted < 0.0
        achieved.add(int(np.argmax(below)) if below.any() else NO_CROSSING_INDEX)
    return achieved


# --------------------------------------------------------------------------
# Task 2 -- convention compatibility
# --------------------------------------------------------------------------


def convention_checks() -> dict[str, object]:
    flat = np.zeros(N_STRICTNESS)
    exact_zero = np.array([2.0, 1, 0, -1, -2, -3, -4, -5, -6])
    rng = np.random.default_rng(20260908)

    endpoints_always = True
    theorem_matches_enumeration = True
    for _ in range(2000):
        curve = rng.integers(-3, 4, size=N_STRICTNESS).astype(float)
        reachable = set(day3.reachable_crossings(curve))
        if not ({0, NO_CROSSING_INDEX} <= reachable):
            endpoints_always = False
        if reachable != reachable_from_theorem(curve):
            theorem_matches_enumeration = False
        if reachable != set(day3.reachable_crossings_by_enumeration(curve)):
            theorem_matches_enumeration = False

    return {
        "indexing": {
            "strictness_levels": "s = 0..8",
            "crossing_support": "j = 0..9, where 9 means never crossed",
            "records_enumerated_over": "j = 1..8 (level 0 has no prefix)",
            "level_zero_is_never_a_record": 0 not in day3.prefix_record_lows(
                np.array(SYNTHETIC["A_scalar_step"])
            ),
        },
        "zero_crossing": {
            "rule": "first level with margin < 0, else 9",
            "margin_zero_counts_as_met": int(
                crossing_index(exact_zero[None, :])[0]
            )
            == 3,
            "all_positive_gives_nine": int(
                crossing_index(np.ones(N_STRICTNESS)[None, :])[0]
            )
            == NO_CROSSING_INDEX,
            "all_zero_gives_nine": int(crossing_index(flat[None, :])[0])
            == NO_CROSSING_INDEX,
        },
        "strict_inequality": {
            "rule": "M(j) < min_{t<j} M(t), strict",
            "flat_curve_has_no_records": day3.prefix_record_lows(flat) == (),
            "flat_curve_reaches_only_endpoints": day3.reachable_crossings(flat)
            == (0, NO_CROSSING_INDEX),
        },
        "ties": {
            "rule": "equality never creates a reachable crossing",
            "tied_level_excluded": 2
            not in day3.reachable_crossings(np.array(SYNTHETIC["D_tie"])),
            "why_incompatible_with_the_model": (
                "The proposition assumes tau_s strictly increasing and g strictly "
                "increasing, so M(s) = g(z_hat - tau_s) is strictly decreasing and "
                "no two levels can share a margin. An observed tie is therefore "
                "already outside the model's assumptions; Day 3's strict rule "
                "records that fact rather than weakening the theorem to admit it."
            ),
        },
        "endpoints": {
            "zero_and_nine_always_reachable": endpoints_always,
            "reason": "translate far negative, or far positive",
        },
        "theorem_matches_independent_enumeration": theorem_matches_enumeration,
        "compatible": bool(endpoints_always and theorem_matches_enumeration),
    }


# --------------------------------------------------------------------------
# Task 3 -- frozen empirical counts
# --------------------------------------------------------------------------


def empirical_counts() -> dict[str, object]:
    metrics = pd.read_parquet(ROOT / "artifacts/day3/metrics_by_artifact.parquet")
    rows: list[dict[str, object]] = []
    all_agree = True

    for model_id, slug in MODEL_SLUGS.items():
        scores = pd.read_parquet(ROOT / f"artifacts/day3/raw_scores/{slug}.parquet")
        n_nontrivial = tr1 = tr0 = 0
        full_reach = strictly_decreasing = 0
        agrees_with_day3 = agrees_with_scan = True

        for _, group in scores.groupby("artifact_id"):
            group = group.sort_values("strictness_index")
            curve = group["margin"].to_numpy(dtype=float)
            family = group["family"].iloc[0]
            latent = int(group["latent_level"].iloc[0])
            # Recomputed from the predicate, not read from the stored column.
            j_star = predicates.true_first_fail_index(family, latent)

            reachable = reachable_from_theorem(curve)
            if reachable != set(day3.reachable_crossings(curve)):
                agrees_with_day3 = False
            if reachable != reachable_by_scanning(curve):
                agrees_with_scan = False
            if len(reachable) == NO_CROSSING_INDEX + 1:
                full_reach += 1
            if sp.is_strictly_decreasing(curve):
                strictly_decreasing += 1

            if 1 <= j_star <= N_STRICTNESS - 1:
                n_nontrivial += 1
                if j_star in reachable:
                    tr1 += 1
                else:
                    tr0 += 1

        trr = Fraction(tr1, n_nontrivial)
        frozen = float(
            metrics[
                (metrics["model_id"] == model_id) & (metrics["nontrivial_boundary"])
            ]["translation_reachable"].mean()
        )
        matches = abs(frozen - float(trr)) < 1e-12
        all_agree = all_agree and matches and agrees_with_day3 and agrees_with_scan

        rows.append(
            {
                "model_id": model_id,
                "short_name": SHORT[model_id],
                "n_nontrivial": n_nontrivial,
                "n_reachable": tr1,
                "n_unreachable": tr0,
                "trr_exact": str(trr),
                "trr": float(trr),
                "trr_percent": round(100 * float(trr), 4),
                "one_minus_trr": float(1 - trr),
                "one_minus_trr_percent": round(100 * float(1 - trr), 4),
                "frozen_report_trr": frozen,
                "matches_frozen_report": matches,
                "agrees_with_day3_module": agrees_with_day3,
                "agrees_with_intercept_scan": agrees_with_scan,
                "n_strictly_decreasing_curves": strictly_decreasing,
                "n_fully_reachable_curves": full_reach,
                "scalar_signature_share": round(full_reach / 324, 6),
            }
        )

    return {
        "definition_of_nontrivial": "j* in 1..8 (0 and 9 are always reachable)",
        "trr_definition": "TR_x = 1[j*_x in R_x]; nontrivial TRR = mean over j* in 1..8",
        "recomputed_from": "frozen raw margins; j* rebuilt from the predicates",
        "by_model": rows,
        "all_agree_with_frozen_report": all_agree,
    }


# --------------------------------------------------------------------------
# Task 4 -- synthetic cases
# --------------------------------------------------------------------------


def synthetic_cases() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for key, values in SYNTHETIC.items():
        curve = np.array(values, dtype=float)
        records = day3.prefix_record_lows(curve)
        reachable = day3.reachable_crossings(curve)
        invariant = all(
            day3.reachable_crossings(curve + a) == reachable
            and day3.prefix_record_lows(curve + a) == records
            for a in TRANSLATIONS
        )
        cases.append(
            {
                "case": key,
                "margins": values,
                "strictly_decreasing": sp.is_strictly_decreasing(curve),
                "prefix_record_lows": list(records),
                "reachable_crossings": list(reachable),
                "reachable_count": len(reachable),
                "raw_crossing": int(crossing_index(curve[None, :])[0]),
                "unreachable_nontrivial": [
                    j for j in range(1, N_STRICTNESS) if j not in reachable
                ],
                "translation_invariant": invariant,
                "matches_scan": set(reachable) == reachable_by_scanning(curve),
                "tie_diagnostics": day3.tie_diagnostics(curve, 2),
            }
        )

    scalar = [
        {
            "z_hat": z,
            "strictly_decreasing": sp.is_strictly_decreasing(
                sp.scalar_model_margins(float(z))
            ),
            "fully_reachable": sp.implies_full_reachability(
                sp.scalar_model_margins(float(z))
            ),
            "labels_are_step": sp.is_nonincreasing_step(
                sp.scalar_model_labels(float(z))
            ),
            "transitions": sp.transition_count(sp.scalar_model_labels(float(z))),
        }
        for z in (-1.0, 0.0, 2.5, 4.0, 9.0)
    ]

    return {
        "cases": cases,
        "scalar_model_instances": scalar,
        "all_scalar_instances_fully_reachable": all(
            r["fully_reachable"] for r in scalar
        ),
        "all_translation_invariant": all(c["translation_invariant"] for c in cases),
        "all_match_scan": all(c["matches_scan"] for c in cases),
    }


# --------------------------------------------------------------------------
# Task 6 -- integrity
# --------------------------------------------------------------------------

FROZEN_TREES = (
    "artifacts/day1",
    "artifacts/day2",
    "artifacts/day3",
    "figures/day1",
    "figures/day2",
    "figures/day3",
    "artifacts/poststudy_baselines",
)
FROZEN_DOCS = (
    "docs/PREREGISTRATION_DAY1.md",
    "docs/PREREGISTRATION_DAY2.md",
    "docs/PREREGISTRATION_DAY3.md",
    "docs/RESULTS_DAY1.md",
    "docs/RESULTS_DAY2.md",
    "docs/RESULTS_DAY3.md",
    "docs/DAY1_ERRATA.md",
    "docs/DAY1_PROVENANCE_AUDIT.md",
    "docs/DAY1_INTEGRITY_ADDENDUM.md",
    "docs/POSTSTUDY_BASELINE_AUDIT.md",
)


def frozen_fingerprint() -> dict[str, object]:
    files: dict[str, str] = {}
    for tree in FROZEN_TREES:
        for path in sorted((ROOT / tree).rglob("*")):
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = provenance.sha256_file(path)
    for relative in FROZEN_DOCS:
        if (ROOT / relative).is_file():
            files[relative] = provenance.sha256_file(ROOT / relative)

    prior = json.loads(
        (ROOT / "artifacts/day3/prior_experiments_manifest.json").read_text("utf-8")
    )
    mismatched = [
        relative
        for relative, digest in prior["files"].items()
        if provenance.sha256_file(ROOT / relative) != digest
    ]
    return {
        "n_files": len(files),
        "combined_sha256": provenance.sha256_text(
            "\n".join(f"{k}:{v}" for k, v in sorted(files.items()))
        ),
        "recorded_prior_manifest_verifies": not mismatched,
        "mismatched": mismatched,
        "files": files,
    }


# --------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    canonical = sp.verify_canonicalization()
    conventions = convention_checks()
    empirical = empirical_counts()
    synthetic = synthetic_cases()
    frozen = frozen_fingerprint()

    checks = {
        "canonicalization_passes": canonical["status"] == "PASS",
        "thresholds_strictly_increasing": canonical["thresholds_strictly_increasing"],
        "day3_conventions_compatible": conventions["compatible"],
        "empirical_matches_frozen_report": empirical["all_agree_with_frozen_report"],
        "synthetic_cases_translation_invariant": synthetic["all_translation_invariant"],
        "synthetic_cases_match_scan": synthetic["all_match_scan"],
        "scalar_model_always_fully_reachable": synthetic[
            "all_scalar_instances_fully_reachable"
        ],
        "frozen_record_verifies": frozen["recorded_prior_manifest_verifies"],
    }
    ok = all(checks.values())

    audit = {
        "title": "FormalCRRC scalar latent-estimation / step-function proposition",
        "classification": "post-study theoretical analysis",
        "disclaimer": (
            "This proposition and its verification were derived after completion "
            "of the three preregistered FormalCRRC studies. They were not part of "
            "any preregistration, must not be represented as preregistered, "
            "modify no frozen experimental outcome, and require no additional "
            "model inference."
        ),
        "preregistered": False,
        "model_inference_performed": False,
        "generated_utc": provenance.utc_now(),
        "checks": checks,
        "status": "PASS" if ok else "FAIL",
        "canonicalization": canonical,
        "day3_conventions": conventions,
        "synthetic": synthetic,
        "empirical": empirical,
        "frozen_scientific_record": {
            k: v for k, v in frozen.items() if k not in ("files",)
        },
        "git": provenance.git_state(ROOT),
        "environment": provenance.environment_snapshot(include_gpu=False),
    }

    audit_path = OUT_DIR / "step_proposition_audit.json"
    provenance.write_json(audit_path, audit)

    csv_path = OUT_DIR / "reachability_by_model.csv"
    pd.DataFrame(empirical["by_model"]).to_csv(csv_path, index=False)

    manifest = {
        "note": (
            "Covers only the additive post-study theory outputs. Prior integrity "
            "manifests are untouched and are not restated here."
        ),
        "generated_utc": provenance.utc_now(),
        "files": provenance.hash_paths(
            {
                "audit": "artifacts/poststudy_theory/step_proposition_audit.json",
                "reachability_by_model": "artifacts/poststudy_theory/reachability_by_model.csv",
            },
            ROOT,
        ),
        "source_manifest": provenance.source_manifest(
            [
                "src/formalcrrc/step_proposition.py",
                "scripts/audit_step_proposition.py",
            ],
            ROOT,
        ),
        "git": provenance.git_state(ROOT),
    }
    provenance.write_json(OUT_DIR / "integrity_manifest.json", manifest)

    # ----------------------------------------------------------------- report
    print("=" * 78)
    print("FORMALCRRC STEP-FUNCTION PROPOSITION AUDIT  (post-study, not preregistered)")
    print("=" * 78)
    print()
    print("PREDICATE-FAMILY CANONICALIZATION: %s" % canonical["status"])
    for family in canonical["families"]:
        print(
            "  %-19s %-22s -> %-14s tau_s = s   %s"
            % (
                family["family"],
                family["formal_predicate"],
                family["canonical_compliance"],
                family["status"],
            )
        )
    print()
    print("DAY-3 CONVENTION COMPATIBILITY: %s"
          % ("PASS" if conventions["compatible"] else "FAIL"))
    print("  strict inequality, ties excluded, crossings 0 and 9 always reachable")
    print()
    print("FROZEN DAY-3 REACHABILITY (recomputed read-only):")
    print("  %-9s %6s %7s %8s %12s %12s  %s"
          % ("judge", "n", "TR=1", "TR=0", "TRR", "1-TRR", "agrees"))
    for row in empirical["by_model"]:
        print(
            "  %-9s %6d %7d %8d %7s=%.4f %12.4f  %s"
            % (
                row["short_name"],
                row["n_nontrivial"],
                row["n_reachable"],
                row["n_unreachable"],
                row["trr_exact"],
                row["trr"],
                row["one_minus_trr"],
                row["matches_frozen_report"]
                and row["agrees_with_day3_module"]
                and row["agrees_with_intercept_scan"],
            )
        )
    print()
    print("SYNTHETIC CASES:")
    for case in synthetic["cases"]:
        print(
            "  %-28s strict_dec=%-5s RC=%-3d unreachable=%s"
            % (
                case["case"],
                case["strictly_decreasing"],
                case["reachable_count"],
                case["unreachable_nontrivial"] or "none",
            )
        )
    print()
    print("INTEGRITY: %d frozen files, combined %s"
          % (frozen["n_files"], frozen["combined_sha256"][:16] + "..."))
    print("  recorded prior manifest verifies: %s"
          % frozen["recorded_prior_manifest_verifies"])
    print()
    print("Checks: %s" % json.dumps(checks))
    print("STATUS: %s" % audit["status"])
    print("Wrote %s" % audit_path.relative_to(ROOT))
    print("Wrote %s" % csv_path.relative_to(ROOT))
    print("Wrote %s" % (OUT_DIR / "integrity_manifest.json").relative_to(ROOT))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
