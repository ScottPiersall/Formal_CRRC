"""Oracle shape-preserving translation bounds.

Day 3 asks the last location question FormalCRRC can ask. Given a judge's raw
margin curve, held exactly fixed in shape, what is the best threshold
localisation that *any* additive translation could achieve?

The admissible class is one constant per curve:

    M'(s) = M(s) + alpha

and nothing stronger. Translation moves where the curve crosses zero; it cannot
reorder margin values, change adjacent differences, remove a reversal, or alter
the span.

The central formal result implemented here is exact. A crossing at level
``j`` in 1..8 is reachable by some translation **iff** ``j`` is a strict prefix
record low of the raw curve::

    M(j) < min_{t < j} M(t)

To place the first negative level at ``j`` an intercept must satisfy
``M(t) + alpha >= 0`` for every ``t < j`` and ``M(j) + alpha < 0``, so
``alpha >= -min_{t<j} M(t)`` and ``alpha < -M(j)``. Such an interval is
non-empty exactly when ``M(j) < min_{t<j} M(t)``. Crossing 0 is always reachable
with a sufficiently negative translation and crossing 9 with a sufficiently
positive one.

The oracle uses formal truth. That is deliberate: it is a diagnostic lower bound
on translation-only localisation error, not a deployable evaluator correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from formalcrrc.day2 import (  # noqa: F401  (shared, unchanged Day-2 primitives)
    margin_monotonicity_violation_rate,
    margin_reversal_magnitude,
    margin_spearman,
    response_span,
    sigmoid,
)
from formalcrrc.day2_bootstrap import crossing_index
from formalcrrc.config import FAMILIES, N_STRICTNESS, NO_CROSSING_INDEX
from formalcrrc.day3_config import ORDER_POOR_MAX_RC, ORDER_RICH_MIN_RC


def _as_curve(margins: Sequence[float]) -> np.ndarray:
    curve = np.asarray(margins, dtype=float)
    if curve.shape != (N_STRICTNESS,):
        raise ValueError(f"expected {N_STRICTNESS} margins, got shape {curve.shape}")
    if not np.all(np.isfinite(curve)):
        raise ValueError("margin curve contains non-finite values")
    return curve


# --------------------------------------------------------------------------
# Reachability -- Method A, the theorem
# --------------------------------------------------------------------------


def prefix_record_lows(margins: Sequence[float]) -> tuple[int, ...]:
    """Levels ``j`` in 1..8 that strictly beat every earlier margin.

    These are exactly the non-extreme crossings a translation can produce.
    """
    curve = _as_curve(margins)
    running_min = curve[0]
    records: list[int] = []
    for j in range(1, N_STRICTNESS):
        if curve[j] < running_min:
            records.append(j)
            running_min = curve[j]
    return tuple(records)


def reachable_crossings(margins: Sequence[float]) -> tuple[int, ...]:
    """The complete set of first-fail positions obtainable by translation.

    Always contains 0 (translate far negative) and 9 (translate far positive).
    """
    return tuple(sorted({0, NO_CROSSING_INDEX, *prefix_record_lows(margins)}))


def reachable_count(margins: Sequence[float]) -> int:
    """``RC``: how many formal boundaries this curve shape can represent."""
    return len(reachable_crossings(margins))


def prefix_record_rate(margins: Sequence[float]) -> float:
    """``PRR``: fraction of levels 1..8 that are strict prefix record lows."""
    return len(prefix_record_lows(margins)) / (N_STRICTNESS - 1)


# --------------------------------------------------------------------------
# Reachability -- Method B, explicit intercept enumeration
# --------------------------------------------------------------------------


def candidate_intercepts(margins: Sequence[float]) -> np.ndarray:
    """Intercepts covering every distinct crossing state of one curve.

    The crossing function is piecewise constant in ``alpha`` with breakpoints at
    ``-M(s)``. Candidates are every breakpoint, the midpoint of every adjacent
    unique pair, one value outside each end, and the immediate float neighbours
    of each breakpoint. The neighbours matter because two breakpoints can be
    closer than their midpoint is representable, which would otherwise skip an
    interval.
    """
    curve = _as_curve(margins)
    return _candidates_from_breakpoints(-curve)


def _candidates_from_breakpoints(breakpoints: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(breakpoints, dtype=float).ravel())
    pieces: list[np.ndarray] = [unique]
    if unique.size > 1:
        pieces.append(0.5 * (unique[:-1] + unique[1:]))
    pieces.append(np.nextafter(unique, np.inf))
    pieces.append(np.nextafter(unique, -np.inf))
    pieces.append(np.array([unique[0] - 1.0, unique[-1] + 1.0]))
    return np.unique(np.concatenate(pieces))


def reachable_crossings_by_enumeration(
    margins: Sequence[float],
) -> tuple[int, ...]:
    """The reachable set found by trying candidate intercepts directly.

    An independent implementation of the same quantity as
    :func:`reachable_crossings`; the two must agree on every curve.
    """
    curve = _as_curve(margins)
    alphas = candidate_intercepts(curve)
    shifted = curve[None, :] + alphas[:, None]
    return tuple(sorted(set(int(j) for j in crossing_index(shifted))))


# --------------------------------------------------------------------------
# Oracle translation TCE
# --------------------------------------------------------------------------


def oracle_translation_tce(
    margins: Sequence[float], true_first_fail_index: int
) -> int:
    """``OTCE``: the best crossing error any additive translation can achieve.

    Because ``alpha = 0`` is admissible, ``OTCE <= TCE_raw`` always.
    """
    target = int(true_first_fail_index)
    return min(abs(j - target) for j in reachable_crossings(margins))


def oracle_translation_tce_by_enumeration(
    margins: Sequence[float], true_first_fail_index: int
) -> int:
    """``OTCE`` via explicit intercept enumeration -- the cross-check path."""
    target = int(true_first_fail_index)
    return min(
        abs(j - target) for j in reachable_crossings_by_enumeration(margins)
    )


def nearest_reachable_crossing(
    margins: Sequence[float], true_first_fail_index: int
) -> int:
    """The reachable crossing closest to the formal boundary.

    Ties are broken toward the smaller index, deterministically.
    """
    target = int(true_first_fail_index)
    return min(reachable_crossings(margins), key=lambda j: (abs(j - target), j))


def translation_reachable(
    margins: Sequence[float], true_first_fail_index: int
) -> bool:
    """``TR``: whether the formal boundary itself is reachable by translation."""
    return int(true_first_fail_index) in reachable_crossings(margins)


# --------------------------------------------------------------------------
# Tie diagnostics
# --------------------------------------------------------------------------


def tie_diagnostics(
    margins: Sequence[float], true_first_fail_index: int
) -> dict[str, Any]:
    """Descriptive record of exact margin equalities.

    Equality never creates a reachable crossing under the strict rule. These
    counts are reported so the frequency of ties is visible, and the rule is not
    changed after seeing them.
    """
    curve = _as_curve(margins)
    target = int(true_first_fail_index)

    adjacent_equal = int(np.sum(curve[1:] == curve[:-1]))
    prefix_ties = 0
    blocked_by_equality = False
    running = curve[0]
    for j in range(1, N_STRICTNESS):
        if curve[j] == running:
            prefix_ties += 1
            if j == target:
                blocked_by_equality = True
        running = min(running, curve[j])

    return {
        "adjacent_equal_margins": adjacent_equal,
        "prefix_tie_levels": prefix_ties,
        "target_blocked_by_equality": bool(
            blocked_by_equality
            and 1 <= target <= N_STRICTNESS - 1
            and not translation_reachable(curve, target)
        ),
    }


# --------------------------------------------------------------------------
# Group oracles -- exact enumeration, no gradient descent
# --------------------------------------------------------------------------


def tce_matrix(
    margins_block: np.ndarray, j_star: np.ndarray, alphas: np.ndarray
) -> np.ndarray:
    """``TCE`` for every (artifact, candidate intercept) pair.

    ``margins_block`` is ``(n_artifacts, 9)``. The result is
    ``(n_artifacts, n_alphas)`` and is the reusable core of both the group
    oracle search and its bootstrap.
    """
    margins_block = np.asarray(margins_block, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    out = np.empty((margins_block.shape[0], alphas.size), dtype=np.int16)
    for index, alpha in enumerate(alphas):
        out[:, index] = np.abs(
            crossing_index(margins_block + alpha) - np.asarray(j_star, dtype=int)
        )
    return out


def group_candidate_intercepts(margins_block: np.ndarray) -> np.ndarray:
    """Candidate intercepts covering every distinct state of a whole group."""
    return _candidates_from_breakpoints(-np.asarray(margins_block, dtype=float))


@dataclass(frozen=True)
class GroupOracle:
    """The truth-optimised intercept for one group, and what it achieves."""

    alpha: float
    mean_tce: float
    n_artifacts: int
    n_candidates: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "mean_tce": self.mean_tce,
            "n_artifacts": self.n_artifacts,
            "n_candidates": self.n_candidates,
        }


def select_group_oracle(
    alphas: np.ndarray, mean_tce: np.ndarray, n_artifacts: int
) -> GroupOracle:
    """Apply the preregistered tie-break to an evaluated candidate set.

    Lowest mean TCE, then smallest ``|alpha|``, then numerically smallest
    ``alpha``. Deterministic for any input.
    """
    alphas = np.asarray(alphas, dtype=float)
    mean_tce = np.asarray(mean_tce, dtype=float)
    best = mean_tce.min()
    tied = np.flatnonzero(mean_tce == best)
    order = sorted(tied, key=lambda i: (abs(alphas[i]), alphas[i]))
    chosen = order[0]
    return GroupOracle(
        alpha=float(alphas[chosen]),
        mean_tce=float(best),
        n_artifacts=int(n_artifacts),
        n_candidates=int(alphas.size),
    )


def oracle_group_intercept(
    margins_block: np.ndarray, j_star: np.ndarray
) -> GroupOracle:
    """Exact truth-optimised single intercept for a group of curves."""
    alphas = group_candidate_intercepts(margins_block)
    matrix = tce_matrix(margins_block, j_star, alphas)
    return select_group_oracle(alphas, matrix.mean(axis=0), margins_block.shape[0])


# --------------------------------------------------------------------------
# Per-artifact assembly
# --------------------------------------------------------------------------

TAXONOMY_COLUMNS: tuple[str, ...] = (
    "translation_solvable",
    "translation_near_solvable",
    "translation_limited",
    "order_rich",
    "order_poor",
)


def stack_margins(
    scores: pd.DataFrame, margin_column: str = "margin"
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Arrange one judge's scores as ``(n_artifacts, 9)`` blocks.

    Returns artifact ids, the margin block, the formal boundaries, and the
    family of each artifact, all in a single canonical order.
    """
    ordered = scores.sort_values(["artifact_id", "strictness_index"])
    ids: list[str] = []
    families: list[str] = []
    margins: list[np.ndarray] = []
    boundaries: list[int] = []
    for artifact_id, group in ordered.groupby("artifact_id", sort=True):
        levels = group["strictness_index"].to_numpy()
        if not np.array_equal(levels, np.arange(N_STRICTNESS)):
            raise ValueError(f"{artifact_id}: expected strictness 0..8 exactly once")
        ids.append(artifact_id)
        families.append(group["family"].iloc[0])
        margins.append(group[margin_column].to_numpy(dtype=float))
        boundaries.append(int(group["true_first_fail_index"].iloc[0]))
    return ids, np.stack(margins), np.asarray(boundaries, dtype=int), families


def compute_artifact_metrics(
    scores: pd.DataFrame,
    oracle_global_alpha: float,
    oracle_family_alphas: dict[str, float],
    margin_column: str = "margin",
) -> pd.DataFrame:
    """One row per artifact: raw localisation, oracle bounds, shape, taxonomy."""
    required = {
        "model_id",
        "artifact_id",
        "family",
        "latent_level",
        "strictness_index",
        margin_column,
        "formal_truth",
        "true_first_fail_index",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores frame is missing columns: {sorted(missing)}")

    model_id = scores["model_id"].iloc[0]
    ids, block, boundaries, families = stack_margins(scores, margin_column)

    rows: list[dict[str, Any]] = []
    for position, artifact_id in enumerate(ids):
        margins = block[position]
        j_star = int(boundaries[position])
        family = families[position]

        reach_a = reachable_crossings(margins)
        otce = min(abs(j - j_star) for j in reach_a)
        j_raw = int(crossing_index(margins[None, :])[0])
        tce_raw = abs(j_raw - j_star)
        j_global = int(crossing_index((margins + oracle_global_alpha)[None, :])[0])
        alpha_f = float(oracle_family_alphas[family])
        j_family = int(crossing_index((margins + alpha_f)[None, :])[0])
        rc = len(reach_a)
        ties = tie_diagnostics(margins, j_star)

        rows.append(
            {
                "model_id": model_id,
                "artifact_id": artifact_id,
                "family": family,
                "latent_level": int(
                    scores.loc[scores["artifact_id"] == artifact_id, "latent_level"].iloc[0]
                ),
                "true_first_fail_index": j_star,
                # raw and group-oracle localisation
                "j_hat_raw": j_raw,
                "tce_raw": tce_raw,
                "j_hat_oracle_global": j_global,
                "tce_oracle_global": abs(j_global - j_star),
                "j_hat_oracle_family": j_family,
                "tce_oracle_family": abs(j_family - j_star),
                # artifact oracle
                "otce": otce,
                "nearest_reachable": nearest_reachable_crossing(margins, j_star),
                "delta_tce_artifact": tce_raw - otce,
                "translation_reachable": j_star in reach_a,
                "nontrivial_boundary": 1 <= j_star <= N_STRICTNESS - 1,
                # geometry
                "reachable_crossings": ",".join(str(j) for j in reach_a),
                "reachable_count": rc,
                "prefix_record_rate": prefix_record_rate(margins),
                # shape, invariant to any translation
                "mmvr": margin_monotonicity_violation_rate(margins),
                "mmvm": margin_reversal_magnitude(margins),
                "span": response_span(margins),
                "margin_spearman": margin_spearman(margins),
                # ties
                **ties,
                # taxonomy
                "translation_solvable": otce == 0,
                "translation_near_solvable": otce == 1,
                "translation_limited": otce > 1,
                "order_rich": rc >= ORDER_RICH_MIN_RC,
                "order_poor": rc <= ORDER_POOR_MAX_RC,
            }
        )
    return pd.DataFrame(rows)


def reachability_table(scores: pd.DataFrame, margin_column: str = "margin") -> pd.DataFrame:
    """Per-artifact reachable-crossing record, with both methods cross-checked."""
    model_id = scores["model_id"].iloc[0]
    ids, block, boundaries, families = stack_margins(scores, margin_column)
    rows: list[dict[str, Any]] = []
    for position, artifact_id in enumerate(ids):
        margins = block[position]
        j_star = int(boundaries[position])
        theorem = reachable_crossings(margins)
        enumerated = reachable_crossings_by_enumeration(margins)
        rows.append(
            {
                "model_id": model_id,
                "artifact_id": artifact_id,
                "family": families[position],
                "true_first_fail_index": j_star,
                "reachable_theorem": ",".join(str(j) for j in theorem),
                "reachable_enumeration": ",".join(str(j) for j in enumerated),
                "methods_agree": theorem == enumerated,
                "reachable_count": len(theorem),
                "prefix_record_lows": ",".join(
                    str(j) for j in prefix_record_lows(margins)
                ),
                "otce_theorem": min(abs(j - j_star) for j in theorem),
                "otce_enumeration": min(abs(j - j_star) for j in enumerated),
                "translation_reachable": j_star in theorem,
            }
        )
    return pd.DataFrame(rows)


def shape_metric_associations(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Spearman association between OTCE and each simple shape descriptor.

    Descriptive only. Correlation here is not evidence of a mechanism; the point
    is to see which cheap curve descriptors track translation reachability.
    """
    descriptors = ("mmvr", "mmvm", "span", "prefix_record_rate", "reachable_count")
    out: list[dict[str, Any]] = []
    for scope, frame in [("overall", metrics)] + [
        (family, metrics[metrics["family"] == family]) for family in FAMILIES
    ]:
        for descriptor in descriptors:
            x = frame["otce"].to_numpy(dtype=float)
            y = frame[descriptor].to_numpy(dtype=float)
            if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
                rho, p = float("nan"), float("nan")
            else:
                result = stats.spearmanr(x, y)
                rho, p = float(result.statistic), float(result.pvalue)
            out.append(
                {
                    "model_id": frame["model_id"].iloc[0] if len(frame) else None,
                    "scope": scope,
                    "descriptor": descriptor,
                    "spearman_rho": rho,
                    "p_value": p,
                    "n": int(len(frame)),
                }
            )
    return out


def assert_oracle_nesting(
    mean_raw: float, mean_global: float, mean_family: float, mean_artifact: float
) -> dict[str, Any]:
    """The nested classes G subset F subset A must give ordered bounds.

    ``alpha = 0`` is admissible in every class, so each optimised bound must
    also be no worse than raw. A violation is an implementation defect.
    """
    checks = {
        "global_le_raw": mean_global <= mean_raw + 1e-12,
        "family_le_global": mean_family <= mean_global + 1e-12,
        "artifact_le_family": mean_artifact <= mean_family + 1e-12,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


# --------------------------------------------------------------------------
# Prior-experiment immutability
# --------------------------------------------------------------------------


def _walk_tree(root, relative_dir: str):
    from pathlib import Path

    base = Path(root) / relative_dir
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file())


def build_prior_manifest(root=None) -> dict[str, Any]:
    """Fingerprint the complete scientific records of Days 1 and 2.

    Day 3 writes nothing into either prior experiment; this manifest is the
    reference against which that is re-verified at the end.
    """
    from pathlib import Path

    from formalcrrc import provenance
    from formalcrrc.day3_config import PRIOR_DOCS, PRIOR_TREES

    root = Path(root) if root is not None else provenance.repo_root()
    by_experiment: dict[str, dict[str, str]] = {"day1": {}, "day2": {}}

    for tree in PRIOR_TREES:
        key = "day1" if tree.endswith("day1") else "day2"
        for path in _walk_tree(root, tree):
            by_experiment[key][path.relative_to(root).as_posix()] = (
                provenance.sha256_file(path)
            )
    for relative in PRIOR_DOCS:
        candidate = root / relative
        if not candidate.is_file():
            continue
        key = "day2" if "DAY2" in relative.upper() else "day1"
        by_experiment[key][relative] = provenance.sha256_file(candidate)

    files = {**by_experiment["day1"], **by_experiment["day2"]}
    return {
        "created_at": provenance.utc_now(),
        "purpose": (
            "Immutable fingerprint of the completed Day-1 and Day-2 experiments. "
            "Day-3 code writes nothing into artifacts/day1, artifacts/day2, "
            "figures/day1 or figures/day2."
        ),
        "trees": list(PRIOR_TREES),
        "documents": list(PRIOR_DOCS),
        "n_files": len(files),
        "by_experiment": {
            key: {
                "n_files": len(block),
                "combined_sha256": _combined(block),
                "files": block,
            }
            for key, block in by_experiment.items()
        },
        "files": files,
        "combined_sha256": _combined(files),
    }


def _combined(files: dict[str, str]) -> str:
    from formalcrrc import provenance

    if not files:
        raise ValueError("refusing to compute a combined hash over an empty file list")
    payload = "\n".join(f"{name}:{digest}" for name, digest in sorted(files.items()))
    return provenance.sha256_text(payload)


def verify_prior_manifest(manifest: dict[str, Any], root=None) -> dict[str, Any]:
    """Re-check a stored prior fingerprint against the working tree."""
    from pathlib import Path

    from formalcrrc import provenance
    from formalcrrc.day3_config import PRIOR_DOCS, PRIOR_TREES

    root = Path(root) if root is not None else provenance.repo_root()

    per_experiment: dict[str, dict[str, Any]] = {}
    for key in ("day1", "day2"):
        recorded = manifest["by_experiment"][key]["files"]
        changed, missing = [], []
        for relative, digest in sorted(recorded.items()):
            candidate = root / relative
            if not candidate.is_file():
                missing.append(relative)
            elif provenance.sha256_file(candidate) != digest:
                changed.append(relative)
        per_experiment[key] = {
            "n_recorded": len(recorded),
            "changed": changed,
            "missing": missing,
            "status": "PASS" if not changed and not missing else "FAIL",
        }

    current: dict[str, str] = {}
    for tree in PRIOR_TREES:
        for path in _walk_tree(root, tree):
            current[path.relative_to(root).as_posix()] = provenance.sha256_file(path)
    for relative in PRIOR_DOCS:
        if (root / relative).is_file():
            current[relative] = provenance.sha256_file(root / relative)
    added = sorted(set(current) - set(manifest["files"]))

    return {
        "checked_at": provenance.utc_now(),
        "day1": per_experiment["day1"],
        "day2": per_experiment["day2"],
        "added": added,
        "status": (
            "PASS"
            if per_experiment["day1"]["status"] == "PASS"
            and per_experiment["day2"]["status"] == "PASS"
            and not added
            else "FAIL"
        ),
    }
