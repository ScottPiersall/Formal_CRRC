"""The scalar latent-estimation / step-function proposition.

**Post-study theoretical analysis.** Added after Days 1-3 closed. It is not part
of any preregistration, it runs no inference, and it changes no frozen result.
It formalises an explanation FormalCRRC can *rule out*, and connects that to the
Day-3 reachability theorem already implemented in :mod:`formalcrrc.day3`.

The two explanations being separated
------------------------------------
1. **Scalar latent-estimation error.** The judge forms one scalar compliance
   estimate for the artifact and compares that same estimate coherently against
   progressively stricter thresholds. Its estimate may be wrong; its *ordering*
   is not.
2. **Ordering failure.** The observed margin ordering does not contain the
   formal boundary at all, so no relocation of a single scalar can produce it.

The proposition, in two tiers
-----------------------------
Let ``tau_s`` be strictly increasing after canonicalization.

*Tier 1 (labels only).* If the judge answers
``Y_hat(s) = 1[z_hat >= tau_s]`` for one fixed scalar ``z_hat``, then
``Y_hat`` is nonincreasing in ``s`` with at most one met-to-not-met transition.
No assumption about margins is needed.

*Tier 2 (margins).* If additionally the canonical margin is
``M(s) = g(z_hat - tau_s)`` with ``g`` strictly increasing, then ``z_hat - tau_s``
is strictly decreasing, so ``M`` is **strictly decreasing**. Every level
``j`` in 1..8 is then a strict prefix record low, hence -- by the Day-3
theorem -- *every* crossing 0..9 is reachable by an additive translation, and in
particular the true boundary is:

    pure scalar latent-estimation model  ==>  TR_x = 1

The contrapositive is the usable direction::

    TR_x = 0  ==>  the response cannot be explained by an erroneous scalar
                   compliance estimate under this order-preserving model.

What this does **not** license
------------------------------
``TR_x = 1`` does not imply a scalar representation is used -- it only leaves
that explanation available. Nothing here speaks to internal representations, and
``TR_x = 0`` falsifies the *conjunction* of "one scalar estimate" and
"order-preserving margin readout ``g(z_hat - tau_s)``" without isolating which
conjunct fails. See the audit document for the full list of claims that must not
be made.

Note that Tier 2 implies strictly more than ``TR_x = 1``: it implies the whole
reachable set is ``{0,...,9}``. ``TR_x`` is used as the headline test because it
is the preregistered Day-3 quantity and the weakest consequence, so the
falsification rests on the least that the model commits to.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from formalcrrc import day3, predicates
from formalcrrc.config import (
    FAMILIES,
    FAMILY_COVERAGE,
    N_STRICTNESS,
    NO_CROSSING_INDEX,
    SCALE_MAX,
)

#: The canonical threshold ladder. After canonicalization every family shares
#: it, and it is strictly increasing -- the proposition's core assumption.
CANONICAL_THRESHOLDS: tuple[int, ...] = tuple(range(N_STRICTNESS))


# --------------------------------------------------------------------------
# Canonicalization
# --------------------------------------------------------------------------


def canonical_compliance(family: str, latent_level: int) -> int:
    """The canonical compliance scalar ``z``: larger means more compliant.

    Coverage stores ``m`` (already oriented that way). The other two families
    store a *defect* count -- violations ``v`` or absolute error ``e`` -- whose
    printed allowance ``8 - s`` falls as strictness rises, so they are reflected
    to ``z = 8 - latent``.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    if not 0 <= int(latent_level) <= SCALE_MAX:
        raise ValueError(f"latent level out of range: {latent_level}")
    if family == FAMILY_COVERAGE:
        return int(latent_level)
    return SCALE_MAX - int(latent_level)


def canonical_threshold(strictness: int) -> int:
    """``tau_s``. Identity by construction, so the ladder is strictly increasing."""
    if not 0 <= int(strictness) < N_STRICTNESS:
        raise ValueError(f"strictness out of range: {strictness}")
    return int(strictness)


def canonical_labels(family: str, latent_level: int) -> tuple[int, ...]:
    """Formal labels rebuilt in canonical form: ``1[z >= tau_s]``."""
    z = canonical_compliance(family, latent_level)
    return tuple(int(z >= canonical_threshold(s)) for s in range(N_STRICTNESS))


def canonical_first_fail(family: str, latent_level: int) -> int:
    """``j*`` in canonical form. Equals ``z + 1`` for every family."""
    return canonical_compliance(family, latent_level) + 1


def thresholds_strictly_increasing() -> bool:
    """The assumption the proposition rests on, checked rather than assumed."""
    taus = [canonical_threshold(s) for s in range(N_STRICTNESS)]
    return all(b > a for a, b in zip(taus, taus[1:]))


def verify_canonicalization() -> dict[str, object]:
    """Check the canonical form reproduces the implemented predicates exactly.

    Read-only, exhaustive over all three families x nine latent levels x nine
    strictness levels. Returns a per-family record plus overall verdicts.
    """
    families: list[dict[str, object]] = []
    truth_ok = jstar_ok = True

    for family in FAMILIES:
        raw = [predicates.raw_threshold(family, s) for s in range(N_STRICTNESS)]
        mismatches: list[dict[str, int]] = []
        for latent in range(N_STRICTNESS):
            expected = predicates.truth_curve(family, latent)
            observed = list(canonical_labels(family, latent))
            if expected != observed:
                truth_ok = False
                mismatches.append({"latent_level": latent})
            if predicates.true_first_fail_index(
                family, latent
            ) != canonical_first_fail(family, latent):
                jstar_ok = False
                mismatches.append({"latent_level": latent, "field": "j_star"})

        families.append(
            {
                "family": family,
                "stored_latent": {
                    FAMILY_COVERAGE: "m, count of required items present",
                }.get(
                    family,
                    "v, count of prohibited items present"
                    if family == "max_violation"
                    else "e, absolute numeric error",
                ),
                "stored_strictness_index": "s = 0..8",
                "formal_predicate": (
                    "1[m >= s]"
                    if family == FAMILY_COVERAGE
                    else "1[latent <= 8 - s]"
                ),
                "raw_printed_threshold": raw,
                "raw_threshold_strictly_increasing": all(
                    b > a for a, b in zip(raw, raw[1:])
                ),
                "transformation_required": family != FAMILY_COVERAGE,
                "canonical_compliance": (
                    "z = m" if family == FAMILY_COVERAGE else "z = 8 - latent"
                ),
                "canonical_threshold": "tau_s = s",
                "canonical_predicate": "1[z >= tau_s]",
                "canonical_threshold_strictly_increasing": True,
                "direction_of_increasing_strictness": "s increasing is stricter",
                "true_first_fail_index": "j* = z + 1, in 1..9",
                "mismatches": mismatches,
                "status": "PASS" if not mismatches else "FAIL",
            }
        )

    return {
        "families": families,
        "canonical_thresholds": list(CANONICAL_THRESHOLDS),
        "thresholds_strictly_increasing": thresholds_strictly_increasing(),
        "labels_match_implementation": truth_ok,
        "first_fail_matches_implementation": jstar_ok,
        "status": "PASS"
        if truth_ok and jstar_ok and thresholds_strictly_increasing()
        else "FAIL",
    }


# --------------------------------------------------------------------------
# The scalar model, and the structure it forces
# --------------------------------------------------------------------------


def scalar_model_labels(z_hat: float) -> np.ndarray:
    """Tier 1: the labels a pure scalar-readout judge emits, ``1[z_hat >= tau_s]``."""
    return np.array(
        [int(z_hat >= canonical_threshold(s)) for s in range(N_STRICTNESS)], dtype=int
    )


def scalar_model_margins(
    z_hat: float, g: Callable[[np.ndarray], np.ndarray] | None = None
) -> np.ndarray:
    """Tier 2: margins ``g(z_hat - tau_s)`` for a strictly increasing ``g``.

    ``g`` defaults to the identity. Any strictly increasing ``g`` yields the same
    ordering, which is the only thing the proposition uses.
    """
    difference = np.array(
        [z_hat - canonical_threshold(s) for s in range(N_STRICTNESS)], dtype=float
    )
    return difference if g is None else np.asarray(g(difference), dtype=float)


def is_nonincreasing_step(labels: Sequence[int]) -> bool:
    """Tier 1's conclusion: nonincreasing, so at most one met-to-not-met switch."""
    values = np.asarray(labels, dtype=int)
    return bool(np.all(np.diff(values) <= 0))


def transition_count(labels: Sequence[int]) -> int:
    """Number of met-to-not-met switches. The proposition allows at most one."""
    values = np.asarray(labels, dtype=int)
    return int(np.sum(np.diff(values) < 0))


def is_strictly_decreasing(margins: Sequence[float]) -> bool:
    """Tier 2's conclusion about the margin curve."""
    curve = np.asarray(margins, dtype=float)
    return bool(np.all(np.diff(curve) < 0.0))


def all_noninitial_are_prefix_record_lows(margins: Sequence[float]) -> bool:
    """Whether levels 1..8 are *all* strict prefix record lows.

    Equivalent to :func:`is_strictly_decreasing` -- proved in the audit and
    asserted by the test suite -- and equivalent to full reachability.
    """
    return len(day3.prefix_record_lows(margins)) == N_STRICTNESS - 1


def implies_full_reachability(margins: Sequence[float]) -> bool:
    """Whether every crossing 0..9 is reachable by an additive translation."""
    return len(day3.reachable_crossings(margins)) == NO_CROSSING_INDEX + 1


def scalar_model_is_falsified(margins: Sequence[float], true_first_fail: int) -> bool:
    """``TR_x = 0``: the conservative, preregistered-quantity falsification.

    True when the formal boundary is unreachable by any additive translation, so
    no relocation of a single scalar estimate can produce it.
    """
    return not day3.translation_reachable(margins, true_first_fail)
