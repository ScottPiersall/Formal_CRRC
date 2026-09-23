"""Reason-then-score measurement for the reasoning-capable external anchor.

The Day-3 panel was scored at the immediate response boundary: the A/B margin
was read from the next-token logits at the first generated position. That
procedure cannot be reused unchanged here. ``Qwen3-30B-A3B-Thinking-2507`` is a
thinking-only model whose native chat template *opens* a reasoning block in the
generation prompt, so the first generated token starts its reasoning rather than
its judgment. Scoring A and B there would measure the beginning of a thought,
not the end of one.

The protocol is therefore two-stage, and both stages are preregistered:

**Stage R** generates reasoning under the model author's recommended sampling
family until the native end-of-thinking delimiter appears or a fixed token
budget is exhausted. Sampling makes the trace part of the measurement, so every
row's seed is derived deterministically from immutable identifiers and every
generated token id is retained.

**Stage D** rebuilds a standardised decision context -- the original formatted
prompt, the exact generated reasoning, the native delimiter, and a fixed
model-native separator -- and scores the two candidate answers as *continuations*
of that context by full sequence log-likelihood::

    M = L_A - L_B

Positive means "criterion met", matching Day 3. The tie convention, crossing
convention and reachability rule are inherited from Day 3 unchanged, so the
localisation and ordering endpoints remain directly defined.

Nothing here removes, rewrites, summarises or regenerates a reasoning trace. The
only normalisation permitted is the fixed transition between the delimiter and
the decision boundary, frozen before inference.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from formalcrrc.config import N_STRICTNESS, NO_CROSSING_INDEX
from formalcrrc.reasoning_anchor_config import (
    CANDIDATE_MET,
    CANDIDATE_NOT_MET,
    EXPERIMENT_SEED,
    SEED_MODULUS,
    THINK_END_TEXT,
)

# --------------------------------------------------------------------------
# Deterministic per-row seeds
# --------------------------------------------------------------------------


def row_seed(
    artifact_id: str,
    strictness_index: int,
    base_seed: int = EXPERIMENT_SEED,
) -> int:
    """Stable generation seed for one study row.

    Derived from immutable identifiers with SHA-256 rather than Python's
    built-in ``hash()``, which is randomised per process and would not reproduce
    across runs or machines. The digest is reduced modulo ``2**31 - 1`` so the
    value lands inside the range every sampler accepts.

    The same ``(artifact_id, strictness_index)`` always yields the same seed;
    two strictness levels of one artifact yield different seeds.
    """
    payload = f"{int(base_seed)}|{artifact_id}|{int(strictness_index)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % SEED_MODULUS


# --------------------------------------------------------------------------
# Stage R -- locating the end of thinking
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReasoningTrace:
    """One generated reasoning trace and how it ended.

    ``reasoning_token_ids`` are the tokens the model produced, *including* the
    end-of-thinking delimiter when it was reached. ``think_end_reached`` is
    False exactly when the token budget ran out first; such a row is marked
    truncated and is excluded from the primary complete-case analysis. The
    delimiter is never fabricated.
    """

    token_ids: tuple[int, ...]
    think_end_reached: bool
    truncated: bool
    stop_reason: str
    n_tokens: int

    @property
    def reasoning_token_ids(self) -> tuple[int, ...]:
        return self.token_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning_token_ids": list(self.token_ids),
            "reasoning_token_count": self.n_tokens,
            "think_end_reached": self.think_end_reached,
            "reasoning_truncated": self.truncated,
            "stop_reason": self.stop_reason,
        }


def split_reasoning(
    generated_token_ids: Sequence[int],
    think_end_id: int,
    budget: int,
) -> ReasoningTrace:
    """Cut a generated sequence at the first end-of-thinking delimiter.

    Everything up to and including the first ``think_end_id`` is the reasoning
    trace. Tokens after it -- a model that kept going -- are discarded, because
    the decision boundary is defined at the delimiter and the free-form answer
    is not used for the primary margin.

    When the delimiter never appears the row is truncated: the trace is returned
    intact, ``think_end_reached`` is False, and no delimiter is appended.
    """
    ids = tuple(int(t) for t in generated_token_ids)
    for position, token in enumerate(ids):
        if token == int(think_end_id):
            kept = ids[: position + 1]
            return ReasoningTrace(
                token_ids=kept,
                think_end_reached=True,
                truncated=False,
                stop_reason="think_end",
                n_tokens=len(kept),
            )
    return ReasoningTrace(
        token_ids=ids,
        think_end_reached=False,
        truncated=True,
        stop_reason="length" if len(ids) >= int(budget) else "stopped_without_think_end",
        n_tokens=len(ids),
    )


# --------------------------------------------------------------------------
# Stage D -- the standardised decision context
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateTokenization:
    """The token ids each candidate answer takes in the real decision context.

    ``contextual`` ids are what the model must actually emit after the frozen
    separator. ``isolated`` records what encoding the bare label alone would
    have produced. The two differ whenever a tokenizer attaches a word-boundary
    marker to a string encoded on its own -- the defect that was caught in Day 1
    for Mistral -- and it is the contextual ids that index the right positions.
    Candidates are never assumed to be a single token.
    """

    met: str
    not_met: str
    met_token_ids: tuple[int, ...]
    not_met_token_ids: tuple[int, ...]
    isolated_met_token_ids: tuple[int, ...] = ()
    isolated_not_met_token_ids: tuple[int, ...] = ()

    @property
    def differs_from_isolated(self) -> bool:
        return (
            self.met_token_ids != self.isolated_met_token_ids
            or self.not_met_token_ids != self.isolated_not_met_token_ids
        )

    @property
    def multi_token(self) -> bool:
        return len(self.met_token_ids) > 1 or len(self.not_met_token_ids) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_met": self.met,
            "candidate_not_met": self.not_met,
            "met_token_ids": list(self.met_token_ids),
            "not_met_token_ids": list(self.not_met_token_ids),
            "isolated_met_token_ids": list(self.isolated_met_token_ids),
            "isolated_not_met_token_ids": list(self.isolated_not_met_token_ids),
            "differs_from_isolated_encoding": self.differs_from_isolated,
            "multi_token": self.multi_token,
            "tokenization_rule": "prompt_continuation_difference",
            "scoring_method": "sequence_loglikelihood",
        }


def continuation_ids(tokenizer: Any, context_text: str, suffix: str) -> tuple[int, ...]:
    """Token ids ``suffix`` adds when it follows ``context_text``.

    Taken as the difference between the encoded context and the encoded
    context-plus-suffix, so the result is what the model is really being asked
    about rather than what the string encodes on its own.
    """
    base = list(tokenizer.encode(context_text, add_special_tokens=False))
    full = list(tokenizer.encode(context_text + suffix, add_special_tokens=False))
    if full[: len(base)] != base:
        raise ValueError(
            f"appending {suffix!r} changed the preceding tokenisation; the "
            "continuation cannot be isolated"
        )
    continuation = tuple(full[len(base) :])
    if not continuation:
        raise ValueError(f"{suffix!r} produced an empty continuation")
    return continuation


def resolve_separator_ids(
    tokenizer: Any, context_text: str, separator: str | None = None
) -> tuple[int, ...]:
    """Token ids of the frozen post-thinking separator in context."""
    from formalcrrc.reasoning_anchor_config import POST_THINK_SEPARATOR

    return continuation_ids(
        tokenizer, context_text, POST_THINK_SEPARATOR if separator is None else separator
    )


def resolve_candidate_tokenization(
    tokenizer: Any, decision_context_text: str
) -> CandidateTokenization:
    """Resolve A and B as continuations of the standardised decision context."""
    return CandidateTokenization(
        met=CANDIDATE_MET,
        not_met=CANDIDATE_NOT_MET,
        met_token_ids=continuation_ids(tokenizer, decision_context_text, CANDIDATE_MET),
        not_met_token_ids=continuation_ids(
            tokenizer, decision_context_text, CANDIDATE_NOT_MET
        ),
        isolated_met_token_ids=tuple(
            tokenizer.encode(CANDIDATE_MET, add_special_tokens=False)
        ),
        isolated_not_met_token_ids=tuple(
            tokenizer.encode(CANDIDATE_NOT_MET, add_special_tokens=False)
        ),
    )


def build_decision_context(
    prompt_token_ids: Sequence[int],
    reasoning_token_ids: Sequence[int],
    separator_token_ids: Sequence[int],
) -> tuple[int, ...]:
    """Assemble the exact context the candidate answers are scored against.

    Concatenation only: the formatted prompt, the generated reasoning including
    its delimiter, then the frozen model-native separator. No token of the
    reasoning is dropped, reordered or rewritten, so the likelihood really is
    conditioned on what the model actually thought.
    """
    return (
        tuple(int(t) for t in prompt_token_ids)
        + tuple(int(t) for t in reasoning_token_ids)
        + tuple(int(t) for t in separator_token_ids)
    )


def sequence_logprob(
    token_logprobs: Sequence[Mapping[int, float] | None],
    start_index: int,
    candidate_token_ids: Sequence[int],
) -> float:
    """Total log-likelihood of a candidate continuation.

    ``token_logprobs[i]`` maps token id to log-probability at position ``i``,
    conditioned on positions before it -- the shape a prompt-logprob request
    returns. Summing over the candidate's own positions gives the exact
    sequence log-likelihood, which reduces to the single-token case when the
    candidate is one token but stays correct when it is not.

    Raises when a candidate token is absent from its position rather than
    silently substituting a floor value; a missing entry means the request did
    not ask for the actual token's log-probability and the score would be wrong.
    """
    ids = [int(t) for t in candidate_token_ids]
    if not ids:
        raise ValueError("candidate continuation is empty")
    total = 0.0
    for offset, token_id in enumerate(ids):
        position = int(start_index) + offset
        if position >= len(token_logprobs):
            raise ValueError(
                f"position {position} is past the end of the log-probability "
                f"sequence (length {len(token_logprobs)})"
            )
        entry = token_logprobs[position]
        if entry is None or token_id not in entry:
            raise ValueError(
                f"no log-probability recorded for token {token_id} at position "
                f"{position}; the candidate cannot be scored"
            )
        total += float(entry[token_id])
    return total


def canonical_margin(logprob_met: float, logprob_not_met: float) -> float:
    """``M = L_A - L_B``. Positive means the criterion is judged met."""
    if not (np.isfinite(logprob_met) and np.isfinite(logprob_not_met)):
        raise ValueError(
            f"non-finite candidate log-likelihoods: met={logprob_met!r} "
            f"not_met={logprob_not_met!r}"
        )
    return float(logprob_met) - float(logprob_not_met)


def semantic_decision(margin: float) -> int:
    """``D = 1[M >= 0]``. A margin of exactly zero counts as 'met', as in Day 3."""
    return int(float(margin) >= 0.0)


# --------------------------------------------------------------------------
# Curve assembly and mechanical completeness
# --------------------------------------------------------------------------


@dataclass
class CurveCompleteness:
    """Which artifact curves carry a valid margin at all nine thresholds."""

    complete: list[str] = field(default_factory=list)
    incomplete: dict[str, list[int]] = field(default_factory=dict)

    @property
    def n_complete(self) -> int:
        return len(self.complete)

    @property
    def n_excluded(self) -> int:
        return len(self.incomplete)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_complete_curves": self.n_complete,
            "n_excluded_curves": self.n_excluded,
            "excluded_artifact_ids": sorted(self.incomplete),
            "missing_levels_by_artifact": {
                k: sorted(v) for k, v in sorted(self.incomplete.items())
            },
        }


def curve_completeness(rows: pd.DataFrame) -> CurveCompleteness:
    """Split artifacts into complete and mechanically incomplete curves.

    A curve is complete when all nine strictness levels carry a finite margin.
    The rule looks only at whether a margin exists, never at its value, so no
    exclusion can depend on how the anchor performed.
    """
    required = {"artifact_id", "strictness_index", "margin"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"rows frame is missing columns: {sorted(missing)}")

    result = CurveCompleteness()
    for artifact_id, group in rows.groupby("artifact_id", sort=True):
        present = {
            int(level)
            for level, margin in zip(
                group["strictness_index"], group["margin"], strict=True
            )
            if margin is not None and np.isfinite(margin)
        }
        absent = sorted(set(range(N_STRICTNESS)) - present)
        if absent:
            result.incomplete[str(artifact_id)] = absent
        else:
            result.complete.append(str(artifact_id))
    return result


def complete_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Only the rows belonging to curves that are complete at all nine levels."""
    keep = set(curve_completeness(rows).complete)
    return rows[rows["artifact_id"].isin(keep)].copy()


# --------------------------------------------------------------------------
# Descriptive reasoning-length associations
# --------------------------------------------------------------------------


def reasoning_length_association(
    metrics: pd.DataFrame,
    length_column: str = "mean_reasoning_tokens",
    outcome_columns: Sequence[str] = ("tce_raw", "translation_reachable"),
) -> list[dict[str, Any]]:
    """Spearman association between reasoning length and each outcome.

    Descriptive only, and labelled as such wherever it is reported. A
    correlation between how long the model thought and how well it localised a
    boundary is not evidence that thinking longer caused either outcome: trace
    length is itself chosen by the model and confounded with item difficulty.
    """
    from scipy import stats

    out: list[dict[str, Any]] = []
    for column in outcome_columns:
        if column not in metrics.columns or length_column not in metrics.columns:
            continue
        x = metrics[length_column].to_numpy(dtype=float)
        y = metrics[column].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
            rho, p = float("nan"), float("nan")
        else:
            result = stats.spearmanr(x, y)
            rho, p = float(result.statistic), float(result.pvalue)
        out.append(
            {
                "length_column": length_column,
                "outcome": column,
                "spearman_rho": rho,
                "p_value": p,
                "n": int(x.size),
                "interpretation": "descriptive association only; not causal",
            }
        )
    return out


# --------------------------------------------------------------------------
# Frozen-record fingerprinting
# --------------------------------------------------------------------------


def _walk_tree(root, relative_dir: str):
    from pathlib import Path

    base = Path(root) / relative_dir
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file())


def _combined(files: Mapping[str, str]) -> str:
    from formalcrrc import provenance

    if not files:
        raise ValueError("refusing to compute a combined hash over an empty file list")
    payload = "\n".join(f"{name}:{digest}" for name, digest in sorted(files.items()))
    return provenance.sha256_text(payload)


def build_frozen_fingerprint(root=None) -> dict[str, Any]:
    """Fingerprint every protected scientific artifact before new inference.

    Covers Days 1-3, both post-study audits and the completed label-swap study.
    New work is additive, so each of these files must hash identically when the
    fingerprint is recomputed after the anchor run.
    """
    from pathlib import Path

    from formalcrrc import provenance
    from formalcrrc.reasoning_anchor_config import PROTECTED_DOCS, PROTECTED_TREES

    root = Path(root) if root is not None else provenance.repo_root()
    files: dict[str, str] = {}
    for tree in PROTECTED_TREES:
        for path in _walk_tree(root, tree):
            files[path.relative_to(root).as_posix()] = provenance.sha256_file(path)
    for relative in PROTECTED_DOCS:
        candidate = root / relative
        if candidate.is_file():
            files[relative] = provenance.sha256_file(candidate)

    return {
        "created_at": provenance.utc_now(),
        "purpose": (
            "Immutable fingerprint of every completed FormalCRRC scientific "
            "artifact prior to reasoning-anchor inference. The anchor study "
            "writes nothing into any of these paths."
        ),
        "trees": list(PROTECTED_TREES),
        "documents": list(PROTECTED_DOCS),
        "n_files": len(files),
        "files": files,
        "combined_sha256": _combined(files),
    }


def verify_frozen_fingerprint(fingerprint: Mapping[str, Any], root=None) -> dict[str, Any]:
    """Re-check a stored fingerprint against the working tree.

    Reports changed, missing and added files separately. A manifest is never
    rewritten to accommodate a change it did not expect.
    """
    from pathlib import Path

    from formalcrrc import provenance
    from formalcrrc.reasoning_anchor_config import PROTECTED_DOCS, PROTECTED_TREES

    root = Path(root) if root is not None else provenance.repo_root()
    recorded = dict(fingerprint["files"])

    changed, missing = [], []
    for relative, digest in sorted(recorded.items()):
        candidate = root / relative
        if not candidate.is_file():
            missing.append(relative)
        elif provenance.sha256_file(candidate) != digest:
            changed.append(relative)

    current: dict[str, str] = {}
    for tree in PROTECTED_TREES:
        for path in _walk_tree(root, tree):
            current[path.relative_to(root).as_posix()] = provenance.sha256_file(path)
    for relative in PROTECTED_DOCS:
        if (root / relative).is_file():
            current[relative] = provenance.sha256_file(root / relative)
    added = sorted(set(current) - set(recorded))

    combined_now = _combined(current) if current else None
    return {
        "checked_at": provenance.utc_now(),
        "n_recorded": len(recorded),
        "changed": changed,
        "missing": missing,
        "added": added,
        "combined_recorded": fingerprint.get("combined_sha256"),
        "combined_now": combined_now,
        "combined_matches": combined_now == fingerprint.get("combined_sha256"),
        "status": (
            "PASS" if not changed and not missing and not added else "FAIL"
        ),
    }
