"""Elicited reason-then-score measurement for the Qwen2.5 protocol ablation.

Day 3 scored this exact model at the immediate response boundary. The
reasoning-anchor study scored a *different* model under a reason-then-score
protocol and found a large difference, but could not say whether that came from
the model or from the readout, because both moved at once. This study holds the
model, the revision, the artifacts and the formal thresholds fixed and moves
only the protocol, so the comparison is within-model and paired.

Three things here need care and are the reason this module exists rather than
reusing the anchor's code wholesale:

**The marker.** Qwen2.5 has no native end-of-thinking delimiter, so the model is
asked to write ``FINAL:`` and generation stops on the complete marker. Stopping
must happen *before* the next token, so the model never samples the label that
the primary margin is supposed to measure.

**The candidates.** They carry a leading space (`` A``, `` B``) because that is
what actually continues the marker, and they are tokenised in that context, not
in isolation.

**The failure decomposition.** When a boundary is unreachable it matters whether
an earlier threshold scored strictly below it -- a genuine ordering inversion --
or merely equal to it, which fails only because the reachability rule requires a
strict record low. Those are different defects and the study reports them apart.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from formalcrrc.config import N_STRICTNESS, NO_CROSSING_INDEX, PROMPT_TEMPLATE
from formalcrrc.rts_ablation_config import (
    CANDIDATE_MET,
    CANDIDATE_NOT_MET,
    FAILURE_NONE,
    FAILURE_STRICT_INVERSION,
    FAILURE_TIE_ONLY,
    MARKER,
    RTS_PROMPT_TEMPLATE,
)

# Reused unchanged from the reasoning-anchor study so the two experiments share
# one seed convention and one likelihood implementation.
from formalcrrc.reasoning_anchor import (  # noqa: F401
    canonical_margin,
    continuation_ids,
    curve_completeness,
    complete_rows,
    row_seed,
    semantic_decision,
    sequence_logprob,
)


# --------------------------------------------------------------------------
# Prompt construction and the machine-verifiable diff
# --------------------------------------------------------------------------


def build_rts_message(rubric_text: str, candidate_response: str) -> str:
    """Render the reason-then-score user message for one row."""
    return RTS_PROMPT_TEMPLATE.format(
        rubric_text=rubric_text, candidate_response=candidate_response
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_diff() -> dict[str, Any]:
    """Prove the intervention is exactly the preregistered scaffold and nothing else.

    The reason-then-score template is built by appending to the frozen Day-3
    template, so the difference must be a pure suffix. Anything else -- a
    reflowed line, a changed placeholder, a dropped blank line -- would mean the
    ablation had quietly altered the prompt in ways beyond the protocol change,
    and the assertions below fail rather than let that pass.
    """
    original, new = PROMPT_TEMPLATE, RTS_PROMPT_TEMPLATE
    is_suffix_extension = new.startswith(original)
    added = new[len(original) :] if is_suffix_extension else None

    diff = list(
        difflib.unified_diff(
            original.splitlines(keepends=False),
            new.splitlines(keepends=False),
            fromfile="day3_prompt_template",
            tofile="reason_then_score_template",
            lineterm="",
            n=2,
        )
    )
    removed = [
        line[1:]
        for line in diff
        if line.startswith("-") and not line.startswith("---")
    ]
    added_lines = [
        line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")
    ]

    return {
        "original_template": original,
        "original_template_sha256": _sha(original),
        "rts_template": new,
        "rts_template_sha256": _sha(new),
        "construction": (
            "RTS = PROMPT_TEMPLATE + '\\n\\n' + REASONING_SCAFFOLD, built "
            "programmatically from the frozen Day-3 template"
        ),
        "is_pure_suffix_extension": is_suffix_extension,
        "added_suffix": added,
        "added_suffix_sha256": _sha(added) if added is not None else None,
        "added_bytes": len(added.encode("utf-8")) if added is not None else None,
        "removed_lines": removed,
        "added_lines": added_lines,
        "unified_diff": diff,
        "placeholders_preserved": (
            "{rubric_text}" in new and "{candidate_response}" in new
        ),
        "marker_present": MARKER in new,
        "checks": {
            "pure_suffix_extension": is_suffix_extension,
            "nothing_removed": not removed,
            "placeholders_preserved": (
                "{rubric_text}" in new and "{candidate_response}" in new
            ),
            "marker_present": MARKER in new,
        },
    }


def bodies_identical(dataset: pd.DataFrame) -> dict[str, Any]:
    """Confirm the rubric and response bodies are untouched for every row.

    Renders both templates for each row and checks that the reason-then-score
    message is exactly the Day-3 message plus the fixed suffix. This is stronger
    than comparing templates: it catches any interaction between substituted
    content and the appended text.
    """
    from formalcrrc.prompts import build_user_message

    suffix = RTS_PROMPT_TEMPLATE[len(PROMPT_TEMPLATE) :]
    mismatches: list[str] = []
    for row in dataset.itertuples(index=False):
        day3 = build_user_message(row.rubric_text, row.candidate_response)
        rts = build_rts_message(row.rubric_text, row.candidate_response)
        if rts != day3 + suffix:
            mismatches.append(row.prompt_id)
    return {
        "n_rows_checked": int(len(dataset)),
        "n_mismatches": len(mismatches),
        "mismatched_prompt_ids": mismatches[:20],
        "status": "PASS" if not mismatches else "FAIL",
    }


# --------------------------------------------------------------------------
# The final-answer marker
# --------------------------------------------------------------------------

#: A line consisting only of A or B, optionally bolded or followed by a stop.
#: Mechanical and fixed before inference; used for a descriptive QC flag only.
PREMATURE_LABEL_PATTERN: str = r"(?m)^\s*(?:\*\*)?[AB](?:\*\*)?\s*[.)]?\s*$"


@dataclass(frozen=True)
class MarkerSplit:
    """Where the elicited reasoning ends and the decision boundary begins."""

    text_through_marker: str
    reasoning_text: str
    marker_reached: bool
    truncated: bool
    stop_reason: str
    premature_label: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker_reached": self.marker_reached,
            "reasoning_truncated": self.truncated,
            "stop_reason": self.stop_reason,
            "premature_label_observed": self.premature_label,
        }


def split_at_marker(generated_text: str, budget_exhausted: bool) -> MarkerSplit:
    """Cut generated text at the end of the first complete marker.

    Everything up to and including the first ``FINAL:`` is kept, and that is
    exactly where Stage-D scoring happens. Text after it -- a label the model
    volunteered anyway -- is discarded, because the primary margin must come
    from the likelihood of the candidates, not from a sampled token.

    A trace that never produced the marker is truncated. The marker is never
    fabricated: a truncated row carries no primary margin at all.
    """
    index = generated_text.find(MARKER)
    if index < 0:
        return MarkerSplit(
            text_through_marker=generated_text,
            reasoning_text=generated_text,
            marker_reached=False,
            truncated=True,
            stop_reason="length" if budget_exhausted else "stopped_without_marker",
            premature_label=bool(re.search(PREMATURE_LABEL_PATTERN, generated_text)),
        )
    end = index + len(MARKER)
    reasoning = generated_text[:index]
    return MarkerSplit(
        text_through_marker=generated_text[:end],
        reasoning_text=reasoning,
        marker_reached=True,
        truncated=False,
        stop_reason="marker",
        premature_label=bool(re.search(PREMATURE_LABEL_PATTERN, reasoning)),
    )


def resolve_marker_tokenization(tokenizer: Any, context_text: str) -> dict[str, Any]:
    """Token ids the marker takes as a continuation of a realistic context."""
    ids = continuation_ids(tokenizer, context_text, MARKER)
    return {
        "marker": MARKER,
        "marker_bytes_hex": MARKER.encode("utf-8").hex(),
        "contextual_token_ids": list(ids),
        "n_tokens": len(ids),
        "isolated_token_ids": list(
            tokenizer.encode(MARKER, add_special_tokens=False)
        ),
        "multi_token": len(ids) > 1,
    }


def resolve_candidate_tokenization(
    tokenizer: Any, decision_context_text: str
) -> dict[str, Any]:
    """Resolve `` A`` and `` B`` as continuations of the real decision context."""
    met = continuation_ids(tokenizer, decision_context_text, CANDIDATE_MET)
    not_met = continuation_ids(tokenizer, decision_context_text, CANDIDATE_NOT_MET)
    return {
        "candidate_met": CANDIDATE_MET,
        "candidate_not_met": CANDIDATE_NOT_MET,
        "met_token_ids": list(met),
        "not_met_token_ids": list(not_met),
        "isolated_met_token_ids": list(
            tokenizer.encode(CANDIDATE_MET, add_special_tokens=False)
        ),
        "isolated_not_met_token_ids": list(
            tokenizer.encode(CANDIDATE_NOT_MET, add_special_tokens=False)
        ),
        "multi_token": len(met) > 1 or len(not_met) > 1,
        "tokenization_rule": "prompt_continuation_difference",
        "scoring_method": "sequence_loglikelihood",
    }


# --------------------------------------------------------------------------
# Unreachable-failure decomposition
# --------------------------------------------------------------------------


def classify_failure(margins: Sequence[float], true_first_fail_index: int) -> str:
    """Why a true boundary is or is not a strict prefix record low.

    Three mutually exclusive outcomes, decided on exact stored values with no
    tolerance anywhere:

    ``reachable``          the boundary beats every earlier margin strictly;
    ``strict_inversion``   some earlier threshold scored strictly *below* it,
                           which is a genuine ordering failure;
    ``tie_only``           nothing scored below it, but something scored exactly
                           equal, so it fails only because the rule demands a
                           strict record low.

    The distinction matters: a tie is a resolution or saturation problem, an
    inversion is the model ranking a laxer threshold below a stricter one.
    """
    curve = np.asarray(margins, dtype=float)
    if curve.shape != (N_STRICTNESS,):
        raise ValueError(f"expected {N_STRICTNESS} margins, got {curve.shape}")
    target = int(true_first_fail_index)
    if not 1 <= target <= N_STRICTNESS - 1:
        # Crossings 0 and 9 are always reachable by translation.
        return FAILURE_NONE
    prior = curve[:target]
    value = curve[target]
    prior_min = float(prior.min())
    if value < prior_min:
        return FAILURE_NONE
    if prior_min < value:
        return FAILURE_STRICT_INVERSION
    return FAILURE_TIE_ONLY


def failure_table(
    margins_by_artifact: Mapping[str, Sequence[float]],
    boundaries: Mapping[str, int],
    condition: str,
) -> pd.DataFrame:
    """Per-curve failure classification for one condition."""
    rows: list[dict[str, Any]] = []
    for artifact_id, margins in margins_by_artifact.items():
        target = int(boundaries[artifact_id])
        if not 1 <= target <= N_STRICTNESS - 1:
            continue
        curve = np.asarray(margins, dtype=float)
        kind = classify_failure(curve, target)
        prior = curve[:target]
        rows.append(
            {
                "condition": condition,
                "artifact_id": artifact_id,
                "true_first_fail_index": target,
                "failure_type": kind,
                "reachable": kind == FAILURE_NONE,
                "n_strict_inversions": int(np.sum(prior < curve[target])),
                "n_exact_ties": int(np.sum(prior == curve[target])),
                "margin_at_boundary": float(curve[target]),
                "prior_min_margin": float(prior.min()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Exact interval for a zero-count result
# --------------------------------------------------------------------------


def clopper_pearson(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial interval.

    Used when the observed unreachable count is zero. A bootstrap over a
    statistic that is identically zero returns ``[0, 0]``, which describes the
    resampling distribution rather than the population, and would overstate what
    the data can support. The exact interval's upper bound is the honest
    statement of how large the unreachable probability could still be.
    """
    from scipy import stats

    if trials <= 0:
        return (float("nan"), float("nan"))
    if not 0 <= successes <= trials:
        raise ValueError(f"successes={successes} outside 0..{trials}")
    alpha = 1.0 - level
    lower = 0.0 if successes == 0 else float(
        stats.beta.ppf(alpha / 2.0, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return (lower, upper)


# --------------------------------------------------------------------------
# Condition O -- recovering the frozen comparator, read only
# --------------------------------------------------------------------------


def recover_comparator(
    day3_scores: pd.DataFrame, dataset: pd.DataFrame
) -> pd.DataFrame:
    """Recompute the frozen Day-3 Qwen2.5 curve metrics from raw margins.

    Deliberately recomputed rather than read from the Day-3 metrics table, so
    that the comparator this study reports is derived by this study's own code
    from the raw scores. Agreement with the frozen table is then a real check
    rather than a tautology.
    """
    from formalcrrc import day3 as d3
    from formalcrrc.day2 import sigmoid
    from formalcrrc.day2_bootstrap import crossing_index

    keep = dataset[["prompt_id", "formal_truth", "true_first_fail_index"]]
    merged = day3_scores.merge(keep, on="prompt_id", how="left", validate="one_to_one")
    if merged["formal_truth"].isna().any():
        raise ValueError("comparator join lost rows against the frozen dataset")

    rows: list[dict[str, Any]] = []
    for artifact_id, group in merged.sort_values(
        ["artifact_id", "strictness_index"]
    ).groupby("artifact_id", sort=True):
        levels = group["strictness_index"].to_numpy()
        if not np.array_equal(levels, np.arange(N_STRICTNESS)):
            raise ValueError(f"{artifact_id}: expected strictness 0..8 exactly once")
        margins = group["margin"].to_numpy(dtype=float)
        truth = group["formal_truth"].to_numpy(dtype=int)
        target = int(group["true_first_fail_index"].iloc[0])
        reach = d3.reachable_crossings(margins)
        j_hat = int(crossing_index(margins[None, :])[0])
        probabilities = sigmoid(margins)
        rows.append(
            {
                "artifact_id": artifact_id,
                "family": group["family"].iloc[0],
                "latent_level": int(group["latent_level"].iloc[0]),
                "true_first_fail_index": target,
                "j_hat_o": j_hat,
                "tce_o": abs(j_hat - target),
                "translation_reachable_o": target in reach,
                "nontrivial_boundary": 1 <= target <= N_STRICTNESS - 1,
                "reachable_count_o": len(reach),
                "prefix_record_rate_o": d3.prefix_record_rate(margins),
                "failure_type_o": classify_failure(margins, target),
                "accuracy_o": float(np.mean((margins >= 0).astype(int) == truth)),
                "brier_o": float(np.mean((probabilities - truth) ** 2)),
                "margins_o": ",".join(f"{m!r}" for m in margins),
            }
        )
    return pd.DataFrame(rows)


def verify_comparator(
    recovered: pd.DataFrame, frozen_metrics: pd.DataFrame
) -> dict[str, Any]:
    """Check the recomputation against the frozen Day-3 metrics table."""
    frozen = frozen_metrics.sort_values("artifact_id").reset_index(drop=True)
    mine = recovered.sort_values("artifact_id").reset_index(drop=True)
    if list(mine["artifact_id"]) != list(frozen["artifact_id"]):
        return {"status": "FAIL", "reason": "artifact id sets differ"}

    tce_identical = bool(
        (mine["tce_o"].to_numpy() == frozen["tce_raw"].to_numpy()).all()
    )
    reach_identical = bool(
        (
            mine["translation_reachable_o"].to_numpy()
            == frozen["translation_reachable"].to_numpy()
        ).all()
    )
    rc_identical = bool(
        (
            mine["reachable_count_o"].to_numpy()
            == frozen["reachable_count"].to_numpy()
        ).all()
    )
    nontrivial = mine[mine["nontrivial_boundary"]]
    checks = {
        "per_artifact_tce_identical": tce_identical,
        "per_artifact_reachability_identical": reach_identical,
        "per_artifact_rc_identical": rc_identical,
    }
    return {
        "n_artifacts": int(len(mine)),
        "n_nontrivial": int(len(nontrivial)),
        "mean_tce": float(mine["tce_o"].mean()),
        "trr": float(nontrivial["translation_reachable_o"].mean()),
        "unreachable_pct": 100.0
        * (1.0 - float(nontrivial["translation_reachable_o"].mean())),
        "mean_reachable_count": float(mine["reachable_count_o"].mean()),
        "accuracy": float(mine["accuracy_o"].mean()),
        "brier": float(mine["brier_o"].mean()),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


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
    """Fingerprint every protected artifact before the ablation runs."""
    from pathlib import Path

    from formalcrrc import provenance
    from formalcrrc.rts_ablation_config import PROTECTED_DOCS, PROTECTED_TREES

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
            "artifact prior to the reason-then-score ablation, including the "
            "Day-3 comparator and the reasoning-anchor study."
        ),
        "trees": list(PROTECTED_TREES),
        "documents": list(PROTECTED_DOCS),
        "n_files": len(files),
        "files": files,
        "combined_sha256": _combined(files),
    }


def verify_frozen_fingerprint(fingerprint: Mapping[str, Any], root=None) -> dict[str, Any]:
    """Re-check a stored fingerprint against the working tree."""
    from pathlib import Path

    from formalcrrc import provenance
    from formalcrrc.rts_ablation_config import PROTECTED_DOCS, PROTECTED_TREES

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
        "status": "PASS" if not changed and not missing and not added else "FAIL",
    }
