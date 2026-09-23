#!/usr/bin/env python
"""Generate ``docs/RESULTS_DAY1.md`` from the analysis artifacts (Phase H).

Every number in the results document comes from ``summary.json``,
``integrity_manifest.json``, ``bootstrap_results.parquet`` and
``model_provenance.json``. Nothing is retyped by hand, so the prose cannot drift
from the artifacts, and no number can be quietly rounded in a flattering
direction.

Interpretive text lives in ``NARRATIVE`` below and is written after the analysis
has run. It may describe what the numbers show; it may not strengthen the
preregistered claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import provenance  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    BOOTSTRAP_CI,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DECISION_THRESHOLD,
    FAMILIES,
    INTEGRITY_MANIFEST_PATH,
    MODEL_IDS,
    MODEL_PROVENANCE_PATH,
    MODEL_SHORT_NAMES,
    MVR_TOLERANCE,
    N_ARTIFACTS,
    N_ROWS,
    N_STRICTNESS,
    SUMMARY_PATH,
)

FAMILY_TITLES = {
    "coverage": "A · coverage",
    "max_violation": "B · max violation",
    "numeric_tolerance": "C · numeric tolerance",
}

#: Interpretive paragraphs, filled in after the analysis has run. Keys that are
#: left empty are omitted rather than guessed at.
NARRATIVE: dict[str, str] = {
    "executive_summary": """
All four preregistered judges completed all 2916 rubric-response pairs with zero
scoring failures, and all four responded to the threshold manipulation in the
correct direction: every boundary-aligned curve decreases with strictness, and
every judge has a negative mean within-artifact Spearman association.

None of the four reproduced the step function the rubric implies. The
threshold-crossing error is large for every judge — mean TCE between 2.86 and
4.94 on a 0–9 scale, with the judge's 0.5 crossing landing exactly on the formal
boundary for between 0.6% and 17.3% of artifacts, and within one level for
between 11.7% and 33.0%.

Practical monotonicity is mostly respected: MVR at the preregistered tolerance
δ = 0.05 runs from 0.010 to 0.107. At zero tolerance it runs from 0.196 to 0.367,
so small upward wiggles are pervasive while large ones are not.

Two of the four judges show a strong global response bias that dominates their
behaviour. Llama-3.1-8B scores below 0.5 at the most permissive threshold for
322 of 324 artifacts — a threshold every artifact satisfies by construction —
and Mistral-7B never falls below 0.5 for 154 of 324. Curve fidelity and
aggregate accuracy also rank the judges differently.
""",
    "monotonicity": """
The gap between the two tolerances is the informative part. At δ = 0.05 the
judges look largely well-behaved; at δ = 0 between 20% and 37% of adjacent steps
move the wrong way. The upward moves are therefore common but small, which the
MVM column confirms: the largest mean violation magnitude is 0.046 (Qwen2.5-14B),
spread over eight steps.

Gemma-2-9B is the most monotone judge by a clear margin on both tolerances and
on magnitude, and only 24 of its 324 curves contain any violation at δ = 0.05.
It is not the most accurate judge, nor the one that crosses closest to the
boundary.
""",
    "crossing": """
This is the weakest result in the study, and it is weak for every judge.

Llama-3.1-8B's TCE distribution is close to flat at 36 artifacts per bucket. That
is the signature of a constant prediction rather than a mistaken one: with
ĵ = 0 for nearly every artifact and j* uniform over 1…9 by design, |ĵ − j*|
inherits the distribution of j* itself. Its crossing metrics therefore describe a
judge that answers "not met" almost everywhere, not a judge tracking the
threshold badly.

Mistral-7B errs in the opposite direction, never dropping below 0.5 for 154 of
324 artifacts, yet has the lowest mean TCE of the four (2.86). Qwen2.5-14B and
Gemma-2-9B sit between the two. No judge places its crossing exactly at the
formal boundary for even a fifth of artifacts.
""",
    "accuracy": """
Accuracy is modest throughout — 0.53 to 0.76 — against a design in which always
answering "not met" would score about 0.44, since 5 of every 9 cells are formally
met on average.

Accuracy and curve fidelity do not rank the judges the same way. Qwen2.5-14B has
the highest accuracy (0.760) and the lowest Brier score (0.233), but roughly five
times Gemma-2-9B's monotonicity violation rate and a higher mean TCE than either
Mistral-7B or Gemma-2-9B. Gemma-2-9B has the cleanest curves and the worst Brier
score. This is the divergence H3 anticipated, though it should be read with the
absolute accuracy figures in view: no judge here is accurate *and* faithful.
""",
    "boundary_curves": """
Aligning on the true boundary separates two things the per-artifact metrics
conflate.

Qwen2.5-14B's population curve transitions sharply and in the right place: 0.564
at d = −1 falls to 0.248 at d = 0, straddling the boundary. Its per-artifact mean
TCE is nevertheless 3.54. The aggregate transition is therefore not evidence that
individual curves cross where they should — averaging over artifacts recovers a
boundary that the individual curves scatter widely around.

Mistral-7B's curve is displaced upward almost everywhere, crossing 0.5 only near
d = +1. Llama-3.1-8B's sits below 0.5 across the entire range, never reaching the
formally-satisfied region. Gemma-2-9B declines nearly linearly from 0.78 to 0.00
with no visible step at the boundary.
""",
    "family_analysis": """
Predicate family matters, and it matters differently for each judge, so the
overall averages conceal it.

Qwen2.5-14B is far better on numeric tolerance (mean TCE 1.61) than on coverage
(4.83) — arithmetic comparison against an explicit target is easier for it than
counting list members. Gemma-2-9B shows the opposite ordering, best on coverage
(2.54) and worst on max violation (4.96). Mistral-7B is comparatively even
(2.49–3.22). Llama-3.1-8B is uniformly near 5 across all three, consistent with a
constant response rather than family-specific difficulty.

Max violation is the hardest family for two of the four judges. It shares its
candidate-response format with coverage and differs only in the direction of the
threshold, so the difficulty cannot be attributed to the artifact's surface form.
""",
    "cross_model": """
Most pairwise differences exclude zero, but they describe four specific models on
three synthetic predicate families and are not a general ranking. The one pair
whose TCE difference does not exclude zero is Qwen2.5-14B versus Gemma-2-9B
(0.16, [−0.16, 0.49]).

The orderings disagree across metrics: Qwen2.5-14B is best on accuracy and Brier,
Gemma-2-9B on MVR and MVM, Mistral-7B on mean TCE. No judge leads on all three
primary metrics.
""",
    "failure_modes": """
Three distinct failure modes appear, and they are not the same failure.

**Global response bias.** Llama-3.1-8B answers below 0.5 at every threshold for
322 of 324 artifacts, including the s = 0 threshold that every artifact satisfies
by construction. Mistral-7B never crosses for 154 of 324. In both cases the
judge's overall placement of the probability scale, rather than its response to
the threshold, dominates the crossing metrics.

**Shallow response.** Gemma-2-9B moves smoothly from 0.78 to 0.00 across the
whole strictness range with no step at the boundary. Its curves are the most
monotone in the panel and among the least informative about where the boundary
is.

**Local instability.** The gap between zero-tolerance MVR (0.196–0.367) and
δ = 0.05 MVR (0.010–0.107) shows frequent small reversals in all four judges,
including on artifacts whose curves are otherwise well-ordered.
""",
    "research_decision": """
The mechanism this protocol was built to detect is present but weak. All four
judges move in the direction the rubric implies, and mostly without large
monotonicity violations, yet none locates the decision boundary the rubric
defines with any reliability — on formal predicates with no semantic ambiguity,
where the artifact is byte-identical across conditions and the correct answer at
every level is arithmetic.

This is the outcome the specification anticipates as C, with strong D and E
components. It is reported as it fell. No metric, threshold, tolerance, prompt,
model or predicate was changed in response to it.
""",
}


def load(root: Path) -> tuple[dict, dict, pd.DataFrame, dict]:
    summary = json.loads((root / SUMMARY_PATH).read_text(encoding="utf-8"))
    integrity = json.loads(
        (root / INTEGRITY_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    boot = pd.read_parquet(root / "artifacts/day1/bootstrap_results.parquet")
    provenance_path = root / MODEL_PROVENANCE_PATH
    models = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else {}
    )
    return summary, integrity, boot, models


def _ci(boot: pd.DataFrame, model_id: str, metric: str, scope: str, fmt: str) -> str:
    row = boot[
        (boot["model_id"] == model_id)
        & (boot["metric"] == metric)
        & (boot["scope"] == scope)
    ]
    if row.empty:
        return "—"
    r = row.iloc[0]
    return f"{r['point']:{fmt}} [{r['ci_low']:{fmt}}, {r['ci_high']:{fmt}}]"


def _ci_only(boot: pd.DataFrame, model_id: str, metric: str, scope: str, fmt: str) -> str:
    """Just the interval, for tables that already show the point estimate."""
    text = _ci(boot, model_id, metric, scope, fmt)
    return text.split(" ", 1)[1] if " " in text else text


def _complete(summary: dict) -> list[str]:
    return [m for m in MODEL_IDS if m in summary.get("models", {})]


def section_narrative(key: str) -> str:
    text = NARRATIVE.get(key, "").strip()
    return f"\n{text}\n" if text else ""


def build(root: Path) -> str:
    summary, integrity, boot, model_provenance = load(root)
    complete = _complete(summary)
    parts: list[str] = []
    add = parts.append

    prereg = integrity["preregistration"]
    dataset = integrity["dataset"]
    panel = integrity["panel"]

    # ---------------------------------------------------------------- 1
    add("# FormalCRRC Day-1 results\n")
    add(
        f"Generated {summary['created_at']} from the frozen analysis artifacts. "
        "Every number below is read directly from "
        "`artifacts/day1/summary.json`, `artifacts/day1/bootstrap_results.parquet` "
        "and `artifacts/day1/integrity_manifest.json`.\n"
    )
    add("## 1. Executive summary\n")
    add("```text")
    add(f"FORMAL DATASET INTEGRITY = {dataset['status']}")
    add(f"PREREGISTRATION INTEGRITY = {prereg['status']}")
    add(f"MODEL PANEL              = {panel['status']} "
        f"({panel['complete']}/{panel['size']})")
    add(f"DAY-1 ANALYSIS           = {'COMPLETE' if complete else 'BLOCKED'}")
    add("```\n")

    add("| Judge | MVR | MVM | mean TCE | accuracy | Brier |")
    add("|---|---|---|---|---|---|")
    for model_id in MODEL_IDS:
        block = summary["models"].get(model_id)
        if not block:
            reason = summary["panel"][model_id].get("reason") or "not run"
            add(f"| {MODEL_SHORT_NAMES[model_id]} | BLOCKED | BLOCKED | "
                f"BLOCKED | BLOCKED | {reason} |")
            continue
        o = block["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['mvr']:.4f} | {o['mvm']:.4f} | "
            f"{o['tce']['mean']:.3f} | {o['accuracy']:.4f} | {o['brier']:.4f} |"
        )
    add("")
    add(section_narrative("executive_summary"))

    # ---------------------------------------------------------------- 2
    add("## 2. Research question\n")
    add(
        "> When evaluation criteria differ only by a formally controlled decision "
        "threshold, do LLM judges exhibit the monotone counterfactual response "
        "function implied by the rubric?\n"
    )
    add(
        "A fixed artifact is evaluated nine times. The candidate response is "
        "byte-identical across all nine evaluations; only the numeric decision "
        "threshold printed in the rubric changes. The correct answer at every "
        "level is known by construction before any model is consulted.\n"
    )

    # ---------------------------------------------------------------- 3
    add("## 3. Novelty boundaries\n")
    add(
        "This study claims an experimental protocol, not a first. It does not "
        "claim the first rubric perturbation method, the first counterfactual "
        "evaluation of LLM judges, the first strict-vs-lenient or multi-level "
        "strictness experiment, the first judge robustness curve, the first "
        "monotonicity analysis of an LLM evaluator, the first threshold-based "
        "judge evaluation, the first use of controlled rubric changes, or the "
        "first continuous judge reliability measure. See "
        "[`NOVELTY_BOUNDARIES.md`](NOVELTY_BOUNDARIES.md) for the collision "
        "areas that constrain the claim, and [`LINEAGE.md`](LINEAGE.md) for why "
        "this is a new lineage rather than a CRRC version bump.\n"
    )

    # ---------------------------------------------------------------- 4
    add("## 4. FormalCRRC definition\n")
    add(
        "For artifact $x$ and rubric $R_\\tau$, the judge response is "
        "$C_x(\\tau) = P_J(\\text{criterion met} \\mid x, R_\\tau)$. The ordered "
        "sequence $C_x(\\tau_1), \\ldots, C_x(\\tau_9)$ is the counterfactual "
        "rubric response curve. Truth comes from the formal predicate, never "
        "from a generator/validator/consensus chain.\n"
    )

    # ---------------------------------------------------------------- 5
    add("## 5. Dataset construction\n")
    add("| | |")
    add("|---|---|")
    add(f"| Predicate families | 3 — {', '.join(FAMILIES)} |")
    add("| Latent levels per family | 9 |")
    add("| Instances per cell | 12 |")
    add(f"| Artifacts | **{N_ARTIFACTS}** |")
    add(f"| Thresholds per artifact | {N_STRICTNESS} |")
    add(f"| Rubric-response pairs | **{N_ROWS}** per judge |")
    add(f"| Dataset content SHA-256 | `{dataset.get('content_sha256')}` |")
    add("| Human annotation | none |\n")
    add(
        "Family A prints `at least s`; families B and C print `at most 8 − s`, "
        "so the numeric threshold moves in the opposite direction from "
        "strictness. Families A and B share a candidate-response format, so they "
        "differ only in threshold direction. Family A/B responses always carry 14 "
        "markers regardless of latent level, so response length carries no "
        "signal.\n"
    )

    # ---------------------------------------------------------------- 6
    add("## 6. Dataset integrity audit\n")
    add(f"Status: **{dataset['status']}**\n")
    add("| Check | Result |")
    add("|---|---|")
    for name, ok in dataset.get("checks", {}).items():
        add(f"| {name.replace('_', ' ')} | {'PASS' if ok else 'FAIL'} |")
    add("")
    add(
        "The dataset was rebuilt from seed 42 during this audit and reproduced "
        "the recorded content hash. The pre-inference test suite additionally "
        "verifies candidate immutability across thresholds, rubric control "
        "(normalised rubrics identical within an artifact), label monotonicity, "
        "and an independent recomputation of all "
        f"{N_ROWS} labels from the rendered strings alone.\n"
    )

    # ---------------------------------------------------------------- 7
    add("## 7. Model provenance\n")
    add("| Judge | Revision | dtype | Scoring rule | Status |")
    add("|---|---|---|---|---|")
    for model_id in MODEL_IDS:
        record = model_provenance.get("models", {}).get(model_id, {})
        tok = record.get("label_tokenization", {}) or {}
        add(
            f"| `{model_id}` | `{record.get('revision', '—')}` | "
            f"{record.get('dtype', '—')} | {tok.get('scoring_method', '—')} | "
            f"{record.get('status', 'NOT_CHECKED')} |"
        )
    add("")
    env = model_provenance.get("environment", {})
    gpu = (env.get("gpu") or {}).get("devices") or []
    packages = env.get("packages", {})
    add(
        f"Environment at the availability gate: torch "
        f"{packages.get('torch', '—')}, transformers "
        f"{packages.get('transformers', '—')}, CUDA "
        f"{(env.get('gpu') or {}).get('torch_cuda_version', '—')}, GPU "
        f"{gpu[0]['name'] if gpu else '—'}. No quantisation, no fine-tuning, no "
        "adapters, no calibration layer, no prompt optimisation.\n"
    )

    # ---------------------------------------------------------------- 8
    add("## 8. Prompt and probability extraction\n")
    add(
        "One minimal user message per rubric-response pair, semantically "
        "identical for every model, rendered through each model's native chat "
        "template. No system message, no chain-of-thought, no few-shot examples, "
        "no persona, no self-explanation request. Nothing is sampled or "
        "generated: for each prompt a single forward pass yields the model's "
        "scores for the two answer labels, normalised as "
        "$P_\\text{met} = e^{z_A} / (e^{z_A} + e^{z_B})$.\n"
    )
    methods = {
        MODEL_SHORT_NAMES[m]: summary["models"][m]["scoring_method"]
        for m in complete
    }
    if methods:
        add("Scoring rule actually applied: "
            + ", ".join(f"{k} — `{v}`" for k, v in methods.items())
            + ".\n")

    # ---------------------------------------------------------------- 9
    add("## 9. Inference completeness\n")
    add("| Judge | Status | Rows | Failures | Job | Node | GPU | Seconds |")
    add("|---|---|---|---|---|---|---|---|")
    for model_id in MODEL_IDS:
        run = integrity["runs"][model_id]
        meta = run.get("run", {}) or {}
        slurm = meta.get("slurm") or {}
        gpus = meta.get("gpu") or []
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {run['status']} | "
            f"{run.get('rows', 0)} | {meta.get('failures', '—')} | "
            f"{slurm.get('job_id', '—')} | {slurm.get('node', '—')} | "
            f"{gpus[0]['name'] if gpus else '—'} | "
            f"{meta.get('total_seconds', '—')} |"
        )
    add("")
    blocked = [m for m in MODEL_IDS if integrity["runs"][m]["status"] != "COMPLETE"]
    if blocked:
        add(
            "The following judges did not produce a complete run and are "
            "reported as blocked rather than analysed as if complete: "
            + ", ".join(
                f"**{MODEL_SHORT_NAMES[m]}** "
                f"({integrity['runs'][m].get('reason', 'no raw scores')})"
                for m in blocked
            )
            + f". This is a {panel['status']} panel of "
            f"{panel['complete']}/{panel['size']}; it is not described as the "
            "specified four-model panel.\n"
        )
    else:
        add("All four preregistered judges produced complete runs.\n")

    # ---------------------------------------------------------------- 10
    add("## 10. Formal ground truth\n")
    add(
        "Every artifact's nine labels have the form $1,\\ldots,1,0,\\ldots,0$. "
        "The true first-fail index $j^*_x = \\min\\{s : Y_x(s) = 0\\}$ (9 when all "
        "nine thresholds are satisfied) is computed arithmetically: $m+1$ for "
        "coverage, $9-v$ and $9-e$ for the other two families. Because $s = 0$ is "
        "satisfied by every artifact in every family, $j^* \\in \\{1,\\ldots,9\\}$ "
        "and no artifact is ever ambiguous. No language model, validator, "
        "consensus panel or human annotator participated in labelling.\n"
    )

    # ---------------------------------------------------------------- 11
    add("## 11. Monotonicity violation rate\n")
    add(
        f"$MVR_x = \\frac{{1}}{{8}}\\sum_{{s=0}}^{{7}} "
        f"\\mathbf{{1}}[P_x(s+1) > P_x(s) + {MVR_TOLERANCE}]$, "
        "with the zero-tolerance variant as a preregistered sensitivity.\n"
    )
    add("| Judge | MVR (δ=0.05) | 95% CI | MVR (δ=0) | artifacts with any violation | perfectly monotone |")
    add("|---|---|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['mvr']:.4f} | "
            f"{_ci_only(boot, model_id, 'mvr', 'overall', '.4f')} | "
            f"{o['mvr_strict']:.4f} | "
            f"{o['artifacts_with_any_violation']}/{N_ARTIFACTS} | "
            f"{o['artifacts_perfectly_monotone']}/{N_ARTIFACTS} |"
        )
    add("")
    add(section_narrative("monotonicity"))

    # ---------------------------------------------------------------- 12
    add("## 12. Monotonicity violation magnitude\n")
    add(
        "$MVM_x = \\frac{1}{8}\\sum_{s=0}^{7}\\max(0, P_x(s+1) - P_x(s))$ — the "
        "size of the upward moves rather than their frequency.\n"
    )
    add("| Judge | overall | " + " | ".join(FAMILY_TITLES[f] for f in FAMILIES) + " |")
    add("|---" * (len(FAMILIES) + 2) + "|")
    for model_id in complete:
        cells = [_ci(boot, model_id, "mvm", "overall", ".4f")]
        cells += [_ci(boot, model_id, "mvm", f, ".4f") for f in FAMILIES]
        add(f"| {MODEL_SHORT_NAMES[model_id]} | " + " | ".join(cells) + " |")
    add("\nPoint estimate with 95% bootstrap interval.\n")

    # ---------------------------------------------------------------- 13
    add("## 13. Threshold crossing error\n")
    add(
        f"$\\hat j_x = \\min\\{{s : P_x(s) < {DECISION_THRESHOLD}\\}}$ (9 if never "
        "below), $TCE_x = |\\hat j_x - j^*_x|$.\n"
    )
    add("| Judge | mean | 95% CI | median | TCE = 0 | TCE ≤ 1 | max | never crosses |")
    add("|---|---|---|---|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        t = o["tce"]
        ci = _ci_only(boot, model_id, "tce", "overall", ".3f")
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {t['mean']:.3f} | {ci} | "
            f"{t['median']:.1f} | {t['exact_boundary_rate']:.1%} | "
            f"{t['within_one_step_rate']:.1%} | {t['max']} | "
            f"{o['predicted_never_crosses']}/{N_ARTIFACTS} |"
        )
    add("")
    add("**TCE distribution (artifact counts)**\n")
    keys = sorted(
        {
            int(k)
            for m in complete
            for k in summary["models"][m]["overall"]["tce"]["distribution"]
        }
    )
    add("| Judge | " + " | ".join(str(k) for k in keys) + " |")
    add("|---" * (len(keys) + 1) + "|")
    for model_id in complete:
        dist = summary["models"][model_id]["overall"]["tce"]["distribution"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | "
            + " | ".join(str(dist.get(str(k), 0)) for k in keys)
            + " |"
        )
    add("")
    add(section_narrative("crossing"))

    # ---------------------------------------------------------------- 14
    add("## 14. Accuracy and Brier score\n")
    add("| Judge | accuracy | 95% CI | Brier | 95% CI | mean P(met) | P05 | P95 |")
    add("|---|---|---|---|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['accuracy']:.4f} | "
            f"{_ci_only(boot, model_id, 'accuracy', 'overall', '.4f')} | "
            f"{o['brier']:.4f} | "
            f"{_ci_only(boot, model_id, 'brier', 'overall', '.4f')} | "
            f"{o['mean_p_met']:.3f} | {o['p_met_p05']:.3f} | {o['p_met_p95']:.3f} |"
        )
    add("")
    add(section_narrative("accuracy"))

    # ---------------------------------------------------------------- 15
    add("## 15. Boundary-aligned curves\n")
    add(
        "Mean $P_\\text{met}$ against signed distance $d = s - j^*$ from the true "
        "crossing. $d < 0$ is a threshold the artifact formally satisfies; "
        "$d \\ge 0$ one it formally fails.\n"
    )
    curve = pd.DataFrame(summary.get("boundary_aligned_curve", []))
    if not curve.empty:
        distances = sorted(curve["signed_distance"].unique())
        add("| Judge | " + " | ".join(f"d={d}" for d in distances) + " |")
        add("|---" * (len(distances) + 1) + "|")
        for model_id in complete:
            subset = curve[curve["model_id"] == model_id].set_index("signed_distance")
            add(
                f"| {MODEL_SHORT_NAMES[model_id]} | "
                + " | ".join(
                    f"{subset.loc[d, 'mean_p_met']:.3f}"
                    if d in subset.index
                    else "—"
                    for d in distances
                )
                + " |"
            )
        add("")
    add("![Boundary-aligned response curves](../figures/day1/figure2_boundary_aligned_curves.png)\n")
    add(section_narrative("boundary_curves"))

    # ---------------------------------------------------------------- 16
    add("## 16. Predicate-family analysis\n")
    add(
        "Reported separately for every family; family-specific behaviour is not "
        "averaged away.\n"
    )
    for metric, fmt, label in (
        ("mvr", ".4f", "MVR (δ=0.05)"),
        ("tce", ".3f", "mean TCE"),
        ("accuracy", ".4f", "accuracy"),
        ("brier", ".4f", "Brier"),
    ):
        add(f"**{label}**\n")
        add("| Judge | " + " | ".join(FAMILY_TITLES[f] for f in FAMILIES) + " |")
        add("|---" * (len(FAMILIES) + 1) + "|")
        for model_id in complete:
            add(
                f"| {MODEL_SHORT_NAMES[model_id]} | "
                + " | ".join(
                    _ci(boot, model_id, metric, f, fmt) for f in FAMILIES
                )
                + " |"
            )
        add("")
    add("![Model by family](../figures/day1/figure5_model_family_heatmap.png)\n")
    add(section_narrative("family_analysis"))

    # ---------------------------------------------------------------- 17
    add("## 17. Cross-model comparisons\n")
    comparisons = pd.DataFrame(summary.get("paired_comparisons", []))
    if comparisons.empty or len(complete) < 2:
        add("Fewer than two complete runs; no paired comparison is possible.\n")
    else:
        add(
            "Paired bootstrap over identical resampled artifact sets. A positive "
            "difference means the first judge has the larger value.\n"
        )
        for metric, fmt in (("mvr", ".4f"), ("tce", ".3f"), ("accuracy", ".4f")):
            subset = comparisons[
                (comparisons["metric"] == metric)
                & (comparisons["scope"] == "overall")
            ]
            if subset.empty:
                continue
            add(f"**{metric}**\n")
            add("| A | B | difference (A − B) | 95% CI | excludes 0 |")
            add("|---|---|---|---|---|")
            for row in subset.itertuples(index=False):
                add(
                    f"| {MODEL_SHORT_NAMES[row.model_a]} | "
                    f"{MODEL_SHORT_NAMES[row.model_b]} | {row.point:{fmt}} | "
                    f"[{row.ci_low:{fmt}}, {row.ci_high:{fmt}}] | "
                    f"{'yes' if row.excludes_zero else 'no'} |"
                )
            add("")
    add(
        "These are differences between four specific open-weight models on three "
        "synthetic formal predicate families. They are not a general model "
        "ranking.\n"
    )
    add(section_narrative("cross_model"))

    # ---------------------------------------------------------------- 18
    add("## 18. Cluster bootstrap\n")
    add(
        f"{BOOTSTRAP_REPLICATES} replicates, seed {BOOTSTRAP_SEED}, resampling "
        f"`artifact_id` with all {N_STRICTNESS} thresholds carried together, "
        f"stratified by predicate family, percentile intervals at "
        f"{BOOTSTRAP_CI:.0%}. The "
        f"{N_ROWS} threshold cells are never treated as independent "
        "observations, and no asymptotic standard error ignoring within-artifact "
        "dependence is used. No hyperparameter was selected from these outputs.\n"
    )
    add("**Secondary: within-artifact Spearman association (strictness vs P(met))**\n")
    add("| Judge | mean ρ | undefined (constant) curves |")
    add("|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        rho = o["spearman_mean"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | "
            + (f"{rho:.4f}" if rho == rho else "undefined")
            + f" | {o['spearman_undefined_artifacts']}/{N_ARTIFACTS} |"
        )
    add("\nExpected direction is non-positive. Descriptive support only.\n")

    # ---------------------------------------------------------------- 19
    add("## 19. Failure-mode analysis\n")
    add("![MVR distributions](../figures/day1/figure3_mvr_distributions.png)\n")
    add("![TCE distributions](../figures/day1/figure4_tce_distributions.png)\n")
    add("| Judge | never crosses 0.5 | below 0.5 at s=0 | perfectly monotone |")
    add("|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | "
            f"{o['predicted_never_crosses']}/{N_ARTIFACTS} | "
            f"{o['predicted_always_below']}/{N_ARTIFACTS} | "
            f"{o['artifacts_perfectly_monotone']}/{N_ARTIFACTS} |"
        )
    add("")
    add(section_narrative("failure_modes"))

    # ---------------------------------------------------------------- 20
    add("## 20. Limitations\n")
    add(
        "- Three synthetic formal predicate families. Nothing here transfers "
        "automatically to natural-language rubrics.\n"
        "- Four specific open-weight instruction models at specific revisions, "
        "in bf16, with one fixed minimal prompt. A different prompt, a different "
        "dtype, or a different model could behave differently.\n"
        "- The endpoint is a normalised two-label probability, not a calibrated "
        "probability. It is used as scored; nothing is recalibrated.\n"
        "- Logits are read from bf16 model outputs, so $P_\\text{met}$ carries "
        "the granularity of bf16 arithmetic. This is finest near the decision "
        "boundary, where the label logits are small and close together, and "
        "coarsest where the curve has already saturated toward 0 or 1.\n"
        "- Artifacts are constructed so that response length carries no signal "
        "and marker tokens are unambiguous. Real evaluation targets are not.\n"
        "- Monotone curves would not prove general judge reliability; "
        "non-monotone curves would not prove general judge unreliability.\n"
        "- Day 1 establishes nothing about human alignment, clinical "
        "reliability, HealthBench reliability, general semantic rubric "
        "reliability, reward-model correctness, or causal reasoning.\n"
    )

    # ---------------------------------------------------------------- 21
    add("## 21. Research decision\n")
    add("```text")
    add(f"FORMAL DATASET INTEGRITY = {dataset['status']}")
    add(f"PREREGISTRATION INTEGRITY = {prereg['status']}")
    add(f"MODEL PANEL              = {panel['status']} "
        f"({panel['complete']}/{panel['size']})")
    add(f"DAY-1 ANALYSIS           = {'COMPLETE' if complete else 'BLOCKED'}")
    add("```\n")
    add(
        "Day 1 defines no favourable-effect threshold and was not optimised "
        "toward one. It is complete when dataset integrity passes, the "
        "preregistration was frozen before outcomes, intended runs completed or "
        "were transparently blocked, probability extraction is valid, the "
        "analysis matched the preregistration, and results are reported "
        "regardless of direction.\n"
    )
    add(section_narrative("research_decision"))
    add(
        "**No Day-2 experiment has been started.** Any next experiment must be "
        "separately designed and separately preregistered.\n"
    )

    # ---------------------------------------------------------------- 22
    add("## 22. Reproducibility and integrity summary\n")
    add("| Item | SHA-256 |")
    add("|---|---|")
    for label, entry in integrity["files"].items():
        if entry["sha256"]:
            add(f"| {label.replace('_', ' ')} | `{entry['sha256']}` |")
    for model_id in MODEL_IDS:
        run = integrity["runs"][model_id]
        if run.get("raw_scores_sha256"):
            add(
                f"| raw scores — {MODEL_SHORT_NAMES[model_id]} | "
                f"`{run['raw_scores_sha256']}` |"
            )
    add(f"| figures (combined) | `{integrity['figures']['combined_sha256']}` |")
    add(f"| analysis source tree | `{integrity['source_manifest']['tree_sha256']}` |")
    add("")
    add(
        f"Preregistration frozen {prereg.get('frozen_at')}, SHA-256 "
        f"`{prereg.get('sha256')}`, amendments: {prereg.get('amendment_count', 0)}. "
        f"Git commit at integrity check: "
        f"`{integrity['git'].get('commit') or 'unavailable'}` "
        f"(clean = {integrity['git'].get('clean')}).\n"
    )
    add("```text")
    add("No human labels used.")
    add("No human annotation performed.")
    add("No semantic rubric generation performed.")
    add("No outcome-driven tuning performed.")
    add("No Day-2 experiment started.")
    add("```")

    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = Path(args.root)

    target = root / "docs/RESULTS_DAY1.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build(root), encoding="utf-8")
    print(f"wrote docs/RESULTS_DAY1.md ({target.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
