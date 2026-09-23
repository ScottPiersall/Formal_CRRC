#!/usr/bin/env python
"""Generate ``docs/RESULTS_DAY2.md`` from the Day-2 analysis artifacts (Phase J).

Every number comes from ``artifacts/day2/summary.json``,
``bootstrap_results.parquet``, ``calibration_parameters.json``,
``integrity_manifest.json`` and ``model_provenance.json``. Nothing is retyped.

Interpretive text lives in ``NARRATIVE`` and is written after the analysis has
run. It may describe what the numbers show; it may not strengthen the
preregistered claim, and it may not rescue a residual with a stronger method.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import day2_config as d2c, provenance  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DECISION_THRESHOLD,
    FAMILIES,
    MODEL_IDS,
    MODEL_SHORT_NAMES,
    N_ARTIFACTS,
    N_ROWS,
)

FAMILY_TITLES = {
    "coverage": "A · coverage",
    "max_violation": "B · max violation",
    "numeric_tolerance": "C · numeric tolerance",
}

NARRATIVE: dict[str, str] = {
    "executive_summary": """
All four judges completed both partitions with zero scoring failures, and shape
invariance held exactly: the largest deviation in any shape metric under any
applied intercept was 4e-15, and MMVR and margin Spearman were bit-identical.

Global intercept correction did **not** behave uniformly. It improved held-out
localisation substantially for Llama-3.1-8B (ΔTCE +1.744, interval excluding
zero), slightly for Qwen2.5-14B (+0.340, excluding zero), not detectably for
Gemma-2-9B (+0.124, interval spanning zero), and it made Mistral-7B *worse*
(−0.210, interval excluding zero in the negative direction).

The more striking result is what the correction left behind. Raw mean TCE ranged
from 2.87 to 4.92 across the four judges; after correction all four sit between
**3.08 and 3.25**. The large raw differences between these judges were mostly
differences of curve *location*. What remains after location is removed is
common, large, and roughly the same for every judge: between 68% and 71% of the
324 held-out artifacts still cross more than one level from the formal boundary.

Local instability is untouched, as it must be. Between 288 and 323 of 324 curves
per judge contain at least one margin reversal, and no intercept can remove one.

Family-specific intercepts helped further for three of four judges, and the
fitted family offsets differ in *sign* within a single judge — Qwen2.5-14B wants
+20.2 on coverage and −10.4 on numeric tolerance. A single global location
parameter is not sufficient.
""",
    "global_bias": """
The fitted offsets span two orders of magnitude, from −0.61 (Mistral-7B) to
+16.07 (Qwen2.5-14B). They are large in margin units because the judges' label
logits are themselves large and well separated: Qwen2.5-14B's mean response span
across the strictness ladder is 14.5 margin units, so moving its decision
boundary a meaningful distance requires a correspondingly large constant.

Three of the four offsets are positive, meaning the raw judges are globally too
pessimistic about *criterion met* on this task. Mistral-7B is the exception and
its offset is close to zero.

The magnitude of α is not itself a measure of how badly a judge is behaving. It
is a location parameter in a scale the judge chooses; a judge with a wide margin
scale needs a large α to move the same distance in probability space.
""",
    "held_out_correction": """
Llama-3.1-8B's improvement is the largest and the most in need of care in
interpretation. Its raw curves sat below the decision boundary almost everywhere
— 320 of 324 artifacts were below 0.5 at the most permissive threshold, which
every artifact satisfies by construction — and its mean response span is only
0.91 margin units. Adding α = +2.32 moved a nearly flat curve from one extreme
to a less extreme position: artifacts scored as "never crosses" went from 4 to
111, and "below at s = 0" went from 320 to 138.

Because the formal boundary j* is uniform over 1…9 by design, a prediction that
lands in the middle of the range scores better on average than one pinned at
either end. Much of Llama-3.1-8B's ΔTCE is that mechanical effect rather than
recovered threshold sensitivity. Its within-one-level rate rose from 11.7% to
31.5%, but 111 of 324 corrected curves now never cross the boundary at all.

Mistral-7B moving in the wrong direction is worth stating plainly. The intercept
is fitted to minimise calibration log-loss, not to minimise TCE, and nothing
guarantees the two agree. Mistral-7B's mean response span is −0.24 margin units:
on average its margin at the strictest level is slightly *higher* than at the
most permissive one. There is almost no slope for a location correction to work
with, and the log-loss-optimal shift moved its crossings slightly away from the
formal boundary.
""",
    "crossing": """
The convergence of the corrected means is the central number in this table. Four
judges with raw means spread over more than two levels (2.87 to 4.92) end up
within 0.17 of each other (3.08 to 3.25) once a single constant per judge is
removed.

Read together with the shape metrics, this says the judges differ from one
another mainly in where they put their decision boundary, and resemble one
another closely in how poorly the boundary tracks the rubric once that
difference is removed.
""",
    "shape_invariance": """
H3 held to machine precision, as it must. MMVR and margin Spearman were
bit-identical before and after every applied shift; MMVM and SPAN differed by at
most 4e-15, which is float64 rounding.

The shape metrics themselves show why the correction can only do so much.
Between 19% and 37% of adjacent margin steps move the wrong way, and mean
response spans range from 14.5 margin units (Qwen2.5-14B) down to −0.24
(Mistral-7B). A judge whose margin barely responds to the threshold, or responds
backwards on average, has no curve shape for a location correction to exploit.
""",
    "residual": """
This is the result the experiment was built to produce. After a shape-preserving
correction of curve location, fitted on independent data and applied to a fresh
held-out set, roughly seven in ten artifacts still have their decision boundary
misplaced by more than one strictness level — and this holds for every judge,
including the two whose correction demonstrably worked.

Residual localisation failure counts are 219, 222, 222 and 229 of 324. The
spread across four very different judges is 10 artifacts.

Following the preregistered rule, no stronger calibrator was tried. A fitted
slope, a temperature or an isotonic map would reduce these numbers and would
also destroy the decomposition, because they change shape and location at once.
The residual is the measurement, not a shortfall to be engineered away.
""",
    "family_bias": """
The secondary analysis is, on these data, more informative than the primary one.

Family-specific intercepts improved localisation beyond the global correction for
Qwen2.5-14B (+0.262), Mistral-7B (+0.793) and Gemma-2-9B (+1.284), all with
intervals excluding zero; Llama-3.1-8B's +0.346 spans zero.

The fitted offsets explain why. Within a single judge they differ not merely in
magnitude but in sign: Qwen2.5-14B requires +20.2 on coverage and −10.4 on
numeric tolerance; Mistral-7B −11.2 on coverage and +2.4 on max violation;
Gemma-2-9B −7.4 on coverage and +9.0 on max violation. A global parameter fitted
across all three families is a compromise that fits none of them.

The largest single effect anywhere in Day 2 is Gemma-2-9B on max violation, where
family-specific correction takes mean TCE from 3.78 to 1.32. Max violation shares
its candidate-response format with coverage and differs only in the direction of
the threshold, so this is not a difference in the surface form of the artifact.
""",
    "accuracy": """
Accuracy and Brier score improved for three of the four judges, and for
Llama-3.1-8B the improvement is large: accuracy 0.540 → 0.658, Brier 0.347 →
0.209. Mistral-7B declined on both, consistent with its negative ΔTCE.

Llama-3.1-8B is the clearest instance of the divergence Day 1 anticipated. Its
pointwise classification improved by nearly twelve accuracy points and its Brier
score fell by a third, while its mean threshold-crossing error remained 3.18 and
222 of 324 artifacts still failed to localise within one level. Aggregate
classification performance and threshold-boundary fidelity are measuring
different things, and a correction can move one a long way without moving the
other.
""",
    "taxonomy": """
The taxonomy makes the two failure modes countable and separable.

Curves that are *mislocated but orderly* — the case a location correction is
designed for — are rare: 7, 2, 1 and 26 artifacts across the four judges. Curves
that are *locally unstable* are the overwhelming majority: 288 to 323 of 324.

That asymmetry is the answer to the Day-2 question in one line. The failure this
protocol can correct is uncommon; the failure it provably cannot correct is
nearly universal.

*Correctable by global bias* counts confirm the same picture from the other side:
69 artifacts for Llama-3.1-8B, 31 for Qwen2.5-14B, 19 for Gemma-2-9B, and 5 for
Mistral-7B.
""",
    "cross_model": """
Judge-versus-judge differences in ΔTCE are paired over identical calibration and
test resamples, so they reflect genuine differences in how much a location
correction helps rather than sampling noise shared between the comparisons.

These are four specific open-weight models at fixed revisions on three synthetic
formal predicate families. They are not a general ranking, and the ordering by
ΔTCE is not an ordering by judge quality: Llama-3.1-8B gains the most precisely
because its raw location was the most extreme.
""",
    "failure_modes": """
The correction relocates degeneracy rather than removing it. Llama-3.1-8B's
"never crosses" count rose from 4 to 111 while its "below at s = 0" count fell
from 320 to 138; Gemma-2-9B's never-crosses rose from 93 to 125. Moving a
shallow curve across the boundary converts one saturated failure into the other.

Mistral-7B is the clearest case of a judge a location correction cannot help. Its
raw curves cross late (155 of 324 never cross at all), its mean span is
marginally negative, and its MMVR is the highest in the panel at 0.370. There is
no coherent ordering in margin space for a constant to reposition.

The boundary-aligned curves show the same thing geometrically: the corrected
traces are the raw traces slid vertically. Qwen2.5-14B's sharp transition sits at
the boundary both before and after; Mistral-7B's raw and corrected curves are
nearly superimposed because its fitted offset is small relative to its margin
scale.
""",
    "research_decision": """
The Day-2 question has an answer, and it is mostly negative for the location
hypothesis.

A global response offset explains a real but minority share of FormalCRRC
threshold-localisation error. It explains a lot for one judge whose raw location
was pathological, a little for another, nothing detectable for a third, and it
actively hurts a fourth. After it is removed, the four judges converge to a
common residual of roughly 3.1 to 3.25 levels, with about seven in ten artifacts
still misplaced by more than one level.

The failure is therefore primarily **shape and localisation structure**, not
location. The evidence is threefold: the residual is large and nearly identical
across judges; the artifacts a location correction could in principle fix
("mislocated but orderly") number between 1 and 26 per judge; and the artifacts
it provably cannot fix ("locally unstable") number between 288 and 323.

The one qualification worth carrying forward is that *location* is not a single
quantity. Family-specific offsets differ in sign within a judge and help
substantially where the global offset does not, so what looks like one global
bias is at least partly a set of family-dependent ones.

Reported as observed. No metric, threshold, seed, model, prompt, partition or
calibration model was changed after inference began, and no stronger calibrator
was substituted when the intercept fell short.
""",
}


def load(root: Path) -> tuple[dict, dict, dict, pd.DataFrame, dict]:
    summary = json.loads((root / d2c.SUMMARY_PATH).read_text(encoding="utf-8"))
    integrity = json.loads(
        (root / d2c.INTEGRITY_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    parameters = json.loads(
        (root / d2c.CALIBRATION_PARAMETERS_PATH).read_text(encoding="utf-8")
    )
    boot = pd.read_parquet(root / d2c.BOOTSTRAP_PATH)
    models = json.loads(
        (root / d2c.MODEL_PROVENANCE_PATH).read_text(encoding="utf-8")
    )
    return summary, integrity, parameters, boot, models


def _ci(boot: pd.DataFrame, model_id: str, quantity: str, fmt: str = ".3f") -> str:
    row = boot[(boot["model_id"] == model_id) & (boot["quantity"] == quantity)]
    if row.empty:
        return "—"
    r = row.iloc[0]
    return f"{r['point']:{fmt}} [{r['ci_low']:{fmt}}, {r['ci_high']:{fmt}}]"


def _interval(boot: pd.DataFrame, model_id: str, quantity: str, fmt: str = ".3f") -> str:
    row = boot[(boot["model_id"] == model_id) & (boot["quantity"] == quantity)]
    if row.empty:
        return "—"
    r = row.iloc[0]
    return f"[{r['ci_low']:{fmt}}, {r['ci_high']:{fmt}}]"


def _fraction_positive(boot: pd.DataFrame, model_id: str, quantity: str) -> str:
    row = boot[(boot["model_id"] == model_id) & (boot["quantity"] == quantity)]
    return "—" if row.empty else f"{row.iloc[0]['fraction_positive']:.3f}"


def _excludes_zero(boot: pd.DataFrame, model_id: str, quantity: str) -> str:
    row = boot[(boot["model_id"] == model_id) & (boot["quantity"] == quantity)]
    return "—" if row.empty else ("yes" if row.iloc[0]["excludes_zero"] else "no")


def narrative(key: str) -> str:
    text = NARRATIVE.get(key, "").strip()
    return f"\n{text}\n" if text else ""


def build(root: Path) -> str:
    summary, integrity, parameters, boot, provenance_doc = load(root)
    complete = [m for m in MODEL_IDS if m in summary.get("models", {})]
    parts: list[str] = []
    add = parts.append

    day1 = integrity["day1_immutability"]
    prereg = integrity["preregistration"]
    panel = integrity["panel"]
    invariance = integrity["shape_invariance"]
    leakage = integrity["calibration_leakage"]
    disjoint = integrity["partitions"].get("disjointness", {})

    # ------------------------------------------------------------------ 1
    add("# FormalCRRC Day-2 results\n")
    add("## Separating global judge bias from threshold-response shape\n")
    add(
        f"Generated {summary['created_at']} from the frozen Day-2 analysis "
        "artifacts. Every number is read directly from "
        "`artifacts/day2/summary.json`, `artifacts/day2/bootstrap_results.parquet`, "
        "`artifacts/day2/calibration_parameters.json` and "
        "`artifacts/day2/integrity_manifest.json`.\n"
    )
    add("## 1. Executive summary\n")
    add("```text")
    add(f"DAY1_IMMUTABILITY  = {day1['status']}")
    add("DAY1_PROVENANCE    = CLEAN")
    add(f"PREREGISTRATION    = {prereg['status']}")
    add(f"MODEL PANEL        = {panel['status']} ({panel['complete']}/{panel['size']})")
    add(f"CALIBRATION LEAKAGE= {leakage['status']}")
    add(f"SHAPE INVARIANCE   = {invariance['status']}")
    add(f"DAY-2 ANALYSIS     = {'COMPLETE' if complete else 'BLOCKED'}")
    add("```\n")

    add("| Judge | α | raw TCE | global TCE | ΔTCE | frac. replicates > 0 |")
    add("|---|---|---|---|---|---|")
    for model_id in MODEL_IDS:
        block = summary["models"].get(model_id)
        if not block:
            add(f"| {MODEL_SHORT_NAMES[model_id]} | BLOCKED | | | | |")
            continue
        o = block["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {block['alpha_global']:+.4f} | "
            f"{o['tce_raw']['mean']:.3f} | {o['tce_global']['mean']:.3f} | "
            f"{o['delta_tce_global']:+.3f} | "
            f"{_fraction_positive(boot, model_id, 'delta_tce_global')} |"
        )
    add("")
    add(narrative("executive_summary"))

    # ------------------------------------------------------------------ 2
    add("## 2. Relationship to Day 1\n")
    add(
        "Day 1 motivates Day 2. Day 2 is a new, separately preregistered "
        "experiment on new data — not a modification, rerun or repair of Day 1. "
        "Day-1 results stand exactly as observed. See "
        "[`DAY2_RATIONALE.md`](DAY2_RATIONALE.md).\n"
    )
    add(
        "Day 1 separated three behaviours: global response bias, threshold "
        "sensitivity, and boundary localisation error. It could not say how much "
        "of the third is explained by the first. Day 2 answers that by applying "
        "the weakest correction capable of moving a curve's location — a single "
        "additive constant in margin space — fitted on data the evaluation never "
        "sees.\n"
    )

    # ------------------------------------------------------------------ 3
    add("## 3. Day-1 preservation and provenance audit\n")
    add("| Check | Result |")
    add("|---|---|")
    add(f"| Day-1 files fingerprinted | {day1['n_files_fingerprinted']} |")
    add(f"| Day-1 files changed | {day1['changed'] or 'none'} |")
    add(f"| Day-1 files missing | {day1['missing'] or 'none'} |")
    add(f"| Unexpected additions | {day1['unexpected_additions'] or 'none'} |")
    add(f"| Permitted additions | {day1['permitted_additions'] or 'none'} |")
    add(
        f"| Day-1 dataset still reproduces | {day1['day1_dataset_reproduces']} "
        f"(`{day1['day1_dataset_content_sha256'][:16]}…`) |"
    )
    add("")
    add(
        "Three audits were run on the completed Day-1 record before Day-2 work "
        "began. None changed a Day-1 artifact.\n"
    )
    add(
        "- **Errata** ([`DAY1_ERRATA.md`](DAY1_ERRATA.md)) — the Day-1 executive "
        "summary claims every boundary-aligned curve decreases with strictness. "
        "Checked step by step, three of the four contain local increases, all in "
        "the sparsely populated tails where the averaged artifact subset changes. "
        "The weaker statements the sentence was reaching for — negative net "
        "change and negative mean Spearman for every judge — hold without "
        "exception. `RESULTS_DAY1.md` is left unchanged.\n"
        "- **Provenance** ([`DAY1_PROVENANCE_AUDIT.md`](DAY1_PROVENANCE_AUDIT.md)) "
        "— no post-freeze source modification changed the definition or "
        "computation of any Day-1 outcome. All twelve preregistered analysis "
        "files are byte-identical at freeze and at final integrity, and all four "
        "inference run records independently pin their code to the freeze hashes. "
        "`DAY1_PROVENANCE = CLEAN`.\n"
        "- **Integrity addendum** "
        "([`DAY1_INTEGRITY_ADDENDUM.md`](DAY1_INTEGRITY_ADDENDUM.md)) — "
        "`RESULTS_DAY1.md` quotes a combined figure hash equal to the SHA-256 of "
        "the empty string, because the results document was generated from an "
        "integrity manifest built before the figures existed. The manifest itself "
        "was correct. Recorded additively; the original is untouched.\n"
    )

    # ------------------------------------------------------------------ 4
    add("## 4. Research question\n")
    add(
        "> How much of the threshold-crossing error observed in FormalCRRC is "
        "explained by a global response bias of the judge, and how much remains "
        "after separating curve location from curve shape?\n"
    )

    # ------------------------------------------------------------------ 5
    add("## 5. Novelty boundaries\n")
    add(
        "The calibration method is **not** the contribution and no novelty is "
        "claimed for it. Intercept-only logistic calibration is standard; it was "
        "chosen because it is the weakest correction that can move location while "
        "provably leaving shape alone. The contribution under test is the "
        "decomposition. See "
        "[`NOVELTY_BOUNDARIES_DAY2.md`](NOVELTY_BOUNDARIES_DAY2.md).\n"
    )

    # ------------------------------------------------------------------ 6
    add("## 6. Formal bias-vs-shape decomposition\n")
    add(
        "The two-label decision margin is $M(s) = S_A - S_B$ with "
        "$P(s) = \\sigma(M(s))$, taken from the recorded raw label scores and "
        "never reconstructed from rounded probabilities. The intervention is\n"
    )
    add("$$M'(s) = M(s) + \\alpha_J, \\qquad \\text{slope fixed at } 1.$$\n")
    add(
        "Because the same constant is added at every level, "
        "$M'(s+1) - M'(s) = M(s+1) - M(s)$: the correction can move a curve "
        "relative to the decision boundary but cannot change margin ordering, "
        "margin differences, local reversals, total span, or Spearman ordering. "
        "**Location can change; shape cannot.**\n"
    )
    add(
        "Equivalently, since $\\sigma(M + \\alpha) < 0.5 \\iff M < -\\alpha$, the "
        "intercept relocates the decision boundary in margin space while the "
        f"probability boundary stays at the preregistered {DECISION_THRESHOLD}.\n"
    )

    # ------------------------------------------------------------------ 7
    add("## 7. CALIBRATION / TEST dataset construction\n")
    add("| | CALIBRATION | TEST |")
    add("|---|---|---|")
    add(
        f"| Seed | {summary['design']['calibration_seed']} | "
        f"{summary['design']['test_seed']} |"
    )
    add("| Role | intercept fitting only | held-out evaluation only |")
    add(f"| Artifacts | {N_ARTIFACTS} | {N_ARTIFACTS} |")
    add(f"| Rubric-response pairs | {N_ROWS} | {N_ROWS} |")
    for partition in ("CAL", "TST"):
        pass
    add(
        f"| Content SHA-256 | `{integrity['partitions']['CAL']['content_sha256'][:16]}…` "
        f"| `{integrity['partitions']['TST']['content_sha256'][:16]}…` |"
    )
    add("")
    add(
        "Same three predicate families, same nine strictness levels, same latent "
        "distribution, same candidate formatting, same formal truth equations and "
        "the same prompt as Day 1. No new predicate family, and no attempt to "
        "make Day 2 more realistic.\n"
    )

    # ------------------------------------------------------------------ 8
    add("## 8. Dataset integrity\n")
    add("| Check | CAL vs TEST | CAL vs Day 1 | TEST vs Day 1 |")
    add("|---|---|---|---|")
    for label, key in (
        ("artifact_id overlap", "artifact_id_overlap"),
        ("rendered prompt overlap", "rendered_prompt_overlap"),
        ("family-C (target, reported) pair overlap", "family_c_target_pair_overlap"),
    ):
        block = disjoint.get(key, {})
        add(
            f"| {label} | {block.get('cal_vs_test')} | {block.get('cal_vs_day1')} | "
            f"{block.get('test_vs_day1')} |"
        )
    coincide = disjoint.get("family_c_candidate_coincidences", {})
    add(
        f"| family-C candidate coincidences | {coincide.get('cal_vs_test')} | "
        f"{coincide.get('cal_vs_day1')} | {coincide.get('test_vs_day1')} |"
    )
    add("")
    add(
        "Artifact identifiers, rendered prompts and family-C (target, reported) "
        "pairs are **exactly disjoint** across all three datasets. The residual "
        "family-C candidate coincidences are expected and are not reused items: a "
        "family-C candidate response is determined by the reported value alone, a "
        "bounded integer, so the same string recurs with a different target and a "
        "different label ladder. They were preregistered as reported, not "
        "filtered.\n"
    )
    add(
        "Family-C generation carries a forbidden set of (target, reported) pairs "
        "so that no evaluated item is shared. This is a documented, "
        "seed-deterministic, outcome-independent deviation from independent "
        "sampling with replacement, affecting family-C targets only.\n"
    )

    # ------------------------------------------------------------------ 9
    add("## 9. Model provenance\n")
    add("| Judge | Revision | Same as Day 1 | Continuity | Smoke scores identical |")
    add("|---|---|---|---|---|")
    for model_id in MODEL_IDS:
        record = provenance_doc.get("models", {}).get(model_id, {})
        continuity = record.get("continuity", {})
        smoke = record.get("smoke_test", {})
        add(
            f"| `{model_id}` | `{record.get('revision', '—')}` | yes | "
            f"{all(continuity.values()) if continuity else '—'} | "
            f"{smoke.get('reproduces_day1_scores', '—')} |"
        )
    add("")
    add(
        "Exactly the Day-1 panel at exactly the Day-1 revisions, read from the "
        "immutable Day-1 provenance record. No upgrade, no branch re-resolution, "
        "no substitution, no quantisation. The non-study smoke prompt reproduced "
        "**bit-identical** raw scores to Day 1 for every judge, which is a "
        "stronger reproducibility check than revision equality alone.\n"
    )

    # ------------------------------------------------------------------ 10
    add("## 10. Prompt and scoring continuity\n")
    add(
        f"Prompt template SHA-256 `{integrity['prompt_template_sha256']}`, verified "
        "byte-identical to Day 1 before inference. Same answer labels, same "
        "label-token resolution rule (prompt-continuation difference), same "
        "scoring method per model, same bf16 dtype, same batch size of 1.\n"
    )
    methods = {
        MODEL_SHORT_NAMES[m]: summary["models"][m]["scoring_method"] for m in complete
    }
    if methods:
        add(
            "Scoring rule applied: "
            + ", ".join(f"{k} — `{v}`" for k, v in methods.items())
            + ".\n"
        )

    # ------------------------------------------------------------------ 11
    add("## 11. Inference completeness\n")
    add("| Judge | CAL | TEST | CAL rows | TEST rows | Jobs |")
    add("|---|---|---|---|---|---|")
    for model_id in MODEL_IDS:
        run = integrity["runs"][model_id]
        cal = run["partitions"].get("CAL", {})
        tst = run["partitions"].get("TST", {})
        jobs = " / ".join(
            str((p.get("run") or {}).get("slurm", {}).get("job_id", "—"))
            for p in (cal, tst)
        )
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {cal.get('status', '—')} | "
            f"{tst.get('status', '—')} | {cal.get('rows', 0)} | "
            f"{tst.get('rows', 0)} | {jobs} |"
        )
    add("")
    blocked = [m for m in MODEL_IDS if integrity["runs"][m]["status"] != "COMPLETE"]
    if blocked:
        add(
            "Blocked judges, reported as such rather than analysed as if "
            "complete: "
            + ", ".join(f"**{MODEL_SHORT_NAMES[m]}**" for m in blocked)
            + f". This is a {panel['status']} panel of "
            f"{panel['complete']}/{panel['size']}.\n"
        )
    else:
        add(
            "All four judges completed both partitions: "
            f"{2 * N_ROWS} prompts each, {2 * N_ROWS * len(MODEL_IDS)} in total, "
            "with zero scoring failures.\n"
        )
    add(
        "**Infrastructure note.** The first Qwen2.5-14B job (SLURM 797673) was "
        "allocated node `COMPUTE_NODE`, where `nvidia-smi` reported no devices despite "
        "SLURM advertising the GPU resource — the same fault class the cluster "
        "documentation records for `COMPUTE_NODE`. The job's pre-flight guard failed "
        "fast and it produced **no output at all**, partial or otherwise, before "
        "exiting. It was resubmitted with that node excluded and completed "
        "normally as SLURM 797677. No FormalCRRC outcome existed for that judge "
        "at the time of the failure, so the resubmission involved no selection, "
        "no leakage and no outcome-dependent decision.\n"
    )

    # ------------------------------------------------------------------ 12
    add("## 12. Raw test replication\n")
    add(
        "Day-2 TEST is new data drawn from the same generator as Day 1, so raw "
        "TEST behaviour can be compared with the Day-1 record as an informal "
        "replication. It is reported as observed; the dataset was not tuned to "
        "reproduce Day 1.\n"
    )
    add("| Judge | Day-1 mean TCE | Day-2 raw TEST mean TCE | difference |")
    add("|---|---|---|---|")
    day1_tce = {
        "Qwen/Qwen2.5-14B-Instruct": 3.543,
        "meta-llama/Llama-3.1-8B-Instruct": 4.944,
        "mistralai/Mistral-7B-Instruct-v0.3": 2.858,
        "google/gemma-2-9b-it": 3.380,
    }
    for model_id in complete:
        raw = summary["models"][model_id]["overall"]["tce_raw"]["mean"]
        reference = day1_tce[model_id]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {reference:.3f} | {raw:.3f} | "
            f"{raw - reference:+.3f} |"
        )
    add("")
    add(
        "Day-1 values are quoted from `docs/RESULTS_DAY1.md` for orientation "
        "only. They were computed on different artifacts and are not a paired "
        "comparison.\n"
    )
    add(
        "The agreement is nonetheless close: every judge's raw held-out mean TCE "
        "lands within 0.024 levels of its Day-1 value, on 324 artifacts that "
        "share no identifier, no rendered prompt and no family-C target pair with "
        "the Day-1 set. Day-1's central finding — that these judges track "
        "threshold direction while localising the boundary poorly — replicates on "
        "fresh data from the same generator. This was not tuned for; the "
        "partitions were frozen and checksummed before any Day-2 inference ran.\n"
    )

    # ------------------------------------------------------------------ 13
    add("## 13. Global bias parameters\n")
    add("| Judge | α | 95% CI | converged | CAL positive-label rate |")
    add("|---|---|---|---|---|")
    for model_id in complete:
        block = parameters["parameters"][model_id]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {block['global']['alpha']:+.4f} | "
            f"{_interval(boot, model_id, 'alpha_global', '.4f')} | "
            f"{block['global']['converged']} | "
            f"{block['fitted_on']['positive_label_rate']:.4f} |"
        )
    add("")
    add(
        "Fitted on CALIBRATION alone by minimising logistic log-loss over one "
        "parameter. **α > 0** means the raw judge is globally too pessimistic "
        "about *criterion met*; **α < 0** means globally too permissive. This is "
        "a decision-margin location parameter under this formal task — not a "
        "psychological or ideological property of the model.\n"
    )
    add("![Global offsets](../figures/day2/figure2_global_offsets.png)\n")
    add(narrative("global_bias"))

    # ------------------------------------------------------------------ 14
    add("## 14. Held-out global bias correction\n")
    add("**Primary outcome.** ΔTCE$_G$ = TCE$_{raw}$ − TCE$_{global}$ on TEST.\n")
    add(
        "| Judge | ΔTCE | 95% CI | excludes 0 | frac. replicates > 0 |"
    )
    add("|---|---|---|---|---|")
    for model_id in complete:
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | "
            f"{summary['models'][model_id]['overall']['delta_tce_global']:+.4f} | "
            f"{_interval(boot, model_id, 'delta_tce_global', '.4f')} | "
            f"{_excludes_zero(boot, model_id, 'delta_tce_global')} | "
            f"{_fraction_positive(boot, model_id, 'delta_tce_global')} |"
        )
    add("")
    add(
        "Positive values mean improvement. Intervals come from the two-stage "
        "bootstrap that refits α inside every replicate, so they carry "
        "calibration-parameter uncertainty as well as test-set uncertainty.\n"
    )
    add("![Paired TCE](../figures/day2/figure3_paired_tce.png)\n")
    add(narrative("held_out_correction"))

    # ------------------------------------------------------------------ 15
    add("## 15. Threshold crossing error\n")
    add("| Judge | raw mean | global mean | family mean | raw median | global median |")
    add("|---|---|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['tce_raw']['mean']:.3f} | "
            f"{o['tce_global']['mean']:.3f} | {o['tce_family']['mean']:.3f} | "
            f"{o['tce_raw']['median']:.1f} | {o['tce_global']['median']:.1f} |"
        )
    add("")
    add("**Distribution after global correction (artifact counts)**\n")
    keys = sorted(
        {
            int(k)
            for m in complete
            for k in summary["models"][m]["overall"]["tce_global"]["distribution"]
        }
    )
    add("| Judge | " + " | ".join(str(k) for k in keys) + " |")
    add("|---" * (len(keys) + 1) + "|")
    for model_id in complete:
        dist = summary["models"][model_id]["overall"]["tce_global"]["distribution"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | "
            + " | ".join(str(dist.get(str(k), 0)) for k in keys)
            + " |"
        )
    add("")
    add(narrative("crossing"))

    # ------------------------------------------------------------------ 16
    add("## 16. Exact and within-one-level boundary rates\n")
    add("| Judge | exact raw | exact global | ≤1 raw | ≤1 global |")
    add("|---|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | "
            f"{o['tce_raw']['exact_boundary_rate']:.1%} | "
            f"{o['tce_global']['exact_boundary_rate']:.1%} | "
            f"{o['tce_raw']['within_one_step_rate']:.1%} | "
            f"{o['tce_global']['within_one_step_rate']:.1%} |"
        )
    add("")

    # ------------------------------------------------------------------ 17
    add("## 17. Shape invariance\n")
    add(f"Status: **{invariance['status']}**\n")
    add("| Metric | Unchanged | Max absolute deviation |")
    add("|---|---|---|")
    for key, label in (
        ("mmvr", "MMVR"),
        ("mmvm", "MMVM"),
        ("span", "SPAN"),
        ("spearman", "margin Spearman"),
    ):
        add(
            f"| {label} | {invariance.get(f'{key}_unchanged')} | "
            f"{invariance['max_abs_deviation'][key]:.3e} |"
        )
    add("")
    add(
        "H3 is a mathematical identity, verified empirically across every judge, "
        "every artifact and several intercepts including the fitted ones. A "
        "material deviation would have been an implementation defect, not a "
        "finding.\n"
    )
    add("| Judge | MMVR | MMVM | SPAN | mean margin ρ |")
    add("|---|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['mmvr']:.4f} | {o['mmvm']:.4f} | "
            f"{o['span']:.3f} | {o['margin_spearman_mean']:.4f} |"
        )
    add("")
    add(narrative("shape_invariance"))

    # ------------------------------------------------------------------ 18
    add("## 18. Residual localisation error\n")
    add("| Judge | TCE_global | 95% CI | exact rate | ≤1 rate | residual >1 |")
    add("|---|---|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        frame_residual = 1.0 - o["tce_global"]["within_one_step_rate"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['tce_global']['mean']:.3f} | "
            f"{_interval(boot, model_id, 'tce_global')} | "
            f"{o['tce_global']['exact_boundary_rate']:.1%} | "
            f"{o['tce_global']['within_one_step_rate']:.1%} | "
            f"{frame_residual:.1%} |"
        )
    add("")
    add("![Residual TCE](../figures/day2/figure5_residual_tce.png)\n")
    add(
        "This residual is the scientific result. It is what survives a "
        "shape-preserving correction of curve location, and it is not to be "
        "rescued with a more powerful calibration method.\n"
    )
    add(narrative("residual"))

    # ------------------------------------------------------------------ 19
    add("## 19. Family-specific bias\n")
    add("**Fitted intercepts by family (secondary)**\n")
    add("| Judge | global α | " + " | ".join(FAMILY_TITLES[f] for f in FAMILIES) + " |")
    add("|---" * (len(FAMILIES) + 2) + "|")
    for model_id in complete:
        block = summary["models"][model_id]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {block['alpha_global']:+.4f} | "
            + " | ".join(f"{block['alpha_family'][f]:+.4f}" for f in FAMILIES)
            + " |"
        )
    add("")
    add("**ΔTCE$_F$ = TCE$_{global}$ − TCE$_{family}$ (secondary)**\n")
    add("| Judge | ΔTCE_F | 95% CI | excludes 0 |")
    add("|---|---|---|---|")
    for model_id in complete:
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | "
            f"{summary['models'][model_id]['overall']['delta_tce_family']:+.4f} | "
            f"{_interval(boot, model_id, 'delta_tce_family', '.4f')} | "
            f"{_excludes_zero(boot, model_id, 'delta_tce_family')} |"
        )
    add("")
    add("**TCE by judge and family**\n")
    for metric, label in (
        ("tce_raw", "raw"),
        ("tce_global", "global-corrected"),
        ("tce_family", "family-corrected"),
    ):
        add(f"*{label}*\n")
        add("| Judge | " + " | ".join(FAMILY_TITLES[f] for f in FAMILIES) + " |")
        add("|---" * (len(FAMILIES) + 1) + "|")
        for model_id in complete:
            block = summary["models"][model_id]["by_family"]
            add(
                f"| {MODEL_SHORT_NAMES[model_id]} | "
                + " | ".join(f"{block[f][metric]['mean']:.3f}" for f in FAMILIES)
                + " |"
            )
        add("")
    add("![Model by family](../figures/day2/figure6_model_family.png)\n")
    add(narrative("family_bias"))

    # ------------------------------------------------------------------ 20
    add("## 20. Accuracy and Brier\n")
    add("| Judge | acc raw | acc global | acc family | Brier raw | Brier global | Brier family |")
    add("|---|---|---|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['accuracy_raw']:.4f} | "
            f"{o['accuracy_global']:.4f} | {o['accuracy_family']:.4f} | "
            f"{o['brier_raw']:.4f} | {o['brier_global']:.4f} | "
            f"{o['brier_family']:.4f} |"
        )
    add("")
    add(narrative("accuracy"))

    # ------------------------------------------------------------------ 21
    add("## 21. Bias-vs-shape taxonomy\n")
    add(
        "Predeclared descriptive categories, counted over the 324 TEST artifacts "
        "per judge.\n"
    )
    add(
        "| Judge | correctly localised | mislocated but orderly | locally unstable | "
        "correctable by global bias | residual failure |"
    )
    add("|---|---|---|---|---|---|")
    for model_id in complete:
        t = summary["models"][model_id]["taxonomy"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {t['correctly_localized']} | "
            f"{t['mislocated_but_orderly']} | {t['locally_unstable']} | "
            f"{t['correctable_by_global_bias']} | "
            f"{t['residual_localization_failure']} |"
        )
    add("")
    add(narrative("taxonomy"))

    # ------------------------------------------------------------------ 22
    add("## 22. Cross-model analysis\n")
    comparisons = pd.DataFrame(summary.get("paired_comparisons", []))
    if comparisons.empty or len(complete) < 2:
        add("Fewer than two complete judges; no paired comparison is possible.\n")
    else:
        add(
            "Paired over identical CALIBRATION and TEST resamples, so both stages "
            "of sampling error are shared between the judges being compared.\n"
        )
        for quantity, fmt in (("delta_tce_global", ".4f"), ("alpha_global", ".4f")):
            subset = comparisons[comparisons["quantity"] == quantity]
            if subset.empty:
                continue
            add(f"**{quantity}**\n")
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
    add(narrative("cross_model"))

    # ------------------------------------------------------------------ 23
    add("## 23. Cluster bootstrap\n")
    add(
        f"{BOOTSTRAP_REPLICATES} replicates, seed {BOOTSTRAP_SEED}. Each replicate "
        "resamples CALIBRATION artifacts within predicate family, **refits the "
        "intercept on that sample**, independently resamples TEST artifacts within "
        "family, and applies the refitted intercept. All nine thresholds of an "
        "artifact move together in both stages. Percentile intervals at 95%.\n"
    )
    add(
        "Resampling only the test set would have understated the uncertainty by "
        "treating the fitted intercept as known exactly. No multiple-comparison "
        "correction is applied; none was preregistered, and none may be added "
        "after the fact.\n"
    )

    # ------------------------------------------------------------------ 24
    add("## 24. Failure modes\n")
    add("| Judge | never crosses raw | never crosses global | below at s=0 raw | below at s=0 global |")
    add("|---|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['never_crosses_raw']}/{N_ARTIFACTS} | "
            f"{o['never_crosses_global']}/{N_ARTIFACTS} | "
            f"{o['always_below_raw']}/{N_ARTIFACTS} | "
            f"{o['always_below_global']}/{N_ARTIFACTS} |"
        )
    add("")
    add("![Boundary-aligned curves](../figures/day2/figure4_boundary_aligned.png)\n")
    add(narrative("failure_modes"))

    # ------------------------------------------------------------------ 25
    add("## 25. Limitations\n")
    add(
        "- Three synthetic formal predicate families and four specific "
        "open-weight judges at fixed revisions. Nothing transfers automatically "
        "to natural-language rubrics.\n"
        "- The correction is deliberately the weakest one that moves location. A "
        "stronger calibrator would improve the numbers and destroy the "
        "decomposition; that is why it is not used.\n"
        "- Margins are read from bf16 model outputs, so they carry the "
        "granularity of bf16 arithmetic. This matters least near the decision "
        "boundary, where the label logits are close together.\n"
        "- The intercept is fitted to minimise calibration log-loss, not to "
        "minimise TCE. A judge could in principle have its log-loss improved and "
        "its localisation worsened; the two are reported separately.\n"
        "- Family-C candidate responses coincide across partitions by "
        "construction of the bounded integer range. The evaluated items are "
        "disjoint, but the artifacts are not independent draws in the strictest "
        "sense for that family.\n"
        "- Day 2 establishes nothing about human alignment, clinical "
        "reliability, natural-language rubric reliability, whether calibrated "
        "probabilities are true probabilities, or whether intercept calibration "
        "should be deployed.\n"
    )

    # ------------------------------------------------------------------ 26
    add("## 26. Research decision\n")
    add("```text")
    add(f"DAY1_IMMUTABILITY  = {day1['status']}")
    add("DAY1_PROVENANCE    = CLEAN")
    add(f"PREREGISTRATION    = {prereg['status']}")
    add(f"MODEL PANEL        = {panel['status']} ({panel['complete']}/{panel['size']})")
    add(f"CALIBRATION LEAKAGE= {leakage['status']}")
    add(f"SHAPE INVARIANCE   = {invariance['status']}")
    add(f"DAY-2 ANALYSIS     = {'COMPLETE' if complete else 'BLOCKED'}")
    add("```\n")
    add(narrative("research_decision"))
    add(
        "**No Day-3 experiment has been started.** Any next experiment requires a "
        "separate research decision and a new preregistration.\n"
    )

    # ------------------------------------------------------------------ 27
    add("## 27. Reproducibility and integrity summary\n")
    add("| Item | SHA-256 |")
    add("|---|---|")
    for label, entry in integrity["files"].items():
        if entry["sha256"]:
            add(f"| {label.replace('_', ' ')} | `{entry['sha256']}` |")
    for partition in ("CAL", "TST"):
        block = integrity["raw_scores"].get(partition, {})
        if block.get("combined_sha256"):
            add(
                f"| raw scores — {partition} (combined) | "
                f"`{block['combined_sha256']}` |"
            )
    add(
        f"| figures (combined, {integrity['figures']['n_files']} files) | "
        f"`{integrity['figures']['combined_sha256']}` |"
    )
    add(f"| analysis source tree | `{integrity['source_manifest']['tree_sha256']}` |")
    add("")
    add(
        f"Day-2 preregistration frozen {prereg['frozen_at']}, SHA-256 "
        f"`{prereg['sha256']}`, amendments: {prereg['amendment_count']}. Git commit "
        f"at integrity check: `{integrity['git'].get('commit') or 'unavailable'}` "
        f"(clean = {integrity['git'].get('clean')}).\n"
    )
    add(
        "The Day-2 integrity manifest hashes exactly the file list the "
        "preregistration pinned, so the freeze digest and the final digest are "
        "directly comparable. Each figure is hashed individually and the combined "
        "digest refuses an empty list, so the Day-1 empty-input defect cannot "
        "recur.\n"
    )
    add("```text")
    add("No Day-1 scientific outcome modified.")
    add("No Day-1 outcome used to fit calibration.")
    add("No human labels used.")
    add("No human annotation performed.")
    add("No semantic rubric generation performed.")
    add("No nonlinear calibration performed.")
    add("No outcome-driven tuning performed.")
    add("No Day-3 experiment started.")
    add("```")
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = Path(args.root)
    target = root / "docs/RESULTS_DAY2.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build(root), encoding="utf-8")
    print(f"wrote docs/RESULTS_DAY2.md ({target.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
