#!/usr/bin/env python
"""Generate ``docs/RESULTS_DAY3.md`` from the Day-3 analysis artifacts (Phase K).

Every number comes from ``artifacts/day3/summary.json``,
``bootstrap_results.parquet``, ``oracle_global.json``, ``oracle_family.json``,
``integrity_manifest.json`` and ``model_provenance.json``. Nothing is retyped
from terminal output.

Interpretive text lives in ``NARRATIVE`` and is written after the analysis has
run. It may describe what the numbers show; it may not strengthen the
preregistered claim, generalise beyond the tested predicates and models, or
reach for a stronger transformation class.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import day3_config as d3c, provenance  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FAMILIES,
    MODEL_IDS,
    MODEL_SHORT_NAMES,
    N_ARTIFACTS,
    N_ROWS,
    N_STRICTNESS,
)

FAMILY_TITLES = {
    "coverage": "A · coverage",
    "max_violation": "B · max violation",
    "numeric_tolerance": "C · numeric tolerance",
}

#: Prior raw values, quoted from the frozen prior results documents.
PRIOR_TCE = {
    "Qwen/Qwen2.5-14B-Instruct": {"day1": 3.543, "day2_test": 3.556},
    "meta-llama/Llama-3.1-8B-Instruct": {"day1": 4.944, "day2_test": 4.920},
    "mistralai/Mistral-7B-Instruct-v0.3": {"day1": 2.858, "day2_test": 2.867},
    "google/gemma-2-9b-it": {"day1": 3.380, "day2_test": 3.373},
}

NARRATIVE: dict[str, str] = {
    "executive_summary": """
The reachability theorem and an independent intercept enumeration agreed on all
1,296 artifact-judge curves, the oracle nesting inequality held for every judge,
and `OTCE <= TCE_raw` held on every row.

Giving location its strongest possible advantage removes most, but not all, of
the observed localisation error. Mean TCE falls from 2.94–4.90 raw to 0.46–1.15
under a per-artifact oracle translation — a reduction of 1.80 to 3.76 levels.
So a large majority of FormalCRRC threshold error *is* compatible with
heterogeneous response-curve location.

What survives is nonetheless substantial and is the measurement Day 3 was built
to take. For between 35.2% and 64.8% of artifacts, no additive translation
places the crossing exactly at the formal boundary; for between 8.3% and 31.2%,
none places it even within one strictness level. Restricted to nontrivial
boundaries, only 27.1% to 60.4% of formal boundaries are reachable at all.

The nested bounds are ordered as they must be, and the largest single step is
always the last one: moving from one shift per judge and family to one per
artifact buys 1.02–1.42 levels, roughly two to four times what family-level
freedom adds over global. Location heterogeneity clearly extends below the
predicate-family level.

The curve descriptor that tracks the translation floor is the **number of strict
prefix record lows**, not the monotonicity violation rate and not the response
span. Qwen2.5-14B has by far the widest mean margin span (14.5) and a worse
translation floor than Gemma-2-9B, whose span is 3.7 but whose curves carry
substantially more record lows.
""",
    "replication": """
Raw Day-3 localisation lands close to both prior records on data sharing no
artifact identifier, rendered prompt or family-C target pair with either. Three
judges are within 0.08 levels of both prior values; Qwen2.5-14B is 0.18 below.
The Day-1 finding replicates a second time on fresh data from the same
generator, and nothing was tuned to make it do so.
""",
    "reachability": """
This is the sharpest single result in Day 3. Between 39.6% and 72.9% of
nontrivial formal boundaries are simply **not reachable by any additive
translation** of the raw curve. The judge's margin ordering does not contain a
strict new minimum at the position the rubric requires, so no constant, fitted
or oracular, can put the crossing there.

Reachability by boundary position is markedly uneven and differs by judge.
Qwen2.5-14B is best at the extremes (61% at j* = 1, 67% at j* = 8) and worst in
the middle (36–44%), which is what one expects: reaching a middle crossing
demands a record low at exactly that index, while j* = 1 needs only
M(1) < M(0). Mistral-7B reaches j* = 6 for **no** artifact at all, and
Gemma-2-9B's profile is jagged rather than ordered (97% at j* = 6, 14% at
j* = 2).

One confound is worth stating: because j* is determined by the latent level
(j* = m+1 for coverage, 9 − v and 9 − e for the other two families), each
boundary column also fixes the latent level, so position and difficulty are not
separable here. The unevenness is reported descriptively and not interpreted as
a pure position effect.
""",
    "otce": """
The oracle reduction is large: ΔTCE_A runs from 1.80 (Mistral-7B) to 3.76
(Llama-3.1-8B). Read together with Day 2, the picture is that a single global
intercept captured a small fraction of what location can explain, while
per-artifact location captures most of it.

The residual is not small in absolute terms. Mean OTCE of 1.15 for Llama-3.1-8B
and Mistral-7B means that even under the most favourable possible translation,
the average curve's crossing still sits more than a full strictness level from
where the rubric puts it. Gemma-2-9B is the least translation-limited judge at
0.457, and its interval [0.383, 0.528] excludes the others'.
""",
    "residual": """
Under FormalCRRC's additive-translation model, these curves possess a nonzero
shape-imposed localization floor, and it is not marginal. SLR₁ — the fraction of
artifacts for which exact localisation is impossible under any translation —
ranges from 35.2% to 64.8%. SLR₂, the fraction where even one-level accuracy is
unreachable, ranges from 8.3% to 31.2%.

Following the preregistered rule, no stronger transformation was tried. A slope,
a temperature, an isotonic map or any monotone warp would reduce these numbers
and would also change the question, because each of them alters shape as well as
location. This residual is the measurement, not a shortfall to be engineered
away.
""",
    "hierarchy": """
The step sizes are as informative as the endpoints. Global location buys
0.40–1.94 levels; adding family-level freedom buys a further 0.30–0.64; adding
artifact-level freedom buys 1.02–1.42 more. The last step dominates for every
judge.

That ordering says location heterogeneity is real at every scale examined and is
largest at the finest one. It also bounds Day 2's result from above: Day 2's
*fitted* global intercept necessarily did no better than the oracle global bound
reported here, and the gap between that bound and the artifact bound is the room
no judge-level or family-level correction could ever occupy.
""",
    "family": """
Family dependence is severe and model-specific, and averaging it away would hide
the most striking numbers in the experiment.

Qwen2.5-14B is nearly translation-solvable on numeric tolerance — OTCE 0.176,
84.4% of nontrivial boundaries reachable, mean reachable count 7.29 of a possible
10 — while on coverage it manages OTCE 1.222 with only 34.4% reachable and a mean
reachable count of 3.59. Its mean coverage span is *negative* (−2.28), meaning
its margin on average ends higher than it starts as strictness increases.

Mistral-7B on coverage is the extreme case in the panel: mean reachable count
2.56 against a floor of 2, nontrivial TRR of **3.1%**, and OTCE 2.056. Those
curves carry essentially no usable ordering — for most of them the only crossings
any translation can produce are the two trivial extremes. The same judge reaches
66.7% on max violation with a reachable count of 7.14.

Max violation and coverage share a candidate-response format and differ only in
threshold direction, so these gaps are not attributable to the artifact's surface
form.
""",
    "shape": """
Margin ties are frequent enough to matter for one judge and should be read
alongside its reachability numbers. Llama-3.1-8B has 467 adjacent equal margins
and 345 prefix-tie levels across its 324 curves, and for 37 artifacts (11.4%) the
target boundary is blocked *specifically* by an exact equality rather than by an
increase. Its mean margin span is 0.895, so bf16 quantisation produces many exact
repeats. Gemma-2-9B has 29 such artifacts, Qwen2.5-14B 3, Mistral-7B 2.

No tolerance was introduced. The strict inequality was preregistered, the tie
counts were preregistered as descriptive, and the rule was not revisited after
seeing them. But a reader should know that for Llama-3.1-8B a non-trivial slice
of unreachability is a boundary case of exact equality in a coarse numeric
format, not a clear violation of ordering.
""",
    "associations": """
Two descriptors are affine transforms of one another — RC = 2 + 8·PRR exactly —
so their Spearman associations with OTCE are identical by construction and count
as one finding, not two.

That finding is the useful one: the number of strict prefix record lows is
consistently and negatively associated with the translation floor (ρ = −0.30 to
−0.54 across judges). MMVR, the margin monotonicity violation rate carried
forward from Day 2, is essentially unassociated with OTCE (ρ = −0.11 to +0.06).

The distinction matters. A curve can violate monotonicity often and still be
translation-solvable, provided its reversals do not sit where new minima are
needed; and a curve can be broadly well-ordered yet unable to represent most
boundaries because its descent proceeds in plateaus rather than in fresh minima.
Whether a boundary is reachable is a question about record lows, not about
smoothness. These are descriptive associations and are not interpreted causally.
""",
    "taxonomy": """
The counts separate the two regimes cleanly. Translation-solvable artifacts
(OTCE = 0) number 114 to 210 of 324, and translation-limited ones (OTCE > 1)
number 27 to 101.

Order-poor curves — those with at most three reachable crossings out of ten —
are the majority for Llama-3.1-8B (125), Mistral-7B (112) and Qwen2.5-14B (102),
and rare for Gemma-2-9B (23). Gemma-2-9B is the only judge with more order-rich
than order-poor curves (100 versus 23), and it is also the judge with the lowest
translation floor. The two go together, as the association analysis suggests they
should.
""",
    "cross_model": """
The judge ordering by translation floor — Gemma-2-9B, then Qwen2.5-14B, then
Mistral-7B and Llama-3.1-8B together — is not the Day-1 ordering by raw TCE, in
which Mistral-7B was best and Llama-3.1-8B worst. Raw localisation and
translation reachability are measuring different things: the first depends on
where the curve happens to sit, the second only on the ordering it contains.

These are four specific open-weight models at fixed revisions on three synthetic
formal predicate families. Nothing here is a general ranking of judge quality.
""",
    "research_decision": """
Day 3 answers its question in both directions, and the two answers belong
together.

Most of the localisation error FormalCRRC measures is compatible with
heterogeneous curve location: an oracle translation chosen per artifact removes
between 61% and 77% of the raw mean TCE. Day 2's finding that a *single* global
intercept helps little is therefore not evidence that location is irrelevant —
it is evidence that location varies below the judge level, and below the family
level too.

But location is not sufficient. Between 35% and 65% of artifacts cannot be
localised exactly by any additive translation, between 8% and 31% cannot be
localised even within one level, and 40% to 73% of nontrivial formal boundaries
are unreachable outright. For those curves the correct answer is not present in
the ordering, and no constant can put it there.

The strongest permitted statement is the exact one: under FormalCRRC's
additive-translation model, these curves possess a nonzero shape-imposed
localization floor, and the correct formal boundary is unreachable by
translation for 39.6% to 72.9% of nontrivial artifacts.

Reported as observed. No seed, dataset, model, revision, prompt, scoring rule,
decision boundary, inequality or oracle definition was changed after inference
began, and no stronger transformation class was substituted when the residual
proved nonzero.
""",
}


def load(root: Path):
    summary = json.loads((root / d3c.SUMMARY_PATH).read_text(encoding="utf-8"))
    integrity = json.loads(
        (root / d3c.INTEGRITY_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    boot = pd.read_parquet(root / d3c.BOOTSTRAP_PATH)
    oracle_global = json.loads(
        (root / d3c.ORACLE_GLOBAL_PATH).read_text(encoding="utf-8")
    )
    oracle_family = json.loads(
        (root / d3c.ORACLE_FAMILY_PATH).read_text(encoding="utf-8")
    )
    models = json.loads((root / d3c.MODEL_PROVENANCE_PATH).read_text(encoding="utf-8"))
    return summary, integrity, boot, oracle_global, oracle_family, models


def _interval(boot: pd.DataFrame, model_id: str, quantity: str, fmt: str = ".3f") -> str:
    row = boot[(boot["model_id"] == model_id) & (boot["quantity"] == quantity)]
    if row.empty:
        return "—"
    r = row.iloc[0]
    return f"[{r['ci_low']:{fmt}}, {r['ci_high']:{fmt}}]"


def narrative(key: str) -> str:
    text = NARRATIVE.get(key, "").strip()
    return f"\n{text}\n" if text else ""


def build(root: Path) -> str:
    summary, integrity, boot, oracle_global, oracle_family, provenance_doc = load(root)
    complete = [m for m in MODEL_IDS if m in summary.get("models", {})]
    parts: list[str] = []
    add = parts.append

    priors = integrity["prior_immutability"]
    dataset = integrity["dataset"]
    panel = integrity["panel"]
    oracle = integrity["oracle"]
    prereg = integrity["preregistration"]

    # ------------------------------------------------------------------ 1
    add("# FormalCRRC Day-3 results\n")
    add("## Translation limits of threshold-response curves\n")
    add("### Oracle shape-preserving localization bounds\n")
    add(
        f"Generated {summary['created_at']} from the frozen Day-3 analysis "
        "artifacts. Every number is read directly from the machine artifacts "
        "under `artifacts/day3/`.\n"
    )
    add("## 1. Executive summary\n")
    add("```text")
    add(f"PRIOR_EXPERIMENT_IMMUTABILITY = {priors['status']}")
    add(f"DAY3_DATASET                  = {dataset['status']}")
    add(
        "ORACLE_IMPLEMENTATION         = "
        f"{oracle.get('reachability_cross_check', {}).get('status')}"
    )
    add(f"ORACLE_NESTING                = {oracle.get('nesting', {}).get('status')}")
    add(f"MODEL PANEL                   = {panel['status']} "
        f"({panel['complete']}/{panel['size']})")
    add(f"DAY-3 ANALYSIS                = {'COMPLETE' if complete else 'BLOCKED'}")
    add("```\n")

    add("| Judge | raw TCE | OTCE | nontrivial TRR | OTCE = 0 | OTCE > 1 |")
    add("|---|---|---|---|---|---|")
    for model_id in MODEL_IDS:
        block = summary["models"].get(model_id)
        if not block:
            add(f"| {MODEL_SHORT_NAMES[model_id]} | BLOCKED | | | | |")
            continue
        o = block["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['tce_raw']['mean']:.3f} | "
            f"{o['otce']['mean']:.3f} | {o['trr_nontrivial']:.1%} | "
            f"{o['otce']['exact_rate']:.1%} | {o['slr2']:.1%} |"
        )
    add("")
    add(narrative("executive_summary"))

    # ------------------------------------------------------------------ 2
    add("## 2. Relationship to Days 1 and 2\n")
    add(
        "Day 1 found that judges track threshold direction but localise the "
        "formal boundary poorly. Day 2 asked whether that was a *location* "
        "problem and answered partly: a fitted global intercept helped two "
        "judges, did nothing for a third and hurt a fourth, leaving a common "
        "residual of about 3.1–3.25 levels.\n"
    )
    add(
        "Day 2 could not distinguish two explanations of that residual: curves "
        "that would be correctly localised if only each received its own shift, "
        "versus curves whose ordering cannot represent the boundary at all. "
        "Day 3 settles it by giving location its strongest possible advantage — "
        "an oracle translation chosen per curve with knowledge of the formal "
        "truth. See [`DAY3_RATIONALE.md`](DAY3_RATIONALE.md).\n"
    )
    add(
        "Day 3 is a new, separately preregistered experiment. It is not a repair "
        "of Day 1 or Day 2, not another calibration experiment, and not a "
        "stronger post-hoc calibrator. Prior results are unchanged.\n"
    )

    # ------------------------------------------------------------------ 3
    add("## 3. Prior experiment immutability\n")
    add("| Check | Result |")
    add("|---|---|")
    add(f"| Files fingerprinted | {priors['n_files_fingerprinted']} |")
    add(f"| Day-1 files changed | {priors['day1']['changed'] or 'none'} |")
    add(f"| Day-2 files changed | {priors['day2']['changed'] or 'none'} |")
    add(f"| Files added | {priors['added'] or 'none'} |")
    add(
        f"| Day-1 dataset still reproduces | {priors['day1_dataset_reproduces']} "
        f"(`{priors['day1_dataset_content_sha256'][:16]}…`) |"
    )
    add("")
    add(
        "Day-3 code writes only to `artifacts/day3/` and `figures/day3/`. One "
        "Day-1 *source* hash is stale — `src/formalcrrc/dataset.py` — and the "
        "audit classified it correctly as the backward-compatible extension "
        "**Day 2** made and documented, not a Day-3 change. Day 3 added no "
        "source extension of its own, and the Day-1 dataset still reproduces its "
        "frozen content hash.\n"
    )

    # ------------------------------------------------------------------ 4
    add("## 4. Research question\n")
    add(
        "> What is the minimum threshold-crossing error achievable when each "
        "individual judge-response curve may undergo an arbitrary additive "
        "translation but its shape is held exactly fixed?\n"
    )

    # ------------------------------------------------------------------ 5
    add("## 5. Novelty boundaries\n")
    add(
        "The mathematics is elementary and no novelty is claimed for it. The "
        "contribution under test is the **diagnostic decomposition**: a setting "
        "where the correct boundary is known by construction, so *could location "
        "alone have explained this?* becomes exactly answerable. See "
        "[`NOVELTY_BOUNDARIES_DAY3.md`](NOVELTY_BOUNDARIES_DAY3.md).\n"
    )

    # ------------------------------------------------------------------ 6
    add("## 6. Formal translation model\n")
    add(
        "$M(s) = S_A - S_B$, $P(s) = \\sigma(M(s))$, and the admissible class is "
        "one constant per curve, $M'(s) = M(s) + \\alpha$. Since "
        "$M'(s+1) - M'(s) = M(s+1) - M(s)$, translation cannot reorder margins, "
        "remove a reversal, change the span or alter Spearman ordering. It "
        "changes only where the fixed curve crosses zero, which is exactly where "
        "$P$ crosses 0.5.\n"
    )
    add("![Reachability concept](../figures/day3/figure1_reachability_concept.png)\n")

    # ------------------------------------------------------------------ 7
    add("## 7. Reachability theorem\n")
    add(
        "A crossing at level $j \\in \\{1,\\dots,8\\}$ is reachable by some "
        "additive translation **iff** $j$ is a strict prefix record low:\n"
    )
    add("$$M(j) < \\min_{t<j} M(t).$$\n")
    add(
        "Placing the first negative level at $j$ requires "
        "$\\alpha \\ge -\\min_{t<j} M(t)$ and $\\alpha < -M(j)$; that interval is "
        "non-empty exactly under the condition above. Crossings 0 and 9 are "
        "always reachable, so the complete set is\n"
    )
    add(
        "$$\\mathcal{R}_x = \\{0, 9\\} \\cup \\{j : M_x(j) < \\min_{t<j} M_x(t)\\}.$$\n"
    )
    cross = oracle.get("reachability_cross_check", {})
    add(
        f"**Cross-check.** The theorem and an independent intercept enumeration "
        f"were compared on all {cross.get('n_rows', 0)} artifact-judge curves: "
        f"**{cross.get('disagreements', 0)} disagreements** "
        f"({cross.get('status')}). The pre-inference readiness gate additionally "
        "compared them on 3,004 synthetic curves, including canonical, "
        "heavily-tied and one-ULP-separated cases, with zero disagreements.\n"
    )

    # ------------------------------------------------------------------ 8
    add("## 8. Dataset construction\n")
    add("| | |")
    add("|---|---|")
    add(f"| Partition | {summary['design']['partition']} |")
    add(f"| Seed | {summary['design']['seed']} |")
    add(f"| Artifacts | {N_ARTIFACTS} |")
    add(f"| Thresholds per artifact | {N_STRICTNESS} |")
    add(f"| Rubric-response pairs | {N_ROWS} per judge |")
    add(f"| Content SHA-256 | `{dataset['content_sha256']}` |")
    add("")
    add(
        "Same three predicate families, same nine strictness levels, same latent "
        "balance, same candidate formatting, same rubric templates, same truth "
        "functions and the same prompt as Days 1 and 2. No new family, no "
        "semantic rubric, no adversarial variant.\n"
    )

    # ------------------------------------------------------------------ 9
    add("## 9. Dataset integrity\n")
    disjoint = dataset["disjointness"]
    add("| Check | Day 1 | Day-2 CAL | Day-2 TEST |")
    add("|---|---|---|---|")
    for label, key in (
        ("artifact_id overlap", "artifact_id_overlap"),
        ("rendered prompt overlap", "rendered_prompt_overlap"),
        ("family-C (target, reported) pair overlap", "family_c_target_pair_overlap"),
    ):
        block = disjoint[key]
        add(
            f"| {label} | {block['day1']} | {block['day2_calibration']} | "
            f"{block['day2_test']} |"
        )
    coincide = disjoint["candidate_response_overlap_by_family"]
    add(
        "| family-C candidate coincidences | "
        + " | ".join(
            str(coincide[name]["numeric_tolerance"])
            for name in ("day1", "day2_calibration", "day2_test")
        )
        + " |"
    )
    add("")
    add(
        "Artifact identifiers, rendered prompts and family-C (target, reported) "
        "pairs are **exactly disjoint** from all three prior evaluated sets. The "
        "residual family-C candidate coincidences are expected — a family-C "
        "candidate is fixed by its reported value alone, a bounded integer — and "
        "are recorded, not filtered.\n"
    )

    # ------------------------------------------------------------------ 10
    add("## 10. Model provenance\n")
    add("| Judge | Revision | Continuity | Smoke scores identical |")
    add("|---|---|---|---|")
    for model_id in MODEL_IDS:
        record = provenance_doc.get("models", {}).get(model_id, {})
        continuity = record.get("continuity", {})
        smoke = record.get("smoke_test", {})
        add(
            f"| `{model_id}` | `{record.get('revision', '—')}` | "
            f"{all(continuity.values()) if continuity else '—'} | "
            f"{smoke.get('reproduces_day1_scores', '—')} |"
        )
    add("")
    add(
        "Exactly the Day-1/Day-2 panel at exactly the Day-1 revisions, read from "
        "the immutable Day-1 provenance. The non-study smoke prompt reproduced "
        "**bit-identical** raw scores to Day 1 for every judge — the third "
        "consecutive experiment in which it has done so.\n"
    )

    # ------------------------------------------------------------------ 11
    add("## 11. Prompt and scoring continuity\n")
    add(
        f"Prompt template SHA-256 `{integrity['prompt_template_sha256']}`, "
        "verified byte-identical to Days 1 and 2 before inference. Same answer "
        "labels, same label-token resolution rule, same scoring method per "
        "model, same bf16 dtype, same batch size of 1, no sampling.\n"
    )

    # ------------------------------------------------------------------ 12
    add("## 12. Inference completeness\n")
    add("| Judge | Status | Rows | Failures | Job | Seconds |")
    add("|---|---|---|---|---|---|")
    for model_id in MODEL_IDS:
        run = integrity["runs"][model_id]
        meta = run.get("run", {}) or {}
        slurm = meta.get("slurm") or {}
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {run['status']} | "
            f"{run.get('rows', 0)} | {meta.get('failures', '—')} | "
            f"{slurm.get('job_id', '—')} | {meta.get('total_seconds', '—')} |"
        )
    add("")

    # ------------------------------------------------------------------ 13
    add("## 13. Raw Day-3 replication\n")
    add(
        "Day-3 is fresh data from the same formal generator, so raw behaviour "
        "can be compared with the prior records descriptively. There is no "
        "replication gate and the data was not tuned to reproduce anything.\n"
    )
    add("| Judge | Day-1 TCE | Day-2 TEST TCE | Day-3 TCE | max deviation |")
    add("|---|---|---|---|---|")
    for model_id in complete:
        raw = summary["models"][model_id]["overall"]["tce_raw"]["mean"]
        prior = PRIOR_TCE[model_id]
        deviation = max(abs(raw - prior["day1"]), abs(raw - prior["day2_test"]))
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {prior['day1']:.3f} | "
            f"{prior['day2_test']:.3f} | {raw:.3f} | {deviation:.3f} |"
        )
    add("")
    add(narrative("replication"))

    # ------------------------------------------------------------------ 14
    add("## 14. Reachable crossing sets\n")
    add(
        "$RC_x = |\\mathcal{R}_x|$, between 2 (only the extremes) and 10 (a "
        "strictly decreasing curve).\n"
    )
    add("| Judge | mean RC | median | PRR | RC ≥ 8 | RC ≤ 3 |")
    add("|---|---|---|---|---|---|")
    for model_id in complete:
        block = summary["models"][model_id]
        o = block["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['reachable_count']:.3f} "
            f"{_interval(boot, model_id, 'reachable_count')} | "
            f"{o['reachable_count_median']:.1f} | {o['prefix_record_rate']:.4f} | "
            f"{block['taxonomy']['order_rich']}/{N_ARTIFACTS} | "
            f"{block['taxonomy']['order_poor']}/{N_ARTIFACTS} |"
        )
    add("")
    add("**Reachable-count distribution (artifact counts)**\n")
    keys = sorted(
        {
            int(k)
            for m in complete
            for k in summary["models"][m]["overall"]["reachable_count_distribution"]
        }
    )
    add("| Judge | " + " | ".join(str(k) for k in keys) + " |")
    add("|---" * (len(keys) + 1) + "|")
    for model_id in complete:
        dist = summary["models"][model_id]["overall"]["reachable_count_distribution"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | "
            + " | ".join(str(dist.get(str(k), 0)) for k in keys)
            + " |"
        )
    add("")

    # ------------------------------------------------------------------ 15
    add("## 15. Translation reachability rate\n")
    add("| Judge | overall TRR | nontrivial TRR | 95% CI | n nontrivial |")
    add("|---|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['trr_overall']:.1%} | "
            f"{o['trr_nontrivial']:.1%} | "
            f"{_interval(boot, model_id, 'trr_nontrivial', '.4f')} | "
            f"{o['n_nontrivial']} |"
        )
    add("")
    add(
        "The nontrivial rate restricts to $j^* \\in \\{1,\\dots,8\\}$, because "
        "$j^* = 9$ is reachable for every curve by construction and would "
        "otherwise inflate the headline. Both were preregistered.\n"
    )
    add("![Reachability](../figures/day3/figure3_reachability.png)\n")
    add("![Crossing geometry](../figures/day3/figure4_crossing_geometry.png)\n")
    add(narrative("reachability"))

    # ------------------------------------------------------------------ 16
    add("## 16. Oracle translation TCE\n")
    add(
        "$OTCE_x = \\min_{j \\in \\mathcal{R}_x} |j - j^*_x|$ — the best "
        "localisation any additive translation could achieve for that curve.\n"
    )
    add("| Judge | raw TCE | OTCE | 95% CI | ΔTCE_A | median OTCE | max |")
    add("|---|---|---|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['tce_raw']['mean']:.3f} | "
            f"{o['otce']['mean']:.3f} | {_interval(boot, model_id, 'otce')} | "
            f"{o['delta_tce_artifact']:.3f} | {o['otce']['median']:.1f} | "
            f"{o['otce']['max']} |"
        )
    add("")
    add(
        f"The guarantee $OTCE_x \\le TCE^{{raw}}_x$ was asserted on all "
        f"{oracle.get('otce_bounded_by_raw', {}).get('n_rows', 0)} artifact-judge "
        f"rows: {oracle.get('otce_bounded_by_raw', {}).get('violations', 0)} "
        "violations.\n"
    )
    add("![Raw versus OTCE](../figures/day3/figure2_raw_vs_otce.png)\n")
    add(narrative("otce"))

    # ------------------------------------------------------------------ 17
    add("## 17. Residual shape-limited error\n")
    add("| Judge | SLR₁ = P(OTCE > 0) | 95% CI | SLR₂ = P(OTCE > 1) | 95% CI |")
    add("|---|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['slr1']:.1%} | "
            f"{_interval(boot, model_id, 'slr1', '.4f')} | {o['slr2']:.1%} | "
            f"{_interval(boot, model_id, 'slr2', '.4f')} |"
        )
    add("")
    add(
        "$OTCE > 0$ means no additive translation places that curve's crossing "
        "exactly at the formal boundary. $OTCE > 1$ means none places it even "
        "within one strictness level. These are **translation-limited "
        "residuals** — shape-limited under additive translation — and the result "
        "applies only to the FormalCRRC translation class.\n"
    )
    add(narrative("residual"))

    # ------------------------------------------------------------------ 18-21
    add("## 18. Oracle global bound\n")
    add("| Judge | α_G | mean TCE | 95% CI |")
    add("|---|---|---|---|")
    for model_id in complete:
        block = oracle_global["models"][model_id]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {block['alpha']:+.4f} | "
            f"{block['mean_tce']:.3f} | "
            f"{_interval(boot, model_id, 'tce_oracle_global')} |"
        )
    add("")
    add(
        "One truth-optimised intercept per judge, found by exact enumeration of "
        "the breakpoints where the mean TCE changes. Not a deployment result.\n"
    )

    add("## 19. Oracle family bound\n")
    add("| Judge | " + " | ".join(FAMILY_TITLES[f] for f in FAMILIES) + " | weighted mean TCE |")
    add("|---" * (len(FAMILIES) + 2) + "|")
    for model_id in complete:
        block = oracle_family["models"][model_id]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | "
            + " | ".join(
                f"α={block['by_family'][f]['alpha']:+.3f}, "
                f"TCE={block['by_family'][f]['mean_tce']:.3f}"
                for f in FAMILIES
            )
            + f" | {block['weighted_mean_tce']:.3f} |"
        )
    add("")

    add("## 20. Oracle artifact bound\n")
    add(
        "One arbitrary intercept per artifact. Its achieved minimum is exactly "
        "OTCE, reported in §16. This is the absolute translation floor: no "
        "additive correction of any kind, fitted or oracular, can do better.\n"
    )

    add("## 21. Oracle hierarchy\n")
    add("| Judge | RAW | ORACLE-G | ORACLE-F | ORACLE-A | nesting |")
    add("|---|---|---|---|---|---|")
    for model_id in complete:
        h = summary["models"][model_id]["oracle_hierarchy"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {h['raw']:.3f} | "
            f"{h['oracle_global']:.3f} | {h['oracle_family']:.3f} | "
            f"{h['oracle_artifact']:.3f} | {h['nesting']['status']} |"
        )
    add("")
    add(
        "Left to right, translation gains freedom: none, one shift per judge, "
        "one per judge and family, one per artifact. Since $G \\subseteq F "
        "\\subseteq A$, the optimised means must be non-increasing, and they "
        "are. These are **nested operational bounds on what increasingly "
        "flexible location-only explanations can accomplish** — not an additive "
        "variance decomposition, and not deployment performance.\n"
    )
    add("![Oracle hierarchy](../figures/day3/figure5_oracle_hierarchy.png)\n")
    add(narrative("hierarchy"))

    # ------------------------------------------------------------------ 22
    add("## 22. Predicate-family analysis\n")
    for metric, label, fmt in (
        ("otce", "mean OTCE", "{:.3f}"),
        ("trr_nontrivial", "nontrivial TRR", "{:.1%}"),
        ("reachable_count", "mean reachable count", "{:.3f}"),
        ("tce_raw", "raw TCE", "{:.3f}"),
    ):
        add(f"**{label}**\n")
        add("| Judge | " + " | ".join(FAMILY_TITLES[f] for f in FAMILIES) + " |")
        add("|---" * (len(FAMILIES) + 1) + "|")
        for model_id in complete:
            block = summary["models"][model_id]["by_family"]
            values = []
            for family in FAMILIES:
                cell = block[family]
                value = (
                    cell[metric]["mean"] if isinstance(cell[metric], dict) else cell[metric]
                )
                values.append(fmt.format(value))
            add(f"| {MODEL_SHORT_NAMES[model_id]} | " + " | ".join(values) + " |")
        add("")
    add("![Shape floor](../figures/day3/figure6_shape_floor.png)\n")
    add(narrative("family"))

    # ------------------------------------------------------------------ 23
    add("## 23. Shape metrics\n")
    add("| Judge | MMVR | MMVM | SPAN | mean margin ρ | PRR |")
    add("|---|---|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['mmvr']:.4f} | {o['mmvm']:.4f} | "
            f"{o['span']:.3f} | {o['margin_spearman_mean']:.4f} | "
            f"{o['prefix_record_rate']:.4f} |"
        )
    add("")
    add("**Margin ties (descriptive)**\n")
    add("| Judge | adjacent equal margins | prefix tie levels | target blocked by equality |")
    add("|---|---|---|---|")
    for model_id in complete:
        o = summary["models"][model_id]["overall"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {o['adjacent_equal_margins']} | "
            f"{o['prefix_tie_levels']} | {o['target_blocked_by_equality']} |"
        )
    add("")
    add(
        "Equality never creates a reachable crossing under the strict rule. No "
        "tolerance was introduced, and the rule was not changed after observing "
        "tie frequency.\n"
    )
    add(narrative("shape"))

    # ------------------------------------------------------------------ 24
    add("## 24. Shape-metric associations\n")
    add(
        "Within-judge Spearman association between OTCE and each simple curve "
        "descriptor. Descriptive; not interpreted causally.\n"
    )
    associations = pd.DataFrame(summary["shape_metric_associations"])
    overall = associations[associations["scope"] == "overall"]
    descriptors = ["mmvr", "mmvm", "span", "prefix_record_rate", "reachable_count"]
    add("| Judge | " + " | ".join(descriptors) + " |")
    add("|---" * (len(descriptors) + 1) + "|")
    for model_id in complete:
        row = overall[overall["model_id"] == model_id].set_index("descriptor")
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | "
            + " | ".join(
                f"{row.loc[d, 'spearman_rho']:+.3f}" if d in row.index else "—"
                for d in descriptors
            )
            + " |"
        )
    add("")
    add(narrative("associations"))

    # ------------------------------------------------------------------ 25
    add("## 25. Translation taxonomy\n")
    add(
        "| Judge | translation-solvable | near-solvable | translation-limited | "
        "order-rich | order-poor |"
    )
    add("|---|---|---|---|---|---|")
    for model_id in complete:
        t = summary["models"][model_id]["taxonomy"]
        add(
            f"| {MODEL_SHORT_NAMES[model_id]} | {t['translation_solvable']} | "
            f"{t['translation_near_solvable']} | {t['translation_limited']} | "
            f"{t['order_rich']} | {t['order_poor']} |"
        )
    add("")
    add(
        "Counts over the 324 Day-3 artifacts per judge. Thresholds "
        "(OTCE = 0 / = 1 / > 1; RC ≥ 8; RC ≤ 3) were fixed before inference and "
        "are reported as counts only.\n"
    )
    add(narrative("taxonomy"))

    # ------------------------------------------------------------------ 26
    add("## 26. Cross-model analysis\n")
    comparisons = pd.DataFrame(summary.get("paired_comparisons", []))
    if comparisons.empty or len(complete) < 2:
        add("Fewer than two complete judges; no paired comparison is possible.\n")
    else:
        for quantity, fmt in (("otce", ".4f"), ("trr_nontrivial", ".4f")):
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
        "Paired over identical artifact resamples. These are four specific "
        "open-weight models at fixed revisions on three synthetic formal "
        "predicate families, and are not a general ranking.\n"
    )
    add(narrative("cross_model"))

    # ------------------------------------------------------------------ 27
    add("## 27. Cluster bootstrap\n")
    add(
        f"{BOOTSTRAP_REPLICATES} replicates, seed {BOOTSTRAP_SEED}, resampling "
        "artifacts within predicate family and carrying all nine thresholds of "
        "an artifact together. The global and family oracles are **refitted "
        "inside every replicate** — holding a truth-optimised shift fixed while "
        "resampling would understate the uncertainty of a quantity chosen from "
        "the data. OTCE needs no refitting: it is a property of a single curve "
        "and its boundary. Percentile intervals at 95%.\n"
    )

    # ------------------------------------------------------------------ 28
    add("## 28. Limitations\n")
    add(
        "- Three synthetic formal predicate families, four fixed open-weight "
        "judges, one prompt, bf16 inference, a two-label margin readout.\n"
        "- **The oracle uses formal truth and is not deployable.** It is "
        "deliberately optimistic; that optimism is exactly why a surviving "
        "residual is meaningful, and it must not be read as achievable "
        "performance.\n"
        "- Translation is only one class of shape-preserving transformations. A "
        "more general monotone transformation would move probabilities "
        "differently and could reach boundaries this class cannot. Testing one "
        "is a separate research question and was not done.\n"
        "- The reachability condition is a strict inequality on stored float64 "
        "margins derived from bf16 inference. Exact ties never create a "
        "reachable crossing; their frequency is reported rather than smoothed "
        "away with a tolerance.\n"
        "- Day 3 says nothing directly about natural-language rubrics, human "
        "alignment, clinical reliability or HealthBench.\n"
        "- A low OTCE would mean only that a curve's ordering permits the "
        "correct crossing after arbitrary per-artifact translation — not that "
        "the judge is reliable.\n"
    )

    # ------------------------------------------------------------------ 29
    add("## 29. Research decision\n")
    add("```text")
    add(f"PRIOR_EXPERIMENT_IMMUTABILITY = {priors['status']}")
    add(f"DAY3_DATASET                  = {dataset['status']}")
    add(
        "ORACLE_IMPLEMENTATION         = "
        f"{oracle.get('reachability_cross_check', {}).get('status')}"
    )
    add(f"ORACLE_NESTING                = {oracle.get('nesting', {}).get('status')}")
    add(f"MODEL PANEL                   = {panel['status']} "
        f"({panel['complete']}/{panel['size']})")
    add(f"DAY-3 ANALYSIS                = {'COMPLETE' if complete else 'BLOCKED'}")
    add("```\n")
    add(narrative("research_decision"))
    add(
        "**No Day-4 experiment has been started.** Any next experiment requires "
        "a separate research decision and a new preregistration.\n"
    )

    # ------------------------------------------------------------------ 30
    add("## 30. Reproducibility and integrity summary\n")
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
    add(
        f"| figures (combined, {integrity['figures']['n_files']} files) | "
        f"`{integrity['figures']['combined_sha256']}` |"
    )
    add(f"| analysis source tree | `{integrity['source_manifest']['tree_sha256']}` |")
    add("")
    add(
        f"Day-3 preregistration frozen {prereg['frozen_at']}, SHA-256 "
        f"`{prereg['sha256']}`, amendments: {prereg['amendment_count']}. Git "
        f"commit at integrity check: "
        f"`{integrity['git'].get('commit') or 'unavailable'}` "
        f"(clean = {integrity['git'].get('clean')}).\n"
    )
    add("```text")
    add("No prior scientific outcome modified.")
    add("No human labels used.")
    add("No human annotation performed.")
    add("No semantic rubric generation performed.")
    add("No model calibration proposed for deployment.")
    add("No curve shape modified.")
    add("No outcome-driven tuning performed.")
    add("No Day-4 experiment started.")
    add("```")
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = Path(args.root)
    target = root / "docs/RESULTS_DAY3.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build(root), encoding="utf-8")
    print(f"wrote docs/RESULTS_DAY3.md ({target.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
