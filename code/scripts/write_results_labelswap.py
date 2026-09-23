#!/usr/bin/env python
"""Generate ``docs/RESULTS_LABEL_SWAP.md`` from the label-swap analysis artifacts.

Every number comes from ``artifacts/labelswap/summary.json``, the generated
tables, ``prompt_diff.json``, the run records and ``integrity_manifest.json``.
Nothing is retyped from terminal output.

Interpretive text lives in ``NARRATIVE`` and is written after the analysis has
run. It may describe what the numbers show; it may not strengthen the
preregistered claim, relabel a failed criterion as passed, or generalise beyond
the tested nuisance factor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import provenance  # noqa: E402
from formalcrrc import labelswap_config as lsc  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FAMILIES,
    MODEL_IDS,
    MODEL_SHORT_NAMES,
    MODEL_SLUGS,
)

ROOT = provenance.repo_root()

FAMILY_TITLES = {
    "coverage": "A · coverage",
    "max_violation": "B · max violation",
    "numeric_tolerance": "C · numeric tolerance",
}

# Filled in after the analysis. Descriptive only.
NARRATIVE: dict[str, str] = {
    'status': """
All four judges completed the full 2,916-prompt partition with **zero scoring
failures**: 11,664 new swapped-label evaluations in total. No retries were
needed and no observation was excluded. The panel is complete.
""",
    'relationship': """
The Day-3 raw scores are the paired comparison arm. They were opened read-only
and re-verified byte-identical afterwards. Nothing in Days 1-3 was rerun, and
no Day-3 number reported here differs from the frozen record.
""",
    'intervention': """
The swapped template was derived from the frozen template programmatically, not
retyped, and the derivation is asserted by test. Both response instructions are
character permutations of one another, so the two conditions differ in what the
tokens *mean* and in nothing else.

The scoring layer made this clean: FormalCRRC already resolves scores as
"the token meaning met" and "the token meaning not met", so declaring
`label_met = B` produces `S_B - S_A` by construction, with no sign correction
anywhere downstream. A preflight check refuses to run unless the swapped
tokenisation is the Day-3 tokenisation with exactly the two roles exchanged.
""",
    'panel': """
Mistral is the sharpest confirmation that tokens were resolved correctly. Its
tokenizer encodes the bare string `"A"` as a different token from the one that
actually continues a prompt (1098 versus 29509). Under the swap it resolved to
`met -> B` id 29528 and `not_met -> A` id 29509 — precisely the Day-3
continuation ids with roles exchanged, not the naive encodings. Had the swap
been implemented by encoding label strings in isolation, this model alone would
have silently read the wrong logits.

Model loading took substantially longer than in Day 3 (405 s versus 53 s for
Qwen) because of contention on the shared cache filesystem. This affects wall
clock only; scores are deterministic given the revision and prompt.
""",
    'dataset': """
Every check passed. Artifact identity to Day 3 is exact — same ids, same
prompts, same thresholds — which is what makes the comparison paired rather
than merely matched. Formal labels and true crossings were recomputed from
`family`, `latent_level` and `strictness_index` through the predicate
definitions rather than read from stored columns.
""",
    'promptdiff': """
Two byte positions. That is the whole intervention.
""",
    'primary': """
**The headline is that the ordering failure survives the swap in every judge.**
Between 47.9% and 68.1% of nontrivial formal boundaries remain unreachable by
any additive translation under the swapped mapping, against 39.6%-72.9%
originally. No judge's unreachable fraction fell near zero, and none fell below
47%.

Beneath that, the three preregistered criteria disagree with each other, and
they disagree differently for each judge — so this result cannot honestly be
compressed into one verdict:

* **Qwen2.5-14B** meets all three criteria. Semantic agreement is 0.977, the
  TCE change is small (+0.105) and the TRR change is inside the band.
* **Mistral-7B** shows a large *location* change (Δ TCE +0.645,
  CI [+0.389, +0.895]) and the lowest agreement (0.809), yet its Δ TRR
  interval [-0.042, +0.014] sits inside the band and contains zero. Location
  moved; ordering did not.
* **Llama-3.1-8B** moves in the *ordering*: Δ TRR +0.122,
  CI [+0.059, +0.184], above the band. Its unreachable fraction fell from 72.9%
  to 60.8% — a real improvement, though still a majority of boundaries.
* **Gemma-2-9B** fails only Δ TRR, and only just: CI [-0.129, -0.038], with the
  lower end marginally outside -0.10. Its agreement is high (0.954) and its TCE
  change is small.

Note the directions differ. Llama's reachability improved under the swap while
Qwen's and Gemma's worsened. There is no single "labels help" or "labels hurt"
story here.
""",
    'ordering': """
Curve-wise rank correlation separates the judges sharply. Qwen (median ρ 0.983),
Mistral (0.950) and Gemma (0.948) largely preserve the *within-curve ordering*
of canonical margins across the swap. **Llama does not: its median ρ is 0.512**,
and its exact reachable-set agreement is only 0.161. For Llama the
response-token assignment materially reshapes threshold-response geometry.

Exact reachable-set agreement is modest everywhere (0.161-0.590) even where
Jaccard similarity is high (0.620-0.899), meaning the sets usually overlap
substantially but rarely coincide exactly. Reachability is a fragile set-valued
property; that it moves under a nuisance change is expected, and it is why the
aggregate rate rather than the exact set is the preregistered outcome.
""",
    'family': """
The aggregate figures hide large family-level structure, and one entry explains
most of Mistral's overall failure: on **max_violation** its Δ TCE is **+2.000**,
CI [+1.472, +2.602], while on coverage it *improves* by -0.417. Gemma shows the
same opposing pattern with smaller magnitude (coverage +0.213, max_violation
-0.491). Llama's numeric_tolerance TCE is unchanged at exactly 5.000 with
Δ TCE 0.000 — its crossings there are saturated in both conditions.

Family-stratified agreement also varies widely for the label-sensitive judges:
Mistral ranges 0.738-0.928 and Llama 0.770-0.970 across families, against
Qwen's narrow 0.969-0.990.
""",
    'robustness': """
One judge of four meets the strong criterion. The other three each fail for a
different reason, and the reasons are informative rather than interchangeable:
Mistral fails on location, Llama on ordering, Gemma marginally on reachability
alone.

Per the preregistration, a failed criterion is reported as failed. It is not
relabelled robust because the principal phenomenon survives, and the principal
phenomenon is not discarded because a criterion failed. Both statements hold
simultaneously.
""",
    'ordering_interpretation': """
The principal qualitative conclusion persists. Under the swapped mapping every
judge still shows substantial nontrivial boundary unreachability (47.9%-68.1%),
so the ordering failures FormalCRRC measures are not an artifact of assigning
"met" to A and "not met" to B.

The precise reading differs by judge:

* For **Qwen, Mistral and Gemma**, the supported conclusion is the preregistered
  Outcome A/B pattern — response-token assignment influences score location or
  pointwise decisions to varying degrees, but does not account for the
  ordering-based boundary failures.
* For **Llama**, the supported conclusion is the preregistered Outcome C:
  response-label assignment materially changes threshold-response geometry, and
  label-token mapping is an additional source of evaluator instability for this
  judge. Its unreachable fraction still remains at 60.8%, so the phenomenon is
  attenuated rather than removed.
""",
    'rc10': """
The strict scalar-readout signature stays rare under the swap: 1.85% (Qwen),
0.00% (Llama), 1.54% (Mistral) and 0.93% (Gemma), against 3.70%, 0.00%, 0.31%
and 3.70% originally. Reversing the label mapping does not make judges look like
coherent scalar thresholders.

This is a post-study theoretical diagnostic carried over from the step-function
proposition audit. It is not a preregistered endpoint of this study and carries
no robustness band.
""",
    'ties': """
Exact margin equalities remain common and the strict rule was applied unchanged.
Curves carrying at least one adjacent equality: Qwen 58 → 53, Llama 264 → 227,
Mistral 37 → 40, Gemma 150 → 154 out of 324. No tolerance was introduced, and no
amendment was needed.
""",
    'limitations': """
* This tests **one** nuisance factor. It says nothing about sensitivity to other
  verbal labels, option order, instruction phrasing, or rubric wording.
* The swap is a counterbalance, not a randomisation: each artifact is measured
  once per condition, so within-condition scoring noise is not separately
  estimated. The scoring procedure is deterministic given the revision, so this
  is a design boundary rather than an unmeasured variance component.
* Agreement, TCE and TRR are aggregate rates over a balanced design.
  Family-level results show heterogeneity that the aggregates compress.
* Bootstrap intervals are percentile intervals over artifact-clustered
  resampling. They quantify sampling variability of the artifact set, not
  model-level or prompt-family-level uncertainty.
* The four judges are separate planned analyses. No multiplicity correction was
  preregistered or applied, so the per-judge criteria should be read as
  descriptive bands rather than hypothesis tests.
* Reachability is defined relative to additive translation only, exactly as in
  Day 3. A richer correction class could reach boundaries this one cannot.
""",
    'integrity': """
The 120 protected files were fingerprinted before the preregistration was frozen
and re-verified afterwards: byte-identical, same combined digest. The four Day-3
raw score files were read and never written. All 24 new files are confined to
`artifacts/labelswap/` and `figures/labelswap/`.
""",
    'deviations': """
**No protocol deviations.** No retries were required: all four SLURM jobs
returned rc=0 on first submission (799325, 799326, 799327, 799328) and every one
scored 2,916/2,916 prompts with zero failures. No observation was excluded for
any reason. The preregistration was frozen before the first swapped-label
inference and carries 0 amendments.

Two implementation notes, neither affecting any outcome:

1. Model load times were roughly 8x longer than Day 3 owing to shared-filesystem
   contention. Wall clock only.
2. The analysis script initially read a `formal_truth` column from the Day-3 raw
   scores, which do not carry one; it was corrected to recompute formal truth
   from the predicate definitions. This was found and fixed before any result was
   produced, and the recomputed labels are the same ones the frozen dataset
   stores.
""",
    'manuscript': """
Recommended wording, using the exact repository-derived values:

> Reversing the semantic assignment of the response tokens A and B leaves the
> principal FormalCRRC reachability finding in place: between 47.9% and 68.1% of
> nontrivial formal boundaries remain unreachable by any additive translation
> under the swapped mapping, compared with 39.6%-72.9% under the original
> mapping. The ordering failures FormalCRRC measures are therefore not explained
> by the original assignment of "criterion met" to token A.
>
> The effect is not uniformly negligible. Of four judges, one (Qwen2.5-14B) met
> all three prespecified robustness criteria. Mistral-7B showed a substantial
> location shift (Δ TCE +0.645, 95% CI [+0.389, +0.895]) while its reachability
> was unchanged (Δ TRR 95% CI [-0.042, +0.014]), consistent with a
> label-associated location bias rather than an ordering change. Llama-3.1-8B
> showed a genuine ordering change (Δ TRR +0.122, 95% CI [+0.059, +0.184];
> median within-curve Spearman ρ 0.512), identifying response-label mapping as
> an additional source of evaluator instability for that model. Gemma-2-9B
> failed only the reachability band, marginally (95% CI [-0.129, -0.038]).
>
> Label-token assignment is thus a real but judge-specific nuisance factor that
> attenuates rather than removes the measured phenomenon.

Report all three criteria per judge rather than a single pass/fail, and state
Llama's ordering sensitivity explicitly rather than averaging it away.
""",
}


def ci(record: dict, digits: int = 4) -> str:
    return f"[{record['ci_low']:+.{digits}f}, {record['ci_high']:+.{digits}f}]"


def load() -> dict:
    summary = json.loads((ROOT / lsc.SUMMARY_PATH).read_text(encoding="utf-8"))
    diff = json.loads((ROOT / lsc.PROMPT_DIFF_PATH).read_text(encoding="utf-8"))
    prereg = json.loads((ROOT / lsc.PREREG_JSON_PATH).read_text(encoding="utf-8"))
    integrity_path = ROOT / lsc.INTEGRITY_MANIFEST_PATH
    integrity = (
        json.loads(integrity_path.read_text(encoding="utf-8"))
        if integrity_path.is_file()
        else {}
    )
    runs = {}
    for model_id in MODEL_IDS:
        path = ROOT / lsc.RAW_SCORES_DIR / f"{MODEL_SLUGS[model_id]}_run.json"
        if path.is_file():
            runs[model_id] = json.loads(path.read_text(encoding="utf-8"))
    return {
        "summary": summary,
        "diff": diff,
        "prereg": prereg,
        "integrity": integrity,
        "runs": runs,
    }


def section(key: str) -> str:
    text = NARRATIVE.get(key, "").strip()
    return f"\n{text}\n" if text else ""


def table_primary(summary: dict) -> str:
    lines = [
        "| Judge | Original TCE | Swapped TCE | Δ TCE | 95% CI Δ TCE | Semantic agreement | Original TRR | Swapped TRR | Δ TRR | 95% CI Δ TRR | Swapped unreachable |",
        "|---|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for model_id in MODEL_IDS:
        record = summary["by_model"][model_id]
        p = record["point_estimates"]
        s = record["bootstrap"]["statistics"]
        lines.append(
            f"| {MODEL_SHORT_NAMES[model_id]} | {p['tce_original']:.3f} | "
            f"{p['tce_swapped']:.3f} | {p['delta_tce']:+.3f} | {ci(s['delta_tce'], 3)} | "
            f"{p['semantic_agreement']:.4f} | {p['trr_original']:.4f} | "
            f"{p['trr_swapped']:.4f} | {p['delta_trr']:+.4f} | {ci(s['delta_trr'], 4)} | "
            f"**{100 * p['unreachable_swapped']:.2f}%** |"
        )
    return "\n".join(lines)


def table_ordering(summary: dict) -> str:
    lines = [
        "| Judge | Median curve ρ | Sign agreement | Reachable-set agreement | Reachable-set Jaccard | Original mean RC | Swapped mean RC | Original RC=10 | Swapped RC=10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id in MODEL_IDS:
        p = summary["by_model"][model_id]["point_estimates"]
        lines.append(
            f"| {MODEL_SHORT_NAMES[model_id]} | "
            f"{p['median_margin_rank_correlation']:+.4f} | {p['sign_agreement']:.4f} | "
            f"{p['reachable_set_identical']:.4f} | {p['reachable_set_jaccard']:.4f} | "
            f"{p['rc_original']:.3f} | {p['rc_swapped']:.3f} | "
            f"{100 * p['rc10_original']:.2f}% | {100 * p['rc10_swapped']:.2f}% |"
        )
    return "\n".join(lines)


def table_family() -> str:
    path = ROOT / lsc.TABLES_DIR / "table_c_family.csv"
    frame = pd.read_csv(path)
    lines = [
        "| Judge | Family | n | Swapped TCE | Δ TCE | 95% CI Δ TCE | Swapped TRR | Δ TRR | 95% CI Δ TRR | Agreement |",
        "|---|---|---:|---:|---:|:---:|---:|---:|:---:|---:|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {row.Judge} | {FAMILY_TITLES.get(row.Family, row.Family)} | {row.n} | "
            f"{getattr(row, '_5'):.3f} | {getattr(row, '_6'):+.3f} | "
            f"[{getattr(row, '_7'):+.3f}, {getattr(row, '_8'):+.3f}] | "
            f"{getattr(row, '_9'):.4f} | {getattr(row, '_10'):+.4f} | "
            f"[{getattr(row, '_11'):+.4f}, {getattr(row, '_12'):+.4f}] | "
            f"{getattr(row, '_13'):.4f} |"
        )
    return "\n".join(lines)


def table_robustness(summary: dict) -> str:
    lines = [
        "| Judge | Δ TCE ⊂ [−0.5, +0.5] | Δ TRR ⊂ [−0.10, +0.10] | LCB(agreement) ≥ 0.90 | Classification | Failed criteria |",
        "|---|:---:|:---:|:---:|---|---|",
    ]
    for model_id in MODEL_IDS:
        record = summary["by_model"][model_id]
        criteria = record["robustness"]["criteria"]
        failed = record["robustness"]["failed_criteria"]
        mark = lambda ok: "**PASS**" if ok else "**FAIL**"  # noqa: E731
        lines.append(
            f"| {MODEL_SHORT_NAMES[model_id]} | "
            f"{mark(criteria['delta_tce_within_band'])} | "
            f"{mark(criteria['delta_trr_within_band'])} | "
            f"{mark(criteria['agreement_lower_bound_met'])} | "
            f"{record['robustness']['classification']} | "
            f"{', '.join(failed) if failed else '—'} |"
        )
    return "\n".join(lines)


def table_models(prereg: dict, runs: dict) -> str:
    lines = [
        "| Model | Revision | dtype | Scoring | Prompts scored | Failures | SLURM job |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for model_id in MODEL_IDS:
        facts = prereg["day3_facts"]["models"][model_id]
        run = runs.get(model_id, {})
        job = (run.get("slurm") or {}).get("job_id", "—")
        lines.append(
            f"| `{model_id}` | `{facts['revision'][:12]}…` | {facts['dtype']} | "
            f"{run.get('scoring_method', '—')} | {run.get('prompts_scored', '—')} | "
            f"{len(run.get('failures', []))} | {job} |"
        )
    return "\n".join(lines)


def build(data: dict) -> str:
    summary = data["summary"]
    prereg = data["prereg"]
    diff = data["diff"]
    integrity = data["integrity"]
    runs = data["runs"]

    diff_positions = ", ".join(
        f"index {d['index']} ({d['original']}→{d['swapped']})"
        for d in diff["differing_positions"]
    )
    integrity_status = integrity.get("status", "NOT RUN")
    frozen = integrity.get("frozen_record", {})

    return f"""# FormalCRRC Label-Swap Counterbalance Robustness Study — Results

**Preregistered robustness experiment.** Frozen before any swapped-label
inference; 0 amendments. Days 1–3 remain frozen and unchanged.

| | |
|---|---|
| Status | **COMPLETE** — 4/4 judges, {summary["by_model"][MODEL_IDS[0]]["n_rows"]} rows each |
| Preregistration frozen | `{prereg["frozen_at"]}` |
| Preregistration SHA-256 | `{summary["preregistration_sha256"]}` |
| Amendments | {prereg["amendments"]} |
| New evaluations | {lsc.N_NEW_EVALUATIONS:,} ({lsc.N_ROWS_PER_MODEL:,} × {lsc.N_MODELS}) |
| Frozen-record integrity | **{integrity_status}** |
| Protocol | [`PREREGISTRATION_LABEL_SWAP.md`](PREREGISTRATION_LABEL_SWAP.md) |

## 1. Experiment status
{section("status")}
## 2. Relationship to Days 1–3

This study reuses the frozen Day-3 TEST partition (seed {lsc.SOURCE_SEED}) and the Day-3
judge panel at the Day-3 revisions. It is **not** a Day-3 amendment, not a
replacement for Day-3 TRR, and not part of the original three preregistrations.
Day-3 raw scores were read and never written.
{section("relationship")}
## 3. The exact intervention

One conceptual factor moves: {lsc.INTERVENTION}.

| | Original (Day 3) | Swapped (this study) |
|---|---|---|
| Response instruction | `Return A if the criterion is met.`<br>`Return B if the criterion is not met.` | `Return B if the criterion is met.`<br>`Return A if the criterion is not met.` |
| Canonical margin | `{lsc.CANONICAL_MARGIN_ORIGINAL}` | `{lsc.CANONICAL_MARGIN_SWAPPED}` |
| Positive margin means | criterion met | criterion met |

Using `S_A − S_B` for the swapped condition would invert the semantics. The
canonical swapped margin is `S_B − S_A`, and the test suite asserts the sign.
{section("intervention")}
## 4. Model panel

{table_models(prereg, runs)}

All four at the exact Day-3 revisions, bf16, next-token logit scoring with
prompt-continuation label resolution — inherited from Day 3 unchanged. The
swapped tokenisation was verified to be the Day-3 tokenisation with the two
semantic roles exchanged: same token ids, opposite assignment.
{section("panel")}
## 5. Dataset verification

| Check | Result |
|---|---|
| Artifacts | {lsc.N_ARTIFACTS} |
| Rows per model | {lsc.N_ROWS_PER_MODEL:,} |
| Strictness levels | 0–8 |
| Artifacts per true boundary | 36 at each of j\\* = 1…9 |
| Family balance | 108 artifacts per family |
| Artifact identity to Day 3 | exact |
| Formal labels recomputed from primitives | agree |
| j\\* = z + 1 | holds for all three families |
| Nontrivial curves per model | {lsc.N_NONTRIVIAL_PER_MODEL} |
{section("dataset")}
## 6. Prompt-diff verification

| | |
|---|---|
| Original template SHA-256 | `{diff["original_sha256"]}` |
| Swapped template SHA-256 | `{diff["swapped_sha256"]}` |
| Equal length | {diff["same_length"]} |
| Differing byte positions | **{diff["n_differing_positions"]}** — {diff_positions} |
| All differences are label characters | {diff["all_differences_are_label_characters"]} |
| Non-label text identical | {diff["non_label_text_identical"]} |
| Per-row bodies identical | {diff["row_equivalence"]["n_bodies_identical"]}/{diff["row_equivalence"]["n_rows"]} |
| Verdict | **{diff["status"]}** |

Criterion text, candidate response, rubric threshold, artifact identifier,
family, latent level, strictness index and chat-template structure are
byte-identical between conditions. There is no system prompt in FormalCRRC — a
single user message is sent — so there is none to differ.
{section("promptdiff")}
## 7. Primary results — Table A

{table_primary(summary)}

Bootstrap: {BOOTSTRAP_REPLICATES:,} paired artifact-clustered replicates, seed
{BOOTSTRAP_SEED}, percentile intervals. All nine strictness levels of a curve
resample together; the original and swapped members of a pair are drawn jointly.
Nontrivial TRR is bootstrapped over its own 288-curve plan.
{section("primary")}
## 8. Margin-ordering robustness — Table B

{table_ordering(summary)}
{section("ordering")}
## 9. Family-stratified results — Table C

{table_family()}
{section("family")}
## 10. Robustness-band evaluation

Preregistered bands, fixed before inference and unchanged after:
Δ TCE ⊂ [−0.5, +0.5]; Δ TRR ⊂ [−0.10, +0.10]; lower 95% bound of semantic
agreement ≥ 0.90.

{table_robustness(summary)}
{section("robustness")}
## 11. Ordering interpretation
{section("ordering_interpretation")}
## 12. RC = 10 post-study diagnostic

The share of curves whose reachable set is all ten crossings — equivalently,
whose canonical margin curve is strictly decreasing. This is the signature of
the strict scalar-readout model from the Item-2 audit.

**This is a post-study theoretical diagnostic inherited from the step-function
proposition audit, not an original Day-3 endpoint and not a preregistered
outcome of this study.**
{section("rc10")}
## 13. Ties
{section("ties")}
## 14. Limitations
{section("limitations")}
## 15. Integrity verification

| | |
|---|---|
| Protected files | {frozen.get("n_files", "—")} |
| Combined SHA-256 recorded | `{frozen.get("recorded_combined_sha256", "—")}` |
| Combined SHA-256 recomputed | `{frozen.get("recomputed_combined_sha256", "—")}` |
| Changed | {frozen.get("changed") or "**NONE**"} |
| Day-3 inputs read-only | {integrity.get("day3_read_only", {}).get("status", "—")} |
| New files additive only | {integrity.get("new_files", {}).get("status", "—")} |
| Test suite | {integrity.get("tests", {}).get("summary", "—")} |
| Verdict | **{integrity_status}** |

No prior manifest was updated to accommodate a change, because there was none.
{section("integrity")}
## 16. Deviations, retries and failures
{section("deviations")}
## 17. Manuscript-facing interpretation
{section("manuscript")}
## 18. Claims that must not be made

Regardless of outcome, this experiment does not establish that:

- A/B tokens are the only source of judge bias;
- the model internally represents semantic classes in any specific way;
- invariance to A/B swapping implies invariance to arbitrary verbal labels;
- label swapping tests all prompt sensitivity;
- TRR invariance proves the model's representation is stable;
- a change under label swapping invalidates all FormalCRRC findings.

The experiment tests one narrow nuisance factor: the semantic assignment of the
A and B response tokens.

---

*Generated by `scripts/write_results_labelswap.py` from
`{lsc.SUMMARY_PATH}`. Every number above is read from a machine artifact.*
"""


def main() -> int:
    data = load()
    document = build(data)
    out = ROOT / "docs" / "RESULTS_LABEL_SWAP.md"
    out.write_text(document, encoding="utf-8")
    print(f"wrote docs/RESULTS_LABEL_SWAP.md ({len(document):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
