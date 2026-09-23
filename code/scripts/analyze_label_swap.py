#!/usr/bin/env python
"""Validate complete paired panels, compute paired CIs, and report missing models.

Historical original subsamples are exported as clearly provisional references
even when GPUs are unavailable. They never become formal paired results here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from formalcrrc import label_swap as ls, provenance
from formalcrrc.config import MODEL_IDS, MODEL_SLUGS
from run_label_swap_inference import load_historical


def markdown_table(frame, columns):
    def cell(value):
        return f"{value:.6f}" if isinstance(value, float) else str(value)
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(cell(row[c]) for c in columns) + " |" for row in frame.to_dict("records")]])


def analyze(root, experiment_id, analysis_id):
    out, config, frames = ls.load_frozen(root, experiment_id)
    # Reuse path validation for a user-supplied single directory component.
    ls.experiment_path(root, analysis_id)
    target = out / "analysis" / analysis_id
    target.mkdir(parents=True, exist_ok=False)
    status, tables, samples, metrics, references = {}, [], [], [], []
    try:
        baseline = ls.baseline_table(frames["original"])
        baseline.to_csv(target / "baselines.csv", index=False)
        for model in MODEL_IDS:
            slug = MODEL_SLUGS[model]
            directory = out / "runs" / slug
            if not (directory / "COMPLETE.json").exists():
                status[model] = {"status": "incomplete", "reason": "no validated paired completion marker"}
                # Historical scores are reference-only until replay has approved reuse.
                try:
                    reference, _, issues = load_historical(root, frames["original"], model, config["models"][model])
                    reference["reuse_status"] = "pending_runtime_replay"
                    reference.to_parquet(target / f"{slug}_historical_reference.parquet", index=False)
                    rm = ls.artifact_metrics(reference)
                    for family in (None, *ls.FAMILIES):
                        group = rm if family is None else rm[rm.family == family]
                        references.append({"model_id": model, "scope": family or "overall",
                            "tce": group.tce.mean(), "trr_percent": 100 * group.loc[group.nontrivial, "reachable"].mean(),
                            "otce": group.otce.mean(), "status": "historical_reference_only", "metadata_issues": "; ".join(issues)})
                except Exception as error:
                    status[model]["historical_reference_error"] = str(error)
                continue
            try:
                completion = ls.read_json(directory / "COMPLETE.json")
                if completion["status"] != "complete":
                    raise ValueError("invalid completion status")
                ls.assert_hashes(directory, completion["files"])
                run = ls.read_json(directory / "run.json")
                if run["frozen_sha256"] != provenance.sha256_file(out / "FROZEN.json"):
                    raise ValueError("run uses different frozen experiment")
                a, b = ls.paired_frames(pd.read_parquet(directory / "original.parquet"),
                    pd.read_parquet(directory / "swapped.parquet"), frames, model)
                for condition, frame in (("original", a), ("swapped", b)):
                    if set(frame.revision) != {config["models"][model]["revision"]}:
                        raise ValueError("score revision mismatch")
                    tokenization = run["tokenizer_check"]["conditions"][condition]["tokenization"]
                    if set(frame.scoring_method) != {tokenization["scoring_method"]}:
                        raise ValueError("score method mismatch")
                table, replicates = ls.paired_statistics(a, b, replicates=config["bootstrap_replicates"], seed=config["bootstrap_seed"])
                # TRR levels are percentages; only their signed differences are pp.
                table.loc[(table.metric == "trr") & (table.condition != "swapped-original"), "unit"] = "percent"
                metrics.extend([ls.artifact_metrics(a), ls.artifact_metrics(b)])
                tables.append(table)
                samples.append(replicates)
                status[model] = {"status": "complete", "original_source": run["original_source"],
                                 "rows_per_condition": len(a), "artifacts_per_condition": a.artifact_id.nunique()}
            except Exception as error:
                status[model] = {"status": "blocked_invalid_pair", "reason": f"{type(error).__name__}: {error}"}
        if tables:
            pd.concat(tables, ignore_index=True).to_csv(target / "paired_results.csv", index=False)
            pd.concat(samples, ignore_index=True).to_parquet(target / "bootstrap_samples.parquet", index=False)
            pd.concat(metrics, ignore_index=True).to_parquet(target / "artifact_metrics.parquet", index=False)
        if references:
            pd.DataFrame(references).to_csv(target / "historical_reference_summary.csv", index=False)
        complete_count = sum(s["status"] == "complete" for s in status.values())
        report = ["# Supplementary A/B label mapping counterbalance", "",
            f"Experiment: `{experiment_id}`. **Completed paired models: {complete_count}/4.**",
            "This is a new supplementary experiment, not part of the original preregistration.", "",
            "81 artifacts (three per family × latent-level cell), nine strictness levels each; 729 prompts per model/condition. "
            "Selection seed 20260909. Both arms use identical selected artifacts, truth, rubric and candidate text. "
            "Only the final A/B mapping suffix changes; the met line remains first.", "",
            "Scores are aligned to met semantics: original A−B, swapped B−A. Zero margin passes; "
            "first-fail sentinel 9; strict record lows with no equality tolerance; nontrivial TRR uses j*=1..8. "
            "OTCE is the artifact translation oracle; no global/family alpha is fitted.", "",
            "5000 paired artifact bootstrap replicates, stratified by family, percentile 95% intervals. "
            "Each artifact carries all nine thresholds. TRR denominators are recomputed in each resample. "
            "Points are actual-data estimates; all signed differences are swapped−original, with TRR differences in percentage points. "
            "Semantic disagreement is the symmetric per-prompt pass/fail mismatch (clustered by artifact); it has no directional condition difference.", "",
            "## Model status", "", *[f"- `{m}`: {s['status']}" + (f" — {s['reason']}" if 'reason' in s else "") for m, s in status.items()], ""]
        if tables:
            combined = pd.concat(tables, ignore_index=True)
            report += ["## Paired results (overall)", "", markdown_table(combined[combined.scope == "overall"],
                ["model_id", "metric", "condition", "point", "ci_low", "ci_high", "unit"]), "",
                "Family-specific estimates and intervals are in `paired_results.csv`.", ""]
        else:
            report += ["**No paired estimates or confidence intervals are available. Swapped inference has not completed.** "
                "Historical originals below do not validate runtime reproducibility and cannot determine mapping sensitivity.", ""]
        report += ["## Checked analytic TCE baselines", "", markdown_table(baseline[baseline.scope == "overall"],
            ["baseline", "tce", "tce_exact", "prediction_support", "boundary_counts"]), ""]
        if references:
            report += ["## Historical original references (pending replay; not paired results)", "",
                markdown_table(pd.DataFrame(references).query("scope == 'overall'"), ["model_id", "tce", "trr_percent", "otce", "status"]), ""]
        report += ["## Interpretation limits", "",
            "A TCE change with similar TRR/OTCE is consistent with a location-shift account. Changes in TRR/OTCE "
            "indicate mapping-sensitive boundary reachability. Small differences support robustness only within this design; "
            "nonsignificance alone does not establish absence of label bias. This design measures mapping sensitivity "
            "at fixed semantic line order and cannot separately identify letter preference or option-position preference. "
            "Both mappings must be reported; neither replaces historical conclusions.", "",
            "## Provenance and execution", "",
            "See `configuration.json`, `FROZEN.json`, tokenizer checks, run metadata and replay decisions. "
            "The historical CRLF discrepancy is recorded separately from verification of raw LF bytes in an independent checkout. "
            "No historical checksum has been changed. See `docs/LABEL_SWAP.md` for preparation, CPU, GPU and analysis commands.", ""]
        (target / "REPORT.md").write_bytes("\n".join(report).encode("utf-8"))
        input_profile = ls.verify_input_profile(root, out, config)
        ls.write_json(target / "status.json", {"experiment_id": experiment_id, "complete_models": complete_count,
            "intended_models": 4, "panel_complete": complete_count == 4, "models": status,
            "historical_input_profile": input_profile,
            "analysis_environment": provenance.environment_snapshot(False),
            "source_manifest": provenance.source_manifest([
                "src/formalcrrc/label_swap.py", "src/formalcrrc/day3.py", "src/formalcrrc/day3_bootstrap.py",
                "src/formalcrrc/day2_bootstrap.py", "src/formalcrrc/baselines.py", "scripts/analyze_label_swap.py"], root=root)})
        ls.write_json(target / "ANALYSIS_WRITTEN.json", {"panel_complete": complete_count == 4,
            "files": ls.file_hashes(target, [p.name for p in target.iterdir() if p.is_file()])}, exclusive=True)
        print(f"{target}: complete paired models {complete_count}/4", flush=True)
        return 0 if complete_count == 4 else 1
    except Exception as error:
        ls.write_json(target / "INCOMPLETE.json", {"error_type": type(error).__name__, "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=provenance.repo_root())
    parser.add_argument("--experiment-id", default=ls.DEFAULT_EXPERIMENT)
    parser.add_argument("--analysis-id", default="final", help="existing analysis directories are never overwritten")
    args = parser.parse_args()
    return analyze(args.root, args.experiment_id, args.analysis_id)


if __name__ == "__main__":
    raise SystemExit(main())
