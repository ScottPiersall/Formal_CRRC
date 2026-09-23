#!/usr/bin/env python
"""Pair, measure and bootstrap the label-swap robustness study.

Reads the frozen Day-3 raw scores **read-only** and the new swapped raw scores,
pairs them by prompt id, and computes every preregistered outcome. Writes only
under ``artifacts/labelswap/``.

Both conditions are analysed on their own canonical margin -- ``S_A - S_B`` for
the original, ``S_B - S_A`` for the swapped -- so that a positive margin means
"criterion met" in both and every Day-3 convention applies unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import labelswap, predicates, provenance  # noqa: E402
from formalcrrc import labelswap_bootstrap as lsb  # noqa: E402
from formalcrrc import labelswap_config as lsc  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FAMILIES,
    MODEL_IDS,
    MODEL_SLUGS,
    N_STRICTNESS,
)

ROOT = provenance.repo_root()
SHORT = {
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "google/gemma-2-9b-it": "Gemma-2-9B",
}


def load_pair(model_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen Day-3 scores (read-only) and the new swapped scores."""
    slug = MODEL_SLUGS[model_id]
    original = pd.read_parquet(ROOT / f"artifacts/day3/raw_scores/{slug}.parquet")
    swapped = pd.read_parquet(ROOT / lsc.RAW_SCORES_DIR / f"{slug}.parquet")
    return original, swapped


def curve_table(model_id: str) -> pd.DataFrame:
    """One row per artifact carrying every paired per-curve quantity."""
    original, swapped = load_pair(model_id)

    if set(original["prompt_id"]) != set(swapped["prompt_id"]):
        raise AssertionError(f"{model_id}: prompt id sets differ between conditions")

    o = original.sort_values(["artifact_id", "strictness_index"])
    s = swapped.sort_values(["artifact_id", "strictness_index"])

    rows: list[dict] = []
    for artifact_id, o_group in o.groupby("artifact_id", sort=True):
        s_group = s[s["artifact_id"] == artifact_id]
        family = o_group["family"].iloc[0]
        latent = int(o_group["latent_level"].iloc[0])
        # Recomputed from the predicate, never read from a stored column.
        j_star = labelswap.true_first_fail(family, latent)

        m_o = o_group["margin"].to_numpy(dtype=float)
        m_s = s_group["margin"].to_numpy(dtype=float)
        # Day-3 raw scores carry no label column; recompute truth from the
        # predicate rather than joining a stored one.
        truth = np.array(
            [
                predicates.formal_truth(family, latent, int(s))
                for s in o_group["strictness_index"]
            ],
            dtype=int,
        )

        met_o = labelswap.curve_metrics(m_o, j_star)
        met_s = labelswap.curve_metrics(m_s, j_star)

        d_o = labelswap.semantic_decision(m_o)
        d_s = labelswap.semantic_decision(m_s)

        acc_o, brier_o = labelswap.row_accuracy_and_brier(m_o, truth)
        acc_s, brier_s = labelswap.row_accuracy_and_brier(m_s, truth)

        same_set, jaccard = labelswap.reachable_set_agreement(
            met_o["reachable_crossings"], met_s["reachable_crossings"]
        )

        with np.errstate(invalid="ignore"):
            rho = stats.spearmanr(m_o, m_s).statistic

        rows.append(
            {
                "model_id": model_id,
                "artifact_id": artifact_id,
                "family": family,
                "latent_level": latent,
                "true_first_fail_index": j_star,
                "nontrivial_boundary": met_o["nontrivial_boundary"],
                # primary
                "semantic_agreement": float(np.mean(d_o == d_s)),
                "sign_agreement": float(np.mean(np.sign(m_o) == np.sign(m_s))),
                "j_hat_original": met_o["j_hat"],
                "j_hat_swapped": met_s["j_hat"],
                "tce_original": met_o["tce"],
                "tce_swapped": met_s["tce"],
                "delta_tce": met_s["tce"] - met_o["tce"],
                "tr_original": int(met_o["translation_reachable"]),
                "tr_swapped": int(met_s["translation_reachable"]),
                "delta_tr": int(met_s["translation_reachable"])
                - int(met_o["translation_reachable"]),
                # secondary
                "accuracy_original": acc_o,
                "accuracy_swapped": acc_s,
                "delta_accuracy": acc_s - acc_o,
                "brier_original": brier_o,
                "brier_swapped": brier_s,
                "delta_brier": brier_s - brier_o,
                "rc_original": met_o["reachable_count"],
                "rc_swapped": met_s["reachable_count"],
                "delta_rc": met_s["reachable_count"] - met_o["reachable_count"],
                "prr_original": met_o["prefix_record_rate"],
                "prr_swapped": met_s["prefix_record_rate"],
                "mmvr_original": met_o["mmvr"],
                "mmvr_swapped": met_s["mmvr"],
                "mmvm_original": met_o["mmvm"],
                "mmvm_swapped": met_s["mmvm"],
                "span_original": met_o["span"],
                "span_swapped": met_s["span"],
                "full_reachability_original": int(met_o["full_reachability"]),
                "full_reachability_swapped": int(met_s["full_reachability"]),
                "reachable_set_identical": int(same_set),
                "reachable_set_jaccard": jaccard,
                "reachable_original": met_o["reachable_crossings"],
                "reachable_swapped": met_s["reachable_crossings"],
                "margin_rank_correlation": float(rho),
                "adjacent_equal_original": met_o["adjacent_equal_margins"],
                "adjacent_equal_swapped": met_s["adjacent_equal_margins"],
            }
        )
    return pd.DataFrame(rows)


def paired_rows(model_id: str) -> pd.DataFrame:
    """Row-level paired frame, retained so the analysis is reproducible."""
    original, swapped = load_pair(model_id)
    o = original[
        ["prompt_id", "artifact_id", "family", "latent_level", "strictness_index",
         "raw_score_met", "raw_score_not_met", "margin", "p_met"]
    ].rename(
        columns={
            "raw_score_met": "logit_a_original",
            "raw_score_not_met": "logit_b_original",
            "margin": "margin_original",
            "p_met": "p_met_original",
        }
    )
    s = swapped[
        ["prompt_id", "logit_a", "logit_b", "margin", "p_met", "formal_truth",
         "true_first_fail_index"]
    ].rename(
        columns={
            "logit_a": "logit_a_swapped",
            "logit_b": "logit_b_swapped",
            "margin": "margin_swapped",
            "p_met": "p_met_swapped",
        }
    )
    merged = o.merge(s, on="prompt_id", validate="one_to_one")
    merged["model_id"] = model_id
    merged["decision_original"] = (merged["margin_original"] >= 0).astype(int)
    merged["decision_swapped"] = (merged["margin_swapped"] >= 0).astype(int)
    merged["decisions_agree"] = (
        merged["decision_original"] == merged["decision_swapped"]
    ).astype(int)
    return merged


def main() -> int:
    out_dir = ROOT / lsc.ARTIFACT_DIR
    tables_dir = ROOT / lsc.TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        m
        for m in MODEL_IDS
        if not (ROOT / lsc.RAW_SCORES_DIR / f"{MODEL_SLUGS[m]}.parquet").is_file()
    ]
    if missing:
        print("REFUSING TO ANALYSE: panel incomplete. Missing:")
        for model_id in missing:
            print(f"  {model_id}")
        return 2

    all_metrics: list[pd.DataFrame] = []
    all_rows: list[pd.DataFrame] = []
    per_model: dict[str, dict] = {}

    for model_id in MODEL_IDS:
        metrics = curve_table(model_id)
        all_metrics.append(metrics)
        all_rows.append(paired_rows(model_id))

        boot = lsb.bootstrap_judge(metrics)
        robustness = lsb.evaluate_robustness(
            boot["statistics"],
            lsc.DELTA_TCE_BAND,
            lsc.DELTA_TRR_BAND,
            lsc.AGREEMENT_LOWER_BOUND,
        )

        families: dict[str, dict] = {}
        for family in FAMILIES:
            subset = metrics[metrics["family"] == family]
            family_boot = lsb.bootstrap_judge(subset)
            families[family] = {
                "n_artifacts": int(len(subset)),
                "n_nontrivial": int(subset["nontrivial_boundary"].sum()),
                "statistics": family_boot["statistics"],
            }

        nontrivial = metrics[metrics["nontrivial_boundary"]]
        per_model[model_id] = {
            "model_id": model_id,
            "short_name": SHORT[model_id],
            "n_artifacts": int(len(metrics)),
            "n_nontrivial": int(len(nontrivial)),
            "n_rows": int(len(metrics) * N_STRICTNESS),
            "bootstrap": boot,
            "robustness": robustness,
            "families": families,
            "point_estimates": {
                "semantic_agreement": float(metrics["semantic_agreement"].mean()),
                "sign_agreement": float(metrics["sign_agreement"].mean()),
                "tce_original": float(metrics["tce_original"].mean()),
                "tce_swapped": float(metrics["tce_swapped"].mean()),
                "delta_tce": float(metrics["delta_tce"].mean()),
                "trr_original": float(nontrivial["tr_original"].mean()),
                "trr_swapped": float(nontrivial["tr_swapped"].mean()),
                "delta_trr": float(
                    nontrivial["tr_swapped"].mean() - nontrivial["tr_original"].mean()
                ),
                "unreachable_original": float(1 - nontrivial["tr_original"].mean()),
                "unreachable_swapped": float(1 - nontrivial["tr_swapped"].mean()),
                "accuracy_original": float(metrics["accuracy_original"].mean()),
                "accuracy_swapped": float(metrics["accuracy_swapped"].mean()),
                "brier_original": float(metrics["brier_original"].mean()),
                "brier_swapped": float(metrics["brier_swapped"].mean()),
                "rc_original": float(metrics["rc_original"].mean()),
                "rc_swapped": float(metrics["rc_swapped"].mean()),
                "rc10_original": float(metrics["full_reachability_original"].mean()),
                "rc10_swapped": float(metrics["full_reachability_swapped"].mean()),
                "reachable_set_identical": float(
                    metrics["reachable_set_identical"].mean()
                ),
                "reachable_set_jaccard": float(metrics["reachable_set_jaccard"].mean()),
                "median_margin_rank_correlation": float(
                    np.nanmedian(metrics["margin_rank_correlation"])
                ),
                "mmvr_original": float(metrics["mmvr_original"].mean()),
                "mmvr_swapped": float(metrics["mmvr_swapped"].mean()),
                "prr_original": float(metrics["prr_original"].mean()),
                "prr_swapped": float(metrics["prr_swapped"].mean()),
                "adjacent_equal_original": int(
                    (metrics["adjacent_equal_original"] > 0).sum()
                ),
                "adjacent_equal_swapped": int(
                    (metrics["adjacent_equal_swapped"] > 0).sum()
                ),
            },
        }
        print(f"analysed {model_id}", flush=True)

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    rows_all = pd.concat(all_rows, ignore_index=True)
    metrics_all.to_parquet(ROOT / lsc.METRICS_PATH, index=False)
    rows_all.to_parquet(ROOT / lsc.PAIRED_ROWS_PATH, index=False)

    # ---- bootstrap results in long form ------------------------------------
    boot_rows = []
    for model_id, record in per_model.items():
        for name, value in record["bootstrap"]["statistics"].items():
            boot_rows.append(
                {
                    "model_id": model_id,
                    "scope": "all",
                    "statistic": name,
                    **{k: v for k, v in value.items() if not isinstance(v, dict)},
                }
            )
        for family, family_record in record["families"].items():
            for name, value in family_record["statistics"].items():
                boot_rows.append(
                    {
                        "model_id": model_id,
                        "scope": family,
                        "statistic": name,
                        **{k: v for k, v in value.items() if not isinstance(v, dict)},
                    }
                )
    pd.DataFrame(boot_rows).to_parquet(ROOT / lsc.BOOTSTRAP_PATH, index=False)

    # ---- Table A: primary robustness ---------------------------------------
    table_a = pd.DataFrame(
        [
            {
                "Judge": SHORT[m],
                "Original TCE": round(r["point_estimates"]["tce_original"], 4),
                "Swapped TCE": round(r["point_estimates"]["tce_swapped"], 4),
                "Delta TCE": round(r["point_estimates"]["delta_tce"], 4),
                "Delta TCE CI low": round(
                    r["bootstrap"]["statistics"]["delta_tce"]["ci_low"], 4
                ),
                "Delta TCE CI high": round(
                    r["bootstrap"]["statistics"]["delta_tce"]["ci_high"], 4
                ),
                "Semantic agreement": round(
                    r["point_estimates"]["semantic_agreement"], 4
                ),
                "Agreement CI low": round(
                    r["bootstrap"]["statistics"]["agreement"]["ci_low"], 4
                ),
                "Original TRR": round(r["point_estimates"]["trr_original"], 4),
                "Swapped TRR": round(r["point_estimates"]["trr_swapped"], 4),
                "Delta TRR": round(r["point_estimates"]["delta_trr"], 4),
                "Delta TRR CI low": round(
                    r["bootstrap"]["statistics"]["delta_trr"]["ci_low"], 4
                ),
                "Delta TRR CI high": round(
                    r["bootstrap"]["statistics"]["delta_trr"]["ci_high"], 4
                ),
                "Swapped unreachable %": round(
                    100 * r["point_estimates"]["unreachable_swapped"], 2
                ),
            }
            for m, r in per_model.items()
        ]
    )
    table_a.to_csv(tables_dir / "table_a_primary.csv", index=False)

    # ---- Table B: margin-ordering robustness -------------------------------
    table_b = pd.DataFrame(
        [
            {
                "Judge": SHORT[m],
                "Median curve Spearman rho": round(
                    r["point_estimates"]["median_margin_rank_correlation"], 4
                ),
                "Sign agreement": round(r["point_estimates"]["sign_agreement"], 4),
                "Reachable-set agreement": round(
                    r["point_estimates"]["reachable_set_identical"], 4
                ),
                "Reachable-set Jaccard": round(
                    r["point_estimates"]["reachable_set_jaccard"], 4
                ),
                "Original mean RC": round(r["point_estimates"]["rc_original"], 4),
                "Swapped mean RC": round(r["point_estimates"]["rc_swapped"], 4),
                "Original RC=10 %": round(100 * r["point_estimates"]["rc10_original"], 2),
                "Swapped RC=10 %": round(100 * r["point_estimates"]["rc10_swapped"], 2),
            }
            for m, r in per_model.items()
        ]
    )
    table_b.to_csv(tables_dir / "table_b_ordering.csv", index=False)

    # ---- Table C: family-stratified ----------------------------------------
    table_c_rows = []
    for model_id, record in per_model.items():
        subset = metrics_all[metrics_all["model_id"] == model_id]
        for family in FAMILIES:
            fam = subset[subset["family"] == family]
            fam_nt = fam[fam["nontrivial_boundary"]]
            stats_ = record["families"][family]["statistics"]
            table_c_rows.append(
                {
                    "Judge": SHORT[model_id],
                    "Family": family,
                    "n": int(len(fam)),
                    "n nontrivial": int(len(fam_nt)),
                    "Swapped TCE": round(float(fam["tce_swapped"].mean()), 4),
                    "Delta TCE": round(float(fam["delta_tce"].mean()), 4),
                    "Delta TCE CI low": round(stats_["delta_tce"]["ci_low"], 4),
                    "Delta TCE CI high": round(stats_["delta_tce"]["ci_high"], 4),
                    "Swapped TRR": round(float(fam_nt["tr_swapped"].mean()), 4),
                    "Delta TRR": round(
                        float(
                            fam_nt["tr_swapped"].mean() - fam_nt["tr_original"].mean()
                        ),
                        4,
                    ),
                    "Delta TRR CI low": round(stats_["delta_trr"]["ci_low"], 4),
                    "Delta TRR CI high": round(stats_["delta_trr"]["ci_high"], 4),
                    "Semantic agreement": round(
                        float(fam["semantic_agreement"].mean()), 4
                    ),
                }
            )
    pd.DataFrame(table_c_rows).to_csv(tables_dir / "table_c_family.csv", index=False)

    summary = {
        "study": lsc.STUDY_NAME,
        "preregistered": True,
        "preregistration_sha256": (
            (ROOT / lsc.PREREG_SHA_PATH).read_text(encoding="utf-8").split()[0]
        ),
        "generated_utc": provenance.utc_now(),
        "panel_complete": True,
        "canonical_margin_original": lsc.CANONICAL_MARGIN_ORIGINAL,
        "canonical_margin_swapped": lsc.CANONICAL_MARGIN_SWAPPED,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "scheme": lsc.BOOTSTRAP_SCHEME,
            "interval": lsc.BOOTSTRAP_CI,
        },
        "robustness_bands": {
            "delta_tce": list(lsc.DELTA_TCE_BAND),
            "delta_trr": list(lsc.DELTA_TRR_BAND),
            "agreement_lower_bound": lsc.AGREEMENT_LOWER_BOUND,
        },
        "by_model": per_model,
    }
    provenance.write_json(ROOT / lsc.SUMMARY_PATH, summary)

    print()
    print(table_a.to_string(index=False))
    print()
    print(table_b.to_string(index=False))
    print()
    for model_id, record in per_model.items():
        print(f"{SHORT[model_id]:<14} {record['robustness']['classification']:<12} "
              f"failed={record['robustness']['failed_criteria'] or 'none'}")
    print()
    print(f"wrote {lsc.METRICS_PATH}")
    print(f"wrote {lsc.PAIRED_ROWS_PATH}")
    print(f"wrote {lsc.BOOTSTRAP_PATH}")
    print(f"wrote {lsc.SUMMARY_PATH}")
    print(f"wrote {lsc.TABLES_DIR}/table_{{a,b,c}}_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
