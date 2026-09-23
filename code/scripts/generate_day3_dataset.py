#!/usr/bin/env python
"""Generate and freeze the Day-3 evaluation set (Phase C).

One fresh partition of 324 artifacts x 9 thresholds, drawn from the same
data-generating process as Days 1 and 2 with master seed 45. Day 3 needs no
calibration partition: the oracle uses formal truth directly.

Family C draws an integer target from a bounded range, so generation carries a
forbidden set of (target, reported) pairs -- Day 1's, Day-2 CAL's and Day-2
TEST's -- and redraws on collision, exactly as Day 2 did. That guarantees no
evaluated item is shared with any prior experiment.

Writes the dataset, its prompt manifest with the frozen inference order, and a
manifest. Runs before any Day-3 model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import dataset as ds  # noqa: E402
from formalcrrc import day2_config as d2c  # noqa: E402
from formalcrrc import day3_config as d3c  # noqa: E402
from formalcrrc import predicates, prompts, provenance  # noqa: E402
from formalcrrc.config import (  # noqa: E402
    FAMILIES,
    FAMILY_NUMERIC_TOLERANCE,
    LIST_SIZE,
    MARKERS_PER_RESPONSE,
    N_ARTIFACTS,
    N_INSTANCES,
    N_LATENT_LEVELS,
    N_ROWS,
    N_STRICTNESS,
)


def family_c_pairs(frame: pd.DataFrame) -> frozenset[tuple[int, int]]:
    """The (target, reported) pairs used by a dataset's family-C artifacts."""
    subset = frame[frame["family"] == FAMILY_NUMERIC_TOLERANCE].drop_duplicates(
        "artifact_id"
    )
    return frozenset(
        zip(subset["target_value"].astype(int), subset["reported_value"].astype(int))
    )


def rendered_prompts(frame: pd.DataFrame) -> set[str]:
    return {
        prompts.build_user_message(rubric, response)
        for rubric, response in zip(
            frame["rubric_text"], frame["candidate_response"], strict=True
        )
    }


def assert_structure(frame: pd.DataFrame) -> None:
    """Every structural invariant the Day-3 partition must satisfy alone."""
    tag = d3c.PARTITION_TAG
    assert len(frame) == N_ROWS, f"expected {N_ROWS} rows, got {len(frame)}"
    assert frame["artifact_id"].nunique() == N_ARTIFACTS, "artifact count"
    assert frame["prompt_id"].is_unique, "prompt ids not unique"
    assert set(frame["seed"]) == {d3c.SEED}, "seed column"
    assert frame["artifact_id"].str.startswith(f"{tag}-").all(), "namespacing"

    for artifact_id, group in frame.groupby("artifact_id", sort=False):
        levels = sorted(group["strictness_index"].tolist())
        assert levels == list(range(N_STRICTNESS)), f"{artifact_id} levels"
        assert group["candidate_response"].nunique() == 1, (
            f"{artifact_id}: candidate response is not byte-identical"
        )
        family = group["family"].iloc[0]
        normalized = {
            predicates.normalize_rubric(family, text) for text in group["rubric_text"]
        }
        assert len(normalized) == 1, (
            f"{artifact_id}: rubrics differ outside the threshold field"
        )
        assert group["rubric_sha256"].nunique() == N_STRICTNESS, (
            f"{artifact_id}: rubric text does not vary with the threshold"
        )
        ordered = group.sort_values("strictness_index")
        truth = ordered["formal_truth"].tolist()
        assert truth == sorted(truth, reverse=True), f"{artifact_id}: monotonicity"
        latent = int(group["latent_level"].iloc[0])
        recomputed = [
            predicates.formal_truth(family, latent, s) for s in range(N_STRICTNESS)
        ]
        assert truth == recomputed, f"{artifact_id}: truth not reproducible"
        zeros = [s for s, y in enumerate(truth) if y == 0]
        expected = zeros[0] if zeros else N_STRICTNESS
        assert set(ordered["true_first_fail_index"]) == {expected}, (
            f"{artifact_id}: first-fail index"
        )

    cells = Counter(zip(frame["family"], frame["latent_level"]))
    for family in FAMILIES:
        for latent in range(N_LATENT_LEVELS):
            assert cells[(family, latent)] == N_INSTANCES * N_STRICTNESS, (
                f"{family}/L{latent}: wrong row count"
            )

    markers = frame[frame["family"] != FAMILY_NUMERIC_TOLERANCE]
    for payload in markers["response_markers_json"].unique():
        assert payload.count("MK-") == MARKERS_PER_RESPONSE, "marker count"
    for payload in markers["list_items_json"].unique():
        assert payload.count("MK-") == LIST_SIZE, "item list size"


def disjointness_report(day3: pd.DataFrame, priors: dict[str, pd.DataFrame]) -> dict:
    """Overlap of the Day-3 partition with every prior evaluated set."""

    def candidates(frame: pd.DataFrame, family: str | None = None) -> set[str]:
        subset = frame if family is None else frame[frame["family"] == family]
        return set(subset.drop_duplicates("artifact_id")["candidate_sha256"])

    day3_ids = set(day3["artifact_id"])
    day3_prompts = rendered_prompts(day3)
    day3_pairs = family_c_pairs(day3)

    report: dict = {
        "artifact_id_overlap": {},
        "rendered_prompt_overlap": {},
        "family_c_target_pair_overlap": {},
        "candidate_response_overlap_by_family": {},
    }
    for name, frame in priors.items():
        report["artifact_id_overlap"][name] = len(day3_ids & set(frame["artifact_id"]))
        report["rendered_prompt_overlap"][name] = len(
            day3_prompts & rendered_prompts(frame)
        )
        report["family_c_target_pair_overlap"][name] = len(
            day3_pairs & family_c_pairs(frame)
        )
        report["candidate_response_overlap_by_family"][name] = {
            family: len(candidates(day3, family) & candidates(frame, family))
            for family in FAMILIES
        }

    for group in (
        "artifact_id_overlap",
        "rendered_prompt_overlap",
        "family_c_target_pair_overlap",
    ):
        for name, value in report[group].items():
            assert value == 0, f"{group} vs {name} = {value}"
    for name in priors:
        for family in FAMILIES:
            if family == FAMILY_NUMERIC_TOLERANCE:
                continue
            value = report["candidate_response_overlap_by_family"][name][family]
            assert value == 0, f"{family} candidate overlap vs {name} = {value}"

    report["family_c_candidate_coincidence_note"] = (
        "A family-C candidate response is determined by the reported value "
        "alone, an integer in a bounded range, so byte-identical candidate "
        "strings recur across independently seeded partitions. They are not "
        "reused items: the (target, reported) pairs and the rendered prompts "
        "are disjoint by construction."
    )
    return report


def leakage_checks(frame: pd.DataFrame) -> dict:
    """The partition tag must never reach a model-facing prompt."""
    offenders_id: list[str] = []
    offenders_word: list[str] = []
    for row in frame.itertuples(index=False):
        message = prompts.build_user_message(row.rubric_text, row.candidate_response)
        if row.artifact_id in message:
            offenders_id.append(row.prompt_id)
        lowered = message.lower()
        if any(w in lowered for w in ("calibration", "partition", "held-out", "oracle")):
            offenders_word.append(row.prompt_id)
    assert not offenders_id, f"artifact id leaked into {len(offenders_id)} prompts"
    assert not offenders_word, f"partition wording leaked into {len(offenders_word)}"
    return {
        "artifact_id_in_prompt": 0,
        "partition_wording_in_prompt": 0,
        "note": (
            "Prompts contain only the rubric and the candidate response. The "
            "partition tag lives in artifact_id and prompt_id, never rendered "
            "into a model-facing message."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = Path(args.root)

    print("rebuilding prior partitions for the forbidden-pair set ...", flush=True)
    day1 = ds.build_dataset()
    day1_hash = ds.dataset_content_hash(day1)
    if day1_hash != d3c.DAY1_DATASET_CONTENT_SHA256:
        print(
            "ABORT: Day-1 dataset no longer reproduces its frozen content hash",
            flush=True,
        )
        return 2
    print(f"Day-1 content hash reproduced: {day1_hash[:16]}", flush=True)

    day2_cal = pd.read_parquet(root / d2c.CALIBRATION_DATASET_PATH)
    day2_test = pd.read_parquet(root / d2c.TEST_DATASET_PATH)
    priors = {"day1": day1, "day2_calibration": day2_cal, "day2_test": day2_test}

    forbidden = (
        family_c_pairs(day1) | family_c_pairs(day2_cal) | family_c_pairs(day2_test)
    )
    print(f"forbidden family-C pairs: {len(forbidden)}", flush=True)

    print(f"building {d3c.PARTITION_NAME} (seed {d3c.SEED}) ...", flush=True)
    frame = ds.build_dataset(
        seed=d3c.SEED,
        partition=d3c.PARTITION_TAG,
        forbidden_pairs=forbidden,
        enforce_unique_targets=True,
    )
    assert_structure(frame)
    cross = disjointness_report(frame, priors)
    leakage = leakage_checks(frame)

    prompt_frame = prompts.build_prompt_manifest(frame)
    prompt_frame = prompt_frame.drop(columns=["order_index", "order_key"])
    prompt_frame["order_key"] = [
        provenance.sha256_text(f"{d3c.INFERENCE_ORDER_SALT}|{pid}")
        for pid in prompt_frame["prompt_id"]
    ]
    prompt_frame = prompt_frame.sort_values("order_key").reset_index(drop=True)
    prompt_frame.insert(0, "order_index", range(len(prompt_frame)))

    dataset_path = root / d3c.DATASET_PATH
    prompt_path = root / d3c.PROMPT_MANIFEST_PATH
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dataset_path, index=False)
    prompt_frame.to_parquet(prompt_path, index=False)

    manifest = {
        "created_at": provenance.utc_now(),
        "experiment": "FormalCRRC Day 3",
        "partition": d3c.PARTITION_NAME,
        "partition_tag": d3c.PARTITION_TAG,
        "role": "single held-out evaluation partition; no calibration partition",
        "seed": d3c.SEED,
        "dimensions": {
            "families": list(FAMILIES),
            "n_latent_levels": N_LATENT_LEVELS,
            "n_instances_per_cell": N_INSTANCES,
            "n_artifacts": int(frame["artifact_id"].nunique()),
            "n_strictness_levels": N_STRICTNESS,
            "n_rows": int(len(frame)),
        },
        "hashes": {
            "dataset_content_sha256": ds.dataset_content_hash(frame),
            "dataset_file_sha256": provenance.sha256_file(dataset_path),
            "prompt_manifest_file_sha256": provenance.sha256_file(prompt_path),
            "prompt_template_sha256": prompts.prompt_template_sha256(),
            "prompt_set_sha256": hashlib.sha256(
                "\n".join(sorted(prompt_frame["user_message_sha256"])).encode("utf-8")
            ).hexdigest(),
        },
        "inference_order": {
            "salt": d3c.INFERENCE_ORDER_SALT,
            "rule": "sort by sha256(salt|prompt_id), frozen before inference",
        },
        "leakage_checks": leakage,
        "disjointness": cross,
        "family_c_uniqueness_rule": (
            "Family-C (target, reported) pairs already used by Day 1, Day-2 CAL "
            "or Day-2 TEST are excluded and redrawn, so no evaluated item is "
            "shared with any prior experiment."
        ),
        "source": provenance.source_manifest(
            [
                "src/formalcrrc/config.py",
                "src/formalcrrc/predicates.py",
                "src/formalcrrc/dataset.py",
                "src/formalcrrc/prompts.py",
                "src/formalcrrc/day3_config.py",
                "scripts/generate_day3_dataset.py",
            ],
            root=root,
        ),
        "git": provenance.git_state(root),
    }
    provenance.write_json(root / d3c.DATASET_MANIFEST_PATH, manifest)

    print(
        f"{d3c.PARTITION_NAME}: {manifest['dimensions']['n_artifacts']} artifacts, "
        f"{manifest['dimensions']['n_rows']} rows, content "
        f"{manifest['hashes']['dataset_content_sha256'][:16]}",
        flush=True,
    )
    print("", flush=True)
    print("overlap with prior evaluated sets (all must be zero):", flush=True)
    for group in (
        "artifact_id_overlap",
        "rendered_prompt_overlap",
        "family_c_target_pair_overlap",
    ):
        print(f"  {group}: {cross[group]}", flush=True)
    print(
        "  family-C candidate coincidences: "
        + str(
            {
                name: block[FAMILY_NUMERIC_TOLERANCE]
                for name, block in cross[
                    "candidate_response_overlap_by_family"
                ].items()
            }
        ),
        flush=True,
    )
    print("", flush=True)
    for path in (d3c.DATASET_PATH, d3c.PROMPT_MANIFEST_PATH, d3c.DATASET_MANIFEST_PATH):
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
