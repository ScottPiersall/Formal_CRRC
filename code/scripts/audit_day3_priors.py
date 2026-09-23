#!/usr/bin/env python
"""Phase A: fingerprint the completed Day-1 and Day-2 experiments.

Read-only with respect to every prior file. Writes only
``artifacts/day3/prior_experiments_manifest.json``, which is the reference the
final Day-3 integrity check re-verifies against.

Also confirms, as a separate check, that no source file either prior experiment
pinned by hash has been modified — Day 3 extends shared modules only by adding
new ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formalcrrc import dataset as ds  # noqa: E402
from formalcrrc import day3, day3_config as d3c, prompts, provenance  # noqa: E402


def check_prior_source_files(root: Path) -> dict:
    """Every source hash a prior integrity manifest pinned must still hold.

    One documented exception exists and predates Day 3. Day 2 extended
    ``src/formalcrrc/dataset.py`` backward-compatibly to accept a seed and a
    partition tag, recording the change in its own preregistration under
    ``DAY1_SOURCE_EXTENSIONS``. That file therefore no longer matches the hash
    Day 1 recorded, while matching the one Day 2 recorded, and Day 1's dataset
    still reproduces its frozen content hash.

    A change is classified EXPLAINED only if it is on that documented list *and*
    agrees with the later manifest. Anything else is UNEXPLAINED, and Day 3 must
    add nothing to either category.
    """
    from formalcrrc import day2_config

    documented = set(day2_config.DAY1_SOURCE_EXTENSIONS)
    day2_manifest = json.loads(
        (root / "artifacts/day2/integrity_manifest.json").read_text(encoding="utf-8")
    )
    day2_recorded = day2_manifest["source_manifest"]["files"]

    result: dict = {}
    for day, relative in (
        ("day1", "artifacts/day1/integrity_manifest.json"),
        ("day2", "artifacts/day2/integrity_manifest.json"),
    ):
        manifest = json.loads((root / relative).read_text(encoding="utf-8"))
        recorded = manifest["source_manifest"]["files"]
        explained, unexplained = [], []
        for path, digest in recorded.items():
            if digest is None:
                continue
            candidate = root / path
            if not candidate.is_file():
                unexplained.append(path)
                continue
            current = provenance.sha256_file(candidate)
            if current == digest:
                continue
            if (
                day == "day1"
                and path in documented
                and day2_recorded.get(path) == current
            ):
                explained.append(path)
            else:
                unexplained.append(path)
        result[day] = {
            "n_files": len(recorded),
            "explained_by_documented_day2_extension": explained,
            "unexplained_changes": unexplained,
            "status": "PASS" if not unexplained else "FAIL",
            "recorded_tree_sha256": manifest["source_manifest"]["tree_sha256"],
        }
    result["documented_day2_extensions"] = sorted(documented)
    result["day3_added_no_extension"] = (
        set(result["day1"]["explained_by_documented_day2_extension"]) <= documented
        and not result["day1"]["unexplained_changes"]
        and not result["day2"]["unexplained_changes"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(provenance.repo_root()))
    args = parser.parse_args()
    root = Path(args.root)

    manifest_path = root / d3c.PRIOR_MANIFEST_PATH
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"prior manifest already exists ({manifest['n_files']} files)", flush=True)
    else:
        manifest = day3.build_prior_manifest(root)
        provenance.write_json(manifest_path, manifest)
        print(
            f"wrote {d3c.PRIOR_MANIFEST_PATH}  ({manifest['n_files']} files: "
            f"{manifest['by_experiment']['day1']['n_files']} Day-1, "
            f"{manifest['by_experiment']['day2']['n_files']} Day-2)",
            flush=True,
        )

    verification = day3.verify_prior_manifest(manifest, root)
    print("", flush=True)
    print(f"Day-1 files: {verification['day1']['status']} "
          f"(changed={verification['day1']['changed'] or 'none'})", flush=True)
    print(f"Day-2 files: {verification['day2']['status']} "
          f"(changed={verification['day2']['changed'] or 'none'})", flush=True)
    print(f"added: {verification['added'] or 'none'}", flush=True)

    sources = check_prior_source_files(root)
    print("", flush=True)
    for day in ("day1", "day2"):
        block = sources[day]
        print(
            f"{day} pinned source files: {block['status']} "
            f"({block['n_files']} files)",
            flush=True,
        )
        if block["explained_by_documented_day2_extension"]:
            print(
                "    explained by the documented Day-2 extension: "
                f"{block['explained_by_documented_day2_extension']}",
                flush=True,
            )
        if block["unexplained_changes"]:
            print(f"    UNEXPLAINED: {block['unexplained_changes']}", flush=True)
    print(f"Day 3 added no source extension: {sources['day3_added_no_extension']}",
          flush=True)

    day1_hash = ds.dataset_content_hash(ds.build_dataset())
    reproduces = day1_hash == d3c.DAY1_DATASET_CONTENT_SHA256
    print("", flush=True)
    print(f"Day-1 dataset reproduces: {reproduces} ({day1_hash[:16]})", flush=True)
    print(
        f"prompt template: {prompts.prompt_template_sha256()[:16]}",
        flush=True,
    )

    ok = (
        verification["status"] == "PASS"
        and sources["day1"]["status"] == "PASS"
        and sources["day2"]["status"] == "PASS"
        and sources["day3_added_no_extension"]
        and reproduces
    )
    print("", flush=True)
    print(f"PRIOR_EXPERIMENT_IMMUTABILITY = {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
