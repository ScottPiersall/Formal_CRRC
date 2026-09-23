"""Supplementary fixed-line-order A/B mapping experiment; no frozen-file writes.

Semantic scores always mean met / not-met, even when their physical tokens are
B / A. Historical scores are references until a GPU replay authorises reuse.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from formalcrrc import baselines, day3, day3_bootstrap, day3_config as d3c
from formalcrrc import prompts, provenance, scoring
from formalcrrc.config import FAMILIES, MODEL_IDS, MODEL_SLUGS
from formalcrrc.day2_bootstrap import crossing_index

SEED = 20260909
REPLICATES = 5000
DEFAULT_EXPERIMENT = "ab_mapping_20260909"
CONDITIONS = ("original", "swapped")
MAPPINGS = {"original": ("A", "B"), "swapped": ("B", "A")}
ORIGINAL_SUFFIX = "Return A if the criterion is met.\nReturn B if the criterion is not met."
SWAPPED_SUFFIX = "Return B if the criterion is met.\nReturn A if the criterion is not met."
META = ["artifact_id", "family", "latent_level", "strictness_index",
        "formal_truth", "true_first_fail_index"]
SCORE_FIELDS = ["raw_score_met", "raw_score_not_met", "margin", "p_met",
                "scoring_method", "n_prompt_tokens", "rendered_prompt_sha256"]


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    """Write actual LF bytes, including on Windows; never rewrite historical hashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb" if exclusive else "wb") as handle:
        handle.write(provenance.canonical_json(value).encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_path(root: Path, experiment_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_id):
        raise ValueError("experiment_id must be a single safe directory name")
    return root / "artifacts" / "label_swap" / experiment_id


def replace_suffix(message: str, condition: str) -> str:
    if condition not in CONDITIONS or not message.endswith(ORIGINAL_SUFFIX):
        raise ValueError("unknown condition or unexpected historical prompt suffix")
    suffix = ORIGINAL_SUFFIX if condition == "original" else SWAPPED_SUFFIX
    return message[:-len(ORIGINAL_SUFFIX)] + suffix


def resolve_labels(tokenizer: Any, rendered_reference: str,
                   label_met: str, label_not_met: str) -> scoring.LabelTokenization:
    if {label_met, label_not_met} != {"A", "B"}:
        raise ValueError("expected an explicit bijective A/B mapping")
    met = scoring.label_continuation_ids(tokenizer, rendered_reference, label_met)
    other = scoring.label_continuation_ids(tokenizer, rendered_reference, label_not_met)
    return scoring.LabelTokenization(
        label_met, label_not_met, met, other, len(met) == len(other) == 1,
        tuple(tokenizer.encode(label_met, add_special_tokens=False)),
        tuple(tokenizer.encode(label_not_met, add_special_tokens=False)))


def semantic_scores(result: dict, condition: str) -> dict:
    """score_prompt has ALREADY aligned semantics. Do not negate its margin."""
    met, other = MAPPINGS[condition]
    return {**result, "label_met": met, "label_not_met": other,
            f"raw_score_{met}": result["raw_score_met"],
            f"raw_score_{other}": result["raw_score_not_met"],
            "margin": result["raw_score_met"] - result["raw_score_not_met"]}


def file_hashes(root: Path, paths) -> dict[str, str]:
    return {str(p): provenance.sha256_file(root / p) for p in sorted(paths)}


def historical_snapshot(root: Path) -> dict[str, str]:
    paths = [str(p.relative_to(root)).replace("\\", "/")
             for day in (1, 2, 3) for p in (root / f"artifacts/day{day}").rglob("*")
             if p.is_file()]
    return file_hashes(root, paths)


def assert_hashes(root: Path, hashes: dict[str, str]) -> None:
    changed = [p for p, h in hashes.items()
               if not (root / p).is_file() or provenance.sha256_file(root / p) != h]
    if changed:
        raise ValueError(f"frozen input bytes changed: {changed}")


def freeze_lf_profile(root: Path, out: Path, lf_root: Path) -> None:
    """Add a separately frozen portability record; never change selection/config.

    Windows working bytes and independent LF checkout bytes remain distinct.
    This additive profile permits the same frozen manifests on the Linux cluster.
    """
    path = out / "HISTORICAL_LF_PROFILE.json"
    if path.exists():
        verify_input_profile(root, out, read_json(out / "configuration.json"))
        return
    audit_history(root, lf_root)
    working = read_json(out / "configuration.json")["historical_input_hashes"]
    hashes = file_hashes(lf_root, working)
    different = []
    for name, digest in hashes.items():
        if digest != working[name]:
            a, b = (root / name).read_bytes(), (lf_root / name).read_bytes()
            if a.replace(b"\r\n", b"\n") != b or b"\r\n" not in a:
                raise ValueError(f"LF checkout differs beyond CRLF: {name}")
            different.append(name)
    write_json(path, {"frozen_manifest_sha256": provenance.sha256_file(out / "FROZEN.json"),
        "historical_input_hashes": hashes, "different_working_byte_paths": different,
        "created_at": provenance.utc_now(), "note": "Separate raw bytes from an independent LF checkout; original Windows hashes are retained."}, exclusive=True)
    (out / "HISTORICAL_LF_PROFILE.sha256").write_bytes((provenance.sha256_file(path) + "\n").encode())


def verify_input_profile(root: Path, out: Path, config: dict) -> str:
    path = out / "HISTORICAL_LF_PROFILE.json"
    profile = None
    if path.exists():
        if provenance.sha256_file(path) != (out / "HISTORICAL_LF_PROFILE.sha256").read_text().strip():
            raise ValueError("LF profile checksum mismatch")
        profile = read_json(path)
        if profile["frozen_manifest_sha256"] != provenance.sha256_file(out / "FROZEN.json"):
            raise ValueError("LF profile belongs to a different experiment freeze")
    try:
        assert_hashes(root, config["historical_input_hashes"])
        return "original_working_raw_bytes"
    except ValueError:
        if profile is None:
            raise
        assert_hashes(root, profile["historical_input_hashes"])
        return "independently_verified_lf_raw_bytes"


def audit_history(root: Path, lf_root: Path | None) -> dict:
    """Distinguish working bytes from independently checked LF historical bytes."""
    dataset_meta = read_json(root / d3c.DATASET_MANIFEST_PATH)
    for path, key in ((d3c.DATASET_PATH, "dataset_file_sha256"),
                      (d3c.PROMPT_MANIFEST_PATH, "prompt_manifest_file_sha256")):
        if provenance.sha256_file(root / path) != dataset_meta["hashes"][key]:
            raise ValueError(f"historical binary input hash mismatch: {path}")
    if dataset_meta["hashes"]["prompt_template_sha256"] != prompts.prompt_template_sha256():
        raise ValueError("frozen prompt template changed")
    expected = (root / d3c.PREREG_SHA_PATH).read_text().split()[0]
    current = provenance.sha256_file(root / d3c.PREREG_JSON_PATH)
    verified_root = root
    if current != expected:
        if lf_root is None or lf_root.resolve() == root.resolve():
            raise ValueError("preregistration byte mismatch: supply independent --lf-root")
        if provenance.sha256_file(lf_root / d3c.PREREG_JSON_PATH) != expected:
            raise ValueError("independent LF checkout does not match historical checksum")
        # This is a separate equality, never called raw-byte equality.
        if (root / d3c.PREREG_JSON_PATH).read_bytes().replace(b"\r\n", b"\n") != (
                lf_root / d3c.PREREG_JSON_PATH).read_bytes():
            raise ValueError("preregistration differs beyond line endings")
        verified_root = lf_root
    sources = {}
    raw = {}
    for model in MODEL_IDS:
        slug = MODEL_SLUGS[model]
        record = read_json(root / d3c.RAW_SCORES_DIR / f"{slug}_run.json")
        path = root / d3c.RAW_SCORES_DIR / f"{slug}.parquet"
        raw[model] = provenance.sha256_file(path) == record["raw_scores_sha256"]
        for name, digest in record["source_manifest"]["files"].items():
            actual = provenance.sha256_file(root / name)
            verified = provenance.sha256_file(verified_root / name)
            if verified != digest:
                raise ValueError(f"historical scoring source mismatch in verified checkout: {name}")
            if (root / name).read_bytes().replace(b"\r\n", b"\n") != (
                    verified_root / name).read_bytes().replace(b"\r\n", b"\n"):
                raise ValueError(f"working scoring source content changed: {name}")
            sources[name] = {"working_raw_sha256": actual, "historical_sha256": digest,
                             "verified_checkout_raw_sha256": verified,
                             "working_raw_equal": actual == digest}
    return {"preregistration": {"working_raw_sha256": current, "historical_sha256": expected,
            "working_raw_equal": current == expected, "independent_lf_verified": current != expected},
            "sources": sources, "raw_scores_match_run_hash": raw,
            "note": "LF checkout bytes independently verified; normalization is not raw-byte equality."}


def build_manifests(dataset: pd.DataFrame, historical: pd.DataFrame,
                    artifact_ids: list[str], experiment_id: str) -> dict[str, pd.DataFrame]:
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("duplicate selected artifact")
    selected = dataset[dataset.artifact_id.isin(artifact_ids)].copy()
    saved = historical[historical.artifact_id.isin(artifact_ids)].copy()
    if selected.prompt_id.duplicated().any() or saved.prompt_id.duplicated().any():
        raise ValueError("duplicate source prompt")
    if set(selected.prompt_id) != set(saved.prompt_id):
        raise ValueError("dataset and saved manifest prompt sets differ")
    selected = selected.set_index("prompt_id").sort_index()
    saved = saved.set_index("prompt_id").sort_index()
    for field in META[:4]:
        if not selected[field].equals(saved[field]):
            raise ValueError(f"source metadata mismatch: {field}")
    for field, text in (("rubric_sha256", "rubric_text"), ("candidate_sha256", "candidate_response")):
        if not (selected[text].map(scoring.sha256_text) == selected[field]).all():
            raise ValueError(f"source text hash mismatch: {text}")
    if not (saved.user_message.map(scoring.sha256_text) == saved.user_message_sha256).all():
        raise ValueError("historical user message hash mismatch")
    expected_text = [prompts.build_user_message(r, c) for r, c in
                     zip(selected.rubric_text, selected.candidate_response, strict=True)]
    if expected_text != saved.user_message.tolist():
        raise ValueError("historical prompt differs from dataset text and frozen template")
    base = selected[META + ["rubric_text", "candidate_response", "rubric_sha256", "candidate_sha256"]].copy()
    base["original_prompt_id"] = base.index
    base["historical_order_index"] = saved.order_index
    base["historical_user_message_sha256"] = saved.user_message_sha256
    base["user_message"] = saved.user_message
    base = base.sort_values("historical_order_index").reset_index(drop=True)
    base["order_index"] = np.arange(len(base))
    result = {}
    for condition in CONDITIONS:
        frame = base.copy()
        frame["condition"] = condition
        frame["prompt_id"] = experiment_id + "|" + condition + "|" + frame.original_prompt_id
        frame["user_message"] = frame.user_message.map(lambda s: replace_suffix(s, condition))
        frame["user_message_sha256"] = frame.user_message.map(scoring.sha256_text)
        frame["label_met"], frame["label_not_met"] = MAPPINGS[condition]
        validate_manifest(frame, condition)
        result[condition] = frame
    return result


def validate_manifest(frame: pd.DataFrame, condition: str) -> None:
    if len(frame) != 729 or frame.artifact_id.nunique() != 81:
        raise ValueError("expected 729 prompts / 81 artifacts")
    if frame.original_prompt_id.duplicated().any() or frame.prompt_id.duplicated().any():
        raise ValueError("duplicate manifest prompt")
    if set(frame.condition) != {condition}:
        raise ValueError("mixed conditions")
    if sorted(frame.order_index) != list(range(729)):
        raise ValueError("invalid fixed order")
    for _, group in frame.groupby("artifact_id"):
        if sorted(group.strictness_index) != list(range(9)):
            raise ValueError("expected strictness 0..8 exactly once")
        for field in ("family", "latent_level", "true_first_fail_index", "candidate_response"):
            if group[field].nunique() != 1:
                raise ValueError(f"inconsistent artifact {field}")
    cells = frame.drop_duplicates("artifact_id").groupby(["family", "latent_level"]).size()
    if len(cells) != 27 or not (cells == 3).all() or set(frame.family) != set(FAMILIES):
        raise ValueError("expected 3 artifacts in each of 27 strata")
    if set(frame.latent_level) != set(range(9)) or not frame.true_first_fail_index.between(1, 9).all():
        raise ValueError("invalid latent level or boundary support")
    if not np.array_equal(frame.formal_truth, (frame.strictness_index < frame.true_first_fail_index).astype(int)):
        raise ValueError("truth / boundary mismatch")
    if not (frame.user_message.map(scoring.sha256_text) == frame.user_message_sha256).all():
        raise ValueError("manifest text hash mismatch")


def prepare(root: Path, experiment_id: str, lf_root: Path | None = None) -> Path:
    out = experiment_path(root, experiment_id)
    if (out / "FROZEN.json").exists():
        load_frozen(root, experiment_id)
        if lf_root is not None:
            freeze_lf_profile(root, out, lf_root)
        return out  # Never redraw a frozen selection.
    if out.exists():
        raise FileExistsError(f"incomplete preparation exists; inspect it: {out}")
    audit = audit_history(root, lf_root)
    snapshot = historical_snapshot(root)
    dataset = pd.read_parquet(root / d3c.DATASET_PATH)
    saved = pd.read_parquet(root / d3c.PROMPT_MANIFEST_PATH)
    artifacts = dataset[["artifact_id", "family", "latent_level"]].drop_duplicates().sort_values("artifact_id")
    if artifacts.artifact_id.duplicated().any():
        raise ValueError("source artifact metadata conflict")
    rng = np.random.default_rng(SEED)
    chosen = []
    for _, cell in artifacts.groupby(["family", "latent_level"], sort=True):
        chosen.extend(rng.choice(cell.artifact_id.to_numpy(), size=3, replace=False).tolist())
    chosen = sorted(chosen)
    manifests = build_manifests(dataset, saved, chosen, experiment_id)
    original = manifests["original"]
    replay = original.sort_values(["artifact_id", "strictness_index"])
    replay = replay[replay.strictness_index == replay.latent_level].groupby(
        ["family", "latent_level"], sort=True).head(1).original_prompt_id.tolist()
    out.mkdir(parents=True, exist_ok=False)
    configuration = {"experiment_id": experiment_id, "supplementary_not_original_preregistration": True,
        "selection_seed": SEED, "bootstrap_seed": SEED, "bootstrap_replicates": REPLICATES,
        "bootstrap": "paired artifact clusters, family-stratified, percentile 95% CI; actual-data points",
        "selection_method": "sort artifact_id; lexicographic family/latent strata; one numpy PCG64 stream; choice(3, replace=False); sort selected IDs",
        "selected_artifact_ids": chosen, "replay_prompt_ids": replay,
        "mappings": MAPPINGS, "condition_order": list(CONDITIONS),
        "prompt_order": "historical Day-3 order restricted to selection, same for both conditions",
        "replay_policy": "reuse only if provenance/environment match and all 27 fixed original probes are exactly equal in both raw scores, margin and p_met, with zero decision changes; otherwise rerun all original before swapped",
        "replay_absolute_tolerance": 0.0,
        "inference": {"dtype": "bfloat16", "batch_size": 1, "eval": True, "sampling": False,
                      "quantization": None, "local_files_only": True, "add_generation_prompt": True,
                      "add_special_tokens": False},
        "models": read_json(root / d3c.MODEL_PROVENANCE_PATH)["models"],
        "history_audit": audit, "historical_input_hashes": snapshot,
        "created_at": provenance.utc_now(), "preparation_environment": provenance.environment_snapshot(False)}
    write_json(out / "configuration.json", configuration, exclusive=True)
    pd.DataFrame({"artifact_id": chosen}).to_csv(out / "selected_artifact_ids.csv", index=False)
    for condition, frame in manifests.items():
        frame.to_parquet(out / f"manifest_{condition}.parquet", index=False)
    frozen_names = ["configuration.json", "selected_artifact_ids.csv", *[f"manifest_{c}.parquet" for c in CONDITIONS]]
    assert_hashes(root, snapshot)
    write_json(out / "FROZEN.json", {"status": "frozen_before_scoring", "files": file_hashes(out, frozen_names)}, exclusive=True)
    if lf_root is not None:
        freeze_lf_profile(root, out, lf_root)
    return out


def load_frozen(root: Path, experiment_id: str) -> tuple[Path, dict, dict[str, pd.DataFrame]]:
    out = experiment_path(root, experiment_id)
    assert_hashes(out, read_json(out / "FROZEN.json")["files"])
    config = read_json(out / "configuration.json")
    verify_input_profile(root, out, config)
    for name, audit in config["history_audit"]["sources"].items():
        if provenance.sha256_file(root / name) not in {audit["working_raw_sha256"], audit["historical_sha256"]}:
            raise ValueError(f"frozen scoring source changed: {name}")
    frames = {c: pd.read_parquet(out / f"manifest_{c}.parquet") for c in CONDITIONS}
    for condition, frame in frames.items():
        validate_manifest(frame, condition)
    a, b = (frames[c].sort_values("original_prompt_id").reset_index(drop=True) for c in CONDITIONS)
    for field in META + ["original_prompt_id", "rubric_text", "candidate_response", "order_index"]:
        if not a[field].equals(b[field]):
            raise ValueError(f"condition manifest mismatch: {field}")
    if a.user_message.map(lambda s: replace_suffix(s, "swapped")).tolist() != b.user_message.tolist():
        raise ValueError("condition changes outside fixed suffix")
    if not (a.user_message_sha256 == a.historical_user_message_sha256).all():
        raise ValueError("original prompt changed")
    return out, config, frames


def check_tokenizer(tokenizer: Any, frames: dict[str, pd.DataFrame], record: dict) -> tuple[dict, dict]:
    """Check every selected real prompt, independently for both conditions, CPU only."""
    tokens, checks = {}, {}
    chat_hash = scoring.sha256_text(tokenizer.chat_template or "")
    if chat_hash != record["chat_template_sha256"]:
        raise ValueError("chat-template hash differs from Day-3 provenance")
    for condition in CONDITIONS:
        reference = scoring.render_chat_prompt(tokenizer, replace_suffix(prompts.label_reference_message(), condition))
        tokenization = resolve_labels(tokenizer, reference, *MAPPINGS[condition])
        details = []
        for entry in frames[condition].itertuples(index=False):
            rendered = scoring.render_chat_prompt(tokenizer, entry.user_message)
            actual = resolve_labels(tokenizer, rendered, *MAPPINGS[condition])
            if actual != tokenization:
                raise ValueError(f"unstable continuation: {condition}/{entry.original_prompt_id}")
            details.append({"original_prompt_id": entry.original_prompt_id,
                            "rendered_prompt_sha256": scoring.sha256_text(rendered)})
        tokens[condition] = tokenization
        checks[condition] = {"tokenization": tokenization.to_dict(),
                             "reference_rendered_sha256": scoring.sha256_text(reference),
                             "real_prompts_checked": len(details), "prompts": details}
    # Mapping reversal is checked against provenance, never against hardcoded IDs.
    if tokens["original"].to_dict() != record["label_tokenization"]:
        raise ValueError("original continuation differs from Day-3 provenance")
    if (tokens["swapped"].met_token_ids != tokens["original"].not_met_token_ids or
            tokens["swapped"].not_met_token_ids != tokens["original"].met_token_ids):
        raise ValueError("swapped continuation differs from physical Day-3 labels")
    return tokens, {"chat_template_sha256": chat_hash, "conditions": checks}


def validate_scores(frame: pd.DataFrame, manifest: pd.DataFrame, model_id: str, condition: str) -> pd.DataFrame:
    """Exact-set one-to-one validation before any metrics or pairing."""
    if len(frame) != len(manifest) or len(frame) != 729:
        raise ValueError("incomplete panel: expected 729 rows")
    if set(frame.model_id) != {model_id} or set(frame.condition) != {condition}:
        raise ValueError("mixed models or conditions")
    keys = ["model_id", "original_prompt_id"]
    if frame.duplicated(keys).any() or manifest.original_prompt_id.duplicated().any():
        raise ValueError("duplicate pairing key")
    if set(frame.original_prompt_id) != set(manifest.original_prompt_id):
        raise ValueError("missing or unexpected original_prompt_id")
    a = frame.set_index("original_prompt_id").sort_index()
    b = manifest.set_index("original_prompt_id").sort_index()
    for field in META + ["user_message_sha256", "order_index"]:
        if a[field].tolist() != b[field].tolist():
            raise ValueError(f"score / manifest metadata mismatch: {field}")
    for field, label in zip(("label_met", "label_not_met"), MAPPINGS[condition]):
        if set(frame[field]) != {label}:
            raise ValueError(f"incorrect semantic mapping: {field}")
    numbers = frame[["raw_score_met", "raw_score_not_met", "raw_score_A", "raw_score_B", "margin", "p_met"]].to_numpy(float)
    if not np.isfinite(numbers).all():
        raise ValueError("non-finite scores")
    if not np.array_equal(frame.margin, frame.raw_score_met - frame.raw_score_not_met):
        raise ValueError("semantic margin is not met minus not-met")
    met, other = MAPPINGS[condition]
    if not np.array_equal(frame.raw_score_met, frame[f"raw_score_{met}"]) or not np.array_equal(frame.raw_score_not_met, frame[f"raw_score_{other}"]):
        raise ValueError("physical / semantic score mismatch")
    probabilities = [scoring.normalized_probability(m, n) for m, n in zip(frame.raw_score_met, frame.raw_score_not_met)]
    if not np.array_equal(probabilities, frame.p_met):
        raise ValueError("p_met mismatch")
    day3.stack_margins(frame)  # A single validated model/condition only.
    return a.reset_index()


def paired_frames(original: pd.DataFrame, swapped: pd.DataFrame, frames: dict, model_id: str) -> tuple:
    a = validate_scores(original, frames["original"], model_id, "original")
    b = validate_scores(swapped, frames["swapped"], model_id, "swapped")
    for field in ["model_id", "original_prompt_id", *META]:
        if a[field].tolist() != b[field].tolist():
            raise ValueError(f"paired metadata conflict: {field}")
    return a, b


def artifact_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.model_id.nunique() != 1 or frame.condition.nunique() != 1:
        raise ValueError("stack_margins requires exactly one model and condition")
    ids, margins, truth, families = day3.stack_margins(frame)
    if not np.isfinite(margins).all():
        raise ValueError("non-finite margins")
    return pd.DataFrame({"artifact_id": ids, "family": families, "j_star": truth,
        "model_id": frame.model_id.iloc[0], "condition": frame.condition.iloc[0],
        "tce": np.abs(crossing_index(margins) - truth),
        "otce": [day3.oracle_translation_tce(m, j) for m, j in zip(margins, truth)],
        "reachable": [day3.translation_reachable(m, j) for m, j in zip(margins, truth)],
        "nontrivial": (truth >= 1) & (truth <= 8)})


def paired_statistics(a: pd.DataFrame, b: pd.DataFrame, *, replicates: int = REPLICATES, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute points on actual data and percentile CIs on paired resamples."""
    am, bm = artifact_metrics(a), artifact_metrics(b)
    if not am[["artifact_id", "family", "j_star"]].equals(bm[["artifact_id", "family", "j_star"]]):
        raise ValueError("artifact metric mismatch")
    plan = day3_bootstrap.build_plan(am.artifact_id, am.family, replicates=replicates, seed=seed)
    aa = a.sort_values(["artifact_id", "strictness_index"])
    bb = b.sort_values(["artifact_id", "strictness_index"])
    if aa.original_prompt_id.tolist() != bb.original_prompt_id.tolist():
        raise ValueError("disagreement pairing mismatch")
    disagreement = ((aa.margin.to_numpy() >= 0) != (bb.margin.to_numpy() >= 0)).reshape(-1, 9).mean(axis=1)
    table, samples = [], []
    for family in (None, *FAMILIES):
        positions, columns = plan.positions(family), plan.columns(family)
        drawn = plan.indices[:, columns]
        scope = family or "overall"
        nt = am.nontrivial.to_numpy()
        for metric in ("tce", "trr", "otce", "semantic_disagreement"):
            if metric == "semantic_disagreement":
                values = {"paired": disagreement}
            elif metric == "trr":
                values = {"original": am.reachable.to_numpy(float) * 100,
                          "swapped": bm.reachable.to_numpy(float) * 100}
            else:
                values = {"original": am[metric].to_numpy(float), "swapped": bm[metric].to_numpy(float)}
            means, boots = {}, {}
            for condition, value in values.items():
                if metric == "trr":
                    mask = nt[positions]
                    means[condition] = value[positions][mask].mean() if mask.any() else np.nan
                    denom = nt[drawn].sum(axis=1)
                    boots[condition] = np.divide((value[drawn] * nt[drawn]).sum(axis=1), denom,
                        out=np.full(replicates, np.nan), where=denom > 0)
                else:
                    means[condition] = value[positions].mean()
                    boots[condition] = value[drawn].mean(axis=1)
            if metric != "semantic_disagreement":
                means["swapped-original"] = means["swapped"] - means["original"]
                boots["swapped-original"] = boots["swapped"] - boots["original"]
            for condition in means:
                finite = boots[condition][np.isfinite(boots[condition])]
                low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
                table.append({"model_id": a.model_id.iloc[0], "scope": scope, "metric": metric,
                    "condition": condition, "point": means[condition], "ci_low": low, "ci_high": high,
                    "n_artifacts": len(positions), "n_nontrivial": int(nt[positions].sum()),
                    "bootstrap_valid_replicates": len(finite),
                    "unit": "percentage_points" if metric == "trr" else "proportion" if metric == "semantic_disagreement" else "strictness_steps"})
                samples.extend({"model_id": a.model_id.iloc[0], "scope": scope, "metric": metric,
                    "condition": condition, "replicate": r, "value": v} for r, v in enumerate(boots[condition]))
    return pd.DataFrame(table), pd.DataFrame(samples)


def baseline_table(manifest: pd.DataFrame) -> pd.DataFrame:
    artifacts = manifest.drop_duplicates("artifact_id")
    rows = []
    for family in (None, *FAMILIES):
        selected = artifacts if family is None else artifacts[artifacts.family == family]
        boundaries = selected.true_first_fail_index.tolist()
        if set(boundaries) != set(baselines.BOUNDARY_SUPPORT):
            raise ValueError("unexpected selected boundary support")
        counts = {j: boundaries.count(j) for j in range(1, 10)}
        for name, value, support in (
            ("constant_5", baselines.constant_crossing_tce(5, boundaries), [5]),
            ("uniform_crossing_1_9", baselines.uniform_random_crossing_tce(boundaries, range(1, 10)), list(range(1, 10))),
            ("uniform_crossing_0_9", baselines.uniform_random_crossing_tce(boundaries, range(10)), list(range(10)))):
            rows.append({"scope": family or "overall", "baseline": name, "tce": float(value),
                         "tce_exact": str(value), "prediction_support": json.dumps(support),
                         "boundary_counts": json.dumps(counts), "n_artifacts": len(selected)})
    return pd.DataFrame(rows)


def reserve_run(out: Path, slug: str, *, debug_id: str | None = None) -> Path:
    if debug_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", debug_id):
        raise ValueError("invalid debug id")
    directory = out / "runs" / slug if debug_id is None else out / "debug" / debug_id / slug
    directory.mkdir(parents=True, exist_ok=False)  # Reject completed AND partial runs, atomically.
    write_json(directory / "INCOMPLETE.json", {"status": "incomplete", "started_at": provenance.utc_now()}, exclusive=True)
    return directory
