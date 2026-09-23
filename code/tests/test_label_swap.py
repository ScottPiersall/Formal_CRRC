"""Boundary tests for the additive label-swap integration; no model downloads."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from formalcrrc import label_swap as ls, scoring, day3, provenance


@pytest.fixture(scope="module")
def manifests():
    root = Path(__file__).resolve().parents[1]
    data = pd.read_parquet(root / ls.d3c.DATASET_PATH)
    saved = pd.read_parquet(root / ls.d3c.PROMPT_MANIFEST_PATH)
    artifacts = data.drop_duplicates("artifact_id").sort_values("artifact_id")
    ids = artifacts.groupby(["family", "latent_level"], sort=True).head(3).artifact_id.tolist()
    return ls.build_manifests(data, saved, ids, "test")


def fake_scores(manifest, margins=None):
    frame = manifest.copy()
    frame["model_id"] = "test-model"
    m = np.asarray(margins) if margins is not None else (frame.true_first_fail_index - frame.strictness_index - 0.5).to_numpy(float)
    frame["raw_score_met"] = m
    frame["raw_score_not_met"] = 0.0
    frame["margin"] = m
    frame["p_met"] = [scoring.normalized_probability(x, 0) for x in m]
    met, other = ls.MAPPINGS[frame.condition.iloc[0]]
    frame[f"raw_score_{met}"] = m
    frame[f"raw_score_{other}"] = 0.0
    return frame


def test_only_suffix_changes_in_real_prompts(manifests):
    a, b = manifests.values()
    assert len(a) == len(b) == 729
    for x, y in zip(a.user_message, b.user_message):
        assert x.encode()[:-len(ls.ORIGINAL_SUFFIX)] == y.encode()[:-len(ls.SWAPPED_SUFFIX)]
        assert y.endswith(ls.SWAPPED_SUFFIX)
    for field in ls.META + ["rubric_text", "candidate_response", "order_index"]:
        assert a[field].equals(b[field])
    assert (a.user_message_sha256 == a.historical_user_message_sha256).all()


@pytest.mark.parametrize("ending", ["\n", " ", ".", ""])
def test_suffix_validation_is_exact(ending):
    text = "family A, artifact B, MK-AB\n" + ls.ORIGINAL_SUFFIX
    if ending:
        with pytest.raises(ValueError, match="suffix"):
            ls.replace_suffix(text + ending, "swapped")
    else:
        assert ls.replace_suffix(text, "swapped").startswith("family A, artifact B, MK-AB\n")


class Tokenizer:
    def __init__(self, multi=False):
        self.multi = multi

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        if text in ("A", "B"):
            return [100 if text == "A" else 101]  # Naive IDs intentionally wrong.
        ids = [7, 8, 9]
        if text.endswith("A"):
            return ids + [2]
        if text.endswith("B"):
            return ids + ([3, 4] if self.multi else [3])
        return ids


@pytest.mark.parametrize("condition,expected", [("original", 5.0), ("swapped", -5.0)])
def test_same_physical_logits_get_correct_semantics_without_second_flip(monkeypatch, condition, expected):
    tokens = ls.resolve_labels(Tokenizer(), "rendered>", *ls.MAPPINGS[condition])
    logits = {2: 7.0, 3: 2.0}
    def score(model, tokenizer, prompt, tokenization):
        return logits[tokenization.met_token_ids[0]], logits[tokenization.not_met_token_ids[0]], 3
    monkeypatch.setattr(scoring, "score_single_token", score)
    result = ls.semantic_scores(scoring.score_prompt(None, None, "rendered>", tokens), condition)
    assert result["margin"] == expected
    assert result["raw_score_A"] == 7
    assert result["raw_score_B"] == 2
    assert (result["p_met"] >= 0.5) == (expected >= 0)
    if condition == "swapped":
        assert tokens.label_met == "B" and tokens.met_token_ids == (3,)
        assert tokens.naive_met_token_ids == (101,)


def test_fallback_keeps_sequence_loglikelihood_name(monkeypatch):
    tokens = ls.resolve_labels(Tokenizer(multi=True), "rendered>", "B", "A")
    monkeypatch.setattr(scoring, "score_sequence_loglikelihood", lambda *args: (-3.0, -5.0, 3))
    result = ls.semantic_scores(scoring.score_prompt(None, None, "rendered>", tokens), "swapped")
    assert result["scoring_method"] == "sequence_loglikelihood"
    assert result["margin"] == 2 and result["raw_score_B"] == -3


def test_zero_all_pass_and_strict_equality_conventions(manifests):
    a = fake_scores(manifests["original"], np.zeros(729))
    metrics = ls.artifact_metrics(a)
    assert np.array_equal(metrics.tce, 9 - metrics.j_star)
    assert day3.reachable_crossings(np.zeros(9)) == (0, 9)
    assert day3.reachable_crossings([3, 3, 2, 2, 1, 1, 1, 1, 1]) == (0, 2, 4, 9)


@pytest.mark.parametrize("corrupt", ["missing", "duplicate", "artifact_id", "family", "latent_level", "strictness_index", "formal_truth", "true_first_fail_index", "prompt_key", "margin", "letter", "probability"])
def test_strict_pairing_rejects_bad_panels(manifests, corrupt):
    a, b = fake_scores(manifests["original"]), fake_scores(manifests["swapped"])
    if corrupt == "missing":
        b = b.iloc[:-1]
    elif corrupt == "duplicate":
        b = pd.concat([b.iloc[:-1], b.iloc[:1]], ignore_index=True)
    elif corrupt == "prompt_key":
        b.loc[0, "original_prompt_id"] = "unexpected"
    elif corrupt == "margin":
        b.loc[0, "margin"] *= -1
    elif corrupt == "letter":
        b.loc[0, "raw_score_A"] += 1
    elif corrupt == "probability":
        b.loc[0, "p_met"] = 0.25
    else:
        b.loc[0, corrupt] = "wrong" if corrupt in ("family", "artifact_id") else 99
    with pytest.raises(ValueError):
        ls.paired_frames(a, b, manifests, "test-model")


def test_unordered_rows_pair_by_original_prompt_key(manifests):
    a, b = (fake_scores(manifests[c]).sample(frac=1, random_state=i) for i, c in enumerate(ls.CONDITIONS))
    aa, bb = ls.paired_frames(a, b, manifests, "test-model")
    assert aa.original_prompt_id.tolist() == bb.original_prompt_id.tolist()


def test_conditions_never_reach_stack_margins_mixed(manifests, monkeypatch):
    combined = pd.concat([fake_scores(manifests[c]) for c in ls.CONDITIONS])
    def forbidden(*args, **kwargs):
        pytest.fail("mixed conditions reached historical stack_margins")
    monkeypatch.setattr(day3, "stack_margins", forbidden)
    with pytest.raises(ValueError, match="one model and condition"):
        ls.artifact_metrics(combined)


def test_formal_output_never_overwrites_and_debug_is_separate(tmp_path):
    formal = ls.reserve_run(tmp_path, "model")
    ls.write_json(formal / "COMPLETE.json", {"status": "complete"})
    before = (formal / "COMPLETE.json").read_bytes()
    with pytest.raises(FileExistsError):
        ls.reserve_run(tmp_path, "model")
    debug = ls.reserve_run(tmp_path, "model", debug_id="smoke")
    assert debug == tmp_path / "debug" / "smoke" / "model"
    assert formal == tmp_path / "runs" / "model"
    assert (formal / "COMPLETE.json").read_bytes() == before
    with pytest.raises(FileExistsError):
        ls.reserve_run(tmp_path, "model", debug_id="smoke")
    with pytest.raises(ValueError):
        ls.reserve_run(tmp_path, "model", debug_id="../../day3")


def test_paired_bootstrap_zero_difference_for_identical_curves(manifests):
    a, b = [fake_scores(manifests[c]) for c in ls.CONDITIONS]
    table, samples = ls.paired_statistics(a, b, replicates=5000)
    delta = table[table.condition == "swapped-original"]
    assert (delta[["point", "ci_low", "ci_high"]] == 0).all().all()
    disagreement = table[table.metric == "semantic_disagreement"]
    assert (disagreement[["point", "ci_low", "ci_high"]] == 0).all().all()
    assert len(samples) == 5000 * len(table)


def test_trr_bootstrap_recomputes_nontrivial_denominator_and_actual_point(manifests):
    a = fake_scores(manifests["original"])
    b = fake_scores(manifests["swapped"], np.ones(729))
    table, samples = ls.paired_statistics(a, b, replicates=97)
    trr = table[(table.metric == "trr") & (table.condition == "swapped-original")]
    assert np.array_equal(trr.point, [-100] * 4)
    assert (trr.ci_low == -100).all() and (trr.ci_high == -100).all()
    point = table.query("scope == 'overall' and metric == 'tce' and condition == 'swapped-original'").point.iloc[0]
    assert point == 4.0  # Actual mean j*=5; original exact, swapped sentinel 9.
    draws = samples.query("scope == 'overall' and metric == 'tce' and condition == 'swapped-original'").value
    assert draws.mean() != point  # Point must not be replaced with this random mean.


def test_baselines_use_observed_support(manifests):
    table = ls.baseline_table(manifests["original"])
    overall = table.query("scope == 'overall'").set_index("baseline")
    assert overall.loc["constant_5", "tce_exact"] == "20/9"
    assert overall.loc["uniform_crossing_1_9", "tce_exact"] == "80/27"
    assert overall.loc["uniform_crossing_0_9", "tce_exact"] == "19/6"


def test_prepare_freezes_reuses_and_does_not_touch_history(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    # Independent disposable LF fixture, without changing repository originals.
    for p in (root / "artifacts/day3").rglob("*"):
        if p.is_file():
            target = tmp_path / p.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes().replace(b"\r\n", b"\n") if p.suffix in (".json", ".sha256") else p.read_bytes())
    record = ls.read_json(root / "artifacts/day3/raw_scores/mistral_7b_instruct_v0_3_run.json")
    for name in record["source_manifest"]["files"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes().replace(b"\r\n", b"\n"))
    before = ls.historical_snapshot(tmp_path)
    actual_before = ls.historical_snapshot(root)
    out = ls.prepare(tmp_path, "experiment")
    frozen = ls.file_hashes(out, [p.name for p in out.iterdir()])
    def no_redraw(*args, **kwargs):
        pytest.fail("frozen selection was redrawn")
    monkeypatch.setattr(np.random, "default_rng", no_redraw)
    assert ls.prepare(tmp_path, "experiment") == out
    ls.assert_hashes(out, frozen)
    assert before == ls.historical_snapshot(tmp_path)
    assert actual_before == ls.historical_snapshot(root)
    config = ls.read_json(out / "configuration.json")
    assert len(config["selected_artifact_ids"]) == 81
    assert len(config["replay_prompt_ids"]) == 27
    # Changed frozen bytes fail closed; no silent regeneration.
    (out / "selected_artifact_ids.csv").write_bytes(b"corrupt\n")
    with pytest.raises(ValueError, match="frozen input bytes changed"):
        ls.prepare(tmp_path, "experiment")


def test_current_crlf_is_not_claimed_raw_equal(tmp_path):
    path = tmp_path / "record.json"
    ls.write_json(path, {"a": 1})
    assert b"\r\n" not in path.read_bytes()
    with pytest.raises(FileExistsError):
        ls.write_json(path, {}, exclusive=True)


def test_replay_detects_numeric_and_decision_changes():
    path = Path(__file__).resolve().parents[1] / "scripts/run_label_swap_inference.py"
    spec = importlib.util.spec_from_file_location("label_swap_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    historical = pd.DataFrame([{"original_prompt_id": "p", "raw_score_met": 1.0,
        "raw_score_not_met": 1.0, "margin": 0.0, "p_met": 0.5, "rendered_prompt_sha256": "hash"}])
    assert module.replay_comparison(historical.copy(), historical)[1]
    current = historical.copy()
    current.loc[0, "margin"] = -1e-12
    report, exact = module.replay_comparison(current, historical)
    assert not exact and report[0]["decision_changed"]
    assert report[0]["delta_margin"] == -1e-12


def test_bundle_preserves_inference_hashes_without_partial_checkpoints(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/reanalyze_label_swap_bundle.py"
    spec = importlib.util.spec_from_file_location("bundle_reanalysis_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = ["run.json", "original.parquet", "swapped.parquet", "historical_reuse_decision.json"]
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    hashes = ls.file_hashes(tmp_path, names)
    hashes["swapped.partial.parquet"] = "omitted redundant checkpoint"
    ls.write_json(tmp_path / "COMPLETE.json", {"status": "complete", "files": hashes})
    module.verify_completed_run(tmp_path)
    # A newly computed outer ZIP hash must not authorize changed inference bytes.
    (tmp_path / "swapped.parquet").write_bytes(b"changed after inference")
    with pytest.raises(ValueError, match="frozen input bytes changed"):
        module.verify_completed_run(tmp_path)


@pytest.mark.parametrize("replay_equal,fail_swap", [(True, False), (False, False), (True, True)])
def test_runner_gates_reuse_before_swap_and_validates_complete_panels(tmp_path, monkeypatch, replay_equal, fail_swap):
    """Execute the real orchestration with an in-memory model, including failure gates.

    This is a CPU control-flow test, never evidence of GPU reproducibility.
    """
    import sys
    import types
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("runner_flow_test", root / "scripts/run_label_swap_inference.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    model_id = ls.MODEL_IDS[0]
    slug = ls.MODEL_SLUGS[model_id]
    dataset = pd.read_parquet(root / ls.d3c.DATASET_PATH)
    saved = pd.read_parquet(root / ls.d3c.PROMPT_MANIFEST_PATH)
    ids = dataset.drop_duplicates("artifact_id").sort_values("artifact_id").groupby(["family", "latent_level"]).head(3).artifact_id.tolist()
    frames = ls.build_manifests(dataset, saved, ids, "flow-test")
    record = ls.read_json(root / ls.d3c.MODEL_PROVENANCE_PATH)["models"][model_id]
    historical, historical_run, issues = runner.load_historical(root, frames["original"], model_id, record)
    assert issues == []
    probes = frames["original"].sort_values("original_prompt_id").head(27).original_prompt_id.tolist()
    config = {"models": {model_id: record}, "replay_prompt_ids": probes, "historical_input_hashes": {},
              "inference": {"dtype": "bfloat16", "batch_size": 1}}
    out = tmp_path / "experiment"
    out.mkdir()
    ls.write_json(out / "FROZEN.json", {"test": True})
    monkeypatch.setattr(ls, "load_frozen", lambda *args: (out, config, frames))
    monkeypatch.setattr(provenance, "environment_snapshot", lambda *args: historical_run["environment"])
    token_check = {"conditions": {"original": {"prompts": historical[["original_prompt_id", "rendered_prompt_sha256"]].to_dict("records")}}}
    monkeypatch.setattr(ls, "check_tokenizer", lambda *args: ({c: c for c in ls.CONDITIONS}, token_check))
    fake_model = types.SimpleNamespace(dtype="BF16", training=False, config=types.SimpleNamespace(_attn_implementation="test"))
    fake_model.to = lambda device: fake_model
    fake_model.eval = lambda: fake_model
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="BF16", cuda=types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True)))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: fake_model)))
    seen = []
    def score_rows(model, tokenizer, tokens, manifest, mid, revision, directory, name):
        if "swapped" in name:
            assert (directory / "historical_reuse_decision.json").exists()
        if name == "swapped" and fail_swap:
            raise RuntimeError("injected partial inference failure")
        seen.append((name, len(manifest)))
        h = historical.set_index("original_prompt_id")
        rows = []
        for entry in manifest.to_dict("records"):
            old = h.loc[entry["original_prompt_id"]]
            met = float(old.raw_score_met) + (0 if replay_equal else 1)
            other = float(old.raw_score_not_met)
            result = {"raw_score_met": met, "raw_score_not_met": other,
                "p_met": scoring.normalized_probability(met, other),
                "rendered_prompt_sha256": old.rendered_prompt_sha256,
                "scoring_method": old.scoring_method, "n_prompt_tokens": old.n_prompt_tokens}
            rows.append({**entry, "model_id": mid, "revision": revision, **ls.semantic_scores(result, entry["condition"])})
        return pd.DataFrame(rows)
    monkeypatch.setattr(runner, "score_rows", score_rows)
    def run(stage):
        monkeypatch.setattr(sys, "argv", ["runner", "--root", str(root), "--model", model_id, "--stage", stage])
        return runner.main()
    assert run("tokenize") == 0
    assert run("smoke") == 0
    result = run("run")
    formal = out / "runs" / slug
    if fail_swap:
        assert result == 1
        assert not (formal / "COMPLETE.json").exists()
        assert (formal / "INCOMPLETE.json").exists()
        assert ls.read_json(formal / "run.json")["error"] == "injected partial inference failure"
        return
    assert result == 0
    decision = ls.read_json(formal / "historical_reuse_decision.json")
    assert decision["reuse_historical_original"] == replay_equal
    assert (("original", 729) in seen) == (not replay_equal)
    assert ("swapped", 729) in seen
    ls.paired_frames(pd.read_parquet(formal / "original.parquet"), pd.read_parquet(formal / "swapped.parquet"), frames, model_id)
    assert (formal / "COMPLETE.json").is_file()
    assert not (formal / "INCOMPLETE.json").exists()
    with pytest.raises(FileExistsError):
        run("run")
