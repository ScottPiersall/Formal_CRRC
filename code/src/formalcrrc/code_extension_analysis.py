# Review adaptation: historical non-English narrative strings omitted; numerical routines retained.
"""Independent audits, task-bootstrap estimates and self-contained v2 evidence."""
from __future__ import annotations
import ast
import csv
import json
import math
import pathlib
import platform
import re
import zipfile
import numpy as np
from formalcrrc import code_extension as cp
from formalcrrc.code_pilot_analysis import accounting
METRICS = ['accuracy', 'strict_accuracy', 'trr', 'fsrr', 'tce', 'oracle_min_errors', 'k0_correct', 'strict_trr', 'strict_fsrr']

def ci(values):
    a = np.array([v for v in values if v is not None], dtype=float)
    if not len(a):
        return dict(n=0, estimate=None, ci95=None)
    rng = np.random.default_rng(cp.SEED)
    boot = a[rng.integers(0, len(a), size=(5000, len(a)))].mean(axis=1)
    return dict(n=len(a), estimate=float(a.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist())

def independent(m, z):
    truth = set(range(z + 1))
    groups = [set()]
    enabled = set()
    for v in sorted(set(m), reverse=True):
        enabled = enabled | {i for i, x in enumerate(m) if x == v}
        groups.append(enabled)
    raw = {i for i, x in enumerate(m) if x >= 0}
    return dict(errors=len(raw ^ truth), oracle_min_errors=min((len(g ^ truth) for g in groups)), tce=abs(next((i for i in range(9) if i not in raw), 9) - (z + 1)), trr=any((next((i for i in range(9) if i not in g), 9) == z + 1 for g in groups)) if z < 8 else None, fsrr=any((g == truth for g in groups)) if z < 8 else None)

def metric_row(task, key, variant, dtype, m, z):
    d = cp.curve_diagnostics(m, z + 1)
    check = independent(m, z)
    for name, value in check.items():
        if d[name] != value:
            raise RuntimeError(f'Independent metric mismatch {name} {task}')
    d.update(task_id=task, model_key=key, variant=variant, dtype=dtype, k0_correct=m[0] >= 0, strict_trr=m[z + 1] < min(m[1:z + 1]) if 1 <= z <= 7 else None, strict_fsrr=min(m[1:z + 1]) > max(m[z + 1:]) if 1 <= z <= 7 else None)
    return d

def summarize_rows(rows):
    sets = {'all': rows, 'nontrivial': [r for r in rows if r['z'] < 8], 'partial': [r for r in rows if 1 <= r['z'] <= 7]}
    return {g: dict(n_tasks=len(rs), **{m: ci([r[m] for r in rs]) for m in METRICS}) for g, rs in sets.items()}

def summarize_development(root):
    root = pathlib.Path(root)
    out = cp.outdir(root)
    cp.assert_development(root)
    if (out / 'development_summary.json').exists():
        existing = cp.load(out / 'development_summary.json')
        cp.assert_manifest(root, existing['evidence_files'])
        if not existing['complete'] or not existing['numeric_self_check_passed']:
            raise RuntimeError('Saved development summary is incomplete')
        print('Development summary already complete and evidence intact', flush=True)
        return existing
    cp.assert_manifest(root, cp.load(out / 'development_prompt_freeze.json')['files'])
    items = cp.load(out / 'development/instances.json')
    rows = []
    raw = {}
    missing = []
    for key in cp.MODELS:
        prompts = {(r['task_id'], r['variant'], r['k']): r for r in cp.load(out / 'development_prompts' / f'{key}.json')['rows']}
        for v in cp.VARIANTS:
            for dtype in cp.DTYPES:
                for item in items:
                    task = item['task_id']
                    records = []
                    for k in range(9):
                        p = out / 'development_scores' / key / v / dtype / f'{cp.score_key(task, k)}.json'
                        if not p.exists():
                            missing.append(str(p.relative_to(root)))
                            continue
                        s = cp.load(p)
                        r = prompts[task, v, k]
                        assert s['rendered_prompt_sha256'] == r['rendered_prompt_sha256']
                        assert s['freeze_sha256'] == cp.sha((out / 'development_freeze.json').read_bytes())
                        assert s['created_at'] >= cp.load(out / 'development_prompt_freeze.json')['created_at']
                        assert math.isfinite(s['margin']) and s['margin'] == s['score_met'] - s['score_not_met']
                        records.append(s)
                        raw[key, v, dtype, task, k] = s
                    if len(records) == 9:
                        z = cp.load(out / 'development/truth' / f'{cp.slug(task)}.json')['z']
                        rows.append(metric_row(task, key, v, dtype, [s['margin'] for s in records], z))
    summaries = {f'{key}/{v}/{dtype}': summarize_rows([r for r in rows if (r['model_key'], r['variant'], r['dtype']) == (key, v, dtype)]) for key in cp.MODELS for v in cp.VARIANTS for dtype in cp.DTYPES}
    endpoint = {}
    numerical = {}
    replication = {}
    for key in cp.MODELS:
        for v in cp.VARIANTS:
            ss = [s for (m, w, d, t, k), s in raw.items() if (m, w, d, k) == (key, v, 'bfloat16', 0)]
            labels = [re.match('^([AB])(?:\\b|$)', s.get('greedy_text', '').strip()) for s in ss]
            endpoint[f'{key}/{v}'] = dict(n=len(ss), conditional_A_count=sum((s['margin'] >= 0 for s in ss)), greedy_A_count=sum((bool(m) and m.group(1) == 'A' for m in labels)), greedy_B_count=sum((bool(m) and m.group(1) == 'B' for m in labels)), greedy_other_count=sum((m is None for m in labels)), mean_full_vocab_AB_mass=float(np.mean([s['full_vocab_p_met'] + s['full_vocab_p_not_met'] for s in ss])) if ss else None)
            pairs = [(s, raw[m, w, 'float32', t, k]) for (m, w, d, t, k), s in raw.items() if (m, w, d) == (key, v, 'bfloat16') and (m, w, 'float32', t, k) in raw]
            numerical[f'{key}/{v}'] = dict(n_scores=len(pairs), max_abs_margin_delta=max((abs(a['margin'] - b['margin']) for a, b in pairs), default=None), sign_flips=sum(((a['margin'] >= 0) != (b['margin'] >= 0) for a, b in pairs)), fsrr_flips=sum((a['fsrr'] != b['fsrr'] for a in rows for b in rows if (a['model_key'], a['variant'], a['dtype']) == (key, v, 'bfloat16') and (b['model_key'], b['variant'], b['dtype'], b['task_id']) == (key, v, 'float32', a['task_id']))))
        olddir = root / 'artifacts/code_pilot_v1/scores/main' / key
        pairs = []
        for item in items:
            for k in range(9):
                new = raw.get((key, 'original', 'bfloat16', item['task_id'], k))
                p = olddir / f"{cp.score_key(item['task_id'], k)}.json"
                if new and p.exists():
                    pairs.append((new, cp.load(p)))
        replication[key] = dict(n=len(pairs), max_abs_margin_delta=max((abs(a['margin'] - b['margin']) for a, b in pairs), default=None), sign_flips=sum(((a['margin'] >= 0) != (b['margin'] >= 0) for a, b in pairs)))
    format_freeze = cp.load(out / 'development_format_freeze.json')
    cp.assert_manifest(root, format_freeze['files'])
    formatting = {}
    format_count = 0
    for key in cp.MODELS:
        for v in cp.VARIANTS:
            correct = 0
            changed = 0
            masses = []
            for item in items:
                p = out / 'development_format_scores' / key / v / f"{cp.slug(item['task_id'])}.json"
                if not p.exists():
                    missing.append(str(p.relative_to(root)))
                    continue
                r = cp.load(p)
                base = raw[key, v, 'bfloat16', item['task_id'], 0]
                current = cp.sha((out / 'development_format_freeze.json').read_bytes())
                if r['freeze_sha256'] == current:
                    assert r['created_at'] >= format_freeze['created_at']
                else:
                    assert r['freeze_sha256'] in format_freeze.get('accepted_previous_score_freezes', {})
                    assert r['created_at'] >= format_freeze['accepted_previous_score_freezes'][r['freeze_sha256']]
                    assert format_freeze['preserved_prior_score_files'][str(p.relative_to(root))] == cp.sha(p.read_bytes())
                assert r['rendered_prompt_sha256'] == base['rendered_prompt_sha256']
                expected = next((x['answer_forms'] for x in cp.load(out / 'development_prompts' / f'{key}__answer_tokens.json')['rows'] if x['task_id'] == item['task_id'] and x['variant'] == v and (x['k'] == 0)))
                assert all((r['answer_events'][name]['token_ids'] == expected[name]['token_ids'] for name in expected))
                assert set(r['answer_events']) == {'A', ' A', '\nA', 'B', ' B', '\nB'}
                events = {tuple(e['token_ids']): (name[-1], e['log_probability']) for name, e in r['answer_events'].items()}
                totals = {label: sum((math.exp(value) for _, (tag, value) in events.items() if tag == label)) for label in ['A', 'B']}
                assert abs(math.log(totals['A']) - math.log(totals['B']) - r['margin']) < 1e-10
                correct += r['margin'] >= 0
                changed += (r['margin'] >= 0) != (base['margin'] >= 0)
                format_count += 1
                masses.append(totals['A'] + totals['B'])
            formatting[f'{key}/{v}'] = dict(n=len(masses), whitespace_union_A_count=correct, classification_changes_from_bare=changed, mean_union_full_vocab_mass=float(np.mean(masses)) if masses else None)
    evidence = list((out / 'development_scores').rglob('*.json')) + list((out / 'development_format_scores').rglob('*.json')) + list((out / 'development_prompts').glob('*.json')) + [out / 'development_prompt_freeze.json', out / 'development_format_freeze.json']
    result = dict(created_at=cp.now(), complete=not missing and len(raw) == 2160 and (format_count == 120), missing=missing, numeric_self_check_passed=bool(raw) and all((s['algebra_error'] <= 1e-05 for s in raw.values())), max_algebra_error=max((s['algebra_error'] for s in raw.values()), default=None), max_teacher_forced_abs_delta=max((abs(s['teacher_forced_delta']) for s in raw.values() if 'teacher_forced_delta' in s), default=None), scores=len(raw), greedy_calls=sum(('greedy_text' in s for s in raw.values())), format_probes=format_count, evidence_files=cp.manifest(root, evidence), formatting=formatting, summaries=summaries, endpoints=endpoint, numerical=numerical, v1_replication=replication, decision='Retain original as primary and explicit as paired secondary, as fixed before diagnostics. Freeze new task panel only after complete validated diagnostics.', caveat='Development results reuse the 30 v1 tasks and are not independent confirmation; full-model float32 starts from the same originally bf16 checkpoint values.')
    if not result['complete']:
        cp.snapshot(out / 'development_partial_status.json', result)
        raise RuntimeError(f'Development missing {len(missing)} rows')
    cp.immutable(out / 'development_summary.json', result)
    cp.immutable(out / 'development_metrics.json', rows)
    print(cp.canonical(dict(complete=result['complete'], endpoints=endpoint, numerical=numerical, formatting=formatting)))

def audit(root):
    root = pathlib.Path(root)
    out = cp.outdir(root)
    a = cp.assert_freeze_a(root)
    b = cp.assert_freeze_b(root)
    cp.assert_development(root)
    cp.assert_manifest(root, cp.load(out / 'development_summary.json')['evidence_files'])
    baseline = cp.load(out / 'prior_v1_baseline.json')['files']
    if (root / '.git').exists():
        cp.assert_manifest(root, baseline)
    else:
        cp.assert_manifest(root, {p: h for p, h in baseline.items() if (root / p).exists()})
    items = cp.load(out / 'instances.json')
    selection = cp.load(out / 'task_selection.json')
    assert len(items) == 83 and len({i['task_id'] for i in items}) == 83
    assert sum((i['split'] == 'main' for i in items)) == 80 and sum((i['split'] == 'smoke' for i in items)) == 3
    assert not {i['task_id'] for i in items} & set(selection['excluded_v1_ids'])
    order = sorted(selection['eligible_unused_order'], key=lambda t: cp.deterministic_order(t, 'new_tasks'))
    pool_path = out / 'upstream/pilot_reference_eligibility_pool.json'
    assert cp.sha(pool_path.read_bytes()) == selection['task_pool_sha256']
    pool = cp.load(pool_path)
    expected_unused = sorted([r['task_id'] for r in pool['tasks'] if r['eligible'] and r['task_id'] not in selection['excluded_v1_ids']], key=lambda t: cp.deterministic_order(t, 'new_tasks'))
    assert expected_unused == order and len(order) == 97
    assert selection['selected_ids'] == order[:83] == [i['task_id'] for i in items]
    numeric = cp.load(out / 'numeric_task_ids.json')
    assert numeric == sorted([i['task_id'] for i in items if i['split'] == 'main'], key=lambda t: cp.deterministic_order(t, 'numeric'))[:10]
    upstream = {r['task_id']: r for r in map(json.loads, (out / 'upstream/HumanEvalPlus-v0.1.10.jsonl').read_text().splitlines())}
    truths = {}
    observations = 0
    compact = lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    for i in items:
        task = i['task_id']
        g = cp.load(out / 'generation' / f'{cp.slug(task)}.json')
        t = cp.load(out / 'truth' / f'{cp.slug(task)}.json')
        assert a['created_at'] <= g['created_at'] <= b['created_at'] and t['created_at'] <= b['created_at']
        assert t['stable'] and t['code_sha256'] == g['code_sha256'] == cp.sha(g['code'])
        assert g['user_message'] == a['protocol']['generation']['message'].format(problem=i['problem'])
        assert g['rendered_prompt_sha256'] == cp.sha(g['rendered_prompt'])
        assert g['model_id'] == cp.MODELS['mistral']['model_id'] and g['revision'] == cp.MODELS['mistral']['revision']
        assert cp.extract(g['response'])[0] == g['code']
        assert g['freeze_A_sha256'] == cp.sha((out / 'freeze_A.json').read_bytes())
        inputs = [compact(x['args']) for x in i['tests']]
        assert len(set(inputs)) == 8 and all((len(v.encode()) <= 512 for v in inputs))
        assert not set(inputs) & set(map(compact, upstream[task]['base_input']))
        assert set(inputs) <= set(map(compact, upstream[task]['plus_input']))
        for test in i['tests']:
            assert cp.exact_type(ast.literal_eval(test['expected_repr'])) and len(test['expected_repr'].encode()) <= 512
        for role in ['execution', 'reference_execution']:
            rr = t[role]['tests']
            assert len(rr) == 16 and t[role]['status'] == 'complete'
            for r in rr:
                assert r['category'] != 'infrastructure_error'
                try:
                    passed = r['category'] == 'ok' and bool(ast.literal_eval(r['observed_repr']) == ast.literal_eval(i['tests'][r['test_index']]['expected_repr']))
                except (ValueError, SyntaxError):
                    passed = False
                assert passed == r['passed']
                if role == 'reference_execution':
                    assert passed
                observations += 1
            assert [r['passed'] for r in rr[:8]] == [r['passed'] for r in rr[8:]]
        bits = [int(r['passed']) for r in t['execution']['tests'][:8]]
        assert bits == t['pass_bits'] and sum(bits) == t['z']
        truths[task] = t
        direct = cp.load(out / 'truth' / f'{cp.slug(task)}__direct_equality.json')
        assert direct['passed'] and direct['code_sha256'] == g['code_sha256'] and (direct['created_at'] <= b['created_at'])
        from formalcrrc.code_extension_validation import equality_source
        for role, code in [('reference', i['reference_code']), ('candidate', g['code'])]:
            check = direct['roles'][role]
            assert check['wrapper_sha256'] == cp.sha(equality_source(code, i['entry_point'], [x['expected_repr'] for x in i['tests']]))
            rr = check['execution']['tests']
            assert len(rr) == 16 and check['agreement']
            expected_bits = [1] * 8 if role == 'reference' else bits
            assert check['pass_bits'] == expected_bits
            for j, r in enumerate(rr):
                observed = r['category'] == 'ok' and ast.literal_eval(r['observed_repr']) is True
                assert r['passed'] == observed == bool(expected_bits[j % 8])
                observations += 1
    raw = {}
    rows = []
    groups = 0
    for key in cp.MODELS:
        prep = cp.load(out / 'prompts' / f'{key}.json')
        indexed = {(r['task_id'], r['variant'], r['k']): r for r in prep['rows']}
        assert len(indexed) == 1494
        for i in items:
            task = i['task_id']
            for v in cp.VARIANTS:
                normalized = []
                for k in range(9):
                    p = indexed[task, v, k]
                    encoded = p['user_message'].encode()
                    start, end = p['threshold_span']
                    assert encoded[start:end] == str(k).encode()
                    normalized.append(encoded[:start] + b'{threshold}' + encoded[end:])
                    assert cp.sha(p['rendered_prompt']) == p['rendered_prompt_sha256']
                    assert p['user_message'] == cp.prompt(i, cp.load(out / 'generation' / f'{cp.slug(task)}.json')['code'], k, v)['user_message']
                assert len(set(normalized)) == 1
                groups += 1
                for dtype in cp.DTYPES if task in numeric else ['bfloat16']:
                    scores = []
                    for k in range(9):
                        s = cp.load(cp.score_path(out, i['split'], key, v, dtype, task, k))
                        p = indexed[task, v, k]
                        assert (s['task_id'], s['k'], s['variant'], s['dtype'], s['model_key']) == (task, k, v, dtype, key)
                        assert s['revision'] == cp.MODELS[key]['revision'] and s['created_at'] >= b['created_at']
                        assert s['freeze_sha256'] == cp.sha((out / 'freeze_B.json').read_bytes())
                        assert s['rendered_prompt_sha256'] == p['rendered_prompt_sha256'] and s['label_tokenization'] == p['label_tokenization']
                        assert s['n_prompt_tokens'] == p['n_prompt_tokens'] and math.isfinite(s['margin'])
                        assert s['margin'] == s['score_met'] - s['score_not_met'] and s['algebra_error'] <= 1e-05
                        from formalcrrc.scoring import normalized_probability
                        assert s['p_met'] == normalized_probability(s['score_met'], s['score_not_met'])
                        scores.append(s)
                        raw[key, v, dtype, task, k] = s
                    if i['split'] == 'main':
                        rows.append(metric_row(task, key, v, dtype, [s['margin'] for s in scores], truths[task]['z']))
    assert len(list((out / 'scores/main').rglob('*.json'))) == 3240 and len(list((out / 'scores/smoke').rglob('*.json'))) == 108
    result = dict(checked_at=cp.now(), passed=True, independently_recomputed_main_curves=len(rows), truth_observations=observations, normalized_prompt_groups=groups, new_main_tasks=80, new_smoke_tasks=3, overlap_with_v1=0, score_records=len(raw), freezes_passed=True, source_membership_passed=True, prior_v1_preserved=True)
    cp.snapshot(out / 'independent_evidence_audit.json', result)
    return (items, truths, raw, rows, result)

def analyze(root):
    root = pathlib.Path(root)
    out = cp.outdir(root)
    report = out / 'analysis'
    report.mkdir(parents=True, exist_ok=True)
    from importlib.metadata import version
    cp.snapshot(report / 'runtime.json', dict(created_at=cp.now(), python=platform.python_version(), packages={name: version(name) for name in ['numpy', 'scipy', 'matplotlib', 'pytest']}, random_generator='numpy.default_rng / PCG64', bootstrap_seed=cp.SEED, bootstrap_replicates=5000, source_sha256=cp.sha(pathlib.Path(__file__).read_bytes())))
    items, truths, raw, rows, audit_result = audit(root)
    from formalcrrc import code_extension_answer_analysis as answer_analysis
    answer_raw, answer_rows, answer_audit = answer_analysis.audit(root, items, truths)
    answer_summary = answer_analysis.summarize(answer_rows, rows)
    answer_fp_raw, answer_fp_rows, answer_numerical = answer_analysis.audit_precision(root, truths, answer_rows, answer_raw)
    answer_summary['numerical'] = answer_numerical
    cp.snapshot(report / 'answer_form_float32_metrics.json', answer_fp_rows)
    answer_audit.update(float32_scores=len(answer_fp_raw), float32_curves=len(answer_fp_rows), tf32_disabled=True)
    audit_result['answer_forms'] = answer_audit
    cp.snapshot(out / 'independent_evidence_audit.json', audit_result)
    main = [i['task_id'] for i in items if i['split'] == 'main']
    partial = [t for t in main if 1 <= truths[t]['z'] <= 7]
    indexed = {(r['model_key'], r['variant'], r['dtype'], r['task_id']): r for r in rows}
    summaries = {f'{key}/{v}/{d}': summarize_rows([r for r in rows if (r['model_key'], r['variant'], r['dtype']) == (key, v, d)]) for key in cp.MODELS for v in cp.VARIANTS for d in cp.DTYPES}
    paired = {}
    for dtype in cp.DTYPES:
        pairs = [(f'{key}/explicit-minus-original/{dtype}', (key, 'explicit', dtype), (key, 'original', dtype)) for key in cp.MODELS]
        pairs += [(f'{v}/qwen-minus-mistral/{dtype}', ('qwen', v, dtype), ('mistral', v, dtype)) for v in cp.VARIANTS]
        for name, left, right in pairs:
            common = sorted({r['task_id'] for r in rows if (r['model_key'], r['variant'], r['dtype']) == left} & {r['task_id'] for r in rows if (r['model_key'], r['variant'], r['dtype']) == right})
            paired[name] = {}
            for group, ids in [('all', common), ('nontrivial', [t for t in common if truths[t]['z'] < 8]), ('partial', [t for t in common if t in partial])]:
                paired[name][group] = dict(n_tasks=len(ids), task_ids=ids, **{m: ci([indexed[*left, t][m] - indexed[*right, t][m] for t in ids if indexed[*left, t][m] is not None and indexed[*right, t][m] is not None]) for m in METRICS})
    robustness = {}
    numeric_ids = set(cp.load(out / 'numeric_task_ids.json'))
    for key in cp.MODELS:
        both = [t for t in partial if all((indexed[key, v, 'bfloat16', t]['fsrr'] is False for v in cp.VARIANTS))]
        numeric = [t for t in partial if t in numeric_ids]
        strong = [t for t in numeric if all((indexed[key, v, d, t]['fsrr_gap'] < -0.01 for v in cp.VARIANTS for d in cp.DTYPES))]
        pairs = [(raw[key, v, 'bfloat16', t, k], raw[key, v, 'float32', t, k]) for v in cp.VARIANTS for t in numeric_ids for k in range(9)]
        robustness[key] = dict(partial_n=len(partial), same_task_both_templates_fsrr_failure_ids=both, same_task_failure_rate=ci([t in both for t in partial]), numerical_partial_n=len(numeric), both_templates_both_dtypes_gap_below_minus_001_ids=strong, numeric_task_n=len(numeric_ids), numerical_max_abs_margin_delta=max((abs(a['margin'] - b['margin']) for a, b in pairs)), numerical_sign_flips=sum(((a['margin'] >= 0) != (b['margin'] >= 0) for a, b in pairs)), numerical_fsrr_flips=sum((indexed[key, v, 'bfloat16', t]['fsrr'] != indexed[key, v, 'float32', t]['fsrr'] for v in cp.VARIANTS for t in numeric_ids)))
    same = []
    cases = {}
    for key in cp.MODELS:
        for v in cp.VARIANTS:
            subset = sorted([r for r in rows if (r['model_key'], r['variant'], r['dtype']) == (key, v, 'bfloat16')], key=lambda r: r['task_id'])
            for e in sorted({r['errors'] for r in subset if r['z'] < 8}):
                group = [r for r in subset if r['errors'] == e and r['z'] < 8]
                pair = next(((a, b) for a in group for b in group if (a['trr'], a['fsrr']) != (b['trr'], b['fsrr'])), None)
                if pair:
                    same.append(dict(model=key, variant=v, errors=e, case_a=pair[0], case_b=pair[1]))
            for name, predicate in [('recoverable', lambda r: r['fsrr'] is True), ('partial_unrecoverable', lambda r: 1 <= r['z'] <= 7 and r['fsrr'] is False), ('trr_yes_fsrr_no', lambda r: r['trr'] is True and r['fsrr'] is False)]:
                r = next((r for r in subset if predicate(r)), None)
                if r is None:
                    cases[f'{key}/{v}/{name}'] = None
                    continue
                t = r['task_id']
                item = next((i for i in items if i['task_id'] == t))
                cases[f'{key}/{v}/{name}'] = dict(metrics=r, code=cp.load(out / 'generation' / f'{cp.slug(t)}.json')['code'], tests=item['tests'], truth=truths[t], margins=[raw[key, v, 'bfloat16', t, k]['margin'] for k in range(9)])
    hist = {str(z): sum((truths[t]['z'] == z for t in main)) for z in range(9)}
    gens = [cp.load(p) for p in (out / 'generation').glob('*.json')]
    attempts = [cp.load(p) for p in (out / 'attempts').rglob('*.json')]
    allocations = accounting(out)
    dev_scores = [cp.load(p) for p in (out / 'development_scores').rglob('*.json')]
    format_scores = [cp.load(p) for p in (out / 'development_format_scores').rglob('*.json')]
    summary = dict(created_at=cp.now(), technical_complete=True, n_main=80, partial_n=len(partial), z_histogram=hist, summaries=summaries, paired=paired, robustness=robustness, same_errors_different_recovery=same, syntax_extraction_counts={s: sum((g['split'] == 'main' and g['extraction_status'] == s for g in gens)) for s in ['ok', 'syntax_error', 'extraction_failure']}, main_generation_truncations=sum((g['split'] == 'main' and g['truncated'] for g in gens)), partial_failure_counts={f'{k}/{v}': dict(n=len(partial), fsrr_failures=sum((indexed[k, v, 'bfloat16', t]['fsrr'] is False for t in partial)), trr_yes_fsrr_no=sum((indexed[k, v, 'bfloat16', t]['trr'] is True and indexed[k, v, 'bfloat16', t]['fsrr'] is False for t in partial)), tiny_gap_failures=sum((indexed[k, v, 'bfloat16', t]['fsrr'] is False and abs(indexed[k, v, 'bfloat16', t]['fsrr_gap']) <= 0.01 for t in partial))) for k in cp.MODELS for v in cp.VARIANTS}, bootstrap='5000 task-level percentile replicates, seed 20260911; paired complete common task sets; no multiplicity-adjusted significance claims.', limitations=['Conditional 80-task sample of unused eligible HumanEval+ tasks; possible pretraining exposure remains.', 'v1 diagnostics are a development set and are excluded from confirmation estimates.', 'One candidate per task from Mistral, which also serves as one judge; two ordinary immediate-score models only.', 'Recovery and oracle thresholds use task-specific truth, not deployable calibration.', 'Float32 arithmetic uses the same originally bf16 checkpoint values; the numeric audit covers only ten predetermined new tasks.', 'An all-zero bootstrap interval is degenerate empirical resampling, not proof of population probability zero.'])
    status = dict(updated_at=cp.now(), status='COMPLETE', planned=cp.protocol(root)['planned'], completed=dict(development_scores=len(dev_scores), development_greedy_calls=sum(('greedy_text' in s for s in dev_scores)), development_format_probes=len(format_scores), generation_responses=len(gens), new_main_truth=80, new_smoke_truth=3, main_bf16_scores=2880, main_float32_scores=360, smoke_scores=108, main_bf16_curves=320, main_float32_curves=40), calls=dict(generation_forward_calls=sum((g['generation_forward_calls'] for g in gens)), development_score_forward_calls=sum((s['forward_calls'] for s in dev_scores)), development_greedy_forward_calls=sum((s.get('greedy_generation_forward_calls', 0) for s in dev_scores)), development_format_forward_calls=sum((s['forward_calls'] for s in format_scores)), main_and_smoke_forward_calls=sum((s['forward_calls'] for s in raw.values())), infrastructure_failed_model_attempts=sum((a['status'] == 'infrastructure_error' for a in attempts))), gpu_accounting=allocations, max_additional_gpu_hours=8, audit=audit_result, blockers=[], recovery_commands=[])
    summary['answer_form_robustness'] = answer_summary
    status['planned'].update(answer_form_main_scores=2880, answer_form_smoke_scores=108, answer_form_main_curves=320, answer_form_float32_scores=360, answer_form_float32_curves=40)
    status['completed'].update(answer_form_main_scores=2880, answer_form_smoke_scores=108, answer_form_main_curves=320, answer_form_float32_scores=360, answer_form_float32_curves=40)
    status['calls']['answer_form_forward_calls'] = sum((r['forward_calls'] for r in answer_raw))
    status['calls']['answer_form_float32_forward_calls'] = sum((r['forward_calls'] for r in answer_fp_raw))
    if allocations['allocated_gpu_seconds'] > cp.MAX_GPU_SECONDS:
        raise RuntimeError('GPU budget exceeded')
    cp.snapshot(report / 'summary.json', summary)
    cp.snapshot(report / 'per_instance_metrics.json', rows)
    cp.snapshot(report / 'auditable_cases.json', cases)
    with (report / 'per_instance_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cp.snapshot(out / 'execution_status.json', status)
    write_reports(root, summary, status, cases)
    answer_analysis.write(report, answer_summary, answer_rows, answer_raw)
    print(cp.canonical(dict(status='COMPLETE', z_histogram=hist, partial_n=len(partial), counts=summary['partial_failure_counts'], robustness=robustness)))

def fmt(stat):
    if stat['estimate'] is None:
        return 'NA (n=0)'
    lo, hi = stat['ci95']
    return f"{stat['estimate']:.3f} [{lo:.3f}, {hi:.3f}], n={stat['n']}"

def write_reports(root, s, status, cases):
    out = cp.outdir(root)
    report = out / 'analysis'
    lines = ['# Code acceptance extension v2 — 80 new tasks', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', '|---|---|---|---|---|']
    for key in cp.MODELS:
        for v in cp.VARIANTS:
            g = s['summaries'][f'{key}/{v}/bfloat16']
            lines.append('| ' + key + ' / ' + v + ' | ' + ' | '.join((fmt(st) for st in [g['all']['k0_correct'], g['all']['strict_accuracy'], g['partial']['fsrr'], g['partial']['strict_fsrr']])) + ' |')
    lines += ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.']
    for key, r in s['robustness'].items():
        lines.append('Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.')
    lines += ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.'] + ['- ' + x for x in s['limitations']]
    (report / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    english = ['# Methods and Results — independent code acceptance extension', '\nWe prospectively fixed 80 new main tasks and three disjoint smoke tasks from the 97 eligible HumanEval+ tasks unused by the pilot. Task ordering used SHA256 with seed 20260911; reference-only input eligibility inherited the pilot rule and seed 20260910. No task was replaced based on candidate behavior or judge outcomes. The 30 pilot main tasks were a separate development set for diagnostics, never pooled into confirmation estimates.', '\nMistral generated one greedy candidate per task, capped at 1024 new tokens and without the additional tests or execution feedback. Deterministic extraction retained failures and truncations. Reference and candidate code each ran twice per test in the pinned no-network/no-mount Docker sandbox with fresh processes, five-second candidate limits and 512 MiB address space. Y(k)=1[z>=k] for k=0,...,8.', '\nBoth pinned ordinary judges scored the original template (primary) and an explicit decision template (paired secondary). Template roles were fixed before development diagnostics. All nine prompts within a task/template differed only in the threshold field. Native chat templates, contextual A/B continuation IDs, batch size one and no truncation were enforced. Primary inference used unquantized bf16 SDPA; ten hash-selected new tasks additionally used full-model float32 arithmetic for both templates. Float32 did not restore information absent from the original checkpoint precision.', '\nThe primary estimand was original-template FSRR among partially correct candidates (1<=z<=7), per model. Secondary endpoints were TRR, raw and k>=1 accuracy, first-failure error, oracle minimum residual errors, paired template differences, k>=1 recovery sensitivity and numerical agreement. Percentile 95% intervals used 5000 task-level bootstrap replicates with seed 20260911, preserving all thresholds and pairing common complete task IDs. No adaptive sample-size or significance stopping rule was used.', f"\nActual completion: 80 main tasks, 3 smoke tasks, 2880 bf16 main scores (320 curves), 360 float32 audit scores (40 curves), and 108 smoke scores. There were {s['partial_n']} partially correct candidates; z histogram: {s['z_histogram']}. Independent audit recomputed all 360 main curves and checked 5312 execution observations, including repeated direct in-container Python equality checks agreeing with the primary oracle. Allocated GPU time including development and handoffs was {status['gpu_accounting']['allocated_gpu_seconds'] / 3600:.4f} hours."]
    for key in cp.MODELS:
        for v in cp.VARIANTS:
            g = s['summaries'][f'{key}/{v}/bfloat16']
            english.append(f"\n{key}, {v}: partial-correct FSRR {fmt(g['partial']['fsrr'])}; partial-correct TRR {fmt(g['partial']['trr'])}; strict-threshold accuracy {fmt(g['all']['strict_accuracy'])}; k=0 accuracy {fmt(g['all']['k0_correct'])}; partial-correct k>=1 FSRR {fmt(g['partial']['strict_fsrr'])}.")
    english += ['\nLimitations:'] + ['- ' + x for x in s['limitations']]
    (report / 'METHODS_RESULTS_EN.md').write_text('\n'.join(english) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen = [(name, c) for name, c in cases.items() if c and '/original/partial_unrecoverable' in name]
    if chosen:
        fig, axes = plt.subplots(len(chosen), 1, figsize=(8, 3.3 * len(chosen)), squeeze=False, constrained_layout=True)
        for ax, (name, c) in zip(axes[:, 0], chosen):
            r = c['metrics']
            task = r['task_id']
            key = r['model_key']
            z = r['z']
            for v, color in [('original', '#2563eb'), ('explicit', '#be123c')]:
                vals = [cp.load(cp.score_path(out, 'main', key, v, 'bfloat16', task, k))['margin'] for k in range(9)]
                ax.plot(range(9), vals, 'o-', label=v, color=color)
            ax.axhline(0, color='#64748b', linestyle='--', linewidth=0.8)
            ax.axvspan(-0.2, z + 0.4, color='#dcfce7', alpha=0.5)
            ax.set(title=f'{task} / {key} / z={z}', xlabel='Required passes k', ylabel='A - B margin', xticks=range(9), xlim=(-0.2, 8.2))
            ax.legend()
            ax.grid(axis='y', alpha=0.2)
        fig.savefig(report / 'paired_template_curves.png', dpi=180)
        fig.savefig(report / 'paired_template_curves.svg')
        plt.close(fig)

def package(root):
    root = pathlib.Path(root)
    out = cp.outdir(root)
    if cp.load(out / 'execution_status.json')['status'] != 'COMPLETE':
        raise RuntimeError('Incomplete evidence')
    files = [p for p in out.rglob('*') if p.is_file() and p.name != 'handoff_manifest.json']
    files += cp.source_files(root)
    needed = {p.stem for p in files if p.parent == root / 'src/formalcrrc'}
    todo = list(needed)
    while todo:
        name = todo.pop()
        tree = ast.parse((root / 'src/formalcrrc' / f'{name}.py').read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'formalcrrc':
                imports.update((x.name for x in node.names))
            elif isinstance(node, ast.ImportFrom) and (node.module or '').startswith('formalcrrc.'):
                imports.add(node.module.split('.')[1])
            elif isinstance(node, ast.Import):
                imports.update((x.name.split('.')[1] for x in node.names if x.name.startswith('formalcrrc.')))
        for module in imports:
            if module not in needed and (root / 'src/formalcrrc' / f'{module}.py').is_file():
                needed.add(module)
                todo.append(module)
    files += [root / 'src/formalcrrc' / f'{name}.py' for name in needed]
    files += list((root / 'tests').glob('test_code_extension*.py')) + [root / 'tests/test_scoring.py', root / 'tests/test_code_pilot.py', root / 'tests/conftest.py', root / 'pyproject.toml', root / 'LICENSE', root / 'docs/CODE_EXTENSION_V2.md']
    files = sorted({p for p in files if p.is_file()})
    cp.snapshot(out / 'handoff_manifest.json', dict(created_at=cp.now(), files=cp.manifest(root, files)))
    files.append(out / 'handoff_manifest.json')
    target = root.parent / 'FormalCRRC_Code_Extension_v2_Handoff.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in files:
            archive.write(p, p.relative_to(root).as_posix())
    digest = cp.sha(target.read_bytes())
    target.with_suffix('.zip.sha256').write_text(f'{digest}  {target.name}\n')
    with zipfile.ZipFile(target) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())) and archive.testzip() is None
        for path, h in cp.load(out / 'handoff_manifest.json')['files'].items():
            assert cp.sha(archive.read(path.replace('\\', '/'))) == h
    print(str(target), digest, flush=True)
