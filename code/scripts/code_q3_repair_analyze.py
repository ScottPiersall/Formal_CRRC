# Review adaptation: historical non-English narrative strings omitted; numerical routines retained.
"""Supplemental versions with unchanged v2 metrics and explicit denominators."""
import pathlib, sys, json, math, itertools, collections, csv
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from formalcrrc import code_extension as cp, code_five as f, code_q3_repair as q
from formalcrrc.code_extension_analysis import metric_row, summarize_rows, METRICS, ci
from formalcrrc.code_extension_answer_tokens import BackendTokenizer
from code_five_analyze import audit_new, audit_strict_recovery, contrast

def write(p, obj):
    cp.snapshot(p, obj)

def read_lines(p):
    return [json.loads(s) for s in p.read_text().splitlines()]

def estimate(r):
    if r['estimate'] is None:
        return 'NA'
    return f"{r['estimate']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"

def analyze():
    o = q.out(ROOT)
    a = o / 'analysis'
    a.mkdir(exist_ok=True)
    freeze = q.verify(ROOT)
    old = f.out(ROOT)
    rows = cp.load(old / 'prompts/qwen3.json')['rows']
    meta = {r['key']: r for r in cp.load(o / 'inputs.json')['rows']}
    baseline = cp.load(old / 'analysis/curve_metrics.json')
    inv = cp.load(old / 'inventory_qwen3.json')
    tokenizer = BackendTokenizer(old / 'tokenizers/qwen3/tokenizer.json')
    truth = {r['task_id']: r['z'] for r in rows if r['split'] == 'main'}
    assert len(truth) == 80
    expected = dict(all=80, nontrivial=sum((z < 8 for z in truth.values())), partial=sum((1 <= z <= 7 for z in truth.values())))
    assert expected == dict(all=80, nontrivial=54, partial=23)
    scores = {}
    traces = {}
    missing = []
    validation = []
    changes = []
    continuation_rows = []
    for row in rows:
        key = f.key(row)
        original = cp.load(f.trace_path(ROOT, row))
        trace = original
        assert cp.sha(f.trace_path(ROOT, row).read_bytes()) == meta[key]['original_trace_sha256']
        if q.trace_path(ROOT, row).exists():
            trace = cp.load(q.trace_path(ROOT, row))
            assert not original['natural_end']
            assert trace['raw_generated_token_ids'] == original['raw_generated_token_ids'] + trace['additional_generated_token_ids']
            assert trace['raw_generated_token_ids'][:8192] == original['raw_generated_token_ids']
            assert trace['continuation_seed'] == q.seed(row) == meta[key]['continuation_seed']
            assert len(trace['additional_generated_token_ids']) == trace['additional_token_count'] <= 24576
            assert len(trace['raw_generated_token_ids']) == trace['reasoning_token_count'] <= 32768
            assert trace['freeze_sha256'] == freeze
            continuation_rows.append(dict(key=key, task_id=row['task_id'], variant=row['variant'], k=row['k'], split=row['split'], z=row['z'], natural_end=trace['natural_end'], additional_token_count=trace['additional_token_count'], total_reasoning_tokens=trace['reasoning_token_count'], finish_reason=trace['finish_reason'], stop_reason=trace['stop_reason']))
        if trace['natural_end']:
            assert trace['raw_generated_token_ids'][-1] == inv['think_end_id'] and trace['finish_reason'] == 'stop'
            assert trace['final_context_token_ids'] == row['prompt_token_ids'] + trace['raw_generated_token_ids'] + inv['separator_ids']
        else:
            assert not trace['final_context_token_ids']
        traces[key] = trace
        path = q.score_path(ROOT, row)
        if path.exists():
            s = cp.load(path)
            try:
                audit_new(s)
            except Exception as exc:
                validation.append(dict(key=key, error=str(exc), source=path.relative_to(ROOT).as_posix()))
                missing.append(dict(key=key, task_id=row['task_id'], variant=row['variant'], k=row['k'], split=row['split'], reason='score_validation_failure', missing_forms=['bare', 'whitespace_union']))
                continue
            assert s['freeze_sha256'] == freeze and s['original_trace_sha256'] == meta[key]['original_trace_sha256']
            assert s['z'] == row['z'] and s['user_message_sha256'] == row['user_message_sha256']
            assert trace['natural_end'] and s['final_context_ids_sha256'] == trace['final_context_ids_sha256']
            assert s['reasoning_sha256'] == cp.sha((ROOT / s['reasoning_path']).read_bytes())
            _, actual_forms = f.paths_for(tokenizer, trace['final_context_text'], trace['final_context_token_ids'])
            assert {name: e['token_ids'] for name, e in s['answer_events'].items()} == {name: e['token_ids'] for name, e in actual_forms.items()} == {name: e['token_ids'] for name, e in meta[key]['answer_forms'].items()}
            assert tokenizer.decode(trace['final_context_token_ids']) == trace['final_context_text']
            for rep, names in (('bare', ('A', 'B')), ('whitespace_union', f.FORMS)):
                unique = {tuple(s['answer_events'][name]['token_ids']): (name[-1], s['answer_events'][name]['probability']) for name in names}
                totals = {label: sum((probability for tag, probability in unique.values() if tag == label)) for label in ('A', 'B')}
                score = s['scores'][rep]
                for field, value in [('full_vocab_p_met', totals['A']), ('full_vocab_p_not_met', totals['B']), ('total_probability_mass', sum(totals.values())), ('p_met', totals['A'] / sum(totals.values()))]:
                    assert math.isclose(score[field], value, rel_tol=1e-09, abs_tol=1e-12), (key, rep, field)
            cp.assert_manifest(ROOT, s['node_receipts'])
            for node in s['prefix_nodes']:
                assert sum((math.exp(v) for v in node['next_log_probabilities'].values())) <= 1.00001
                receipt = cp.load(q.node_path(ROOT, row, node['token_ids']))
                assert receipt['next_log_probabilities'] == node['next_log_probabilities']
                assert receipt['prompt_ids_sha256'] == cp.sha(cp.canonical(trace['final_context_token_ids'] + node['token_ids']))
            scores[key] = s
            prior = f.score_path(ROOT, 'qwen3', row)
            if original['natural_end'] and prior.exists():
                oldscore = cp.load(prior)
                for rep in ('bare', 'whitespace_union'):
                    x, y = (oldscore['scores'][rep], s['scores'][rep])
                    changes.append(dict(key=key, task_id=row['task_id'], variant=row['variant'], k=row['k'], split=row['split'], representation=rep, old_margin=x['margin'], new_margin=y['margin'], margin_delta=y['margin'] - x['margin'], p_met_delta=y['p_met'] - x['p_met'], sign_changed=(x['margin'] >= 0) != (y['margin'] >= 0), old_mass=x['total_probability_mass'], new_mass=y['total_probability_mass'], old_accepted_mass=oldscore['scores']['whitespace_union']['total_probability_mass'] <= 1.00001))
        if key not in scores:
            reason = 'continuation_not_attempted' if not original['natural_end'] and (not q.trace_path(ROOT, row).exists()) else 'terminal_reasoning_failure' if not trace['natural_end'] else 'score_missing_or_failed'
            missing.append(dict(key=key, task_id=row['task_id'], variant=row['variant'], k=row['k'], split=row['split'], reason=reason, missing_forms=['bare', 'whitespace_union']))
    curves_by_version = {'original_8k_old_scorer': baseline}
    statuses = {}
    for version in ('original_8k_shared_prefix_scorer', 'two_stage_32k_shared_prefix_scorer'):
        use = {key: s for key, s in scores.items() if version.startswith('two_stage') or meta[key]['original_natural_end']}
        curves = [dict(r) for r in baseline if r['model_key'] != 'qwen3']
        status = {}
        for split, n in (('main', 1440), ('smoke', 54)):
            selected = [r for r in rows if r['split'] == split]
            valid = sum((f.key(r) in use for r in selected))
            complete = failed = unattempted = 0
            for task in sorted({r['task_id'] for r in selected}):
                for v in ('original', 'explicit'):
                    rr = [next((r for r in selected if r['task_id'] == task and r['variant'] == v and (r['k'] == k))) for k in range(9)]
                    if all((f.key(r) in use for r in rr)):
                        complete += 1
                        if split == 'main':
                            for rep in ('bare', 'whitespace_union'):
                                m = [use[f.key(r)]['scores'][rep]['margin'] for r in rr]
                                metric = metric_row(task, 'qwen3', v, 'bfloat16', m, truth[task])
                                audit_strict_recovery(m, truth[task], metric)
                                curves.append(dict(metric, representation=rep, margins=m))
                    else:
                        terminal = any(((not meta[f.key(r)]['original_natural_end'] if version.startswith('original') else q.trace_path(ROOT, r).exists() and (not traces[f.key(r)]['natural_end'])) or any((x['key'] == f.key(r) for x in validation)) for r in rr))
                        failed += int(terminal)
                        unattempted += int(not terminal)
            status[split] = dict(expected_contexts=n, valid_contexts=valid, valid_score_records=2 * valid, missing_contexts=n - valid, expected_curves=len(selected) // 9, complete_curves=complete, failed_curves=failed, missing_curves=unattempted)
        statuses[version] = status
        curves_by_version[version] = curves
    summaries = {}
    common = {}
    paired = {}
    tables = []
    paired_templates = {}
    paired_forms = {}
    for version, curves in curves_by_version.items():
        groups = {(m, v, rep): [r for r in curves if (r['model_key'], r['variant'], r['representation']) == (m, v, rep)] for m in f.MODELS for v in ('original', 'explicit') for rep in ('bare', 'whitespace_union')}
        summaries[version] = {}
        common[version] = {}
        paired[version] = {}
        paired_templates[version] = {f'{m}/{rep}': contrast(groups[m, 'explicit', rep], groups[m, 'original', rep]) for m in f.MODELS for rep in ('bare', 'whitespace_union')}
        paired_forms[version] = {f'{m}/{v}': contrast(groups[m, v, 'whitespace_union'], groups[m, v, 'bare']) for m in f.MODELS for v in ('original', 'explicit')}
        for key, group in groups.items():
            result = summarize_rows(group)
            for subset, values in result.items():
                values['expected_tasks'] = expected[subset]
                for metric in METRICS:
                    tables.append(dict(version=version, model=key[0], template=key[1], representation=key[2], subset=subset, expected_tasks=expected[subset], valid_curves=values['n_tasks'], metric=metric, **values[metric]))
            summaries[version]['/'.join(key)] = result
        for v in ('original', 'explicit'):
            for rep in ('bare', 'whitespace_union'):
                ids = set.intersection(*[{r['task_id'] for r in groups[m, v, rep]} for m in f.MODELS])
                common[version][v + '/' + rep] = dict(n_tasks=len(ids), task_ids=sorted(ids), summaries={m: summarize_rows([r for r in groups[m, v, rep] if r['task_id'] in ids]) for m in f.MODELS})
                for m, n in itertools.combinations(f.MODELS, 2):
                    paired[version][f'{m}-minus-{n}/{v}/{rep}'] = dict(pairwise_all_valid=contrast(groups[m, v, rep], groups[n, v, rep]), five_model_common_complete=contrast([r for r in groups[m, v, rep] if r['task_id'] in ids], [r for r in groups[n, v, rep] if r['task_id'] in ids]))
        joint = set.intersection(*[{r['task_id'] for r in rs} for rs in groups.values()])
        common[version]['joint_all_templates_forms'] = dict(n_tasks=len(joint), task_ids=sorted(joint))
    write(a / 'curve_metrics_by_version.json', curves_by_version)
    write(a / 'model_all_valid_summaries.json', summaries)
    write(a / 'five_model_common_complete.json', common)
    write(a / 'paired_model_comparisons.json', paired)
    write(a / 'paired_template_comparisons.json', paired_templates)
    write(a / 'paired_representation_comparisons.json', paired_forms)
    paired_versions = {}
    for earlier, later in itertools.combinations(curves_by_version, 2):
        for v in ('original', 'explicit'):
            for rep in ('bare', 'whitespace_union'):
                old_curves = [r for r in curves_by_version[earlier] if (r['model_key'], r['variant'], r['representation']) == ('qwen3', v, rep)]
                new_curves = [r for r in curves_by_version[later] if (r['model_key'], r['variant'], r['representation']) == ('qwen3', v, rep)]
                old_ids = {r['task_id'] for r in old_curves}
                new_ids = {r['task_id'] for r in new_curves}
                paired_versions[f'{later}-minus-{earlier}/{v}/{rep}'] = dict(earlier_valid_tasks=len(old_ids), later_valid_tasks=len(new_ids), newly_complete_tasks=sorted(new_ids - old_ids), no_longer_complete_tasks=sorted(old_ids - new_ids), paired_common_complete=contrast(new_curves, old_curves), note='The paired effect uses only tasks complete in both versions. Newly recovered curves are reported separately; changes in all-valid aggregate rates also reflect composition.')
    write(a / 'paired_version_comparisons.json', paired_versions)
    prior_curves = {(r['task_id'], r['variant'], r['representation']): r for r in curves_by_version['original_8k_old_scorer'] if r['model_key'] == 'qwen3'}
    same_context_curve_changes = []
    same_context_curves_checked = 0
    for r in curves_by_version['original_8k_shared_prefix_scorer']:
        key = (r['task_id'], r['variant'], r['representation'])
        if r['model_key'] != 'qwen3' or key not in prior_curves:
            continue
        same_context_curves_checked += 1
        differences = {metric: dict(old=prior_curves[key][metric], new=r[metric]) for metric in METRICS if prior_curves[key][metric] != r[metric]}
        if differences:
            same_context_curve_changes.append(dict(task_id=key[0], variant=key[1], representation=key[2], differences=differences))
    write(a / 'same_context_curve_metric_changes.json', dict(compared_complete_curves_with_forms=same_context_curves_checked, changed_curves_with_forms=len(same_context_curve_changes), changes=same_context_curve_changes, note='Only original natural reasoning and tasks complete under both scorers; every metric is checked per curve, so offsetting aggregate changes cannot hide individual differences.'))
    write(a / 'continuation_outcomes.json', continuation_rows)
    write(a / 'same_context_scorer_changes.json', changes)
    write(a / 'missing_score_conditions.json', missing)
    write(a / 'score_validation_failures.json', validation)
    with (a / 'metric_summary.csv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=tables[0])
        w.writeheader()
        w.writerows(tables)
    calls = [r for p in (o / 'runtime').glob('engine_calls_*.jsonl') for r in read_lines(p)]
    completed = [r for r in calls if r.get('event') == 'completed']
    submitted = [r for r in calls if r.get('event') == 'submitted']
    copied_runtime_jobs = {cp.load(p)['job_id'] for p in (o / 'runtime').glob('*.json')}
    log_coverage = copied_runtime_jobs.issubset({r['job_id'] for r in submitted})
    ledger = read_lines(o / 'submissions.jsonl') if (o / 'submissions.jsonl').exists() else []
    charged = sum((r['reserved_gpu_seconds'] for r in ledger))
    reconciled_jobs = {r['job_id'] for r in ledger if r['stage'] == 'reconciliation'}
    reconciled_charge = sum((r['charged_gpu_seconds'] for r in ledger if r['stage'] == 'reconciliation'))
    memory = []
    for p in o.glob('memory_*.txt'):
        for line in p.read_text().splitlines():
            parts = [x.strip() for x in line.split(',')]
            if len(parts) == 3 and parts[1].isdigit():
                memory.append(int(parts[1]))
    usage = dict(prior_conservative_gpu_seconds=17925, new_charged_or_reserved_gpu_seconds=charged, combined_cap=28800, new_reconciled_actual_gpu_seconds=reconciled_charge, active_job_reserved_gpu_seconds=sum((r['reserved_gpu_seconds'] for r in ledger if r['job_id'] not in reconciled_jobs)), remaining_before_unreconciled_job_consumption_gpu_seconds=10875 - reconciled_charge, remaining_conservative_budget_seconds=10875 - charged, original_budget_not_reset=True, scheduler_ledger=ledger, all_jobs_reconciled=all((any((x['job_id'] == r['job_id'] and x['stage'] == 'reconciliation' for x in ledger)) for r in ledger)), continuation_request_attempts=sum((r['logical_requests'] for r in submitted if r['phase'] == 'continuation')), shared_prefix_score_request_attempts=sum((r['logical_requests'] for r in submitted if r['phase'] == 'shared_prefix_score')), known_forward_calls=sum((r['meter']['forward_calls'] for r in completed if r.get('meter'))), scheduled_token_positions=sum((r['meter']['scheduled_token_positions'] for r in completed if r.get('meter'))), forward_count_complete=log_coverage and len(submitted) == len(completed) and all((r.get('meter') for r in completed)), engine_log_coverage_complete_for_copied_runtime_jobs=log_coverage, request_count_note='Known driver attempts from copied engine logs. Live immutable-only copies can precede mutable logs; final full pull and terminal accounting are required for complete counts.', observed_engine_seconds_by_phase={phase: sum((r['elapsed_seconds'] for r in completed if r['phase'] == phase)) for phase in ('continuation', 'shared_prefix_score')}, additional_reasoning_tokens=sum((t['additional_token_count'] for t in continuation_rows)), ignored_score_tokens=sum((s['scoring_extra_generated_tokens'] for s in scores.values())), sampled_peak_gpu_used_mib=max(memory, default=None), memory_sample_count=len(memory), independent_main_tasks=80, new_score_records=2 * len(scores), natural_contexts_uniformly_rescored=sum((meta[k]['original_natural_end'] for k in scores)))
    if (o / 'host_memory_administrative_change.json').exists():
        usage['host_memory_administrative_change'] = cp.load(o / 'host_memory_administrative_change.json')
    continuation_status = {s: dict(expected=60 if s == 'main' else 2, attempted=sum((t['split'] == s for t in continuation_rows)), natural_complete=sum((t['split'] == s and t['natural_end'] for t in continuation_rows)), terminal_failures=sum((t['split'] == s and (not t['natural_end']) for t in continuation_rows))) for s in ('main', 'smoke')}
    version = 'two_stage_32k_shared_prefix_scorer'
    finished = statuses[version]['main']['valid_contexts'] == 1440 and statuses[version]['smoke']['valid_contexts'] == 54
    execution_finished = len(continuation_rows) == 62 and (not validation) and all((x['reason'] == 'terminal_reasoning_failure' for x in missing))
    identity = {m: dict(spec, reused=m != 'qwen3', reused_from_study=('code_extension_v2' if m in ('qwen', 'mistral') else 'code_extension_v3_five_models') if m != 'qwen3' else None, main_contexts=1440 if m != 'qwen3' else statuses[version]['main']['valid_contexts'], smoke_contexts=54 if m != 'qwen3' else statuses[version]['smoke']['valid_contexts'], status='COMPLETE_REUSED' if m != 'qwen3' else 'COMPLETE' if finished else 'ALL_REQUESTS_ATTEMPTED_WITH_TERMINAL_MISSING' if execution_finished else 'PARTIAL_WITH_EXPLICIT_MISSING', scoring_protocol=spec['protocol'] if m != 'qwen3' else 'native reason-then-score, two-stage 32k, shared conditional prefix scorer') for m, spec in f.MODELS.items()}
    for m, s in identity.items():
        s.update(expected_main_contexts=1440, expected_smoke_contexts=54, main_missing_contexts=1440 - s['main_contexts'], smoke_missing_contexts=54 - s['smoke_contexts'], main_score_records=2 * s['main_contexts'], smoke_score_records=2 * s['smoke_contexts'], independent_main_tasks=80, dtype='bfloat16', quantization=None, new_full_model_float32_audit=False)
    status = dict(created_at=cp.now(), models=identity, versions=statuses, continuations=continuation_status, resources=usage, validation_failures=len(validation), all_score_conditions_complete=finished, all_continuations_attempted=len(continuation_rows) == 62, missing_score_conditions=missing, infrastructure_failure_files=[p.relative_to(ROOT).as_posix() for p in (o / 'failures').glob('*.json')], resume_command='python3 scripts/code_q3_repair_resume.py --stage main --check; only after terminal-job reconciliation and verified infrastructure or previously unattempted work, use --label main_resume_1 --minutes MINUTES_WITHIN_REMAINING_BUDGET --reason CONFIRMED_REASON. Frozen runtime skips saved outputs and never resamples terminal reasoning.', all_authorized_execution_finished=execution_finished and usage['all_jobs_reconciled'], no_remaining_run_required=execution_finished and usage['all_jobs_reconciled'], completion_note='Execution of every authorized request is distinct from complete scoring conditions; native terminal failures are retained and must not be resampled.')
    write(o / 'execution_status.json', status)
    audit = dict(created_at=cp.now(), passed=not validation, accepted_scores_independently_checked=len(scores), independent_cut_enumeration_new_q3_curves=sum((sum((r['model_key'] == 'qwen3' for r in rs)) for v, rs in curves_by_version.items() if v != 'original_8k_old_scorer')), other_model_curve_audits='Reused verified v3 evidence; not counted again as newly enumerated curves', original_prefix_and_context_hash_checks=True, all_natural_contexts_uniformly_scored=sum((meta[k]['original_natural_end'] for k in scores)) == 1432, shared_node_normalization_checked=True, both_forms_all_probability_fields_reconstructed=True, final_context_continuations_rechecked_with_saved_backend_tokenizer=True, expected_subsets=expected, bootstrap_replicates=5000, bootstrap_seed=cp.SEED, original_scorer_sign_flips={rep: sum((r['sign_changed'] for r in changes if r['representation'] == rep)) for rep in ('bare', 'whitespace_union')}, scorer_comparison_contexts=len(changes) // 2, same_context_complete_curves_with_forms_checked=same_context_curves_checked, same_context_complete_curves_with_metric_changes=len(same_context_curve_changes), former_mass_failure=[r for r in changes if r['key'] == 'main/explicit/HumanEval_138__k2'])
    runtime = [cp.load(p) for p in (o / 'runtime').glob('*.json')]
    assert runtime and all((r['engine'] == cp.load(o / 'protocol.json')['engine'] for r in runtime))
    assert all((r['model'] == f.MODELS['qwen3'] and r['dtype'] == 'bfloat16' and (r['quantization'] is None) for r in runtime))
    assert len({json.dumps({k: r[k] for k in ('torch', 'vllm', 'source_files')}, sort_keys=True) for r in runtime}) == 1
    audit['runtime_model_engine_and_installed_scoring_source_consistent'] = True
    audit['runtime_jobs_checked'] = [r['job_id'] for r in runtime]
    write(a / 'independent_audit.json', audit)
    report(o, a, identity, statuses, summaries, common, continuation_status, usage, audit, changes)
    print(json.dumps(dict(continuations=continuation_status, versions=statuses, resources=usage, audit=audit), ensure_ascii=False))

def report(o, a, identity, statuses, summaries, common, cont, usage, audit, changes):
    version = 'two_stage_32k_shared_prefix_scorer'
    summary = summaries[version]
    table = ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', '|---|---|---|---:|---:|---|']
    for m, s in identity.items():
        table.append(f"|{s['model_id']}|{s['revision']}|{s['scoring_protocol']}|{s['main_contexts']}|{s['smoke_contexts']}|{s['status']}|")
    table += ['', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', '|---|---|---|---:|---:|---|---|---|---|---|---|']
    for m in f.MODELS:
        for v in ('original', 'explicit'):
            for rep in ('bare', 'whitespace_union'):
                s = summary[f'{m}/{v}/{rep}']
                x = s['all']
                p = s['partial']
                table.append(f"|{m}|{v}|{rep}|{x['n_tasks']}|{p['n_tasks']}|{estimate(x['accuracy'])}|{estimate(x['strict_accuracy'])}|{estimate(p['trr'])}|{estimate(p['fsrr'])}|{estimate(x['tce'])}|{estimate(x['oracle_min_errors'])}|")
    (a / 'FIVE_MODEL_SUMMARY.md').write_text('\n'.join(table) + '\n', encoding='utf-8')
    statuses_table = ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', '|---|---:|---:|---:|---:|---:|', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.']
    for v, s in statuses.items():
        statuses_table.append(f"|{v}|{s['main']['valid_contexts']}|{s['main']['complete_curves']}|{s['main']['failed_curves']}|{s['main']['missing_curves']}|{s['smoke']['valid_contexts']}|")
    outcomes = 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.'
    final = statuses[version]
    bad = audit['former_mass_failure']
    mass = '；'.join((f"{r['representation']}: {r['old_mass']:.12f} → {r['new_mass']:.12f}" for r in bad))
    qo = summary['qwen3/original/bare']
    qe = summary['qwen3/explicit/bare']
    text = 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.' + '\n'.join(statuses_table) + 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.'
    (a / 'REPORT_ZH.md').write_text(text, encoding='utf-8')
    english = f"# Methods and Results: post-observation Qwen3 supplement\n\n## Methods\n\nWe preserved the delivered 8,192-token study and conducted a separately authorized, prospectively frozen supplement after observing its outcomes. The same 80 main and three smoke tasks, candidate programs, eight tests, verified truth values, two templates, and nine thresholds were retained. Four immediate-readout models were reused without additional inference. Qwen3-30B-A3B-Thinking-2507 revision 144afc2f379b542fdd4e85a1fcd5e1f79112d95d used unquantized bfloat16 native reason-then-score.\n\nEvery originally length-truncated prefix (60 main, two smoke) was eligible for exactly one continuation, using the original prompt IDs followed by the saved 8,192 generated IDs. The additional cap was 24,576 tokens (32,768 total reasoning tokens), temperature 0.6, top-p 0.95, top-k 20, and min-p 0. Each task/template/threshold received a frozen deterministic continuation seed. We inserted no continuation instruction or closing marker. Only native completion at </think> qualified for scoring. Because the original sampler RNG state was unavailable, this is a two-stage seeded continuation protocol, not an uninterrupted single-shot 32k run. Raw failures remain in the denominator.\n\nWe uniformly rescored every originally natural context and every naturally finished continuation. Bare A/B and the union of bare, space-prefixed, and newline-prefixed answer paths shared the same reasoning context. Paths were resolved contextually, deduplicated, and checked for semantic confusion and prefix overlap. At each conditional prefix, all needed outgoing log probabilities were extracted from one full-vocabulary float32 log-softmax on bf16 model logits using vLLM raw_logprobs and logprob_token_ids. The additional sampled scoring token was discarded and counted separately. Event probabilities were reconstructed independently from stored conditional components; the pre-existing 1.00001 probability-mass validation limit was unchanged.\n\nWe retained the v2 accuracy, k=1–8 accuracy, TRR, FSRR, TCE, and oracle minimum-error definitions, with independent cut-point enumeration. All80, nontrivial54, and partially correct23 task denominators are explicit. Paired comparisons use common task IDs while retaining all nine thresholds within each bootstrap unit (5,000 replicates, seed{cp.SEED}). Each model's all-valid set and the five-model common-complete set are reported separately. The original 8k scores, repaired 8k scores, and two-stage32k scores are distinct versions.\n\nFollowing v2, recovery sensitivities excluding k=0 are defined on z=1–7; k=1–8 accuracy is evaluated on all valid curves. Template contrasts are explicit minus original and answer-form contrasts are union minus bare. Version contrasts pair only tasks complete in both versions and list newly completed tasks separately; changes in all-valid aggregate rates can also reflect changing composition.\n\n## Results\n\nThe supplement attempted {cont['main']['attempted']}/60 main continuations, of which {cont['main']['natural_complete']} completed naturally and {cont['main']['terminal_failures']} remained unsuccessful; smoke outcomes were {cont['smoke']['natural_complete']} natural completions among {cont['smoke']['attempted']}/2 attempts. We uniformly rescored {usage['natural_contexts_uniformly_rescored']}/1,432 originally natural contexts. The final supplemental Qwen3 dataset contains {final['main']['valid_contexts']}/1,440 valid main contexts ({final['main']['complete_curves']}/160 complete task-template curves) and {final['smoke']['valid_contexts']}/54 smoke contexts. It retains {final['main']['missing_contexts']} main and {final['smoke']['missing_contexts']} smoke missing scoring contexts.\n\nThe uniform scorer comparison covered {audit['scorer_comparison_contexts']} identical saved reasoning contexts and observed {audit['original_scorer_sign_flips']['bare']} bare and {audit['original_scorer_sign_flips']['whitespace_union']} union sign changes. The formerly invalid mass row is retained in the old version and independently evaluated in the new version: {mass or 'not yet scored'}. Full effect estimates and task-bootstrap confidence intervals are in FIVE_MODEL_SUMMARY.md and metric_summary.csv; task-level paired contrasts and common-set estimates accompany them as JSON.\n\nUnder bare scoring, Qwen3 accuracy on each template's all-valid set was {estimate(qo['all']['accuracy'])} for original and {estimate(qe['all']['accuracy'])} for explicit. Partially correct-subset FSRR was {estimate(qo['partial']['fsrr'])} for original ({qo['partial']['n_tasks']}/23 valid tasks) and {estimate(qe['partial']['fsrr'])} for explicit ({qe['partial']['n_tasks']}/23 valid tasks). These all-valid estimates are distinct from the paired common-task contrasts.\n\n## Limitations\n\nThis is a post-observation supplement; continuation uses a new stage seed and does not reproduce the original sampler state. A single reasoning draw per condition means task bootstrap intervals do not quantify repeated-reasoning variability. Four immediate-readout models plus one native reasoning model cannot isolate the causal effect of reasoning. No new full-model float32 audit was performed. Failed or missing curves are reported explicitly rather than excluded without denominators.\n\nThe scorer comparison changes both the extraction API and batching arrangements, so numerical differences cannot be attributed exclusively to one implementation component. Shared-node normalization and identical saved natural reasoning contexts are directly verified.\n"
    (a / 'METHODS_RESULTS_EN.md').write_text(english, encoding='utf-8')
if __name__ == '__main__':
    analyze()
