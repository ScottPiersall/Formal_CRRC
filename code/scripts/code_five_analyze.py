# Review adaptation: historical non-English narrative strings omitted; numerical routines retained.
"""Five-model, task-clustered analysis with explicit missing-data denominators."""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, math, pathlib, re, sys, zipfile, subprocess
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from formalcrrc import code_five as f, code_extension as cp
from formalcrrc.code_extension_analysis import metric_row, METRICS, ci, summarize_rows

def write(path, data):
    cp.snapshot(path, data)

class ProbabilityMassError(AssertionError):
    """Preserved numeric failure; the pre-existing mass tolerance is unchanged."""

def audit_new(record):
    assert record['status'] == 'ok' and record['dtype'] == 'bfloat16' and (record['quantization'] is None)
    assert record['revision'] == f.MODELS[record['model_key']]['revision']
    events = record['answer_events']
    assert set(events) == set(f.FORMS)
    dedup = {}
    nodes = {tuple(n['token_ids']): {int(k): v for k, v in n['next_log_probabilities'].items()} for n in record.get('prefix_nodes', [])}
    for name, event in events.items():
        path = tuple(event['token_ids'])
        value = sum((nodes[path[:j]][token] for j, token in enumerate(path))) if nodes else sum(event['token_conditional_log_probabilities'])
        assert math.isclose(value, event['log_probability'], rel_tol=0, abs_tol=1e-10)
        assert math.isclose(math.exp(value), event['probability'], rel_tol=0, abs_tol=1e-10)
        if path in dedup:
            assert dedup[path] == (name[-1], value)
        dedup[path] = (name[-1], value)
    assert not any((len(a) < len(b) and b[:len(a)] == a for a in dedup for b in dedup))
    totals = {label: sum((math.exp(value) for tag, value in dedup.values() if tag == label)) for label in ('A', 'B')}
    union = record['scores']['whitespace_union']
    assert math.isclose(union['margin'], math.log(totals['A']) - math.log(totals['B']), rel_tol=0, abs_tol=1e-09)
    assert math.isclose(union['p_met'], totals['A'] / sum(totals.values()), rel_tol=0, abs_tol=1e-09)
    assert math.isclose(union['total_probability_mass'], sum(totals.values()), rel_tol=0, abs_tol=1e-09)
    bare = record['scores']['bare']
    la = events['A']['log_probability']
    lb = events['B']['log_probability']
    assert math.isclose(bare['margin'], la - lb, rel_tol=0, abs_tol=1e-05)
    for score in record['scores'].values():
        assert score['margin'] == score['score_met'] - score['score_not_met'] and math.isfinite(score['margin'])
    if sum(totals.values()) > 1.00001:
        raise ProbabilityMassError(f'Disjoint event mass {sum(totals.values()):.17g} exceeds existing check 1.00001')

def get_accounting(o):
    ledger = [json.loads(s) for s in (o / 'submissions.jsonl').read_text().splitlines()] if (o / 'submissions.jsonl').exists() else []
    jobs = {}
    for path in sorted((o / 'accounting').glob('*.txt')):
        for line in path.read_text().splitlines():
            parts = line.split('|')
            if len(parts) != 6 or not parts[0].isdigit():
                continue
            job, state, exit_code, elapsed, tres, node = parts
            match = re.search('(?:^|,)gres/gpu=(\\d+)(?:,|$)', tres)
            jobs[job] = dict(state=state, exit_code=exit_code, elapsed_seconds=int(elapsed), gpu_count=int(match.group(1)) if match else 0, node=node)
    actual = sum((j['elapsed_seconds'] * j['gpu_count'] for j in jobs.values()))
    reserved = sum((j['reserved_gpu_seconds'] for j in ledger))
    return dict(jobs=jobs, submissions=ledger, new_allocated_gpu_seconds=actual, prior_v2_allocated_gpu_seconds=6824, combined_gpu_seconds=actual + 6824, combined_cap_gpu_seconds=28800, reserved_v3_gpu_seconds=reserved, remaining_unreserved_original_cap_gpu_seconds=21976 - reserved, all_observed_jobs_terminal=all((not any((s in j['state'] for s in ('RUNNING', 'PENDING', 'COMPLETING'))) for j in jobs.values())), remaining_after_v2_cap_seconds=21976, initial_conservative_reservation_remainder_seconds=10800, budget_reconciliation_source='budget_administrative_reconciliation.json', original_cap_not_reset=True)

def audit_strict_recovery(m, z, metric):
    if not 1 <= z <= 7:
        return
    enabled = set()
    events = [set()]
    for cut in sorted(set(m[1:]), reverse=True):
        enabled = enabled | {k for k in range(1, 9) if m[k] == cut}
        events.append(enabled)
    truth = set(range(1, z + 1))
    fsrr = any((event == truth for event in events))
    trr = any((next((k for k in range(1, 9) if k not in event), 9) == z + 1 for event in events))
    assert metric['strict_fsrr'] == fsrr and metric['strict_trr'] == trr

def summarize(selected):
    result = summarize_rows(selected)
    for subset, r in result.items():
        r['expected_n_tasks'] = 80 if subset == 'all' else None
    return result

def contrast(left, right):
    a = {r['task_id']: r for r in left}
    b = {r['task_id']: r for r in right}
    ids = sorted(a.keys() & b.keys())
    groups = {'all': ids, 'nontrivial': [t for t in ids if a[t]['z'] < 8], 'partial': [t for t in ids if 1 <= a[t]['z'] <= 7]}
    return {g: dict(n_paired_tasks=len(ts), task_ids=ts, **{m: ci([a[t][m] - b[t][m] for t in ts if a[t][m] is not None and b[t][m] is not None]) for m in METRICS}) for g, ts in groups.items()}

def analyze():
    o = f.out(ROOT)
    a = o / 'analysis'
    a.mkdir(exist_ok=True)
    inputs = cp.load(o / 'inputs.json')['rows']
    main = [r for r in inputs if r['split'] == 'main']
    tasks = sorted({r['task_id'] for r in main})
    truth = {r['task_id']: r['z'] for r in main}
    expected_subsets = {'all': 80, 'nontrivial': sum((z < 8 for z in truth.values())), 'partial': sum((1 <= z <= 7 for z in truth.values()))}
    assert len(tasks) == 80 and expected_subsets['partial'] == 23
    curves = []
    raw = []
    statuses = {}
    audit_counts = {}
    missing_conditions = []
    validation_failures = []
    v2 = ROOT / 'artifacts/code_extension_v2'
    for model, spec in f.MODELS.items():
        prepared = {(r['task_id'], r['variant'], r['k']): r for r in cp.load(o / 'prompts' / f'{model}.json')['rows']} if (o / 'prompts' / f'{model}.json').exists() else {}
        model_freeze_hash = cp.sha((o / f'freeze_{model}.json').read_bytes()) if (o / f'freeze_{model}.json').exists() else None
        counts = dict(expected_main_contexts=1440, expected_main_score_records=2880, expected_smoke_contexts=54, expected_smoke_score_records=108, main_contexts=0, main_score_records=0, smoke_contexts=0, smoke_score_records=0, reasoning_generation_requests_main=0, raw_saved_main_contexts=0, raw_saved_smoke_contexts=0, validation_failed_main_contexts=0, validation_failed_smoke_contexts=0, reasoning_generation_requests_smoke=0, reasoning_tokens_main=0, reasoning_tokens_smoke=0, natural_reasoning_main=0, natural_reasoning_smoke=0, truncated_or_wrong_boundary_main=0, truncated_or_wrong_boundary_smoke=0, known_forward_calls=0, forward_calls_fully_observed=model != 'qwen3', scoring_path_requests=0, scoring_extra_generated_tokens=0, evaluated_unique_answer_paths=0, independent_main_tasks=80, measured_context_scoring_wall_seconds=0, peak_torch_allocated_bytes=0, peak_torch_reserved_bytes=0, main_complete_curves=0, main_failed_curves=0, main_missing_curves=0, expected_main_curves=160, smoke_complete_curves=0, smoke_failed_curves=0, smoke_missing_curves=0, expected_smoke_curves=6)
        values = {}
        failed = set()
        sources = {}
        for attempt_path in (o / 'attempts' / model).rglob('*.json'):
            attempt = cp.load(attempt_path)
            if attempt.get('status') != 'score_saved':
                failed.add((attempt['task_id'], attempt['variant']))
        for row in inputs:
            validation_error = None
            task, variant, k, split = (row['task_id'], row['variant'], row['k'], row['split'])
            idx = (task, variant, k)
            if spec['reused']:
                bare_path = cp.score_path(v2, split, model, variant, 'bfloat16', task, k)
                union_path = v2 / 'scores_answer_forms' / split / model / variant / f'{cp.score_key(task, k)}.json'
                bare = cp.load(bare_path)
                union = cp.load(union_path)
                record = dict(scores={'bare': bare, 'whitespace_union': union}, forward_calls=bare.get('forward_calls', 1) + union['forward_calls'])
                sources[idx] = {'bare': str(bare_path.relative_to(ROOT)), 'whitespace_union': str(union_path.relative_to(ROOT))}
            else:
                path = f.score_path(ROOT, model, row)
                if model == 'qwen3' and f.trace_path(ROOT, row).exists():
                    trace = cp.load(f.trace_path(ROOT, row))
                    counts[f'reasoning_generation_requests_{split}'] += 1
                    counts[f'reasoning_tokens_{split}'] += trace['reasoning_token_count']
                    assert trace['generation_seed'] == row['generation_seed'] and len(trace['raw_generated_token_ids']) == trace['reasoning_token_count']
                    counts[f'natural_reasoning_{split}'] += int(trace['natural_end'])
                    counts[f'truncated_or_wrong_boundary_{split}'] += int(not trace['natural_end'])
                    if not trace['natural_end']:
                        failed.add((task, variant))
                    else:
                        inv = cp.load(o / 'inventory_qwen3.json')
                        assert trace['raw_generated_token_ids'][-1] == inv['think_end_id']
                        prompt = prepared[idx]
                        assert trace['final_context_token_ids'] == prompt['prompt_token_ids'] + trace['raw_generated_token_ids'] + inv['separator_ids']
                if not path.exists():
                    reason = 'not_scored'
                    if model == 'qwen3':
                        reason = 'not_generated' if not f.trace_path(ROOT, row).exists() else 'decision_score_missing' if trace['natural_end'] else 'truncated_or_wrong_boundary'
                    missing_conditions.append(dict(model_key=model, task_id=task, variant=variant, k=k, split=split, z=row['z'], reason=reason, score_forms_missing=['bare', 'whitespace_union']))
                    continue
                record = cp.load(path)
                try:
                    audit_new(record)
                except ProbabilityMassError as exc:
                    validation_error = str(exc)
                assert record['user_message_sha256'] == row['user_message_sha256'] and record['z'] == row['z']
                assert record['freeze_sha256'] == model_freeze_hash
                if model == 'qwen3':
                    assert record['reasoning_sha256'] == cp.sha(f.trace_path(ROOT, row).read_bytes())
                sources[idx] = {rep: str(path.relative_to(ROOT)) for rep in record['scores']}
            counts[f'raw_saved_{split}_contexts'] += 1
            counts['known_forward_calls'] += record.get('forward_calls') or 0
            counts['scoring_path_requests'] += record.get('scoring_requests', 0)
            if not spec['reused']:
                counts['evaluated_unique_answer_paths'] += len({tuple(e['token_ids']) for e in record['answer_events'].values()})
            counts['scoring_extra_generated_tokens'] += record.get('scoring_extra_generated_tokens', 0)
            counts['measured_context_scoring_wall_seconds'] += record.get('elapsed_seconds', 0)
            counts['peak_torch_allocated_bytes'] = max(counts['peak_torch_allocated_bytes'], record.get('peak_gpu_memory_allocated_bytes', 0))
            counts['peak_torch_reserved_bytes'] = max(counts['peak_torch_reserved_bytes'], record.get('peak_gpu_memory_reserved_bytes', 0))
            if validation_error is not None:
                counts[f'validation_failed_{split}_contexts'] += 1
                failed.add((task, variant))
                failure = dict(model_key=model, task_id=task, variant=variant, k=k, split=split, z=row['z'], reason='probability_mass_validation_failure', source=str(path.relative_to(ROOT)), source_sha256=cp.sha(path.read_bytes()), error=validation_error, observed_mass=record['scores']['whitespace_union']['total_probability_mass'], unchanged_mass_limit=1.00001, score_forms_missing=['bare', 'whitespace_union'], raw_preserved=True, retried=False)
                validation_failures.append(failure)
                missing_conditions.append(failure)
                continue
            counts[f'{split}_contexts'] += 1
            counts[f'{split}_score_records'] += 2
            for rep, s in record['scores'].items():
                values[task, variant, k, rep] = s['margin']
                raw.append(dict(model_key=model, **spec, task_id=task, split=split, variant=variant, k=k, z=row['z'], representation=rep, margin=s['margin'], p_met=s['p_met'], source=sources[idx][rep], newly_generated=not spec['reused']))
        for task in tasks:
            for variant in ('original', 'explicit'):
                complete = all(((task, variant, k, 'bare') in values for k in range(9)))
                counts['main_complete_curves'] += int(complete)
                counts['main_failed_curves'] += int(not complete and (task, variant) in failed)
                counts['main_missing_curves'] += int(not complete and (task, variant) not in failed)
                for rep in ('bare', 'whitespace_union'):
                    if not all(((task, variant, k, rep) in values for k in range(9))):
                        continue
                    m = [values[task, variant, k, rep] for k in range(9)]
                    metric = metric_row(task, model, variant, 'bfloat16', m, truth[task])
                    metric.update(representation=rep, margins=m)
                    audit_strict_recovery(m, truth[task], metric)
                    curves.append(metric)
        for task in sorted({r['task_id'] for r in inputs if r['split'] == 'smoke'}):
            for variant in ('original', 'explicit'):
                complete = all(((task, variant, k, 'bare') in values for k in range(9)))
                counts['smoke_complete_curves'] += int(complete)
                counts['smoke_failed_curves'] += int(not complete and (task, variant) in failed)
                counts['smoke_missing_curves'] += int(not complete and (task, variant) not in failed)
        counts.update(main_missing_contexts=1440 - counts['main_contexts'], main_missing_score_records=2880 - counts['main_score_records'], smoke_missing_contexts=54 - counts['smoke_contexts'], complete_curve_fraction=counts['main_complete_curves'] / 160)
        counts['infrastructure_and_scoring_failure_records'] = len(list((o / 'attempts' / model).rglob('*.json')))
        if model == 'qwen3':
            engine = [json.loads(line) for p in (o / 'runtime').glob('qwen3_engine_calls_*.jsonl') for line in p.read_text().splitlines()]
            counts['engine_forward_calls'] = sum((r['forward_calls'] for r in engine))
            counts['known_forward_calls'] = counts['engine_forward_calls']
            counts['forward_calls_by_split_and_phase'] = {f'{s}/{p}': sum((r['forward_calls'] for r in engine if r['split'] == s and r['phase'] == p)) for s in ('smoke', 'main') for p in ('reasoning', 'decision')}
            counts['engine_scheduled_token_positions'] = sum((r['scheduled_token_positions'] for r in engine))
            counts['measured_successful_batched_wall_seconds'] = {f'{s}/{p}': sum((r['elapsed_seconds'] for r in engine if r['split'] == s and r['phase'] == p and (r['status'] == 'ok'))) for s in ('smoke', 'main') for p in ('reasoning', 'decision')}
            counts['mean_reasoning_tokens_by_split'] = {s: counts[f'reasoning_tokens_{s}'] / counts[f'reasoning_generation_requests_{s}'] if counts[f'reasoning_generation_requests_{s}'] else None for s in ('main', 'smoke')}
            counts['reasoning_tokens_per_observed_engine_second_by_split'] = {s: counts[f'reasoning_tokens_{s}'] / counts['measured_successful_batched_wall_seconds'][f'{s}/reasoning'] if counts['measured_successful_batched_wall_seconds'][f'{s}/reasoning'] else None for s in ('main', 'smoke')}
            counts['measured_context_scoring_wall_seconds'] = None
            counts['peak_torch_allocated_bytes'] = None
            counts['peak_torch_reserved_bytes'] = None
            counts['forward_calls_fully_observed'] = False
            counts['unobserved_failed_batches'] = [dict(job_id='803705', phase='decision', split='smoke', natural_contexts=52, logical_path_requests=312, physical_forward_calls=None, reason='CUDA OOM terminated engine before post-call counter RPC; 54 raw chunk failure rows include two truncated traces that were never decision-scored.')]
            counts['successful_engine_decision_path_requests_by_split'] = {s: sum((r['logical_requests'] for r in engine if r['phase'] == 'decision' and r['split'] == s and (r['status'] == 'ok'))) for s in ('main', 'smoke')}
            counts['saved_reasoning_traces_main'] = counts['reasoning_generation_requests_main']
            counts['saved_reasoning_traces_smoke'] = counts['reasoning_generation_requests_smoke']
            counts['reasoning_request_count_note'] = 'Saved native outputs and completed engine batches are counted here. In-flight requests during RUNNING snapshots are not yet included. Inspect raw attempt records for any failed generation batch before claiming total attempts.'
            counts['engine_reasoning_request_attempts_by_split'] = {s: sum((r['logical_requests'] for r in engine if r['phase'] == 'reasoning' and r['split'] == s)) for s in ('main', 'smoke')}
            memory = []
            for p in (o / 'runtime').glob('qwen3_memory_*.txt'):
                for line in p.read_text(errors='replace').splitlines():
                    fields = [v.strip() for v in line.split(',')]
                    if len(fields) == 3 and fields[1].isdigit() and fields[2].isdigit():
                        memory.append(dict(timestamp=fields[0], used_mib=int(fields[1]), utilization_percent=int(fields[2]), source=p.name))
            counts['gpu_memory_observation'] = dict(samples=len(memory), interval_seconds=5, max_observed_used_mib=max((r['used_mib'] for r in memory), default=None), by_source={source: dict(samples=sum((r['source'] == source for r in memory)), max_observed_used_mib=max((r['used_mib'] for r in memory if r['source'] == source))) for source in sorted({r['source'] for r in memory})}, note='nvidia-smi device memory sampled in the same allocated GPU job; observed maximum, not an exact instantaneous peak; raw timestamps retained.')
            write(a / 'qwen3_gpu_memory_observations.json', memory)
        attempted_all = model == 'qwen3' and counts['reasoning_generation_requests_main'] == 1440 and (counts['reasoning_generation_requests_smoke'] == 54)
        counts['all_reasoning_requests_attempted'] = attempted_all if model == 'qwen3' else None
        counts['all_score_conditions_complete'] = counts['main_contexts'] == 1440 and counts['smoke_contexts'] == 54
        counts['status'] = 'COMPLETE_REUSED' if spec['reused'] else 'COMPLETE' if counts['all_score_conditions_complete'] else 'ALL_REASONING_ATTEMPTED_WITH_MISSING_SCORES' if attempted_all else 'PARTIAL_OR_BLOCKED'
        counts['independent_main_tasks_with_any_score'] = len({task for task, variant, k, rep in values if task in tasks})
        counts['main_curve_counts_by_template'] = {v: dict(expected=80, complete=sum((r['model_key'] == model and r['variant'] == v and (r['representation'] == 'bare') for r in curves)), failed=sum(((t, v) in failed and (not all(((t, v, k, 'bare') in values for k in range(9)))) for t in tasks)), missing=sum(((t, v) not in failed and (not all(((t, v, k, 'bare') in values for k in range(9)))) for t in tasks))) for v in ('original', 'explicit')}
        loads = [cp.load(p) for p in (o / 'runtime').glob(model + '_*.json')]
        counts['measured_model_load_seconds_by_job'] = {r['job_id']: r['load_seconds'] for r in loads if 'load_seconds' in r}
        if spec['reused']:
            counts['measured_context_scoring_wall_seconds'] = None
            counts['peak_torch_allocated_bytes'] = None
            counts['peak_torch_reserved_bytes'] = None
        counts['resource_measurement_note'] = 'Wall measurements and sampled device/allocator peaks; not exclusive GPU kernel duration. See globally charged allocation jobs; shared jobs are not independent per-model GPU allocations.'
        counts['scoring_request_count_note'] = 'Legacy scoring_path_requests sums the saved scoring_requests field: one context-level invocation per immediate-readout record, six distinct teacher-forcing path requests per Q3 record. These are different API units, not independent samples. evaluated_unique_answer_paths separately counts distinct scored paths in new saved records; failed Q3 decision requests are listed in unobserved_failed_batches.'
        if spec['reused']:
            counts['evaluated_unique_answer_paths'] = None
        statuses[model] = dict(**spec, **counts)
        audit_counts[model] = dict(independently_checked_new_contexts=0 if spec['reused'] else counts['main_contexts'] + counts['smoke_contexts'], independently_enumerated_main_curves=sum((r['model_key'] == model for r in curves)))
    grouped = {(m, v, r): [x for x in curves if (x['model_key'], x['variant'], x['representation']) == (m, v, r)] for m in f.MODELS for v in ('original', 'explicit') for r in ('bare', 'whitespace_union')}
    summaries = {'/'.join(k): summarize_rows(v) for k, v in grouped.items()}
    for summary in summaries.values():
        for subset, result in summary.items():
            result['expected_n_tasks'] = expected_subsets[subset]
    common = {}
    common_summary = {}
    pairs = {}
    all_pairs = {}
    for v in ('original', 'explicit'):
        for rep in ('bare', 'whitespace_union'):
            sets = [{r['task_id'] for r in grouped[m, v, rep]} for m in f.MODELS]
            ids = set.intersection(*sets)
            common[v + '/' + rep] = dict(n_tasks=len(ids), expected_tasks=80, task_ids=sorted(ids))
            for m in f.MODELS:
                common_summary[f'{m}/{v}/{rep}'] = summarize_rows([r for r in grouped[m, v, rep] if r['task_id'] in ids])
            for m, n in itertools.combinations(f.MODELS, 2):
                tag = f'{m}-minus-{n}/{v}/{rep}'
                all_pairs[tag] = contrast(grouped[m, v, rep], grouped[n, v, rep])
                pairs[tag] = contrast([r for r in grouped[m, v, rep] if r['task_id'] in ids], [r for r in grouped[n, v, rep] if r['task_id'] in ids])
    representation_pairs = {f'{m}/{v}': contrast(grouped[m, v, 'whitespace_union'], grouped[m, v, 'bare']) for m in f.MODELS for v in ('original', 'explicit')}
    template_pairs = {f'{m}/{r}': contrast(grouped[m, 'explicit', r], grouped[m, 'original', r]) for m in f.MODELS for r in ('bare', 'whitespace_union')}
    write(a / 'curve_metrics.json', curves)
    write(a / 'model_all_valid_summaries.json', summaries)
    joint = set.intersection(*[{r['task_id'] for r in rs} for rs in grouped.values()])
    write(a / 'five_model_common_complete.json', dict(sets=common, summaries=common_summary, joint_all_models_templates_forms=dict(n_tasks=len(joint), task_ids=sorted(joint), summaries={'/'.join(k): summarize_rows([r for r in rs if r['task_id'] in joint]) for k, rs in grouped.items()})))
    write(a / 'paired_model_comparisons.json', dict(direction='left model minus right model', pairwise_valid=all_pairs, five_model_common_complete=pairs))
    write(a / 'paired_representation_comparisons.json', representation_pairs)
    write(a / 'paired_template_comparisons.json', template_pairs)
    write(a / 'missing_score_conditions.json', missing_conditions)
    write(a / 'score_validation_failures.json', validation_failures)
    with (a / 'model_identity_and_completion.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        keys = ['model_key', 'model_id', 'revision', 'protocol', 'reused', 'status', 'main_contexts', 'main_score_records', 'smoke_contexts', 'main_complete_curves', 'main_failed_curves', 'main_missing_curves']
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for model, data in statuses.items():
            writer.writerow({key: model if key == 'model_key' else data[key] for key in keys})
    with (a / 'metric_summary.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=['model_key', 'template', 'representation', 'subset', 'expected_tasks', 'complete_tasks', 'metric', 'endpoint_tasks', 'estimate', 'ci95_low', 'ci95_high'])
        writer.writeheader()
        for tag, summary in summaries.items():
            model, template, representation = tag.split('/')
            for subset, data in summary.items():
                for metric in METRICS:
                    r = data[metric]
                    bounds = r['ci95'] or [None, None]
                    writer.writerow(dict(model_key=model, template=template, representation=representation, subset=subset, expected_tasks=expected_subsets[subset], complete_tasks=data['n_tasks'], metric=metric, endpoint_tasks=r['n'], estimate=r['estimate'], ci95_low=bounds[0], ci95_high=bounds[1]))
    with (a / 'score_index.jsonl').open('w', encoding='utf-8') as handle:
        for r in raw:
            handle.write(json.dumps(r, ensure_ascii=False) + '\n')
    resource = get_accounting(o)
    resource['per_model_allocation_links'] = {m: dict(new_gpu_job_ids=[r['job_id'] for r in resource['submissions'] if r['stage'].startswith('qwen3' if m == 'qwen3' else 'immediate')], allocation_shared_with=['gemma' if m == 'llama' else 'llama'] if m in ('llama', 'gemma') else [], measurements_location=f'models.{m}') if m in f.NEW else dict(reused=True, new_gpu_seconds=0, prior_v2_shared_job_ids=['803090', '803174']) for m in f.MODELS}
    blockers = []
    for m in f.NEW:
        if statuses[m]['main_missing_contexts'] or statuses[m]['smoke_missing_contexts']:
            reasons = {r['reason'] for r in missing_conditions if r['model_key'] == m}
            blockers.append(dict(model=m, missing_main_contexts=statuses[m]['main_missing_contexts'], missing_smoke_contexts=statuses[m]['smoke_missing_contexts'], reason_counts={reason: sum((r['model_key'] == m and r['reason'] == reason for r in missing_conditions)) for reason in sorted(reasons)}, recoverable_without_resampling=bool(reasons - {'truncated_or_wrong_boundary', 'probability_mass_validation_failure'})))
    status = dict(updated_at=cp.now(), status='COMPLETE' if not blockers else 'PARTIAL', models=statuses, resource=resource, blockers=blockers, all_planned_requests_attempted=all((statuses[m]['all_score_conditions_complete'] for m in ('llama', 'gemma'))) and statuses['qwen3']['all_reasoning_requests_attempted'], recoverable_missing_score_conditions=sum((r['reason'] not in ('truncated_or_wrong_boundary', 'probability_mass_validation_failure') for r in missing_conditions)), raw_saved_new_main_contexts=sum((statuses[m]['raw_saved_main_contexts'] for m in f.NEW)), raw_saved_new_main_score_records=2 * sum((statuses[m]['raw_saved_main_contexts'] for m in f.NEW)), validation_failed_contexts=len(validation_failures), new_main_contexts=sum((statuses[m]['main_contexts'] for m in f.NEW)), new_main_score_records=sum((statuses[m]['main_score_records'] for m in f.NEW)), new_smoke_contexts=sum((statuses[m]['smoke_contexts'] for m in f.NEW)), new_smoke_score_records=sum((statuses[m]['smoke_score_records'] for m in f.NEW)), qwen3_main_reasoning_requests=statuses['qwen3']['reasoning_generation_requests_main'], recovery_commands=[f'python scripts/code_five_CLUSTER status', 'python scripts/code_five_CLUSTER pull', 'python scripts/code_five_analyze.py', 'Only if recoverable missing requests remain, after checking remaining authorization and no active duplicate job: inside the authorized allocation run bash slurm/run_code_five_gpu.sbatch qwen3. This sets the finite scheduler deadline and invokes the frozen runtime-v3 driver. Never resample terminal truncated traces or repeat quarantined numerical score failures. See docs/CODE_FIVE_MODELS.md.'], limitations=['Post-v2 model extension; not original v2 preregistration', 'Four immediate-readout models and one native reason-then-score model; no identified causal reasoning effect', 'One sampled reasoning per Q3 row: task bootstrap does not estimate repeated-reasoning variability', 'No new float32 audit; only v2 Qwen2.5 and Mistral ten-task audit retained', 'Q3 successful batched forward invocations are observed; the 803705 OOM decision batch has unavailable physical forward counts. Shared per-row calls are not fabricated. Initialization/warmup excluded but allocated time included.', 'Original 54 Q3 smoke traces preserve batch elapsed time but per-request latency was unavailable. Main completion receipts measure caller-observed submission-to-finish latency including batch scheduling, not exclusive GPU compute.'])
    write(o / 'execution_status.json', status)
    write(a / 'independent_audit.json', dict(passed=not validation_failures, all_accepted_scores_passed=True, preserved_probability_mass_failures=len(validation_failures), counts=audit_counts, expected_subsets=expected_subsets, bootstrap_replicates=5000, bootstrap_seed=20260911))
    report(o, status, summaries, common, expected_subsets)
    comparison_tables(o)
    print(json.dumps(dict(status=status['status'], new_main_contexts=status['new_main_contexts'], new_smoke_contexts=status['new_smoke_contexts'], models={m: statuses[m]['main_contexts'] for m in f.MODELS})))

def val(s, key, digits=3):
    r = s[key]
    if r['estimate'] is None:
        return '—'
    return f"{r['estimate']:.{digits}f} [{r['ci95'][0]:.{digits}f}, {r['ci95'][1]:.{digits}f}]"

def comparison_tables(o):
    common = cp.load(o / 'analysis/five_model_common_complete.json')
    pairs = cp.load(o / 'analysis/paired_model_comparisons.json')
    lines = ['# Common complete sets and paired model comparisons', '', 'All estimates use complete nine-threshold task curves. The planned task panel remains 80 (23 partially correct); each row explicitly gives its valid denominator. JSON files also retain task identifiers and all metrics. Intervals are 95% task-bootstrap intervals.', '']
    for subset, expected in [('all', 80), ('nontrivial', 54), ('partial', 23)]:
        lines += [f'## Five-model common complete set: {subset}, planned N={expected}', '', '|Model/template/form|Valid tasks|Accuracy|k>=1 accuracy|TRR|FSRR|TCE|Oracle minimum errors|', '|---|---:|---|---|---|---|---|---|']
        for tag, summary in common['summaries'].items():
            s = summary[subset]
            lines.append(f"|{tag}|{s['n_tasks']}/{expected}|{val(s, 'accuracy')}|{val(s, 'strict_accuracy')}|{val(s, 'trr')} (n={s['trr']['n']})|{val(s, 'fsrr')} (n={s['fsrr']['n']})|{val(s, 'tce')}|{val(s, 'oracle_min_errors')}|")
        lines.append('')
    joint = common['joint_all_models_templates_forms']
    lines += [f"## Joint complete set across all five models, both templates and both forms: {joint['n_tasks']}/80", '', 'Task IDs: ' + ', '.join(joint['task_ids']), '', '|Model/template/form|All-task accuracy|Partial valid tasks/23|Partial TRR|Partial FSRR|', '|---|---|---:|---|---|']
    for tag, s in joint['summaries'].items():
        lines.append(f"|{tag}|{val(s['all'], 'accuracy')}|{s['partial']['n_tasks']}/23|{val(s['partial'], 'trr')}|{val(s['partial'], 'fsrr')}|")
    for scope in ('pairwise_valid', 'five_model_common_complete'):
        lines += ['', f'## Paired differences: {scope}', '', 'Direction is the left model minus the right model. Pairwise-valid comparisons and five-model-common comparisons are distinct analyses.', '', '|Pair/template/form|Paired all tasks/80|Accuracy difference|TCE difference|Paired partial tasks/23|TRR difference|FSRR difference|', '|---|---:|---|---|---:|---|---|']
        for tag, s in pairs[scope].items():
            lines.append(f"|{tag}|{s['all']['n_paired_tasks']}/80|{val(s['all'], 'accuracy')}|{val(s['all'], 'tce')}|{s['partial']['n_paired_tasks']}/23|{val(s['partial'], 'trr')}|{val(s['partial'], 'fsrr')}|")
    (o / 'analysis/COMMON_COMPLETE_AND_PAIRED.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def report(o, status, summaries, common, denoms):
    metered = status['models']['qwen3'].get('forward_calls_fully_observed', False)
    forward_note = 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.'
    identity = ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', '|---|---|---|---:|---:|---:|---|']
    for m, s in status['models'].items():
        identity.append(f"|{s['model_id']}|`{s['revision']}`|{s['protocol']}|{s['main_contexts']}|{s['main_score_records']}|{s['smoke_contexts']}|{s['status']}|")
    table = ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', '|---|---|---|---:|---|---|---|---|---|']
    for m in f.MODELS:
        for v in ('original', 'explicit'):
            for rep in ('bare', 'whitespace_union'):
                s = summaries[f'{m}/{v}/{rep}']
                p = s['partial']
                a = s['all']
                table.append(f"|{m}|{v}|{rep}|{p['n_tasks']}|{val(a, 'accuracy')}|{val(a, 'strict_accuracy')}|{val(p, 'trr')}|{val(p, 'fsrr')}|{val(a, 'tce')}|")
    joined = '\n'.join(identity)
    results = '\n'.join(table)
    observed = []
    observed_en = []
    for model in f.MODELS:
        original = summaries[f'{model}/original/bare']['partial']
        explicit = summaries[f'{model}/explicit/bare']['partial']

        def fraction(result):
            metric = result['fsrr']
            return f"{round(metric['estimate'] * metric['n'])}/{metric['n']}" if metric['estimate'] is not None else 'NA/0'
        observed.append(f'{model}：original {fraction(original)}，explicit {fraction(explicit)}')
        observed_en.append(f'{model}: original {fraction(original)}, explicit {fraction(explicit)}')
    findings = '；'.join(observed)
    findings_en = '; '.join(observed_en)
    identity_en = '\n'.join(['|Model|Revision|Readout protocol|Main contexts /1440|Main score records /2880|Smoke contexts /54|Status|', identity[1]] + identity[2:])
    results_en = '\n'.join(['|Model|Template|Score form|Valid partial-correct tasks /23|Accuracy (all valid tasks)|Accuracy k=1..8|TRR (partial)|FSRR (partial)|TCE (all valid tasks)|', table[1]] + table[2:])
    (o / 'analysis/FIVE_MODEL_SUMMARY.md').write_text(joined + 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.' + results + '\n', encoding='utf-8')
    detailed = []
    for subset, description in [('all', 'All main tasks, z=0..8'), ('nontrivial', 'Nontrivial, z=0..7'), ('partial', 'Partially correct, z=1..7')]:
        detailed += [f'## {description}; expected N={denoms[subset]}', '', '|Model/template/form|Complete tasks|Accuracy|Accuracy k=1..8|TCE|Oracle minimum errors|TRR|FSRR|', '|---|---:|---|---|---|---|---|---|']
        for tag, summary in summaries.items():
            s = summary[subset]
            detailed.append(f"|{tag}|{s['n_tasks']}/{denoms[subset]}|{val(s, 'accuracy')}|{val(s, 'strict_accuracy')}|{val(s, 'tce')}|{val(s, 'oracle_min_errors')}|{val(s, 'trr')} (n={s['trr']['n']})|{val(s, 'fsrr')} (n={s['fsrr']['n']})|")
        detailed.append('')
    detailed += ['## k=0 and sensitivity excluding k=0', '', 'The k>=1 recovery endpoints use z=1..7 as in v2; endpoint-trivial tasks are not silently added to that denominator. k>=1 accuracy for all tasks is reported above.', '', '|Model/template/form|k=0 accuracy, all complete tasks|k>=1 TRR, partial tasks|k>=1 FSRR, partial tasks|', '|---|---|---|---|']
    for tag, s in summaries.items():
        detailed.append(f"|{tag}|{val(s['all'], 'k0_correct')} (n={s['all']['k0_correct']['n']})|{val(s['partial'], 'strict_trr')} (n={s['partial']['strict_trr']['n']})|{val(s['partial'], 'strict_fsrr')} (n={s['partial']['strict_fsrr']['n']})|")
    (o / 'analysis/DETAILED_METRIC_TABLES.md').write_text('\n'.join(detailed) + '\n', encoding='utf-8')
    counts = '\n'.join(('Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.' for m, s in status['models'].items()))
    text = 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.'
    (o / 'analysis/REPORT_ZH.md').write_text(text, encoding='utf-8')
    english = f"# Methods / Results draft\n\n## Methods\n\nWe conducted a model extension specified after observing the code_extension_v2 results. It is not part of the original v2 preregistration. We reused exactly 80 main tasks and three disjoint smoke tasks, the frozen Mistral-generated candidates, eight tests and validated outcomes, thresholds k=0,...,8, and the original and explicit templates. The earlier 30 tasks remained development-only. Model identities and exact revisions appear in FIVE_MODEL_SUMMARY.md.\n\nFour models use immediate readout; Qwen3-30B-A3B-Thinking-2507 uses native reason-then-score. The three added models use unquantized bfloat16 weights. Actual full-context answer continuations are verified with native chat templates. Single-token bare continuations use immediate logits; multi-token continuations use complete conditional sequence likelihood. The fixed alternative representation sums distinct, prefix-disjoint token paths for A, space+A and newline+A, and the corresponding B forms, using full-vocabulary probabilities. No form is selected using outcomes.\n\nQwen3 samples exactly one reasoning trace per task/template/threshold using temperature 0.6, top-p 0.95, top-k 20, and an 8192-token cap. SHA256 maps immutable identifiers to fixed per-request seeds before inference. Generation must naturally reach the native end-of-thinking token; the decision context adds the native two-newline separator. Bare and union scores share that exact trace. Raw generated IDs, text, termination evidence and decision contexts are retained. Truncations and wrong boundaries are not force-closed or resampled.\n\nWe retain v2 accuracy, k>=1 accuracy, TRR, FSRR, TCE, oracle minimum errors and sensitivity excluding k=0. Analyses report all 80 tasks, the {denoms['nontrivial']}-task nontrivial subset (z=0,...,7), and 23 partial-correct tasks (z=1,...,7). Percentile 95% confidence intervals use 5,000 task-cluster bootstrap replicates (seed 20260911), preserving all nine thresholds within each resampled task. Contrasts pair identical tasks. Model-specific valid sets and the five-model common complete set are reported separately, with complete, failed and missing denominators. Independent score-cut enumeration and saved-probability reconstruction audit recovery metrics and answer-form aggregation.\n\nThe margin is the log-probability difference between acceptance and rejection events; the uncalibrated decision accepts at margin >= 0. Ground truth accepts exactly when k <= z. TCE is the absolute distance between the first rejected threshold and z+1, taking the first rejection as 9 if all thresholds are accepted. TRR asks whether any score cut recovers that first rejection; FSRR requires one cut to recover the entire nine-decision pattern. Oracle minimum errors minimizes the number of incorrect decisions across all cuts. Tied scores move together. Recovery endpoints exclude z=8; the k>=1 recovery sensitivity uses z=1,...,7, as in v2.\n\n## Results\n\nExecution status is {status['status']}. Newly obtained main contexts: {status['new_main_contexts']}/4320; corresponding score records: {status['new_main_score_records']}/8640. New smoke contexts: {status['new_smoke_contexts']}/162. Qwen3 main reasoning requests: {status['qwen3_main_reasoning_requests']}/1440. Detailed estimates and intervals are provided in FIVE_MODEL_SUMMARY.md and the JSON outputs; missing conditions have no imputed estimates. Per-model completion counts are shown below.\n\nThe archive preserves {status['raw_saved_new_main_contexts']} newly saved raw main scoring contexts ({status['raw_saved_new_main_score_records']} form records). Across main and smoke, {status['validation_failed_contexts']} context(s) failed the existing disjoint-event mass check of <=1.00001. Both forms of each affected context are excluded from valid-score counts and complete-curve statistics, while their raw components and hashes remain available. The check was not relaxed, the values were not renormalized, and neither scoring nor reasoning was repeated. This post-observation validation handling is documented in analysis_validation_review.json. The exact numerical cause remains undetermined; no additional precision experiment was performed. Accordingly, the all-raw-scores audit does not pass even though every accepted score passes its checks.\n\nBare FSRR counts on partially correct candidates (successful tasks / valid tasks) are {findings_en}. The planned denominator is 23 per model and template. A smaller denominator explicitly identifies the complete-curve subset, not completion of the planned panel. Observed model differences were retained without adapting tasks, prompts or answer events.\n\n{identity_en}\n\n{results_en}\n\nCommon complete-set estimates and paired differences are separately tabulated in [COMMON_COMPLETE_AND_PAIRED.md](COMMON_COMPLETE_AND_PAIRED.md), including the joint set across all models, templates and forms. The table above uses each model's entire valid set.\n\nThe protocol difference prevents identifying a causal effect of reasoning from the five-model comparison. One Qwen3 reasoning sample per row means the task bootstrap does not quantify repeated-reasoning variability. The prior ten-task float32 audit applies only to Qwen2.5 and Mistral; it was not expanded to the three new models. A separately frozen observability adapter counts positive-token vLLM runner executions by split and phase ({status['models']['qwen3'].get('engine_forward_calls', 0)} recorded calls; completeness={metered}); the failed 803705 decision batch lost its final counter after CUDA OOM, so its physical forward count remains unknown. Initialization is excluded from that call count but included in allocated time. Batched executions are not attributed as independent per-row forwards or independent samples. The initial 54 smoke traces retain batch timing but lack per-request latency; main completion receipts capture caller-observed batch submission to request completion, including scheduling. A pre-main runtime amendment capped prefill workspace at 512 tokens after smoke scoring OOM; it did not truncate contexts or change weights, sampling or probability events. Saved smoke reasoning was reused, including two terminal truncations. No claim of successful completion is made for missing conditions.\n"
    (o / 'analysis/METHODS_RESULTS_EN.md').write_text(english, encoding='utf-8')

def package():
    o = f.out(ROOT)
    dest = ROOT.parent / 'FormalCRRC_Code_Five_Models_Handoff.zip'
    paths = [p for p in o.rglob('*') if p.is_file()]
    paths += [p for p in (ROOT / 'artifacts/code_extension_v2').rglob('*') if p.is_file()]
    paths += [ROOT / name for name in cp.load(o / 'v2_dependency_manifest.json')['files'] if not name.startswith('artifacts/code_extension_v2/')]
    paths += list((ROOT / 'src/formalcrrc').glob('code_five*.py')) + list((ROOT / 'scripts').glob('code_five*.py')) + list((ROOT / 'slurm').glob('run_code_five*.sbatch')) + [ROOT / 'tests/test_code_five.py']
    paths += [ROOT / 'docs/CODE_FIVE_MODELS.md', ROOT / 'tests/test_code_five_meter.py', ROOT / 'tests/test_code_five_runtime.py']
    paths = sorted(set(paths) - {o / 'handoff_manifest.json'})
    manifest = {p.relative_to(ROOT).as_posix(): cp.sha(p.read_bytes()) for p in paths}
    for name, expected in cp.load(o / 'v2_dependency_manifest.json')['files'].items():
        assert manifest[name] == expected, f'Historical evidence changed before packaging: {name}'
    write(o / 'handoff_manifest.json', dict(created_at=cp.now(), files=manifest, dependencies_included='v2 full evidence and checked original shared sources; no model weights or credentials'))
    paths.append(o / 'handoff_manifest.json')
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in paths:
            z.write(p, p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for name, expected in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    digest = cp.sha(dest.read_bytes())
    dest.with_suffix('.zip.sha256').write_text(digest + '  ' + dest.name + '\n')
    print('PACKAGED', dest, digest)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--package', action='store_true')
    a = p.parse_args()
    analyze()
    if a.package:
        package()
