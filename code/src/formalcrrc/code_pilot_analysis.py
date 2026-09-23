# Review adaptation: historical non-English narrative strings omitted; numerical routines retained.
"""Task-clustered pilot statistics and evidence-only handoff packaging."""
from __future__ import annotations
import csv
import io
import json
import pathlib
import platform
import subprocess
import zipfile
import numpy as np
from formalcrrc import code_pilot as cp
METRICS = ['accuracy', 'strict_accuracy', 'trr', 'fsrr', 'tce', 'oracle_min_errors']

def ci(values, seed=cp.SEED):
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return dict(n=0, estimate=None, ci95=None)
    rng = np.random.default_rng(seed)
    estimates = a[rng.integers(0, len(a), size=(5000, len(a)))].mean(axis=1)
    return dict(n=len(a), estimate=float(a.mean()), ci95=np.quantile(estimates, [0.025, 0.975]).tolist())

def summarize(rows):
    groups = {'all': rows, 'nontrivial': [r for r in rows if r['z'] < 8], 'partial': [r for r in rows if 1 <= r['z'] <= 7]}
    return {group: {metric: ci([r[metric] for r in selected if r[metric] is not None]) for metric in METRICS} | {'n_tasks': len(selected)} for group, selected in groups.items()}

def fmt(stat):
    if stat['estimate'] is None:
        return 'NA (n=0)'
    a, b = stat['ci95']
    return f"{stat['estimate']:.4f} [{a:.4f}, {b:.4f}], n={stat['n']}"

def accounting(out):
    files = sorted((out / 'accounting').glob('*.txt'))
    jobs = {}
    if files:
        for line in files[-1].read_text().splitlines():
            fields = line.split('|')
            if len(fields) < 7 or not fields[0].isdigit():
                continue
            job, name, state, exitcode, elapsed, tres, node = fields[:7]
            gpu = next((int(v.split('=')[-1]) for v in tres.split(',') if v.startswith('gres/gpu=')), 0)
            jobs[job] = dict(state=state, exit_code=exitcode, elapsed_seconds=int(elapsed or 0), gpu_count=gpu, node=node)
    return dict(jobs=jobs, allocated_gpu_seconds=sum((r['elapsed_seconds'] * r['gpu_count'] for r in jobs.values())), source=str(files[-1].name) if files else None)

def analyze(root):
    root = pathlib.Path(root)
    out = cp.outdir(root)
    report = out / 'analysis'
    report.mkdir(parents=True, exist_ok=True)
    audit = cp.integrity(root)
    cp.snapshot(out / 'integrity.json', audit)
    instances = cp.load(out / 'instances.json') if (out / 'instances.json').exists() else []
    main = [i for i in instances if i['split'] == 'main']
    truth = {}
    gens = {}
    for item in instances:
        task = item['task_id']
        for directory, dest in [('truth', truth), ('generation', gens)]:
            p = out / directory / f'{cp.slug(task)}.json'
            if p.exists():
                dest[task] = cp.load(p)
    rows = []
    margins = {}
    all_scores = []
    for key in cp.MODELS:
        for p in (out / 'scores').glob(f'*/{key}/*.json'):
            all_scores.append(cp.load(p))
        for item in main:
            task = item['task_id']
            t = truth.get(task)
            paths = [out / 'scores/main' / key / f'{cp.score_key(task, k)}.json' for k in range(9)]
            if not t or not t['stable'] or (not all((p.exists() for p in paths))):
                continue
            m = [cp.load(p)['margin'] for p in paths]
            row = dict(task_id=task, model_key=key, **cp.curve_diagnostics(m, t['z'] + 1))
            rows.append(row)
            margins[key, task] = m
    fields = ['task_id', 'model_key', 'z', 'j_star', 'accuracy', 'strict_accuracy', 'errors', 'trr', 'fsrr', 'tce', 'predicted_first_fail', 'oracle_min_errors', 'trr_gap', 'fsrr_gap']
    with (report / 'per_instance_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    summaries = {k: summarize([r for r in rows if r['model_key'] == k]) for k in cp.MODELS}
    validsets = {k: {r['task_id'] for r in rows if r['model_key'] == k} for k in cp.MODELS}
    common = set.intersection(*validsets.values())
    indexed = {(r['model_key'], r['task_id']): r for r in rows}
    paired = {}
    for group in ['all', 'nontrivial', 'partial']:
        tasks = sorted((t for t in common if group == 'all' or (truth[t]['z'] < 8 if group == 'nontrivial' else 1 <= truth[t]['z'] <= 7)))
        paired[group] = {'n_tasks': len(tasks), 'task_ids': tasks, 'qwen_minus_mistral': {}}
        for metric in METRICS:
            differences = [indexed['qwen', t][metric] - indexed['mistral', t][metric] for t in tasks if indexed['qwen', t][metric] is not None and indexed['mistral', t][metric] is not None]
            paired[group]['qwen_minus_mistral'][metric] = ci(differences)
    histogram = {str(z): sum((t['stable'] and t['z'] == z for task, t in truth.items() if task in {i['task_id'] for i in main})) for z in range(9)}
    n_partial = sum((histogram[str(z)] for z in range(1, 8)))
    n_truth = sum(histogram.values())
    counts = {k: dict(partial_fsrr_unrecoverable=sum((r['model_key'] == k and 1 <= r['z'] <= 7 and (r['fsrr'] is False) for r in rows)), partial_trr_yes_fsrr_no=sum((r['model_key'] == k and 1 <= r['z'] <= 7 and (r['trr'] is True) and (r['fsrr'] is False) for r in rows))) for k in cp.MODELS}
    technical = audit['passed'] and n_truth == 30 and (len(rows) == 60)
    if not technical:
        decision = 'TECHNICAL_BLOCKED/PARTIAL'
    elif n_partial < 10:
        decision = 'REDESIGN_NEEDED'
    elif any((c['partial_fsrr_unrecoverable'] >= 3 or c['partial_trr_yes_fsrr_no'] >= 3 for c in counts.values())):
        decision = 'GO_FOR_LARGER_STUDY'
    else:
        decision = 'INCONCLUSIVE/NO_GO_FOR_CURRENT_DESIGN'
    failures = [r for r in rows if 1 <= r['z'] <= 7 and r['fsrr'] is False]
    tiny = sum((abs(r['fsrr_gap']) <= 0.01 for r in failures))
    timeout_cases = []
    execution_categories = {}
    main_ids = {i['task_id'] for i in main}
    for task, t in truth.items():
        for test in t['execution']['tests']:
            if task in main_ids:
                execution_categories[test['category']] = execution_categories.get(test['category'], 0) + 1
            if test['category'] == 'candidate_timeout':
                timeout_cases.append(dict(task_id=task, split=t['split'], test_index=test['test_index'], repeat=test['repeat'], process_seconds=test['process_seconds'], candidate_seconds=test.get('candidate_seconds'), expected_repr=t['expected'][test['test_index']], code_sha256=t['code_sha256']))
    cp.snapshot(report / 'timeout_cases.json', timeout_cases)
    recheck = decision == 'GO_FOR_LARGER_STUDY' and tiny > len(failures) / 2
    same_error = []
    for key in cp.MODELS:
        selected = sorted([r for r in rows if r['model_key'] == key and r['z'] < 8], key=lambda r: r['task_id'])
        for error in sorted({r['errors'] for r in selected}):
            subset = [r for r in selected if r['errors'] == error]
            pair = next(((a, b) for a in subset for b in subset if (a['trr'], a['fsrr']) != (b['trr'], b['fsrr'])), None)
            if pair:
                same_error.append(dict(model_key=key, errors=error, case_a=pair[0], case_b=pair[1]))
    cases = {}
    ordered = sorted(rows, key=lambda r: (r['task_id'], r['model_key']))
    rules = {'fsrr_recoverable': lambda r: r['fsrr'] is True, 'trr_unrecoverable': lambda r: r['trr'] is False, 'trr_yes_fsrr_no': lambda r: r['trr'] is True and r['fsrr'] is False}
    for key in cp.MODELS:
        for name, predicate in rules.items():
            row = next((r for r in ordered if r['model_key'] == key and predicate(r)), None)
            label = key + '/' + name
            if row is None:
                cases[label] = None
                continue
            task = row['task_id']
            item = next((i for i in main if i['task_id'] == task))
            cases[label] = dict(metrics=row, margins=margins[key, task], code=gens[task]['code'], tests=item['tests'], truth=truth[task])
    summary = dict(created_at=cp.now(), decision=decision, numerical_recheck_required=recheck, technical_complete=technical, z_histogram=histogram, complete_truth_tasks=n_truth, missing_truth_tasks=30 - n_truth, partial_correct_tasks=n_partial, all_true_proportion=histogram['8'] / n_truth if n_truth else None, model_summaries=summaries, paired=paired, decision_counts=counts, same_errors_different_recovery=same_error, near_zero_failure_count=tiny, partial_fsrr_failure_curves=len(failures), generator_main_status={s: sum((g.get('extraction_status') == s for t, g in gens.items() if t in {i['task_id'] for i in main})) for s in ['ok', 'syntax_error', 'extraction_failure']}, generation_truncations=sum((g['truncated'] for t, g in gens.items() if t in {i['task_id'] for i in main})), execution_categories_main_repeated_tests=execution_categories, timeout_main_tasks=len({t['task_id'] for t in timeout_cases if t['split'] == 'main'}), timeout_main_distinct_tests=len({(t['task_id'], t['test_index']) for t in timeout_cases if t['split'] == 'main'}), timeout_main_repeated_test_events=sum((t['split'] == 'main' for t in timeout_cases)), bootstrap='5000 task bootstrap replicates, seed 20260910, percentile 95% CI; eligible subsets resampled within subset; model differences paired by task', caveats=['Exploratory adapted eight-test acceptance task; not official EvalPlus aggregate benchmarking.', 'Mistral is both candidate generator and one judge.', 'TRR/FSRR and minimum errors use truth-informed per-instance score cuts, not deployable calibration.', '30 tasks and 2 immediate-score models do not support universal reliability or reasoning claims.'])
    cp.snapshot(report / 'summary.json', summary)
    cp.snapshot(report / 'auditable_cases.json', cases)
    cp.snapshot(report / 'curve_margins.json', [dict(model_key=k, task_id=t, margins=m) for (k, t), m in margins.items()])
    plot_cases(report, cases)
    allocations = accounting(out)
    attempts = [cp.load(p) for p in (out / 'attempts').rglob('*.json')]
    generation_main = sum((t in gens for t in {i['task_id'] for i in main}))
    generation_smoke = sum((i['task_id'] in gens for i in instances if i['split'] == 'smoke'))
    smoke_scores = [r for r in all_scores if r['split'] == 'smoke']
    main_scores = [r for r in all_scores if r['split'] == 'main']
    failures_log = [a for a in attempts if a['status'] == 'infrastructure_error']
    blockers = [] if technical else audit['issues'] + [f'truth complete {n_truth}/30; model curves complete {len(rows)}/60']
    status = dict(updated_at=cp.now(), decision=decision, stages=dict(code_implemented=True, unit_tests='unit_tests_pass.xml', sandbox_engineering_smoke=cp.load(out / 'sandbox_smoke.json')['passed'], disjoint_end_to_end_smoke_complete=len(smoke_scores) == 54 and all((truth.get(i['task_id'], {}).get('stable', False) for i in instances if i['split'] == 'smoke')), actual_generation_main_completed=generation_main, actual_generation_smoke_completed=generation_smoke, real_truth_main_completed=n_truth, main_judge_completed=len(main_scores), statistics_completed=True), planned=dict(main_generation=30, smoke_generation=3, main_judge_requests=540, smoke_judge_requests=54, main_curves=60, max_gpu_hours=8), completed=dict(main_generation=generation_main, smoke_generation=generation_smoke, main_judge_requests=len(main_scores), smoke_judge_requests=len(smoke_scores), main_curves=len(rows), truth_main=n_truth), models=audit['models'], gpu_accounting=allocations, calls=dict(generation_responses=len(gens), generation_forward_calls=sum((g['generation_forward_calls'] for g in gens.values())), main_judge_logical_requests=len(main_scores), main_judge_forward_calls=sum((r['forward_calls'] for r in main_scores)), smoke_judge_logical_requests=len(smoke_scores), smoke_judge_forward_calls=sum((r['forward_calls'] for r in smoke_scores)), infrastructure_failed_attempts=len(failures_log), retry_attempts=sum((a['attempt'] > 0 for a in attempts)), judge_inference_seconds=sum((r['elapsed_seconds'] for r in all_scores)), generation_seconds=sum((g['elapsed_seconds'] for g in gens.values()))), failures=failures_log, blockers=blockers, recovery_commands=[] if technical else ['wsl -d Ubuntu-24.04 -- python3 scripts/code_pilot_CLUSTER status', 'wsl -d Ubuntu-24.04 -- python3 scripts/code_pilot_CLUSTER pull', 'wsl -d Ubuntu-24.04 -- .remote/code_pilot_v1_venv/bin/python scripts/code_pilot.py verify-truth', 'wsl -d Ubuntu-24.04 -- .remote/code_pilot_v1_venv/bin/python scripts/code_pilot.py freeze', 'wsl -d Ubuntu-24.04 -- python3 scripts/code_pilot_CLUSTER push', 'wsl -d Ubuntu-24.04 -- python3 scripts/code_pilot_CLUSTER submit --stage run'])
    cp.snapshot(out / 'execution_status.json', status)
    write_reports(report, summary, status, cases)
    print(cp.canonical(dict(decision=decision, truth=n_truth, main_scores=len(main_scores), z_histogram=histogram)))

def plot_cases(report, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen = []
    for category in ['fsrr_recoverable', 'trr_unrecoverable', 'trr_yes_fsrr_no']:
        case = next(((name, c) for name, c in cases.items() if name.endswith('/' + category) and c is not None), None)
        if case:
            chosen.append(case)
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 1, figsize=(8, 3.1 * len(chosen)), squeeze=False, constrained_layout=True)
    for ax, (name, c) in zip(axes[:, 0], chosen):
        r = c['metrics']
        z = r['z']
        ax.axhline(0, color='#64748b', linestyle='--', linewidth=0.8)
        ax.axvspan(-0.15, z + 0.4, color='#d1fae5', alpha=0.5, label='True criterion')
        ax.plot(range(9), c['margins'], 'o-', color='#244c8b', linewidth=1.5)
        ax.set(title=f"{r['task_id']} / {r['model_key']} / z={z} / {name}", xlabel='Required passes k', ylabel='A − B score margin', xticks=range(9), xlim=(-0.15, 8.15))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.2)
    fig.savefig(report / 'auditable_curves.png', dpi=180)
    fig.savefig(report / 'auditable_curves.svg')
    plt.close(fig)

def write_reports(report, s, status, cases):
    lines = ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', '|---|---|---|---|---|---|']
    for key, groups in s['model_summaries'].items():
        a = groups['all']
        n = groups['nontrivial']
        lines.append('| ' + key + ' | ' + ' | '.join((fmt(x) for x in [a['accuracy'], a['strict_accuracy'], n['trr'], n['fsrr'], a['tce']])) + ' |')
    lines += ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.' + ('Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.' if s['same_errors_different_recovery'] else 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.'), 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.']
    for name, c in cases.items():
        lines.append(f'- {name}: ' + ('Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.' if c is None else f"{c['metrics']['task_id']} / {c['metrics']['model_key']}; margins={c['margins']}"))
    lines += (['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.'] if any((c is not None for c in cases.values())) else ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.']) + ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.']
    if s['decision'] == 'REDESIGN_NEEDED':
        lines.append('Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.')
    if status['blockers']:
        lines.append('Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.' + str(status['blockers']))
    (report / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    methods = f"# Methods and Results draft — exploratory code acceptance pilot v1\n\nWe adapted HumanEval+ v0.1.10 using EvalPlus v0.3.1, commit e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2. The dataset SHA256 is recorded in the upstream manifest. This is an eight-test acceptance pilot, not the official EvalPlus aggregate evaluation. Suitability screening used only upstream problems, inputs, checkers, and reference solutions. Exact deterministic standard-library tasks with eight distinct non-example inputs and input/output serialization limits of 512 UTF-8 bytes were eligible. Seed 20260910 determined input and task ordering. Three disjoint engineering smoke tasks preceded 30 main tasks.\n\nFreeze A recorded selection, extraction, scoring, metrics and decisions before candidate generation; Freeze B hashed actual instances, replies, extracted code, truth logs, rendered prompts and revisions before main judge inference. Both are prospectively timestamped exploratory freezes, not retroactive claims of formal preregistration. Mistral-7B-Instruct-v0.3 generated one greedy response per task (maximum 1024 new tokens), seeing only the original problem. Extraction was deterministic; failures and truncations were retained.\n\nReference and candidate code were evaluated twice, with a fresh Python process per test, in a pinned Docker image without network access or host mounts, running as uid 65534 with a read-only root, dropped capabilities, no-new-privileges, a 32-process cap, one CPU and a 64 MiB temporary filesystem. Each test had a 512 MiB address-space limit and a five-second code-load/function/serialization wall-time limit. The container had a separate 768 MiB supervisor-inclusive memory limit. Candidate errors and timeouts failed the test; infrastructure failures remained missing. Truth was Y(k)=1[z≥k], k=0,…,8, including universally true k=0.\n\nQwen2.5-14B-Instruct and Mistral-7B-Instruct-v0.3 used fixed existing revisions, unquantized bf16 and SDPA immediate scoring. Labels A/B meant criterion met/not met. Continuation token IDs were checked on every rendered prompt. The two-label probability was sigmoid(score_A−score_B), not the probability of A under the full vocabulary. Every request was an independent conversation; only the dedicated threshold field varied. No prompts were truncated. Mistral's dual role as generator and judge is a limitation.\n\nWe report task-level accuracy, strict-threshold accuracy (k=1,…,8), first-failure error, TRR, FSRR and the minimum residual errors from enumerating all distinct score cuts. Ties were inseparable. z=8 curves were excluded from primary recovery denominators, while z=0 remained nontrivial but was not classified as partially correct. Percentile 95% intervals used 5000 task-level bootstrap replicates (seed 20260910). Paired differences resampled common complete tasks, preserving all nine thresholds. These truth-informed per-instance recovery bounds are not deployable calibrators.\n\n## Results\n\nActual main generation completed {status['completed']['main_generation']}/30 tasks, stable truth {s['complete_truth_tasks']}/30, main judge scoring {status['completed']['main_judge_requests']}/540 logical requests, and {status['completed']['main_curves']}/60 complete curves. Main z histogram: {s['z_histogram']}. There were {s['partial_correct_tasks']} partially correct candidates. The paired complete set contained {s['paired']['all']['n_tasks']} tasks. The prespecified resource decision was {s['decision']}. The main syntax/extraction status counts were {s['generator_main_status']}; generation truncations: {s['generation_truncations']}.\n"
    for k, g in s['model_summaries'].items():
        methods += f"\n{k}: accuracy {fmt(g['all']['accuracy'])}; strict accuracy {fmt(g['all']['strict_accuracy'])}; nontrivial TRR {fmt(g['nontrivial']['trr'])}; nontrivial FSRR {fmt(g['nontrivial']['fsrr'])}; TCE {fmt(g['all']['tce'])}; oracle residual errors {fmt(g['all']['oracle_min_errors'])}.\n"
    methods += f"\nThere were {len(s['same_errors_different_recovery'])} groups exhibiting equal raw error counts but different recoverability. Numerical margins, case-selection absences, model-specific denominators, partial-correct subsets and paired intervals are provided in the accompanying machine-readable tables. The study used {status['gpu_accounting']['allocated_gpu_seconds'] / 3600:.4f} allocated GPU hours. No simulated or prior-study scores are counted. These small-sample findings do not establish general judge reliability or a reasoning-model advantage. Any redesigned or expanded experiment requires new tasks and a new freeze.\n"
    (report / 'METHODS_RESULTS_EN.md').write_text(methods, encoding='utf-8')

def package(root):
    root = pathlib.Path(root)
    out = cp.outdir(root)
    if not (out / 'execution_status.json').exists():
        analyze(root)
    sources = sorted(list((root / 'src/formalcrrc').glob('code_pilot*.py')) + [root / 'src/formalcrrc/sequence_recovery.py'] + list((root / 'scripts').glob('code_pilot*.py')) + list((root / 'slurm').glob('run_code_pilot_v1*.sbatch')) + [root / 'tests/test_code_pilot.py', root / 'docs/CODE_PILOT_V1.md'])
    diff = []
    for p in sources:
        if not p.is_file():
            continue
        name = str(p.relative_to(root))
        data = p.read_text().splitlines()
        diff += ['diff --git a/' + name + ' b/' + name, 'new file mode 100644', '--- /dev/null', '+++ b/' + name, f'@@ -0,0 +1,{len(data)} @@'] + ['+' + line for line in data]
    patch = out / 'code_pilot_v1.patch'
    patch.write_text('\n'.join(diff) + '\n')
    target = root.parent / 'FormalCRRC_Code_Pilot_v1_Handoff.zip'
    files = [p for p in out.rglob('*') if p.is_file() and p.name != 'handoff_manifest.json'] + sources
    files += [root / 'src/formalcrrc/scoring.py', root / 'src/formalcrrc/provenance.py', root / 'LICENSE', root / 'FormalCRRC_Code_Pilot_Codex_Task.md']
    files += list((root / 'src/formalcrrc').glob('*.py'))
    files = sorted(set((p for p in files if p.is_file())))
    cp.snapshot(out / 'handoff_manifest.json', dict(created_at=cp.now(), files=cp.manifest(root, files)))
    files.append(out / 'handoff_manifest.json')
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in files:
            archive.write(p, str(p.relative_to(root)))
    digest = cp.sha(target.read_bytes())
    target.with_suffix('.zip.sha256').write_text(f'{digest}  {target.name}\n')
    print(str(target), digest)
