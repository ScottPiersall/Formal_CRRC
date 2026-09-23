# Review adaptation: historical non-English narrative strings omitted; numerical routines retained.
"""Independent audit and paired estimates for answer-form robustness."""
import csv, math
from formalcrrc import code_extension as cp

def audit(root, items, truths):
    out = cp.outdir(root)
    spec = cp.load(out / 'answer_form_protocol.json')
    assert cp.load(out / 'freeze_A.json')['protocol']['answer_form_robustness']['protocol_file_sha256'] == cp.sha((out / 'answer_form_protocol.json').read_bytes())
    cp.assert_manifest(root, spec['source_files'])
    from formalcrrc.code_extension_analysis import metric_row
    raw = []
    rows = []
    for key in cp.MODELS:
        prompts = {(r['task_id'], r['variant'], r['k']): r for r in cp.load(out / 'prompts' / f'{key}.json')['rows']}
        for item in items:
            task = item['task_id']
            split = item['split']
            for variant in cp.VARIANTS:
                scores = []
                for k in range(9):
                    r = cp.load(out / 'scores_answer_forms' / split / key / variant / f'{cp.score_key(task, k)}.json')
                    p = prompts[task, variant, k]
                    assert (r['task_id'], r['model_key'], r['variant'], r['k'], r['split'], r['dtype']) == (task, key, variant, k, split, 'bfloat16')
                    assert r['revision'] == cp.MODELS[key]['revision'] and r['created_at'] >= cp.load(out / 'freeze_B.json')['created_at']
                    assert r['freeze_B_sha256'] == cp.sha((out / 'freeze_B.json').read_bytes()) and r['protocol_sha256'] == cp.sha((out / 'answer_form_protocol.json').read_bytes())
                    assert r['rendered_prompt_sha256'] == p['rendered_prompt_sha256'] and r['n_prompt_tokens'] == p['n_prompt_tokens']
                    assert set(r['answer_events']) == {'A', ' A', '\nA', 'B', ' B', '\nB'}
                    nodes = {tuple(n['token_ids']): {int(t): v for t, v in n['next_log_probabilities'].items()} for n in r['prefix_nodes']}
                    events = {}
                    for text, event in r['answer_events'].items():
                        path = tuple(event['token_ids'])
                        value = sum((nodes[path[:j]][t] for j, t in enumerate(path)))
                        assert abs(value - event['log_probability']) < 1e-10
                        if path in events:
                            assert events[path][0] == text[-1]
                        events[path] = (text[-1], value)
                    assert not any((len(a) < len(b) and b[:len(a)] == a for a in events for b in events))
                    total = {label: sum((math.exp(value) for tag, value in events.values() if tag == label)) for label in ['A', 'B']}
                    a, b = (math.log(total['A']), math.log(total['B']))
                    assert abs(r['score_met'] - a) < 1e-10 and abs(r['score_not_met'] - b) < 1e-10
                    assert abs(r['margin'] - (a - b)) < 1e-10 and math.isfinite(r['margin'])
                    assert abs(r['p_met'] - total['A'] / (total['A'] + total['B'])) < 1e-10
                    assert abs(r['full_vocab_union_mass'] - sum(total.values())) < 1e-10 and r['full_vocab_union_mass'] <= 1.00001
                    assert r['forward_calls'] == len(nodes)
                    assert r['answer_events']['A']['token_ids'] == p['label_tokenization']['met_token_ids']
                    assert r['answer_events']['B']['token_ids'] == p['label_tokenization']['not_met_token_ids']
                    raw.append(r)
                    scores.append(r)
                if split == 'main':
                    rows.append(metric_row(task, key, variant, 'bfloat16', [r['margin'] for r in scores], truths[task]['z']))
    assert len(list((out / 'scores_answer_forms/main').rglob('*.json'))) == 2880 and len(list((out / 'scores_answer_forms/smoke').rglob('*.json'))) == 108
    return (raw, rows, dict(passed=True, main_curves=len(rows), main_scores=2880, smoke_scores=108, independent_prefix_probability_audit=True))

def summarize(rows, bare_rows):
    from formalcrrc.code_extension_analysis import summarize_rows, ci, METRICS
    summaries = {f'{key}/{v}': summarize_rows([r for r in rows if (r['model_key'], r['variant']) == (key, v)]) for key in cp.MODELS for v in cp.VARIANTS}
    base = {(r['model_key'], r['variant'], r['task_id']): r for r in bare_rows if r['dtype'] == 'bfloat16'}
    paired = {}
    failures = {}
    for key in cp.MODELS:
        for v in cp.VARIANTS:
            selected = sorted([r for r in rows if (r['model_key'], r['variant']) == (key, v)], key=lambda r: r['task_id'])
            paired[f'{key}/{v}'] = {}
            for name, group in [('all', selected), ('nontrivial', [r for r in selected if r['z'] < 8]), ('partial', [r for r in selected if 1 <= r['z'] <= 7])]:
                paired[f'{key}/{v}'][name] = dict(n_tasks=len(group), **{m: ci([r[m] - base[key, v, r['task_id']][m] for r in group if r[m] is not None]) for m in METRICS})
            part = [r for r in selected if 1 <= r['z'] <= 7]
            failures[f'{key}/{v}'] = dict(n=len(part), fsrr_failure_ids=[r['task_id'] for r in part if r['fsrr'] is False], strict_fsrr_failure_ids=[r['task_id'] for r in part if r['strict_fsrr'] is False], trr_yes_fsrr_no_ids=[r['task_id'] for r in part if r['trr'] is True and r['fsrr'] is False], near_zero_failure_ids=[r['task_id'] for r in part if r['fsrr'] is False and abs(r['fsrr_gap']) <= 0.01])
    return dict(summaries=summaries, paired_union_minus_bare=paired, partial_failures=failures, interpretation='Secondary scoring representation specified from development observations before independent-task freeze A. These are paired scores on the same 80 tasks, not new independent samples. The three forms are a finite predefined whitelist, not all possible free-form answers.')

def audit_precision(root, truths, bf_rows, bf_raw):
    from formalcrrc.code_extension_analysis import metric_row, summarize_rows
    out = cp.outdir(root)
    tasks = cp.load(out / 'numeric_task_ids.json')
    raw = []
    rows = []
    baseline = {(r['model_key'], r['variant'], r['task_id'], r['k']): r for r in bf_raw if r['split'] == 'main'}
    for key in cp.MODELS:
        for variant in cp.VARIANTS:
            for task in tasks:
                scores = []
                for k in range(9):
                    r = cp.load(out / 'scores_answer_forms_float32' / key / variant / f'{cp.score_key(task, k)}.json')
                    b = baseline[key, variant, task, k]
                    assert (r['model_key'], r['variant'], r['task_id'], r['k'], r['dtype']) == (key, variant, task, k, 'float32')
                    assert r['revision'] == cp.MODELS[key]['revision'] and r['freeze_B_sha256'] == cp.sha((out / 'freeze_B.json').read_bytes())
                    assert r['protocol_sha256'] == cp.sha((out / 'answer_form_protocol.json').read_bytes()) and r['created_at'] >= cp.load(out / 'freeze_B.json')['created_at']
                    assert r['rendered_prompt_sha256'] == b['rendered_prompt_sha256'] and r['n_prompt_tokens'] == b['n_prompt_tokens']
                    runtime = cp.load(out / 'runtime' / f"answer_forms_float32_{key}_{r['job_id']}.json")
                    assert runtime['cuda_matmul_allow_tf32'] is False and runtime['float32_matmul_precision'] == 'highest' and (runtime['dtype'] == 'float32')
                    assert set(r['answer_events']) == set(b['answer_events'])
                    nodes = {tuple(n['token_ids']): {int(t): v for t, v in n['next_log_probabilities'].items()} for n in r['prefix_nodes']}
                    events = {}
                    for name, e in r['answer_events'].items():
                        assert e['token_ids'] == b['answer_events'][name]['token_ids']
                        path = tuple(e['token_ids'])
                        value = sum((nodes[path[:j]][t] for j, t in enumerate(path)))
                        assert abs(value - e['log_probability']) < 1e-10
                        events[path] = (name[-1], value)
                    totals = {tag: sum((math.exp(v) for label, v in events.values() if label == tag)) for tag in ['A', 'B']}
                    assert abs(r['margin'] - (math.log(totals['A']) - math.log(totals['B']))) < 1e-10
                    assert abs(r['p_met'] - totals['A'] / sum(totals.values())) < 1e-10
                    assert r['forward_calls'] == len(nodes) and math.isfinite(r['margin'])
                    raw.append(r)
                    scores.append(r)
                rows.append(metric_row(task, key, variant, 'float32', [r['margin'] for r in scores], truths[task]['z']))
    assert len(list((out / 'scores_answer_forms_float32').rglob('*.json'))) == 360
    indexed = {(r['model_key'], r['variant'], r['task_id']): r for r in bf_rows}
    numerical = {}
    for key in cp.MODELS:
        for v in cp.VARIANTS:
            group = [r for r in rows if (r['model_key'], r['variant']) == (key, v)]
            pairs = [(r, baseline[key, v, r['task_id'], r['k']]) for r in raw if (r['model_key'], r['variant']) == (key, v)]
            partial = [r for r in group if 1 <= r['z'] <= 7]
            numerical[f'{key}/{v}'] = dict(n_tasks=len(group), partial_n=len(partial), summary=summarize_rows(group), max_abs_margin_delta=max((abs(a['margin'] - b['margin']) for a, b in pairs)), sign_flips=sum(((a['margin'] >= 0) != (b['margin'] >= 0) for a, b in pairs)), fsrr_changed_ids=[r['task_id'] for r in group if r['fsrr'] != indexed[key, v, r['task_id']]['fsrr']], partial_failure_both_gaps_below_minus_001_ids=[r['task_id'] for r in partial if r['fsrr_gap'] < -0.01 and indexed[key, v, r['task_id']]['fsrr_gap'] < -0.01])
    return (raw, rows, numerical)

def write(report, summary, rows, raw):
    from formalcrrc.code_extension_analysis import fmt
    cp.snapshot(report / 'answer_form_summary.json', summary)
    cp.snapshot(report / 'answer_form_metrics.json', rows)
    with (report / 'answer_form_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', '|---|---|---|---|---|']
    for name, g in summary['summaries'].items():
        lines.append('| ' + name + ' | ' + ' | '.join((fmt(s) for s in [g['all']['k0_correct'], g['all']['strict_accuracy'], g['partial']['fsrr'], g['partial']['strict_fsrr']])) + ' |')
    lines += ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.']
    lines += ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.']
    for name, r in summary['numerical'].items():
        lines.append('Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.')
    (report / 'ANSWER_FORMS_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (report / 'REPORT_ZH.md').open('a', encoding='utf-8') as f:
        f.write('Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.' + '\n'.join(lines[2:]) + '\n')
    with (report / 'METHODS_RESULTS_EN.md').open('a', encoding='utf-8') as f:
        f.write('\n## Development-informed answer-form robustness\n\nBefore independent-task generation, we additionally fixed a paired secondary representation analysis over the same 80 tasks. We summed disjoint exact token-sequence probabilities for A/B with no prefix, one space or one newline, deduplicating identical sequences and never selecting a prefix by results. The original bare-token primary analysis was retained. All 2880 additional main scores and 108 secondary smoke scores were completed; shared teacher-forced prefixes reduced computational forwards without changing the event definition. These are paired representations, not additional independent samples.\n')
        for name, g in summary['summaries'].items():
            f.write(f"\n{name}: union k=0 accuracy {fmt(g['all']['k0_correct'])}; partial FSRR {fmt(g['partial']['fsrr'])}; partial k>=1 FSRR {fmt(g['partial']['strict_fsrr'])}.\n")
        f.write('\nThe same ten preselected numerical tasks also received the answer-form union in full-model float32 with TF32 disabled (360 additional scores, 40 curves); numerical flips and non-negligible residual failures are listed separately.\n')
    base = cp.load(report / 'per_instance_metrics.json')
    baseline = {(r['model_key'], r['variant'], r['task_id']): r for r in base if r['dtype'] == 'bfloat16'}
    out = report.parent
    illustrations = {}
    for key in cp.MODELS:
        group = sorted([r for r in rows if r['model_key'] == key and r['variant'] == 'original' and (1 <= r['z'] <= 7)], key=lambda r: r['task_id'])
        chosen = next((r for r in group if baseline[key, 'original', r['task_id']]['fsrr'] is False and r['fsrr'] is True), None)
        rule = 'first_partial_bare_failure_union_recovery'
        if chosen is None:
            chosen = next((r for r in group if r['fsrr'] is False), None)
            rule = 'first_partial_union_failure'
        if chosen is None:
            illustrations[key] = None
            continue
        task = chosen['task_id']
        bare = [cp.load(cp.score_path(out, 'main', key, 'original', 'bfloat16', task, k))['margin'] for k in range(9)]
        union = [next((s['margin'] for s in raw if s['model_key'] == key and s['variant'] == 'original' and (s['task_id'] == task) and (s['k'] == k))) for k in range(9)]
        illustrations[key] = dict(selection_rule=rule, metrics=chosen, bare_margins=bare, union_margins=union, code=cp.load(out / 'generation' / f'{cp.slug(task)}.json')['code'], tests=next((i['tests'] for i in cp.load(out / 'instances.json') if i['task_id'] == task)))
    cp.snapshot(report / 'answer_form_cases.json', illustrations)
    selected = [(key, c) for key, c in illustrations.items() if c is not None]
    if selected:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(selected), 1, figsize=(8, 3.4 * len(selected)), squeeze=False, constrained_layout=True)
        for ax, (key, c) in zip(axes[:, 0], selected):
            z = c['metrics']['z']
            for label, margins, color in [('bare A/B', c['bare_margins'], '#2563eb'), ('answer-form union', c['union_margins'], '#be123c')]:
                ax.plot(range(9), margins, 'o-', label=label, color=color)
            ax.axhline(0, color='#64748b', linestyle='--', linewidth=0.8)
            ax.axvspan(-0.2, z + 0.4, color='#dcfce7', alpha=0.5)
            ax.set(title=f"{c['metrics']['task_id']} / {key} / z={z}", xlabel='Required passes k', ylabel='Met minus not-met score', xticks=range(9), xlim=(-0.2, 8.2))
            ax.legend()
            ax.grid(axis='y', alpha=0.2)
        fig.savefig(report / 'answer_form_curves.png', dpi=180)
        fig.savefig(report / 'answer_form_curves.svg')
        plt.close(fig)
