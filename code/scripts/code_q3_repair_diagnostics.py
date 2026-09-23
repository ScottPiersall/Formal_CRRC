# Review adaptation: historical non-English narrative strings omitted; numerical routines retained.
"""Post-observation description of saved continuations; never affects inference."""
import pathlib, json, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from formalcrrc import code_extension as cp, code_q3_repair as q, code_five as f

def periodic_tail(ids):
    tail = ids[-8192:]
    best = (0, None)
    for period in range(1, min(129, len(tail) // 2 + 1)):
        matched = 0
        for i in range(len(tail) - 1, period - 1, -1):
            if tail[i] != tail[i - period]:
                break
            matched += 1
        length = matched + period
        if matched >= period and length > best[0]:
            best = (length, period)
    return dict(exact_periodic_suffix_tokens=best[0], period_tokens=best[1], inspected_tail_tokens=len(tail), definition='Longest exactly periodic suffix within final8192 continuation tokens, candidate periods1..128; purely descriptive, not a stopping or exclusion rule.')

def main():
    o = q.out(ROOT)
    rows = []
    for p in sorted((o / 'continuations').rglob('*.json')):
        t = cp.load(p)
        rows.append(dict(key=f"{t['split']}/{t['variant']}/{p.stem}", task_id=t['task_id'], split=t['split'], variant=t['variant'], k=t['k'], natural_end=t['natural_end'], additional_token_count=t['additional_token_count'], source_sha256=cp.sha(p.read_bytes()), terminal_text_tail=t['additional_reasoning_text'][-450:] if not t['natural_end'] else None, **periodic_tail(t['additional_generated_token_ids'])))
    original = []
    for row in cp.load(o / 'inputs.json')['rows']:
        if row['original_natural_end']:
            continue
        path = ROOT / row['original_trace_path'] if 'original_trace_path' in row else f.trace_path(ROOT, row)
        trace = cp.load(path)
        original.append(dict(key=row['key'], split=row['split'], finish_reason=trace['finish_reason'], reasoning_token_count=trace['reasoning_token_count'], natural_end=trace['natural_end'], source_sha256=cp.sha(path.read_bytes()), **periodic_tail(trace['raw_generated_token_ids'])))
    summary = {split: dict(original_truncated=sum((r['split'] == split for r in original)), original_length_8192=sum((r['split'] == split and r['finish_reason'] == 'length' and (r['reasoning_token_count'] == 8192) for r in original)), original_periodic_suffix_at_least_4096=sum((r['split'] == split and r['exact_periodic_suffix_tokens'] >= 4096 for r in original)), continued_terminal_failures=sum((r['split'] == split and (not r['natural_end']) for r in rows)), terminal_periodic_suffix_at_least_4096=sum((r['split'] == split and (not r['natural_end']) and (r['exact_periodic_suffix_tokens'] >= 4096) for r in rows))) for split in ('main', 'smoke')}
    cp.snapshot(o / 'analysis/continuation_periodicity_descriptive.json', dict(created_at=cp.now(), post_observation_diagnostic=True, inference_unchanged=True, all_saved_continuations_included=True, rows=rows, original_truncated_prefixes=original, summary=summary, interpretation='Literal suffix inspection only. A unique native closing token can yield a zero periodic suffix for a natural completion despite earlier repetition. The 4096-token descriptive threshold was chosen after observation; it never affects stopping, inclusion or scoring.'))
    explanation = 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.'
    for row in rows:
        if not row['natural_end']:
            explanation += 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.'
    if {r['key'] for r in rows if not r['natural_end']} == {'main/original/HumanEval_116__k7', 'main/original/HumanEval_142__k3', 'smoke/original/HumanEval_119__k3'}:
        explanation += 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.'
    explanation += 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.'
    (o / 'analysis/ROOT_CAUSE_AND_REPAIR_ZH.md').write_text(explanation, encoding='utf-8')
    print(json.dumps(summary))
if __name__ == '__main__':
    main()
