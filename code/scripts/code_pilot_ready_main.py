"""Validate actual disjoint smoke, record metrics/timings, and release the main gate."""
import argparse
import pathlib
import math
import statistics
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from formalcrrc import code_pilot as cp, scoring

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--release',action='store_true'); args=parser.parse_args()
    root=pathlib.Path(__file__).resolve().parents[1]; out=cp.outdir(root)
    cp.assert_freeze_a(root); cp.assert_freeze_b(root)
    smoke=[i for i in cp.load(out/'instances.json') if i['split']=='smoke']
    assert len(smoke)==3
    curves=[]; timing={}
    for key in cp.MODELS:
        expected={(i['task_id'],k) for i in smoke for k in range(9)}
        paths=list((out/'scores/smoke'/key).glob('*.json')); rows=[cp.load(p) for p in paths]
        assert len(rows)==27 and {(r['task_id'],r['k']) for r in rows}==expected
        prompts={(r['task_id'],r['k']):r for r in cp.load(out/'prompts'/f'{key}.json')['rows']}
        mean=statistics.mean(r['elapsed_seconds'] for r in rows)
        main_prompts=[r for r in prompts.values() if r['split']=='main']
        scale=max(1,max(r['n_prompt_tokens'] for r in main_prompts)/statistics.mean(r['n_prompt_tokens'] for r in rows))
        timing[key]=dict(n=27,mean_seconds=mean,median_seconds=statistics.median(r['elapsed_seconds'] for r in rows),
                        prompt_length_conservative_scale=scale,planned_logical_requests=270,estimated_forward_seconds=270*mean*scale**2)
        for row in rows:
            assert row['margin']==row['score_met']-row['score_not_met'] and math.isfinite(row['margin'])
            assert row['p_met']==scoring.normalized_probability(row['score_met'],row['score_not_met'])
            assert row['rendered_prompt_sha256']==prompts[(row['task_id'],row['k'])]['rendered_prompt_sha256']
            assert row['n_prompt_tokens']==prompts[(row['task_id'],row['k'])]['n_prompt_tokens']
        for item in smoke:
            task=item['task_id']; truth=cp.load(out/'truth'/f'{cp.slug(task)}.json'); assert truth['stable']
            margins=[next(r['margin'] for r in rows if r['task_id']==task and r['k']==k) for k in range(9)]
            curves.append(dict(task_id=task,model_key=key,margins=margins,**cp.curve_diagnostics(margins,truth['z']+1)))
    record=dict(created_at=cp.now(),passed=True,logical_requests=54,curves=curves,
                no_outcome_based_gate=True,checks='counts, identities, stable truth, frozen prompt hashes, score sign, probability, finite margins and token counts')
    path=out/'engineering_smoke_summary.json'
    if not path.exists(): cp.immutable(path,record)
    estimate_path=out/'main_runtime_estimate.json'
    if not estimate_path.exists():
        cp.immutable(estimate_path,dict(created_at=cp.now(),smoke_logical_requests=54,planned_main_logical_requests=540,
           models=timing,estimated_forward_seconds=sum(t['estimated_forward_seconds'] for t in timing.values()),
           model_loading_and_io_allowance_seconds=120,
           formula='27 actual smoke latencies per model, conservatively scaled by squared max-main/mean-smoke token ratio, plus 120 seconds loading and I/O allowance; estimated, not observed main runtime.'))
    estimate=cp.load(estimate_path)
    assert estimate['smoke_logical_requests']==54 and estimate['planned_main_logical_requests']==540
    if args.release and not (out/'main_run_gate.json').exists():
        cp.immutable(out/'main_run_gate.json',dict(created_at=cp.now(),smoke_logical_requests=54,
           freeze_B_sha256=cp.sha((out/'freeze_B.json').read_bytes()),
           planned_main_logical_requests=540,estimated_seconds=estimate['estimated_forward_seconds']+estimate['model_loading_and_io_allowance_seconds'],
           authorized_scope='User taskbook: complete this pilot within 8 GPU-hours; no additional authorization required',
           smoke_integrity_passed=True))
    elif args.release:
        assert cp.load(out/'main_run_gate.json')['freeze_B_sha256']==cp.sha((out/'freeze_B.json').read_bytes())
    print('Disjoint end-to-end smoke validated; main gate', 'released' if args.release else 'not written')

if __name__=='__main__': main()
