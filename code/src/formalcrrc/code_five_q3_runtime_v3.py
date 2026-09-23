"""Bound prompt-logprob workspace and retain native per-request completion receipts."""
import json,os,pathlib,time
from formalcrrc import code_extension as cp,code_five as f,reasoning_anchor as ra
from formalcrrc.code_five_q3_meter_v2 import named_rpc_metered_class

def bounded_class(base,root):
    root=pathlib.Path(root);o=f.out(root)
    rows=cp.load(o/'prompts/qwen3.json')['rows']
    indexed={tuple(r['prompt_token_ids']):r for r in rows}
    class BoundedLLM(named_rpc_metered_class(base,root)):
        def __init__(self,*args,**kwargs):
            kwargs.update(max_num_batched_tokens=512,enable_chunked_prefill=True)
            super().__init__(*args,**kwargs)
            self._receipt_phase=None;self._receipt_started=None
            original=self.llm_engine.step
            def observed_step(*args,**kwargs):
                outputs=original(*args,**kwargs)
                if self._receipt_phase=='reasoning':
                    for output in outputs:
                        if not getattr(output,'finished',False):continue
                        r=indexed.get(tuple(output.prompt_token_ids))
                        if r is None:raise RuntimeError('Unrecognized native completion prompt')
                        c=output.outputs[0]
                        p=o/'completion_receipts'/str(os.environ.get('SLURM_JOB_ID','local'))/(f.key(r)+'.json')
                        if not p.exists():
                            cp.immutable(p,dict(created_at=cp.now(),task_id=r['task_id'],split=r['split'],variant=r['variant'],k=r['k'],
                                seed=r['generation_seed'],request_id=output.request_id,raw_generated_token_ids=list(c.token_ids),
                                finish_reason=c.finish_reason,stop_reason=c.stop_reason,batch_submitted_wall_time=self._receipt_started,
                                observed_finished_wall_time=time.time(),
                                latency_source='Caller observed batch submission to final RequestOutput; includes scheduling/queueing in that batch, not GPU-exclusive per-request compute time.'))
                return outputs
            self.llm_engine.step=observed_step
        def generate(self,prompts_,sampling_params=None,*args,**kwargs):
            p=sampling_params[0] if isinstance(sampling_params,list) else sampling_params
            self._receipt_phase='reasoning' if p.max_tokens==8192 else 'decision'
            self._receipt_started=time.time()
            outputs=super().generate(prompts_,sampling_params,*args,**kwargs)
            if self._receipt_phase=='reasoning':
                from types import SimpleNamespace
                for output in outputs:
                    r=indexed[tuple(output.prompt_token_ids)]
                    receipt=cp.load(o/'completion_receipts'/str(os.environ.get('SLURM_JOB_ID','local'))/(f.key(r)+'.json'))
                    assert receipt['raw_generated_token_ids']==list(output.outputs[0].token_ids)
                    # Actual observed wall timestamps, explicitly described in the receipt.
                    output.metrics=SimpleNamespace(arrival_time=receipt['batch_submitted_wall_time'],finished_time=receipt['observed_finished_wall_time'],
                        first_scheduled_time=None,first_token_time=None,last_token_time=None)
            return outputs
    return BoundedLLM

def restore_receipts(root):
    from transformers import AutoTokenizer
    from formalcrrc.code_five_inference import common
    root=pathlib.Path(root);o=f.out(root);inv=cp.load(o/'inventory_qwen3.json')
    rows={f.key(r):r for r in cp.load(o/'prompts/qwen3.json')['rows']}
    receipts=list((o/'completion_receipts').rglob('*.json'))
    if not receipts:return
    tok=AutoTokenizer.from_pretrained(str(f.snapshot('qwen3')),trust_remote_code=False)
    freeze=f.verify_frozen(root,'qwen3')
    for path in receipts:
        receipt=cp.load(path);r=rows[f.key(receipt)];dest=f.trace_path(root,r)
        if dest.exists():continue
        raw=receipt['raw_generated_token_ids'];assert receipt['seed']==r['generation_seed']
        trace=ra.split_reasoning(raw,inv['think_end_id'],8192)
        natural=bool(raw and raw[-1]==inv['think_end_id'] and trace.think_end_reached and len(raw)<=8192 and receipt['finish_reason']=='stop')
        ctx=list(ra.build_decision_context(r['prompt_token_ids'],trace.token_ids,inv['separator_ids'])) if natural else []
        entry=common(r,'qwen3',freeze)
        entry.update(job_id=path.relative_to(o/'completion_receipts').parts[0],
            status='natural_complete' if natural else 'truncated_or_wrong_boundary',generation_seed=r['generation_seed'],raw_generated_token_ids=raw,
            reasoning_text=tok.decode(raw,skip_special_tokens=False),natural_end=natural,think_end_reached=trace.think_end_reached,
            reasoning_truncated=not natural,reasoning_token_count=len(raw),finish_reason=receipt['finish_reason'],stop_reason=receipt['stop_reason'],
            generated_requests=1,engine_batch_seconds=None,engine_batch_size=None,request_timing=receipt,
            request_elapsed_seconds=receipt['observed_finished_wall_time']-receipt['batch_submitted_wall_time'],
            final_context_token_ids=ctx,final_context_text=tok.decode(ctx,skip_special_tokens=False) if ctx else None,
            final_context_ids_sha256=cp.sha(cp.canonical(ctx)) if ctx else None,restored_from_native_receipt=path.relative_to(root).as_posix(),
            native_receipt_sha256=cp.sha(path.read_bytes()))
        cp.immutable(dest,entry)

def recovered_smoke_estimate(root,model,seconds,n_rows,load_seconds,extra=None):
    """Account for already saved generation, not just this scoring-only resume."""
    from formalcrrc.code_five_inference import remaining_seconds
    assert model=='qwen3'
    o=f.out(root);rows=cp.load(o/'prompts/qwen3.json')['rows']
    smoke=[r for r in rows if r['split']=='smoke']
    traces=[cp.load(f.trace_path(root,r)) for r in smoke]
    assert len(traces)==54
    records=[json.loads(line) for path in (o/'runtime').glob('qwen3_engine_calls_*.jsonl') for line in path.read_text().splitlines()]
    generation=[r for r in records if r['split']=='smoke' and r['phase']=='reasoning' and r['status']=='ok']
    assert sum(r['logical_requests'] for r in generation)==54
    generation_seconds=sum(r['elapsed_seconds'] for r in generation)
    natural=sum(t['natural_end'] for t in traces)
    assert n_rows==natural and all(f.score_path(root,model,r).exists() for r,t in zip(smoke,traces) if t['natural_end'])
    decision_seconds=extra['measured_decision_seconds']
    remaining=sum(not f.score_path(root,model,r).exists() and not f.trace_path(root,r).exists() for r in rows if r['split']=='main')
    # Conservatively assume every remaining main context will need decision scoring.
    estimate=1.5*remaining*(generation_seconds/54+decision_seconds/natural)
    data=dict(created_at=cp.now(),model=model,measured_smoke_contexts=54,natural_smoke_contexts=natural,
        truncated_smoke_contexts=54-natural,model_load_seconds=load_seconds,
        measured_generation_seconds=generation_seconds,measured_decision_seconds=decision_seconds,
        current_scoring_resume_wall_seconds=seconds,main_contexts_remaining=remaining,
        estimated_main_gpu_seconds_with_50_percent_margin=estimate,remaining_allocation_seconds=remaining_seconds(),
        mean_reasoning_tokens=sum(t['reasoning_token_count'] for t in traces)/54,
        basis='Saved 54-sample smoke generation plus recovered scoring on all natural traces. Generation was measured before the prefill-workspace cap; scoring measured after it. 50% margin; all main requests conservatively assumed to need scoring. No failed trace resampled.')
    cp.immutable(o/f'smoke_estimate_qwen3_{os.environ.get("SLURM_JOB_ID","local")}.json',data)
    print('RECOVERED_SMOKE_ESTIMATE',json.dumps(data),flush=True)
    return estimate
