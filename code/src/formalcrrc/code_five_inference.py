"""Checkpointed immediate readout and native Q3 reason-then-score."""
from __future__ import annotations
import gc, json, math, os, pathlib, signal, subprocess, time, traceback
from formalcrrc import code_five as f, code_extension as cp, provenance, reasoning_anchor as ra

STOP=False
def stop_handler(*_):
    global STOP
    STOP=True

def remaining_seconds():
    return float(os.environ.get('FIVE_DEADLINE_EPOCH','inf'))-time.time()

def runtime(root,model,extra=None):
    import torch
    r=dict(created_at=cp.now(),model_key=model,**f.MODELS[model],job_id=os.environ.get('SLURM_JOB_ID'),
        environment=provenance.environment_snapshot(True),dtype='bfloat16',quantization=None,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision(),**(extra or {}))
    cp.immutable(f.out(root)/'runtime'/f'{model}_{os.environ.get("SLURM_JOB_ID","local")}.json',r)
    return r

def common(row,model,freeze):
    return {**{k:row[k] for k in ('task_id','split','variant','k','z','user_message_sha256','rendered_prompt_sha256','n_prompt_tokens')},
        'created_at':cp.now(),'model_key':model,**f.MODELS[model],'dtype':'bfloat16','quantization':None,
        'job_id':os.environ.get('SLURM_JOB_ID'),'freeze_sha256':freeze}

def failure(root,model,row,stage,exc,attempt=0):
    data=dict(**common(row,model,cp.sha((f.out(root)/f'freeze_{model}.json').read_bytes())),stage=stage,
        status='infrastructure_error' if isinstance(exc,(OSError,TimeoutError)) or 'out of memory' in str(exc).lower() else 'scoring_or_protocol_error',
        error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),attempt=attempt)
    cp.immutable(f.out(root)/'attempts'/model/(f.key(row)+f'__{stage}_{attempt}.json'),data)

def immediate_score(model,tok,row):
    import torch
    ids=row['prompt_token_ids']; paths,forms=f.paths_for(tok,row['rendered_prompt'],ids)
    assert forms==row['answer_forms_preflight']
    prefixes=sorted({p[:i] for p in paths for i in range(len(p))},key=lambda x:(len(x),x))
    logp_nodes={}; raw=None
    for prefix in prefixes:
        inputs=torch.tensor([ids+list(prefix)],dtype=torch.long,device=model.device)
        with torch.inference_mode(): logits=model(input_ids=inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        logp=torch.log_softmax(logits,dim=-1)
        needed={p[len(prefix)] for p in paths if len(p)>len(prefix) and p[:len(prefix)]==prefix}
        logp_nodes[prefix]={t:float(logp[t]) for t in needed}
        if not prefix and all(len(forms[a]['token_ids'])==1 for a in ('A','B')):
            raw={a:float(logits[forms[a]['token_ids'][0]]) for a in ('A','B')}
    likelihood={p:sum(logp_nodes[p[:i]][t] for i,t in enumerate(p)) for p in paths}
    scores,events=f.finish_scores(forms,likelihood,raw)
    return dict(status='ok',scores=scores,answer_events=events,
        prefix_nodes=[dict(token_ids=list(p),next_log_probabilities={str(t):v for t,v in values.items()}) for p,values in logp_nodes.items()],
        forward_calls=len(prefixes),scoring_requests=1,reasoning_generation_requests=0,
        peak_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_memory_reserved_bytes=torch.cuda.max_memory_reserved())

def smoke_estimate(root,model,seconds,n_rows,load_seconds,extra=None):
    remaining=sum(not f.score_path(root,model,r).exists() for r in cp.load(f.out(root)/'prompts'/f'{model}.json')['rows'] if r['split']=='main')
    estimate=seconds/max(n_rows,1)*remaining*1.5
    data=dict(created_at=cp.now(),model=model,measured_smoke_contexts=n_rows,measured_smoke_seconds=seconds,model_load_seconds=load_seconds,
        main_contexts_remaining=remaining,estimated_main_gpu_seconds_with_50_percent_margin=estimate,
        remaining_allocation_seconds=remaining_seconds(),basis='Measured full three-task smoke throughput, allocation time includes load; no historical chat estimate',**(extra or {}))
    cp.immutable(f.out(root)/f'smoke_estimate_{model}_{os.environ.get("SLURM_JOB_ID","local")}.json',data)
    print('SMOKE_ESTIMATE',json.dumps(data),flush=True)
    return estimate

def run_immediate(root,model_key):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    freeze=f.verify_frozen(root,model_key); rows=cp.load(f.out(root)/'prompts'/f'{model_key}.json')['rows']
    if all(f.score_path(root,model_key,r).exists() for r in rows): return
    spec=f.MODELS[model_key]; kw=dict(revision=spec['revision'],cache_dir=f.CACHE,local_files_only=True,trust_remote_code=False)
    start=time.monotonic(); tok=AutoTokenizer.from_pretrained(spec['model_id'],**kw)
    model=AutoModelForCausalLM.from_pretrained(spec['model_id'],dtype=torch.bfloat16,attn_implementation='sdpa',**kw).to('cuda').eval()
    assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False) and model.config._commit_hash==spec['revision']
    load_seconds=time.monotonic()-start; runtime(root,model_key,dict(load_seconds=load_seconds,attention_backend='sdpa'))
    for split in ('smoke','main'):
        if split=='main' and any(not f.score_path(root,model_key,r).exists() for r in rows if r['split']=='smoke'): raise RuntimeError('Incomplete smoke gate')
        pending=[r for r in rows if r['split']==split and not f.score_path(root,model_key,r).exists()]
        begin=time.monotonic(); done=0
        for row in pending:
            if STOP or remaining_seconds()<120: print('BOUNDED_STOP',model_key,done,flush=True); return
            start=time.monotonic(); torch.cuda.reset_peak_memory_stats()
            try:
                result=immediate_score(model,tok,row); torch.cuda.synchronize()
                result.update(common(row,model_key,freeze),elapsed_seconds=time.monotonic()-start)
                cp.immutable(f.score_path(root,model_key,row),result); done+=1
                if done%18==0: print('SCORED',model_key,split,done,len(pending),flush=True)
            except Exception as e: failure(root,model_key,row,'score',e); raise
        if split=='smoke' and done:
            estimate=smoke_estimate(root,model_key,time.monotonic()-begin,done,load_seconds)
            if estimate+120>remaining_seconds():
                print('BUDGET_GATE_MAIN_DEFERRED',model_key,flush=True); return
    del model; gc.collect();torch.cuda.empty_cache()

def run_qwen3(root):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.inputs import TokensPrompt
    model='qwen3'; freeze=f.verify_frozen(root,model); o=f.out(root)
    rows=cp.load(o/'prompts/qwen3.json')['rows']; inv=cp.load(o/'inventory_qwen3.json')
    tok=AutoTokenizer.from_pretrained(str(f.snapshot(model)),trust_remote_code=False)
    think_end=inv['think_end_id']; separator=inv['separator_ids']
    start=time.monotonic()
    llm=LLM(model=str(f.snapshot(model)),tokenizer=str(f.snapshot(model)),dtype='bfloat16',quantization=None,
        max_model_len=16384,gpu_memory_utilization=.92,tensor_parallel_size=1,seed=42,
        max_num_seqs=32,enforce_eager=False,enable_prefix_caching=False)
    load_seconds=time.monotonic()-start
    runtime(root,model,dict(load_seconds=load_seconds,engine='vllm',max_model_len=16384,max_num_seqs=32,chunk_contexts=54,
        forward_calls=None,forward_count_note='vLLM batches model forwards across requests; do not equate requested token positions with actual engine forward invocations.'))
    for split in ('smoke','main'):
        selected=[r for r in rows if r['split']==split]
        if split=='main':
            smoke=[r for r in rows if r['split']=='smoke']
            if any(not f.trace_path(root,r).exists() for r in smoke): raise RuntimeError('Every smoke reasoning must be attempted')
        pending=[r for r in selected if not f.score_path(root,model,r).exists() and not (f.trace_path(root,r).exists() and not cp.load(f.trace_path(root,r))['natural_end'])]
        begin=time.monotonic(); done=0; lengths=[]; generated_seconds=0; scoring_seconds=0
        for start_index in range(0,len(pending),54):
            if STOP or remaining_seconds()<180: print('BOUNDED_STOP_Q3',done,flush=True); return
            chunk=pending[start_index:start_index+54]
            generate=[r for r in chunk if not f.trace_path(root,r).exists()]
            if generate:
                params=[SamplingParams(n=1,temperature=.6,top_p=.95,top_k=20,min_p=0.,max_tokens=8192,
                    seed=r['generation_seed'],stop_token_ids=[think_end],include_stop_str_in_output=True) for r in generate]
                t=time.monotonic()
                try: outputs=llm.generate([TokensPrompt(prompt_token_ids=r['prompt_token_ids']) for r in generate],params,use_tqdm=False)
                except Exception as e:
                    for r in generate: failure(root,model,r,'reasoning_infrastructure',e)
                    raise
                elapsed=time.monotonic()-t; generated_seconds+=elapsed
                for r,output in zip(generate,outputs,strict=True):
                    completion=output.outputs[0]; raw=list(completion.token_ids)
                    trace=ra.split_reasoning(raw,think_end,8192)
                    natural=bool(raw and raw[-1]==think_end and trace.think_end_reached and len(raw)<=8192 and completion.finish_reason=='stop')
                    ctx=list(ra.build_decision_context(r['prompt_token_ids'],trace.token_ids,separator)) if natural else []
                    metrics=getattr(output,'metrics',None)
                    timing={k:getattr(metrics,k,None) for k in ('arrival_time','first_scheduled_time','first_token_time','last_token_time','finished_time')} if metrics else {}
                    per_seconds=timing.get('finished_time')-timing.get('arrival_time') if timing.get('finished_time') is not None and timing.get('arrival_time') is not None else None
                    entry=dict(**common(r,model,freeze),status='natural_complete' if natural else 'truncated_or_wrong_boundary',
                        generation_seed=r['generation_seed'],raw_generated_token_ids=raw,reasoning_text=tok.decode(raw,skip_special_tokens=False),
                        natural_end=natural,think_end_reached=trace.think_end_reached,reasoning_truncated=not natural,
                        reasoning_token_count=len(raw),finish_reason=completion.finish_reason,stop_reason=completion.stop_reason,
                        generated_requests=1,engine_batch_seconds=elapsed,engine_batch_size=len(generate),request_timing=timing,request_elapsed_seconds=per_seconds,
                        final_context_token_ids=ctx,final_context_text=tok.decode(ctx,skip_special_tokens=False) if ctx else None,
                        final_context_ids_sha256=cp.sha(cp.canonical(ctx)) if ctx else None)
                    cp.immutable(f.trace_path(root,r),entry); lengths.append(len(raw))
                    print('REASONED',split,r['task_id'],r['variant'],r['k'],len(raw),natural,flush=True)
            score_prompts=[]; score_index=[]; prepared={}
            for position,r in enumerate(chunk):
                trace=cp.load(f.trace_path(root,r))
                if not trace['natural_end']: continue
                ctx=trace['final_context_token_ids']; context=trace['final_context_text']
                paths,forms=f.paths_for(tok,context,ctx); prepared[position]=(r,trace,paths,forms)
                for path in paths:
                    score_prompts.append(TokensPrompt(prompt_token_ids=ctx+list(path)))
                    score_index.append((position,path,len(ctx)))
            if score_prompts:
                t=time.monotonic()
                try: outputs=llm.generate(score_prompts,SamplingParams(max_tokens=1,temperature=0.,prompt_logprobs=0),use_tqdm=False)
                except Exception as e:
                    for r in chunk: failure(root,model,r,'decision_infrastructure',e)
                    raise
                elapsed=time.monotonic()-t; scoring_seconds+=elapsed; values={}; components={}
                for (position,path,nctx),output in zip(score_index,outputs,strict=True):
                    probabilities=[float(output.prompt_logprobs[nctx+i][token].logprob) for i,token in enumerate(path)]
                    values[position,path]=sum(probabilities);components[position,path]=probabilities
                for position,(r,trace,paths,forms) in prepared.items():
                    likelihood={path:values[position,path] for path in paths}
                    scores,events=f.finish_scores(forms,likelihood)
                    for name,e in events.items(): e['token_conditional_log_probabilities']=components[position,tuple(e['token_ids'])]
                    result=dict(**common(r,model,freeze),status='ok',scores=scores,answer_events=events,
                        reasoning_path=str(f.trace_path(root,r).relative_to(pathlib.Path(root))),reasoning_sha256=cp.sha(f.trace_path(root,r).read_bytes()),
                        final_context_ids_sha256=trace['final_context_ids_sha256'],scoring_requests=len(paths),
                        scoring_extra_generated_tokens=len(paths),teacher_forced_answer_token_positions=sum(map(len,paths)),
                        forward_calls=None,forward_count_note='Engine forward invocations not instrumented; shared batched execution, not independent samples.',
                        reasoning_generation_requests=0,decision_batch_seconds=elapsed,decision_batch_contexts=len(prepared))
                    cp.immutable(f.score_path(root,model,r),result)
            done+=len(chunk)
            print('Q3_CHUNK_SAVED',split,done,len(pending),'seconds',round(time.monotonic()-begin,1),flush=True)
        if split=='smoke' and done:
            import numpy as np
            estimate=smoke_estimate(root,model,time.monotonic()-begin,done,load_seconds,dict(reasoning_lengths=lengths,
                mean_reasoning_tokens=float(np.mean(lengths)) if lengths else None,p95_reasoning_tokens=float(np.quantile(lengths,.95)) if lengths else None,
                measured_generation_seconds=generated_seconds,measured_decision_seconds=scoring_seconds))
            if estimate+180>remaining_seconds(): print('BUDGET_GATE_Q3_MAIN_DEFERRED',flush=True); return

def run(root,stage):
    import torch
    signal.signal(signal.SIGUSR1,stop_handler)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.zeros(8,dtype=torch.bfloat16,device='cuda').sum().item()
    if stage=='immediate':
        for model in ('llama','gemma'):
            run_immediate(root,model)
    else: run_qwen3(root)
