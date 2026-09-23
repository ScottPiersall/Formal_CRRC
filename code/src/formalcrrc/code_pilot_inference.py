"""Pinned offline generation and immediate scoring; code is never executed here."""
from __future__ import annotations
import gc
import json
import os
import pathlib
import time
import traceback

from formalcrrc import scoring, provenance
from formalcrrc import code_pilot as cp

def tokenizer_for(key,cache):
    from transformers import AutoTokenizer, AutoConfig
    spec=cp.MODELS[key]
    kwargs=dict(revision=spec['revision'],cache_dir=cache,local_files_only=True,trust_remote_code=False)
    config=AutoConfig.from_pretrained(spec['model_id'],**kwargs)
    if config._commit_hash!=spec['revision']: raise RuntimeError('Resolved config commit mismatch')
    tokenizer=AutoTokenizer.from_pretrained(spec['model_id'],**kwargs)
    return tokenizer,config

def inventory(root,cache):
    records={}
    for key,spec in cp.MODELS.items():
        tok,config=tokenizer_for(key,cache)
        snap=pathlib.Path(cache)/('models--'+spec['model_id'].replace('/','--'))/'snapshots'/spec['revision']
        idx=cp.load(snap/'model.safetensors.index.json')
        files={}
        for p in sorted(snap.iterdir()):
            if p.is_file() and (p.suffix in ('.json','.jinja') or p.name.endswith('.model')):
                files[p.name]=dict(sha256=cp.sha(p.read_bytes()),size=p.stat().st_size,blob=p.resolve().name)
        shards=[]
        for name in sorted(set(idx['weight_map'].values())):
            p=snap/name
            if not p.is_file(): raise RuntimeError('Missing weight shard: '+str(p))
            shards.append(dict(name=name,size=p.stat().st_size,blob=p.resolve().name))
        records[key]=dict(**spec,resolved_config_commit=config._commit_hash,snapshot=str(snap),files=files,shards=shards,
                          max_position_embeddings=config.max_position_embeddings,
                          chat_template_sha256=cp.sha(tok.chat_template),model_type=config.model_type)
    result=dict(created_at=cp.now(),cache_dir=cache,models=records,environment=provenance.environment_snapshot(False),
                revision_basis='Existing Day-1 fixed model_id/revision records, independently resolved by offline AutoConfig from repo_id+revision, checked cached configs and complete shard index. Not inferred from folder name alone.')
    path=cp.outdir(root)/'model_inventory.json'
    if path.exists():
        old=cp.load(path)
        if old['models']!=result['models']: raise RuntimeError('Model inventory drift')
    else: cp.immutable(path,result)
    print('Verified model inventory:',','.join(records),flush=True)

def load_model(key,cache):
    import torch
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA bf16 GPU unavailable')
    spec=cp.MODELS[key]; tok,config=tokenizer_for(key,cache)
    model=AutoModelForCausalLM.from_pretrained(spec['model_id'],revision=spec['revision'],cache_dir=cache,
        local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa')
    model=model.to('cuda').eval()
    if model.dtype!=torch.bfloat16 or model.training or getattr(model,'is_quantized',False):
        raise RuntimeError('Model violates bf16/unquantized/eval protocol')
    if getattr(model.config,'_attn_implementation',None)!='sdpa': raise RuntimeError('Attention backend drift')
    return model,tok

def runtime_record(root,stage,key):
    record=dict(created_at=cp.now(),stage=stage,model=key,environment=provenance.environment_snapshot(True),
                job_id=os.environ.get('SLURM_JOB_ID'),dtype='bfloat16',attention_backend='sdpa')
    cp.immutable(cp.outdir(root)/'runtime'/f'{stage}_{key}_{os.environ.get("SLURM_JOB_ID","local")}.json',record)
    return record

def generate(root,cache,split='all'):
    import torch
    out=cp.outdir(root); frozen=cp.assert_freeze_a(root); key=frozen['protocol']['generator']
    items=[i for i in cp.load(out/'instances.json') if split=='all' or i['split']==split]
    pending=[i for i in items if not (out/'generation'/f'{cp.slug(i["task_id"])}.json').exists()]
    if not pending: print('Generation already complete'); return
    inventory(root,cache)
    model,tok=load_model(key,cache); runtime=runtime_record(root,'generation',key)
    for item in pending:
        dest=out/'generation'/f'{cp.slug(item["task_id"])}.json'
        message=frozen['protocol']['generation']['message'].format(problem=item['problem'])
        rendered=scoring.render_chat_prompt(tok,message)
        inputs=tok(rendered,return_tensors='pt',add_special_tokens=False)
        n=int(inputs['input_ids'].shape[1])
        if n+1024>32768: raise RuntimeError('Generator context limit; no truncation')
        previous=list((out/'attempts/generation').glob(f'{cp.slug(item["task_id"])}__*.json'))
        for attempt in range(len(previous),3):
            start=time.monotonic(); entry=dict(task_id=item['task_id'],attempt=attempt,started_at=cp.now())
            try:
                with torch.inference_mode():
                    output=model.generate(**{k:v.to('cuda') for k,v in inputs.items()},do_sample=False,
                                          max_new_tokens=1024,pad_token_id=tok.eos_token_id)
                torch.cuda.synchronize()
                tokens=output[0,n:].tolist(); response=tok.decode(tokens,skip_special_tokens=True)
                code,extraction=cp.extract(response)
                row=dict(task_id=item['task_id'],split=item['split'],model_key=key,**cp.MODELS[key],
                  created_at=cp.now(),user_message=message,rendered_prompt=rendered,n_prompt_tokens=n,
                  rendered_prompt_sha256=cp.sha(rendered),response=response,generated_token_ids=tokens,
                  generated_tokens=len(tokens),truncated=len(tokens)>=1024 and tokens[-1]!=tok.eos_token_id,
                  code=code,code_sha256=cp.sha(code),**extraction,elapsed_seconds=time.monotonic()-start,
                  generation_forward_calls=len(tokens),runtime=runtime,freeze_A_sha256=cp.sha((out/'freeze_A.json').read_bytes()))
                cp.immutable(dest,row)
                entry.update(status='response_saved',elapsed_seconds=row['elapsed_seconds'],generated_tokens=len(tokens))
                cp.immutable(out/'attempts/generation'/f'{cp.slug(item["task_id"])}__{attempt}.json',entry)
                print('GENERATED',item['task_id'],len(tokens),round(row['elapsed_seconds'],2),flush=True)
                break
            except Exception as e:
                entry.update(status='infrastructure_error',error=str(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
                cp.immutable(out/'attempts/generation'/f'{cp.slug(item["task_id"])}__{attempt}.json',entry)
                if dest.exists() or attempt==2: raise
                torch.cuda.empty_cache()
        else: raise RuntimeError('Infrastructure retry allowance exhausted: '+item['task_id'])
    del model; gc.collect(); torch.cuda.empty_cache()

def prepare_prompts(root,cache):
    out=cp.outdir(root); cp.assert_freeze_a(root)
    instances=cp.load(out/'instances.json')
    for key in cp.MODELS:
        dest=out/'prompts'/f'{key}.json'
        if dest.exists(): continue
        tok,config=tokenizer_for(key,cache); rows=[]; methods=set()
        for item in instances:
            code=cp.load(out/'generation'/f'{cp.slug(item["task_id"])}.json')['code']
            norms=set()
            for k in range(9):
                entry=cp.prompt(item,code,k); norms.add(entry['normalized_sha256'])
                data=entry['user_message'].encode(); a,b=entry['threshold_span']
                if data[a:b]!=str(k).encode() or cp.sha(data[:a]+b'{threshold}'+data[b:])!=entry['normalized_sha256']:
                    raise RuntimeError('Threshold byte invariance failed')
                rendered=scoring.render_chat_prompt(tok,entry['user_message'])
                tokens=scoring.resolve_label_tokenization(tok,rendered)
                methods.add(tokens.scoring_method)
                n=len(tok.encode(rendered,add_special_tokens=False))
                if n+max(len(tokens.met_token_ids),len(tokens.not_met_token_ids))>min(config.max_position_embeddings,32768):
                    raise RuntimeError('Prompt over context; no truncation: '+item['task_id'])
                rows.append(dict(task_id=item['task_id'],split=item['split'],k=k,model_key=key,**entry,
                                 rendered_prompt=rendered,rendered_prompt_sha256=cp.sha(rendered),
                                 n_prompt_tokens=n,label_tokenization=tokens.to_dict()))
            if len(norms)!=1: raise RuntimeError('Prompt invariance failed')
        method='single_token_next_logit' if methods=={'single_token_next_logit'} else 'sequence_loglikelihood'
        rows.sort(key=lambda r:cp.deterministic_order(f'{key}/{r["task_id"]}/{r["k"]}','requests'))
        cp.immutable(dest,dict(model_key=key,**cp.MODELS[key],scoring_method=method,rows=rows,
                      chat_template_sha256=cp.sha(tok.chat_template),created_at=cp.now()))
        print('PROMPTS',key,len(rows),'max tokens',max(r['n_prompt_tokens'] for r in rows),method,flush=True)

def run(root,cache,split='main',model_key='all'):
    import torch
    out=cp.outdir(root)
    cp.assert_freeze_b(root)
    for key in cp.MODELS if model_key=='all' else [model_key]:
        prep=cp.load(out/'prompts'/f'{key}.json')
        rows=[r for r in prep['rows'] if r['split']==split]
        pending=[r for r in rows if not (out/'scores'/split/key/f'{cp.score_key(r["task_id"],r["k"])}.json').exists()]
        if not pending: continue
        if split=='main':
            smoke=list((out/'scores/smoke'/key).glob('*.json'))
            if len(smoke)!=27: raise RuntimeError('27 disjoint smoke scores required per model')
        model,tok=load_model(key,cache); runtime=runtime_record(root,split,key)
        for row in pending:
            dest=out/'scores'/split/key/f'{cp.score_key(row["task_id"],row["k"])}.json'
            tokens=scoring.resolve_label_tokenization(tok,row['rendered_prompt'])
            if tokens.to_dict()!=row['label_tokenization']: raise RuntimeError('Continuation tokens changed')
            if scoring.render_chat_prompt(tok,row['user_message'])!=row['rendered_prompt']: raise RuntimeError('Chat template changed')
            attempts_dir=out/'attempts'/split/key
            previous=list(attempts_dir.glob(f'{cp.score_key(row["task_id"],row["k"])}__*.json'))
            for attempt in range(len(previous),3):
                if split=='smoke':
                    used=sum(1 for p in (out/'attempts/smoke').rglob('*.json'))
                    if used>=60: raise RuntimeError('60 judge smoke attempts cap reached')
                start=time.monotonic(); entry=dict(task_id=row['task_id'],k=row['k'],attempt=attempt,started_at=cp.now())
                try:
                    method=prep['scoring_method']
                    scorer=scoring.score_single_token if method=='single_token_next_logit' else scoring.score_sequence_loglikelihood
                    a,b,n=scorer(model,tok,row['rendered_prompt'],tokens); torch.cuda.synchronize()
                    result=dict(task_id=row['task_id'],k=row['k'],split=split,model_key=key,**cp.MODELS[key],
                       score_met=a,score_not_met=b,margin=a-b,p_met=scoring.normalized_probability(a,b),
                       scoring_method=method,label_tokenization=tokens.to_dict(),n_prompt_tokens=n,
                       rendered_prompt_sha256=row['rendered_prompt_sha256'],elapsed_seconds=time.monotonic()-start,
                       forward_calls=1 if method=='single_token_next_logit' else 2,
                       created_at=cp.now(),job_id=os.environ.get('SLURM_JOB_ID'),
                       freeze_B_sha256=cp.sha((out/'freeze_B.json').read_bytes()))
                    cp.immutable(dest,result)
                    entry.update(status='score_saved',forward_calls=result['forward_calls'],elapsed_seconds=result['elapsed_seconds'])
                    cp.immutable(attempts_dir/f'{cp.score_key(row["task_id"],row["k"])}__{attempt}.json',entry)
                    print('SCORED',split,key,row['task_id'],row['k'],round(result['elapsed_seconds'],3),flush=True)
                    break
                except Exception as e:
                    entry.update(status='infrastructure_error',error=str(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
                    cp.immutable(attempts_dir/f'{cp.score_key(row["task_id"],row["k"])}__{attempt}.json',entry)
                    if dest.exists() or attempt==2: raise
                    torch.cuda.empty_cache()
            else: raise RuntimeError('Retry allowance exhausted')
        del model; gc.collect(); torch.cuda.empty_cache()

def continue_allocation(root,cache):
    """Optional bounded handoff: local isolated truth -> freeze -> smoke -> main.

    All idle and model time remains inside the existing SLURM time limit and
    eight-hour budget. The gate is an engineering readiness record within the
    user's already authorized experiment, not a request for new permission.
    """
    out=cp.outdir(root); request=out/'allocation_continuation_request.json'
    if not request.exists(): return
    spec=cp.load(request)
    if not spec.get('enabled') or spec.get('job_id')!=os.environ.get('SLURM_JOB_ID'): return
    ready=out/'generation_and_prompts_complete.json'
    cp.immutable(ready,dict(created_at=cp.now(),job_id=os.environ.get('SLURM_JOB_ID'),generation_responses=33,rendered_prompts=594))
    print('WAITING_FOR_LOCAL_TRUTH_AND_FREEZE_B',flush=True)
    deadline=time.monotonic()+spec['wait_limit_seconds']
    while not (out/'freeze_B_checksum.json').exists():
        if time.monotonic()>deadline: raise RuntimeError('Bounded local truth handoff timed out; resume smoke in another allocation')
        time.sleep(5)
    cp.assert_freeze_b(root)
    run(root,cache,'smoke')
    cp.immutable(out/'judge_smoke_complete.json',dict(created_at=cp.now(),job_id=os.environ.get('SLURM_JOB_ID'),
       logical_rows=sum(1 for p in (out/'scores/smoke').rglob('*.json'))))
    print('WAITING_FOR_MAIN_RUNTIME_ESTIMATE_GATE',flush=True)
    deadline=time.monotonic()+spec['wait_limit_seconds']
    while not (out/'main_run_gate.json').exists():
        if time.monotonic()>deadline: raise RuntimeError('Bounded timing handoff timed out; resume main in another allocation')
        time.sleep(5)
    gate=cp.load(out/'main_run_gate.json')
    if gate['freeze_B_sha256']!=cp.sha((out/'freeze_B.json').read_bytes()) or gate['smoke_logical_requests']!=54:
        raise RuntimeError('Main readiness gate is invalid')
    run(root,cache,'main')
    cp.immutable(out/'main_judge_complete.json',dict(created_at=cp.now(),job_id=os.environ.get('SLURM_JOB_ID'),
       logical_rows=sum(1 for p in (out/'scores/main').rglob('*.json'))))
