"""Pinned offline generation and immediate scoring; code is never executed here."""
from __future__ import annotations
import gc
import json
import os
import pathlib
import time
import traceback

from formalcrrc import scoring, provenance
from formalcrrc import code_extension as cp

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

def prepare_prompts(root,cache,development=False):
    out=cp.outdir(root)
    if development: cp.assert_development(root)
    else: cp.assert_freeze_a(root)
    data=out/'development' if development else out
    target=out/'development_prompts' if development else out/'prompts'
    items=cp.load(data/'instances.json')
    for key in cp.MODELS:
        dest=target/f'{key}.json'
        if dest.exists(): continue
        tok,config=tokenizer_for(key,cache); rows=[]; methods=set()
        for item in items:
            code=cp.load(data/'generation'/f'{cp.slug(item["task_id"])}.json')['code']
            for variant in cp.VARIANTS:
                norms=set()
                for k in range(9):
                    entry=cp.prompt(item,code,k,variant); raw=entry['user_message'].encode(); a,b=entry['threshold_span']
                    if raw[a:b]!=str(k).encode() or cp.sha(raw[:a]+b'{threshold}'+raw[b:])!=entry['normalized_sha256']: raise RuntimeError('Threshold invariance failed')
                    norms.add(entry['normalized_sha256'])
                    rendered=scoring.render_chat_prompt(tok,entry['user_message'])
                    tokens=scoring.resolve_label_tokenization(tok,rendered); methods.add(tokens.scoring_method)
                    n=len(tok.encode(rendered,add_special_tokens=False))
                    if n+8>min(config.max_position_embeddings,32768): raise RuntimeError('Context limit; no truncation')
                    rows.append(dict(task_id=item['task_id'],split='development' if development else item['split'],variant=variant,k=k,
                        model_key=key,**entry,rendered_prompt=rendered,rendered_prompt_sha256=cp.sha(rendered),n_prompt_tokens=n,label_tokenization=tokens.to_dict()))
                if len(norms)!=1: raise RuntimeError('Prompt changes outside threshold')
        if methods!={'single_token_next_logit'}: raise RuntimeError('Unexpected tokenizer change: requires explicit protocol amendment before scoring')
        rows.sort(key=lambda r:cp.deterministic_order(f'{key}/{r["variant"]}/{r["task_id"]}/{r["k"]}','requests'))
        cp.immutable(dest,dict(created_at=cp.now(),model_key=key,**cp.MODELS[key],chat_template_sha256=cp.sha(tok.chat_template),rows=rows))
        print('PROMPTS',key,len(rows),'max',max(r['n_prompt_tokens'] for r in rows),flush=True)
    if development:
        path=out/'development_prompt_freeze.json'
        if path.exists(): cp.assert_manifest(pathlib.Path(root),cp.load(path)['files'])
        else: cp.immutable(path,dict(created_at=cp.now(),files=cp.manifest(pathlib.Path(root),list(target.glob('*.json')))))

def score_diagnostic(model,tok,row,greedy=False,teacher_forced=False):
    """Full-vocabulary evidence and independent label likelihood cross-check."""
    import torch
    tokens=scoring.resolve_label_tokenization(tok,row['rendered_prompt'])
    if tokens.to_dict()!=row['label_tokenization']: raise RuntimeError('Continuation token drift')
    if scoring.render_chat_prompt(tok,row['user_message'])!=row['rendered_prompt']: raise RuntimeError('Template drift')
    inputs=tok(row['rendered_prompt'],return_tensors='pt',add_special_tokens=False)
    inputs={k:v.to(model.device) for k,v in inputs.items()}
    with torch.inference_mode(): logits=model(**inputs).logits[0,-1,:].float()
    ia,ib=tokens.met_token_ids[0],tokens.not_met_token_ids[0]
    a,b=float(logits[ia]),float(logits[ib]); logp=torch.log_softmax(logits,dim=-1)
    la,lb=float(logp[ia]),float(logp[ib]); margin=a-b
    values,ids=torch.topk(logp,5)
    result=dict(score_met=a,score_not_met=b,margin=margin,p_met=scoring.normalized_probability(a,b),
        full_vocab_p_met=float(logp[ia].exp()),full_vocab_p_not_met=float(logp[ib].exp()),
        logprob_margin=la-lb,algebra_error=abs(margin-(la-lb)),forward_calls=1,
        top_tokens=[dict(token_id=i,text=tok.decode([i]),log_probability=v) for i,v in zip(ids.tolist(),values.tolist())])
    if teacher_forced:
        ta,tb,n=scoring.score_sequence_loglikelihood(model,tok,row['rendered_prompt'],tokens)
        result.update(teacher_forced_margin=ta-tb,teacher_forced_delta=ta-tb-margin,forward_calls=3)
    if greedy:
        with torch.inference_mode(): output=model.generate(**inputs,do_sample=False,max_new_tokens=8,pad_token_id=tok.eos_token_id)
        generated=output[0,inputs['input_ids'].shape[1]:].tolist()
        result.update(greedy_token_ids=generated,greedy_text=tok.decode(generated,skip_special_tokens=True),
                      greedy_generation_forward_calls=len(generated),greedy_max_new_tokens=8)
    torch.cuda.synchronize()
    return result

def score_rows(root,model,tok,key,rows,dtype,development=False):
    import torch
    out=cp.outdir(root)
    for row in rows:
        split=row['split']
        base=out/'development_scores' if development else out/'scores'/split
        dest=base/key/row['variant']/dtype/f'{cp.score_key(row["task_id"],row["k"])}.json'
        if dest.exists(): continue
        attempt_dir=out/'attempts'/('development' if development else split)/key/row['variant']/dtype
        previous=list(attempt_dir.glob(f'{cp.score_key(row["task_id"],row["k"])}__*.json'))
        for attempt in range(len(previous),3):
            if split=='smoke' and len(list((out/'attempts/smoke').rglob('*.json')))>=120: raise RuntimeError('120 smoke attempt cap')
            start=time.monotonic(); entry=dict(task_id=row['task_id'],variant=row['variant'],dtype=dtype,k=row['k'],attempt=attempt,started_at=cp.now())
            try:
                result=score_diagnostic(model,tok,row,greedy=development and dtype=='bfloat16' and row['k']==0,
                    teacher_forced=development and row['variant']=='original' and row['k']==0)
                if result['algebra_error']>1e-5: raise RuntimeError('Logit/log-probability algebra mismatch')
                freeze='development_freeze.json' if development else 'freeze_B.json'
                result.update(task_id=row['task_id'],variant=row['variant'],dtype=dtype,k=row['k'],split=split,model_key=key,**cp.MODELS[key],
                    created_at=cp.now(),job_id=os.environ.get('SLURM_JOB_ID'),n_prompt_tokens=row['n_prompt_tokens'],
                    label_tokenization=row['label_tokenization'],rendered_prompt_sha256=row['rendered_prompt_sha256'],
                    freeze_sha256=cp.sha((out/freeze).read_bytes()),elapsed_seconds=time.monotonic()-start)
                cp.immutable(dest,result); entry.update(status='score_saved',elapsed_seconds=result['elapsed_seconds'])
                cp.immutable(attempt_dir/f'{cp.score_key(row["task_id"],row["k"])}__{attempt}.json',entry)
                print('SCORED',split,key,row['variant'],dtype,row['task_id'],row['k'],round(result['elapsed_seconds'],3),flush=True)
                break
            except Exception as e:
                entry.update(status='infrastructure_error',error=str(e),traceback=traceback.format_exc())
                cp.immutable(attempt_dir/f'{cp.score_key(row["task_id"],row["k"])}__{attempt}.json',entry)
                if dest.exists() or attempt==2: raise
                torch.cuda.empty_cache()
        else: raise RuntimeError('Retry cap exhausted')

def run(root,cache,split='main',model_key='all',development=False):
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    out=cp.outdir(root)
    if development:
        cp.assert_development(root); inventory(root,cache); prepare_prompts(root,cache,True)
        cp.assert_manifest(pathlib.Path(root),cp.load(out/'development_prompt_freeze.json')['files'])
    else: cp.assert_freeze_a(root); cp.assert_freeze_b(root)
    numeric=set() if development else set(cp.load(out/'numeric_task_ids.json'))
    for key in cp.MODELS if model_key=='all' else [model_key]:
        rows=cp.load(out/('development_prompts' if development else 'prompts')/f'{key}.json')['rows']
        if not development: rows=[r for r in rows if r['split']==split]
        if not development and split=='main':
            smoke=list((out/'scores/smoke').rglob('*.json'))
            if len(smoke)!=108: raise RuntimeError('108 disjoint smoke scores required')
        model=None
        for dtype in cp.DTYPES:
            selected=rows if dtype=='bfloat16' or development else [r for r in rows if r['task_id'] in numeric and split=='main']
            base=out/'development_scores' if development else out/'scores'/split
            pending=[r for r in selected if not (base/key/r['variant']/dtype/f'{cp.score_key(r["task_id"],r["k"])}.json').exists()]
            if not pending: continue
            if model is None: model,tok=load_model(key,cache)
            # Preserve pretrained float32 rotary buffers during the bf16 baseline.
            # Casting an already-bf16 model again would round those buffers and
            # break exact replication of the pilot loading procedure.
            if model.dtype!=getattr(torch,dtype): model=model.to(dtype=getattr(torch,dtype))
            model=model.eval(); torch.cuda.empty_cache()
            if any(p.is_floating_point() and p.dtype!=getattr(torch,dtype) for p in model.parameters()): raise RuntimeError('Incomplete dtype conversion')
            cp.immutable(out/'runtime'/f'{"development" if development else split}_{key}_{dtype}_{os.environ.get("SLURM_JOB_ID","local")}.json',
                dict(created_at=cp.now(),model=key,dtype=dtype,attention_backend='sdpa',job_id=os.environ.get('SLURM_JOB_ID'),environment=provenance.environment_snapshot(True),
                    float32_matmul_precision=torch.get_float32_matmul_precision(),cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
            score_rows(root,model,tok,key,pending,dtype,development)
        if model is not None: del model
        gc.collect(); torch.cuda.empty_cache()

def continue_allocation(root,cache):
    out=cp.outdir(root)
    cp.immutable(out/'generation_and_prompts_complete.json',dict(created_at=cp.now(),job_id=os.environ.get('SLURM_JOB_ID'),generation_responses=83,rendered_prompts=2988))
    print('WAITING_FOR_LOCAL_TRUTH_AND_FREEZE_B',flush=True)
    deadline=time.monotonic()+900
    while not (out/'freeze_B_checksum.json').exists():
        if time.monotonic()>deadline: raise RuntimeError('Bounded truth handoff expired; resume smoke/run separately')
        time.sleep(5)
    cp.assert_freeze_a(root); cp.assert_freeze_b(root)
    run(root,cache,'smoke')
    smoke=[cp.load(p) for p in (out/'scores/smoke').rglob('*.json')]
    if len(smoke)!=108: raise RuntimeError('Incomplete smoke panel')
    estimate=sum(s['elapsed_seconds'] for s in smoke)/len(smoke)*3240*3+360
    cp.immutable(out/'main_runtime_estimate.json',dict(created_at=cp.now(),smoke_requests=108,conservative_seconds=estimate,
        formula='Mean smoke request time x 3240 main requests x 3 safety factor + 360 seconds model/dtype overhead; full allocation including handoff remains subject to SLURM one-hour cap.'))
    cp.immutable(out/'judge_smoke_complete.json',dict(created_at=cp.now(),logical_rows=108))
    run(root,cache,'main')
    cp.immutable(out/'main_judge_complete.json',dict(created_at=cp.now(),logical_rows=len(list((out/'scores/main').rglob('*.json')))))
