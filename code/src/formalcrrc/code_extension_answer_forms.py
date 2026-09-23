"""Paired confirmation with an explicit union of semantically equal answer forms."""
import gc,math,os,time,traceback
from formalcrrc import code_extension as cp
from formalcrrc import code_extension_inference as inf
from formalcrrc import scoring
from formalcrrc.code_extension_format import PREFIXES,logsumexp
from formalcrrc import code_extension_answer_tokens as answer_tokens

def paths_for(tok,prompt):
    paths={}; forms={}
    for label in ['A','B']:
        for prefix in PREFIXES:
            text=prefix+label; ids,_=answer_tokens.resolve(tok,prompt,text)
            if ids in paths and paths[ids]!=label: raise RuntimeError('Opposite labels share a token event')
            paths[ids]=label; forms[text]=ids
    if any(len(a)<len(b) and b[:len(a)]==a for a in paths for b in paths): raise RuntimeError('Overlapping answer events')
    return paths,forms

def score(model,tok,row):
    import torch
    prompt=row['rendered_prompt']; paths,forms=paths_for(tok,prompt)
    ids=tok(prompt,return_tensors='pt',add_special_tokens=False)['input_ids'].to(model.device)
    if ids.shape[1]+max(map(len,paths))>32768: raise RuntimeError('Context overflow; no truncation')
    prefixes=sorted({path[:i] for path in paths for i in range(len(path))},key=lambda x:(len(x),x))
    next_logp={}
    for prefix in prefixes:
        full=torch.cat([ids,torch.tensor([list(prefix)],dtype=ids.dtype,device=ids.device)],dim=1) if prefix else ids
        with torch.inference_mode(): logits=model(input_ids=full,use_cache=False).logits[0,-1,:].float()
        logp=torch.log_softmax(logits,dim=-1)
        needed={p[len(prefix)] for p in paths if len(p)>len(prefix) and p[:len(prefix)]==prefix}
        next_logp[prefix]={token:float(logp[token]) for token in needed}
    likelihood={path:sum(next_logp[path[:i]][token] for i,token in enumerate(path)) for path in paths}
    total={label:logsumexp([value for path,value in likelihood.items() if paths[path]==label]) for label in ['A','B']}
    bare=likelihood[forms['A']]-likelihood[forms['B']]
    return dict(score_met=total['A'],score_not_met=total['B'],margin=total['A']-total['B'],
        p_met=scoring.normalized_probability(total['A'],total['B']),bare_margin_recomputed=bare,
        full_vocab_union_mass=sum(math.exp(v) for v in likelihood.values()),
        answer_events={text:dict(token_ids=list(path),log_probability=likelihood[path]) for text,path in forms.items()},
        prefix_nodes=[dict(token_ids=list(p),next_log_probabilities={str(k):v for k,v in next_logp[p].items()}) for p in prefixes],
        forward_calls=len(prefixes),n_prompt_tokens=int(ids.shape[1]))

def run(root,cache):
    import torch
    out=cp.outdir(root); cp.assert_freeze_a(root); cp.assert_freeze_b(root)
    spec=cp.load(out/'answer_form_protocol.json')
    assert spec['main_tasks']==80 and spec['main_scores']==2880 and spec['smoke_scores']==108
    if not (out/'main_judge_complete.json').exists(): raise RuntimeError('Baseline panel must finish before paired representation job')
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    for key in cp.MODELS:
        rows=cp.load(out/'prompts'/f'{key}.json')['rows']; model=None
        manifest={(r['task_id'],r['variant'],r['k']):r for r in cp.load(out/'prompts'/f'{key}__answer_tokens.json')['rows']}
        for split in ['smoke','main']:
            selected=[r for r in rows if r['split']==split]
            pending=[r for r in selected if not (out/'scores_answer_forms'/split/key/r['variant']/f'{cp.score_key(r["task_id"],r["k"])}.json').exists()]
            if split=='main' and len(list((out/'scores_answer_forms/smoke'/key).rglob('*.json')))!=54: raise RuntimeError('54 paired-format smoke rows required per model')
            if not pending: continue
            if model is None:
                model,tok=inf.load_model(key,cache); inf.runtime_record(root,'answer_forms',key)
            for row in pending:
                dest=out/'scores_answer_forms'/split/key/row['variant']/f'{cp.score_key(row["task_id"],row["k"])}.json'
                attempts=out/'attempts/answer_forms'/split/key/row['variant']
                previous=list(attempts.glob(f'{cp.score_key(row["task_id"],row["k"])}__*.json'))
                for attempt in range(len(previous),3):
                    if split=='smoke' and len(list((out/'attempts/answer_forms/smoke').rglob('*.json')))>=120: raise RuntimeError('Answer-form smoke attempt cap')
                    entry=dict(task_id=row['task_id'],k=row['k'],attempt=attempt,started_at=cp.now()); start=time.monotonic()
                    try:
                        assert scoring.render_chat_prompt(tok,row['user_message'])==row['rendered_prompt']
                        result=score(model,tok,row); torch.cuda.synchronize()
                        expected=manifest[row['task_id'],row['variant'],row['k']]['answer_forms']
                        assert all(result['answer_events'][name]['token_ids']==expected[name]['token_ids'] for name in expected)
                        bare=cp.load(cp.score_path(out,split,key,row['variant'],'bfloat16',row['task_id'],row['k']))
                        result.update(created_at=cp.now(),model_key=key,**cp.MODELS[key],variant=row['variant'],dtype='bfloat16',representation='whitespace_union',
                            split=split,task_id=row['task_id'],k=row['k'],job_id=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-start,
                            rendered_prompt_sha256=row['rendered_prompt_sha256'],freeze_B_sha256=cp.sha((out/'freeze_B.json').read_bytes()),
                            protocol_sha256=cp.sha((out/'answer_form_protocol.json').read_bytes()),bare_margin_delta=result['bare_margin_recomputed']-bare['margin'])
                        cp.immutable(dest,result); entry.update(status='score_saved',forward_calls=result['forward_calls'])
                        cp.immutable(attempts/f'{cp.score_key(row["task_id"],row["k"])}__{attempt}.json',entry)
                        print('ANSWER_FORMS',split,key,row['variant'],row['task_id'],row['k'],flush=True); break
                    except Exception as e:
                        entry.update(status='infrastructure_error',error=str(e),traceback=traceback.format_exc())
                        cp.immutable(attempts/f'{cp.score_key(row["task_id"],row["k"])}__{attempt}.json',entry)
                        if dest.exists() or attempt==2: raise
                        torch.cuda.empty_cache()
                else: raise RuntimeError('Answer-form retry cap exhausted')
        if model is not None: del model
        gc.collect(); torch.cuda.empty_cache()
    cp.immutable(out/'answer_forms_complete.json',dict(created_at=cp.now(),job_id=os.environ.get('SLURM_JOB_ID'),main_scores=2880,smoke_scores=108))
