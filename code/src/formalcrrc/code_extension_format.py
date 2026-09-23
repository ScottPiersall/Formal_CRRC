"""Predeclared k=0 development probe of whitespace-prefixed answer events."""
import gc,math,os,pathlib,time
from formalcrrc import code_extension as cp
from formalcrrc import code_extension_inference as inf
from formalcrrc import scoring
from formalcrrc import code_extension_answer_tokens as answer_tokens

PREFIXES=('', ' ', '\n')

def freeze(root):
    root=pathlib.Path(root); out=cp.outdir(root)
    files=[root/'src/formalcrrc/code_extension_format.py',root/'scripts/code_extension_allocation.py']
    cp.immutable(out/'development_format_freeze.json',dict(created_at=cp.now(),
        scope='All 30 v1 main tasks, k=0 only, both templates and two judges, bf16. Separate exploratory formatting diagnosis, not added to the new main sample.',
        answer_events=['A',' A','\nA','B',' B','\nB'],
        rule='Exact contextual token-sequence likelihoods, deduplicate identical sequences, reject prefix-overlapping sequences, sum disjoint event probabilities per label. Do not select a prefix based on results.',
        planned_logical_probes=120,max_teacher_forced_forwards=720,files=cp.manifest(root,files)))

def logsumexp(values):
    peak=max(values); return peak+math.log(sum(math.exp(v-peak) for v in values))

def run(root,cache):
    import torch
    root=pathlib.Path(root); out=cp.outdir(root)
    frozen=cp.load(out/'development_format_freeze.json'); cp.assert_manifest(root,frozen['files'])
    cp.assert_development(root)
    for key in cp.MODELS:
        rows=[r for r in cp.load(out/'development_prompts'/f'{key}.json')['rows'] if r['k']==0]
        pending=[r for r in rows if not (out/'development_format_scores'/key/r['variant']/f'{cp.slug(r["task_id"])}.json').exists()]
        if not pending: continue
        model,tok=inf.load_model(key,cache)
        for row in pending:
            start=time.monotonic(); prompt=row['rendered_prompt']; paths={}; candidates={}
            for label in ['A','B']:
                for prefix in PREFIXES:
                    answer=prefix+label
                    ids,_=answer_tokens.resolve(tok,prompt,answer)
                    if ids in paths and paths[ids]!=label: raise RuntimeError('Answer event shared by opposite labels')
                    paths[ids]=label; candidates[answer]=ids
            ids_list=list(paths)
            if any(len(a)<len(b) and b[:len(a)]==a for a in ids_list for b in ids_list): raise RuntimeError('Answer events overlap as token prefixes')
            encoded=tok(prompt,return_tensors='pt',add_special_tokens=False)['input_ids'].to(model.device)
            values={}
            for ids in paths: values[ids]=scoring._sequence_loglikelihood(model,tok,encoded,ids)
            totals={label:logsumexp([value for ids,value in values.items() if paths[ids]==label]) for label in ['A','B']}
            result=dict(created_at=cp.now(),task_id=row['task_id'],model_key=key,variant=row['variant'],k=0,dtype='bfloat16',**cp.MODELS[key],
                margin=totals['A']-totals['B'],p_met=scoring.normalized_probability(totals['A'],totals['B']),
                answer_events={text:dict(token_ids=list(ids),log_probability=values[ids]) for text,ids in candidates.items()},
                unique_event_count=len(paths),forward_calls=len(paths),elapsed_seconds=time.monotonic()-start,job_id=os.environ.get('SLURM_JOB_ID'),
                rendered_prompt_sha256=row['rendered_prompt_sha256'],freeze_sha256=cp.sha((out/'development_format_freeze.json').read_bytes()))
            cp.immutable(out/'development_format_scores'/key/row['variant']/f'{cp.slug(row["task_id"])}.json',result)
            print('FORMAT_PROBE',key,row['variant'],row['task_id'],flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
