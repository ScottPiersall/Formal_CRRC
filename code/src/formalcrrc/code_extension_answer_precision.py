"""The same ten preselected numerical tasks, under the answer-form union."""
import gc,os,time,traceback
from formalcrrc import code_extension as cp
from formalcrrc import code_extension_inference as inf
from formalcrrc import code_extension_answer_forms as forms

def run(root,cache):
    import torch
    out=cp.outdir(root); cp.assert_freeze_a(root); cp.assert_freeze_b(root)
    task_ids=set(cp.load(out/'numeric_task_ids.json'))
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    for key in cp.MODELS:
        rows=[r for r in cp.load(out/'prompts'/f'{key}.json')['rows'] if r['task_id'] in task_ids]
        pending=[r for r in rows if not (out/'scores_answer_forms_float32'/key/r['variant']/f'{cp.score_key(r["task_id"],r["k"])}.json').exists()]
        if not pending: continue
        model,tok=inf.load_model(key,cache); model=model.float().eval(); torch.cuda.empty_cache()
        assert all(not p.is_floating_point() or p.dtype==torch.float32 for p in model.parameters())
        cp.immutable(out/'runtime'/f'answer_forms_float32_{key}_{os.environ.get("SLURM_JOB_ID")}.json',dict(created_at=cp.now(),model=key,dtype='float32',
            job_id=os.environ.get('SLURM_JOB_ID'),cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision(),environment=inf.provenance.environment_snapshot(True)))
        for row in pending:
            dest=out/'scores_answer_forms_float32'/key/row['variant']/f'{cp.score_key(row["task_id"],row["k"])}.json'
            attempt_dir=out/'attempts/answer_forms_float32'/key/row['variant']
            previous=list(attempt_dir.glob(f'{cp.score_key(row["task_id"],row["k"])}__*.json'))
            for attempt in range(len(previous),3):
                entry=dict(task_id=row['task_id'],k=row['k'],attempt=attempt,started_at=cp.now()); start=time.monotonic()
                try:
                    r=forms.score(model,tok,row); torch.cuda.synchronize()
                    r.update(created_at=cp.now(),task_id=row['task_id'],k=row['k'],model_key=key,**cp.MODELS[key],variant=row['variant'],dtype='float32',
                        split='main',representation='whitespace_union',job_id=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-start,
                        rendered_prompt_sha256=row['rendered_prompt_sha256'],freeze_B_sha256=cp.sha((out/'freeze_B.json').read_bytes()),
                        protocol_sha256=cp.sha((out/'answer_form_protocol.json').read_bytes()))
                    cp.immutable(dest,r); entry['status']='score_saved'; cp.immutable(attempt_dir/f'{cp.score_key(row["task_id"],row["k"])}__{attempt}.json',entry)
                    print('ANSWER_FORMS_FP32',key,row['variant'],row['task_id'],row['k'],flush=True); break
                except Exception as e:
                    entry.update(status='infrastructure_error',error=str(e),traceback=traceback.format_exc())
                    cp.immutable(attempt_dir/f'{cp.score_key(row["task_id"],row["k"])}__{attempt}.json',entry)
                    if dest.exists() or attempt==2: raise
                    torch.cuda.empty_cache()
            else: raise RuntimeError('Numerical answer-form retry cap')
        del model; gc.collect(); torch.cuda.empty_cache()
    cp.immutable(out/'answer_forms_float32_complete.json',dict(created_at=cp.now(),logical_scores=360,job_id=os.environ.get('SLURM_JOB_ID')))
