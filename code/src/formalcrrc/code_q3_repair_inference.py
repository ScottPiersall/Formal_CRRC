"""Native continuation receipts and coherent conditional prefix scoring."""
import json,os,pathlib,signal,time,traceback,math
from formalcrrc import code_extension as cp,code_five as f,code_q3_repair as q

STOP=False
def stop(*_):
    global STOP
    STOP=True
def remaining():return float(os.environ.get('Q3_REPAIR_DEADLINE_EPOCH','inf'))-time.time()
def append(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as h:h.write(json.dumps(data)+'\n');h.flush();os.fsync(h.fileno())

class Runner:
    def __init__(self,root,stage):
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        self.root=pathlib.Path(root);self.o=q.out(root);self.stage=stage;self.freeze=q.verify(root)
        self.rows=cp.load(f.out(root)/'prompts/qwen3.json')['rows']
        self.meta={r['key']:r for r in cp.load(self.o/'inputs.json')['rows']}
        self.inv=cp.load(f.out(root)/'inventory_qwen3.json');self.job=os.environ.get('SLURM_JOB_ID','local')
        self.tok=AutoTokenizer.from_pretrained(str(f.snapshot('qwen3')),trust_remote_code=False)
        self.SamplingParams=SamplingParams; self.active={};self.call=None
        # Verify the installed implementation computes full-vocabulary raw logprobs.
        import vllm
        pkg=pathlib.Path(vllm.__file__).parent;source=(pkg/'v1/sample/sampler.py').read_text()
        assert 'log_softmax(dim=-1, dtype=torch.float32)' in source
        assert 'gather_specific_token_logprobs' in source and 'raw_logprobs' in source
        test=SamplingParams(max_tokens=1,temperature=0.,logprob_token_ids=[32,33])
        assert test.num_logprobs==2
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        engine=cp.load(self.o/'protocol.json')['engine'];start=time.monotonic()
        self.llm=LLM(model=str(f.snapshot('qwen3')),tokenizer=str(f.snapshot('qwen3')),dtype='bfloat16',quantization=None,
            worker_extension_cls='formalcrrc.code_five_q3_meter_v2.FiveMeterWorkerExtension',**engine)
        self.load_seconds=time.monotonic()-start
        self.llm.collective_rpc('five_install_meter',timeout=30)
        cp.immutable(self.o/'runtime'/f'{self.job}.json',dict(created_at=cp.now(),job_id=self.job,stage=stage,load_seconds=self.load_seconds,
            model=f.MODELS['qwen3'],dtype='bfloat16',quantization=None,engine=engine,
            torch=torch.__version__,vllm=vllm.__version__,source_files={str(p.relative_to(pkg)):cp.sha(p.read_bytes()) for p in [pkg/'v1/sample/sampler.py',pkg/'sampling_params.py']},
            max_context_token_limit=engine['max_model_len'],forward_definition='Positive-token GPUModelRunner.execute_model invocations after initialization; shared across requests, TP=PP=1, not independent samples.'))
        original=self.llm.llm_engine.step
        def observed(*args,**kwargs):
            outputs=original(*args,**kwargs)
            for output in outputs:
                if getattr(output,'finished',False):self.save_receipt(output)
            return outputs
        self.llm.llm_engine.step=observed

    def common(self,row):
        return dict(**{k:row[k] for k in ('task_id','split','variant','k','z','user_message_sha256','rendered_prompt_sha256')},
            created_at=cp.now(),model_key='qwen3',**f.MODELS['qwen3'],dtype='bfloat16',quantization=None,
            job_id=self.job,freeze_sha256=self.freeze,original_trace_sha256=self.meta[f.key(row)]['original_trace_sha256'])

    def save_receipt(self,output):
        if not self.active:return
        item=self.active.get(tuple(output.prompt_token_ids))
        if item is None:raise RuntimeError('Unexpected output context')
        row,prefix,needed,dest=item
        if dest.exists():return
        c=output.outputs[0]
        data=dict(**self.common(row),request_id=output.request_id,phase=self.call['phase'],
            prompt_ids_sha256=cp.sha(cp.canonical(list(output.prompt_token_ids))),
            raw_generated_token_ids=list(c.token_ids),finish_reason=c.finish_reason,stop_reason=c.stop_reason,
            submitted_wall_time=self.call['started_wall_time'],observed_finished_wall_time=time.time(),
            timing_note='Batch submission to observed completion; includes queueing, not exclusive GPU time.')
        if prefix is None:data['continuation_seed']=q.seed(row)
        else:
            returned=c.logprobs[0]
            assert all(t in returned for t in needed)
            data.update(token_ids=list(prefix),next_log_probabilities={str(t):float(returned[t].logprob) for t in needed},
                conditional_vector='One full-vocabulary raw log_softmax; outgoing token values selected after normalization',
                ignored_scoring_generated_tokens=len(c.token_ids))
        cp.immutable(dest,data)

    def generate(self,items,phase,split):
        if not items:return
        prompts=[];params=[];self.active={}
        for row,ids,prefix,needed,dest in items:
            assert not dest.exists() and len(ids)<q.MODEL_LENGTH
            self.active[tuple(ids)]=(row,prefix,needed,dest);prompts.append(dict(prompt_token_ids=ids))
            if phase=='continuation':
                params.append(self.SamplingParams(n=1,temperature=.6,top_p=.95,top_k=20,min_p=0.,max_tokens=q.ADDITIONAL_BUDGET,
                    seed=q.seed(row),stop_token_ids=[self.inv['think_end_id']],include_stop_str_in_output=True))
            else:params.append(self.SamplingParams(max_tokens=1,temperature=0.,logprob_token_ids=needed))
        assert len(self.active)==len(items)
        # Count infrastructure submissions per logical request before starting.
        history=[json.loads(s) for p in (self.o/'runtime').glob('engine_calls_*.jsonl') for s in p.read_text().splitlines()]
        keys=[str(x[-1].relative_to(self.o)) for x in items]
        for key in keys:assert sum(key in r.get('request_keys',[]) and r.get('event')=='submitted' for r in history)<2
        log=self.o/'runtime'/f'engine_calls_{self.job}.jsonl'
        self.call=dict(created_at=cp.now(),job_id=self.job,phase=phase,split=split,request_keys=keys,
            logical_requests=len(items),prompt_token_positions=sum(len(x[1]) for x in items),started_wall_time=time.time())
        append(log,dict(self.call,event='submitted'))
        before=self.llm.collective_rpc('five_read_meter',timeout=30);t=time.monotonic();status='ok';error=None;delta=None
        try:
            outputs=self.llm.generate(prompts,params,use_tqdm=False)
            for output in outputs:self.save_receipt(output)
            assert all(x[-1].exists() for x in items)
        except Exception as exc:
            status='infrastructure_or_runtime_failure';error=traceback.format_exc()
            cp.immutable(self.o/'failures'/f'{self.job}_{time.time_ns()}.json',dict(self.call,status=status,error=error));raise
        finally:
            try:
                after=self.llm.collective_rpc('five_read_meter',timeout=30)
                delta={k:sum(r[k] for r in after)-sum(r[k] for r in before) for k in before[0]}
            except Exception:pass
            append(log,dict(self.call,event='completed',status=status,error=error,elapsed_seconds=time.monotonic()-t,
                meter=delta,completed_receipts=sum(x[-1].exists() for x in items)))
            self.active={}

    def restore(self,rows):
        for row in rows:
            if q.trace_path(self.root,row).exists() or not q.receipt_path(self.root,row).exists():continue
            old=cp.load(f.trace_path(self.root,row));receipt=cp.load(q.receipt_path(self.root,row))
            expected=row['prompt_token_ids']+old['raw_generated_token_ids']
            assert receipt['prompt_ids_sha256']==cp.sha(cp.canonical(expected))
            result=q.continuation(old,row,receipt,self.tok,self.inv)
            cp.immutable(q.trace_path(self.root,row),dict(**self.common(row),**result,continuation_seed=q.seed(row),
                original_generation_seed=row['generation_seed'],original_trace_path=f.trace_path(self.root,row).relative_to(self.root).as_posix(),
                receipt_path=q.receipt_path(self.root,row).relative_to(self.root).as_posix(),receipt_sha256=cp.sha(q.receipt_path(self.root,row).read_bytes()),
                request_elapsed_seconds=receipt['observed_finished_wall_time']-receipt['submitted_wall_time'],generated_requests=1))
            print('CONTINUED',f.key(row),result['additional_token_count'],result['natural_end'],flush=True)

    def current_trace(self,row):
        old=cp.load(f.trace_path(self.root,row))
        path=f.trace_path(self.root,row) if old['natural_end'] else q.trace_path(self.root,row)
        return (path,cp.load(path)) if path.exists() else (None,None)

    def score(self,rows):
        todo=[];prepared=[]
        for row in rows:
            if q.score_path(self.root,row).exists():continue
            path,trace=self.current_trace(row)
            if trace is None or not trace['natural_end']:continue
            ctx=trace['final_context_token_ids'];paths,forms=f.paths_for(self.tok,trace['final_context_text'],ctx)
            expected=self.meta[f.key(row)]['answer_forms']
            assert {k:v['token_ids'] for k,v in forms.items()}=={k:v['token_ids'] for k,v in expected.items()}
            nodes=q.trie(paths);prepared.append((row,path,trace,forms,nodes))
            for prefix,needed in nodes.items():
                dest=q.node_path(self.root,row,prefix)
                if not dest.exists():todo.append((row,ctx+list(prefix),prefix,needed,dest))
        self.generate(todo,'shared_prefix_score',rows[0]['split'])
        for row,path,trace,forms,nodes in prepared:
            entries=[cp.load(q.node_path(self.root,row,prefix)) for prefix in nodes]
            for prefix,entry in zip(nodes,entries,strict=True):
                assert entry['prompt_ids_sha256']==cp.sha(cp.canonical(trace['final_context_token_ids']+list(prefix)))
            values={tuple(e['token_ids']):{int(k):v for k,v in e['next_log_probabilities'].items()} for e in entries}
            scores,events=q.combine(forms,values)
            result=dict(**self.common(row),status='ok',scores=scores,answer_events=events,
                prefix_nodes=[dict(token_ids=e['token_ids'],next_log_probabilities=e['next_log_probabilities']) for e in entries],
                reasoning_path=path.relative_to(self.root).as_posix(),reasoning_sha256=cp.sha(path.read_bytes()),
                final_context_ids_sha256=trace['final_context_ids_sha256'],original_natural_context=self.meta[f.key(row)]['original_natural_end'],
                scoring_requests=len(nodes),scoring_extra_generated_tokens=sum(e['ignored_scoring_generated_tokens'] for e in entries),
                forward_calls=None,reasoning_generation_requests=0,
                node_receipts={q.node_path(self.root,row,p).relative_to(self.root).as_posix():cp.sha(q.node_path(self.root,row,p).read_bytes()) for p in nodes})
            cp.immutable(q.score_path(self.root,row),result)
        if prepared:print('SCORED',rows[0]['split'],len(prepared),'remaining_allocation',round(remaining()),flush=True)

    def observed_calls(self):
        return [json.loads(line) for path in (self.o/'runtime').glob('engine_calls_*.jsonl') for line in path.read_text().splitlines()
                if json.loads(line).get('event')=='completed' and json.loads(line).get('status')=='ok']

    def estimate(self):
        calls=self.observed_calls();gen=[x for x in calls if x['phase']=='continuation'];score=[x for x in calls if x['phase']=='shared_prefix_score']
        traces=[cp.load(p) for p in (self.o/'continuations').rglob('*.json')]
        added=sum(t['additional_token_count'] for t in traces)
        gen_time=sum(c['elapsed_seconds'] for c in gen);score_time=sum(c['elapsed_seconds'] for c in score)
        score_tokens=sum(c['prompt_token_positions'] for c in score)
        pending=[r for r in self.rows if r['split']=='main' and not self.meta[f.key(r)]['original_natural_end'] and not q.trace_path(self.root,r).exists()]
        natural=[r for r in self.rows if r['split']=='main' and self.meta[f.key(r)]['original_natural_end'] and not q.score_path(self.root,r).exists()]
        mean=added/max(len(traces),1);tps=added/gen_time if gen_time else 0
        position_rate=score_tokens/score_time if score_time else 0
        old_positions=sum(2*len(cp.load(f.trace_path(self.root,r))['final_context_token_ids'])+1 for r in natural)
        pending_positions=sum(2*(len(r['prompt_token_ids'])+8192+mean+len(self.inv['separator_ids']))+1 for r in pending)
        estimate=1.5*(len(pending)*mean/tps+(old_positions+pending_positions)/position_rate) if tps and position_rate else None
        return dict(created_at=cp.now(),continuation_mean_additional_tokens=mean,continuation_tokens_per_observed_engine_second=tps,
            score_prompt_positions_per_observed_engine_second=position_rate,pending_main_continuations=len(pending),pending_old_natural_main_scores=len(natural),
            main_estimated_gpu_seconds_with_50_percent_margin=estimate,remaining_allocation_seconds=remaining(),
            basis='Measured new smoke/finished batches; no concurrency speedup assumed. Length estimate is observed mean, not a guarantee. Per-batch deadline gates use maximum additional budget.')

    def run(self):
        rows=[r for r in self.rows if r['split']==self.stage]
        self.restore(rows)
        if self.stage=='main':
            smoke=[r for r in self.rows if r['split']=='smoke']
            for r in smoke:
                _,t=self.current_trace(r);assert t is not None and (not t['natural_end'] or q.score_path(self.root,r).exists())
            assert (self.o/'smoke_estimate.json').exists()
            # Repair every original natural context, including the former mass failure.
            old=[r for r in rows if self.meta[f.key(r)]['original_natural_end']]
            for i in range(0,len(old),32):
                if STOP or remaining()<180:return
                self.score(old[i:i+32])
        truncated=[r for r in rows if not self.meta[f.key(r)]['original_natural_end']]
        for i in range(0,len(truncated),8):
            batch=truncated[i:i+8];pending=[r for r in batch if not q.receipt_path(self.root,r).exists()]
            if pending:
                est=self.estimate()
                rate=est['continuation_tokens_per_observed_engine_second']
                bound=1.5*len(pending)*q.ADDITIONAL_BUDGET/rate if rate else 1200
                if STOP or remaining()<max(180,bound+180):
                    cp.immutable(self.o/'budget_stops'/f'{self.job}_{i}.json',dict(est,next_batch_upper_length_estimate_seconds=bound,pending_keys=[f.key(r) for r in pending]));return
                items=[]
                for r in pending:
                    old=cp.load(f.trace_path(self.root,r));ids=r['prompt_token_ids']+old['raw_generated_token_ids']
                    items.append((r,ids,None,None,q.receipt_path(self.root,r)))
                self.generate(items,'continuation',self.stage)
            self.restore(batch);self.score(batch)
        if self.stage=='smoke':
            for i in range(0,len(rows),32):self.score(rows[i:i+32])
            cp.immutable(self.o/'smoke_estimate.json',self.estimate())
            print('SMOKE_GATE',json.dumps(self.estimate()),flush=True)
        cp.immutable(self.o/'stages'/f'{self.stage}_{self.job}.json',dict(created_at=cp.now(),status='all_requested_rows_processed',stage=self.stage,estimate=self.estimate()))

def run(root,stage):
    assert stage in ('smoke','main')
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    runner=Runner(root,stage);runner.run()
