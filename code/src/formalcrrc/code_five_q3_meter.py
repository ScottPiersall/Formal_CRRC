"""Observe vLLM scheduled forward executions without changing sampling or logits."""
import functools,json,os,pathlib,time
from formalcrrc import code_extension as cp

def install_meter(worker):
    runner=worker.model_runner
    if hasattr(worker,'_five_counts'): raise RuntimeError('Duplicate execution meter')
    assert worker.vllm_config.parallel_config.tensor_parallel_size==1
    assert worker.vllm_config.parallel_config.pipeline_parallel_size==1
    assert not worker.vllm_config.parallel_config.use_ubatching
    assert worker.vllm_config.speculative_config is None
    worker._five_counts=dict(forward_calls=0,scheduled_token_positions=0,zero_token_runner_calls=0)
    original=runner.execute_model
    @functools.wraps(original)
    def counted(scheduler_output,*args,**kwargs):
        n=int(scheduler_output.total_num_scheduled_tokens)
        if n:
            worker._five_counts['forward_calls']+=1
            worker._five_counts['scheduled_token_positions']+=n
        else:worker._five_counts['zero_token_runner_calls']+=1
        return original(scheduler_output,*args,**kwargs)
    runner.execute_model=counted
    return dict(worker._five_counts)

def read_meter(worker):return dict(worker._five_counts)

def metered_class(base,root):
    root=pathlib.Path(root);o=root/'artifacts/code_extension_v3_five_models'
    prompts=cp.load(o/'prompts/qwen3.json')['rows']
    class MeteredLLM(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.collective_rpc(install_meter,timeout=30)
            self._five_call_index=0
        def generate(self,prompts_,sampling_params=None,*args,**kwargs):
            before=self.collective_rpc(read_meter,timeout=30)
            started=time.monotonic()
            status='ok';error=None
            try:return super().generate(prompts_,sampling_params,*args,**kwargs)
            except Exception as exc:status='infrastructure_failure';error=type(exc).__name__;raise
            finally:
                after=self.collective_rpc(read_meter,timeout=30)
                delta={key:sum(r[key] for r in after)-sum(r[key] for r in before) for key in before[0]}
                first=prompts_[0]['prompt_token_ids'] if prompts_ else []
                matched=[r for r in prompts if first[:len(r['prompt_token_ids'])]==r['prompt_token_ids']]
                split=matched[0]['split'] if matched else 'unknown'
                params=sampling_params[0] if isinstance(sampling_params,list) else sampling_params
                phase='reasoning' if params.max_tokens==8192 else 'decision'
                record=dict(created_at=cp.now(),job_id=os.environ.get('SLURM_JOB_ID'),call_index=self._five_call_index,
                    phase=phase,split=split,logical_requests=len(prompts_),status=status,error_type=error,
                    elapsed_seconds=time.monotonic()-started,**delta,
                    definition='Positive-scheduled-token GPUModelRunner.execute_model invocations after initialization, including CUDA graph replay. TP=PP=1, no microbatching, no speculation. Zero-token calls excluded. Shared across requests, never independent samples.')
                path=o/'runtime'/f'qwen3_engine_calls_{os.environ.get("SLURM_JOB_ID","local")}.jsonl'
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('a') as handle:handle.write(json.dumps(record)+'\n');handle.flush();os.fsync(handle.fileno())
                self._five_call_index+=1
    return MeteredLLM
