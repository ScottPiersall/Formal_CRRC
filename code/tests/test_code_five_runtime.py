import json
from types import SimpleNamespace as NS
from formalcrrc.code_five_q3_runtime_v3 import bounded_class

def test_bounded_runtime_records_actual_finished_output_without_changing_tokens(tmp_path,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','123')
    p=tmp_path/'artifacts/code_extension_v3_five_models/prompts/qwen3.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'rows':[dict(prompt_token_ids=[7,8],task_id='HumanEval/1',split='main',variant='original',k=2,generation_seed=37)]}))
    output=NS(finished=True,prompt_token_ids=[7,8],request_id='0',outputs=[NS(token_ids=[91,92],finish_reason='stop',stop_reason=92)])
    class Base:
        def __init__(self,**kwargs):
            self.options=kwargs;self.count=0;self.llm_engine=NS(step=lambda:[output])
        def collective_rpc(self,method,**kwargs):return [dict(forward_calls=self.count,scheduled_token_positions=self.count,zero_token_runner_calls=0)]
        def generate(self,*args,**kwargs):
            self.count+=1;return self.llm_engine.step()
    llm=bounded_class(Base,tmp_path)(max_num_batched_tokens=8192)
    result=llm.generate([{'prompt_token_ids':[7,8]}],NS(max_tokens=8192))
    assert result[0] is output and output.outputs[0].token_ids==[91,92]
    assert llm.options['max_num_batched_tokens']==512 and llm.options['enable_chunked_prefill'] is True
    receipt=json.loads(next((p.parents[1]/'completion_receipts').rglob('*.json')).read_text())
    assert receipt['seed']==37 and receipt['raw_generated_token_ids']==[91,92]
    assert output.metrics.finished_time>=output.metrics.arrival_time
    assert output.metrics.finished_time==receipt['observed_finished_wall_time']

def test_recovery_budget_includes_preserved_generation_cost(tmp_path,monkeypatch):
    from formalcrrc import code_five as f,code_extension as cp
    from formalcrrc.code_five_q3_runtime_v3 import recovered_smoke_estimate
    monkeypatch.setenv('SLURM_JOB_ID','recovery')
    monkeypatch.setenv('FIVE_DEADLINE_EPOCH','4102444800')
    o=f.out(tmp_path);rows=[]
    for i in range(54):
        row=dict(task_id=f'HumanEval/{i}',split='smoke',variant='original',k=0);rows.append(row)
        cp.immutable(f.trace_path(tmp_path,row),dict(natural_end=i<52,reasoning_token_count=100))
        if i<52:cp.immutable(f.score_path(tmp_path,'qwen3',row),{})
    rows += [dict(task_id=f'HumanEval/{100+i}',split='main',variant='original',k=0) for i in range(2)]
    cp.immutable(o/'prompts/qwen3.json',dict(rows=rows))
    runtime=o/'runtime';runtime.mkdir()
    (runtime/'qwen3_engine_calls_initial.jsonl').write_text(json.dumps(dict(split='smoke',phase='reasoning',status='ok',logical_requests=54,elapsed_seconds=54))+'\n')
    estimate=recovered_smoke_estimate(tmp_path,'qwen3',30,52,10,dict(measured_decision_seconds=26))
    assert estimate==4.5  # 2 main rows * (1 s reasoning + .5 s scoring) * 1.5 margin.
    saved=cp.load(o/'smoke_estimate_qwen3_recovery.json')
    assert saved['measured_generation_seconds']==54 and saved['truncated_smoke_contexts']==2
