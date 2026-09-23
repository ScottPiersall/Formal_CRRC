from types import SimpleNamespace as NS
from formalcrrc.code_five_q3_meter import install_meter,read_meter

def test_execution_meter_preserves_result_and_excludes_empty_steps():
    expected=object();calls=[]
    def execute(scheduler,*args,**kwargs):calls.append(scheduler.total_num_scheduled_tokens);return expected
    worker=NS(model_runner=NS(execute_model=execute),vllm_config=NS(speculative_config=None,
        parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,use_ubatching=False)))
    install_meter(worker)
    for n in [32,0,8]:assert worker.model_runner.execute_model(NS(total_num_scheduled_tokens=n)) is expected
    assert calls==[32,0,8]
    assert read_meter(worker)==dict(forward_calls=2,scheduled_token_positions=40,zero_token_runner_calls=1)

def test_named_rpc_adapter_sends_only_strings(tmp_path):
    import json
    from formalcrrc.code_five_q3_meter_v2 import named_rpc_metered_class
    p=tmp_path/'artifacts/code_extension_v3_five_models/prompts/qwen3.json'
    p.parent.mkdir(parents=True);p.write_text(json.dumps({'rows':[]}))
    class Base:
        def __init__(self,**kwargs):self.options=kwargs;self.calls=[]
        def collective_rpc(self,method,**kwargs):
            assert isinstance(method,str);self.calls.append(method);return [{}]
    model=named_rpc_metered_class(Base,tmp_path)()
    model.collective_rpc(read_meter)
    assert model.calls==['five_install_meter','five_read_meter']
    assert model.options['worker_extension_cls']=='formalcrrc.code_five_q3_meter_v2.FiveMeterWorkerExtension'
