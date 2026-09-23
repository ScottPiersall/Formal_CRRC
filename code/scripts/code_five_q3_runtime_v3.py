"""OOM recovery without regenerating saved reasoning or changing scoring events."""
import pathlib,sys,os
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_extension as cp,code_five as f

def main():
    o=f.out(ROOT);freeze=o/'qwen3_runtime_v3_freeze.json'
    for name in ['qwen3_engine_meter_freeze.json','qwen3_engine_meter_v2_freeze.json']:
        cp.assert_manifest(ROOT,cp.load(o/name)['files'])
    if '--freeze' in sys.argv:
        files=[ROOT/'src/formalcrrc/code_five_q3_runtime_v3.py',pathlib.Path(__file__),o/'qwen3_engine_meter_v2_freeze.json']
        traces=list((o/'reasoning/qwen3').rglob('*.json'))
        assert len(traces)==54 and not list((o/'reasoning/qwen3/main').rglob('*.json'))
        cp.immutable(freeze,dict(created_at=cp.now(),files=cp.manifest(ROOT,files),preserved_smoke_traces=cp.manifest(ROOT,traces),
            max_num_batched_tokens=512,enable_chunked_prefill=True,
            reason='803705 completed all 54 smoke reasoning samples (52 natural, 2 budget-truncated), then CUDA OOM while computing full-vocabulary prompt logprobs. Limit prefill workspace; no context truncation and no altered likelihood, answer forms, seeds or reasoning budget.',
            retry='Recover only missing decision scores for saved natural traces; do not regenerate any saved trace. Main traces gain durable final-output receipts and observed completion latency. Preserve original failures and meter versions.',
            failure_reclassification='Raw EngineDeadError entries were broadly labeled scoring_or_protocol_error; native engine traceback establishes a CUDA allocation OOM infrastructure failure. The 54 row entries describe the failed chunk; only 52 natural contexts had decision requests.',
            main_reasoning_before_amendment=0))
        print('BOUNDED_Q3_RUNTIME_FROZEN',cp.sha(freeze.read_bytes()));return
    cp.assert_manifest(ROOT,cp.load(freeze)['files']);cp.assert_manifest(ROOT,cp.load(freeze)['preserved_smoke_traces']);f.verify_frozen(ROOT,'qwen3')
    import vllm
    from formalcrrc.code_five_q3_runtime_v3 import bounded_class,restore_receipts,recovered_smoke_estimate
    restore_receipts(ROOT)
    vllm.LLM=bounded_class(vllm.LLM,ROOT)
    from formalcrrc import code_five_inference as inference
    inference.smoke_estimate=recovered_smoke_estimate
    inference.run(ROOT,'qwen3')
if __name__=='__main__':main()
