"""Observability-only Q3 launcher; frozen scoring implementation remains unchanged."""
import pathlib,sys,hashlib,json,os
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_extension as cp,code_five as f

def main():
    o=f.out(ROOT)
    files=[ROOT/'src/formalcrrc/code_five_q3_meter.py',pathlib.Path(__file__),o/'freeze_qwen3.json']
    freeze=o/'qwen3_engine_meter_freeze.json'
    if '--freeze' in sys.argv:
        from importlib.metadata import distribution
        dist=distribution('vllm')
        runtime_sources={}
        for name in ['vllm/entrypoints/llm.py','vllm/v1/worker/gpu_worker.py','vllm/v1/worker/gpu_model_runner.py']:
            p=pathlib.Path(dist.locate_file(name));runtime_sources[name]=cp.sha(p.read_bytes())
        cp.immutable(freeze,dict(created_at=cp.now(),files=cp.manifest(ROOT,files),runtime_source_hashes=runtime_sources,
            scope='Observability-only addition before any Q3 smoke/main inference. No changes to prompt, reasoning sample, seed, logits, answer paths or analysis endpoints.',
            definition='Count positive-token runner.execute_model calls after model initialization. One worker, TP=PP=1, no microbatching or speculative decoding; record phase, split and shared logical request count separately.'))
        print('Q3_METER_FROZEN',cp.sha(freeze.read_bytes()));return
    cp.assert_manifest(ROOT,cp.load(freeze)['files'])
    f.verify_frozen(ROOT,'qwen3')
    import vllm
    from formalcrrc.code_five_q3_meter import metered_class
    vllm.LLM=metered_class(vllm.LLM,ROOT)
    from formalcrrc.code_five_inference import run
    run(ROOT,'qwen3')
if __name__=='__main__':main()
