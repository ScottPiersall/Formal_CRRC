"""Named-RPC engineering recovery, before the first actual reasoning request."""
import pathlib,sys,os
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_extension as cp,code_five as f

def main():
    o=f.out(ROOT);freeze=o/'qwen3_engine_meter_v2_freeze.json'
    original=o/'qwen3_engine_meter_freeze.json'
    cp.assert_manifest(ROOT,cp.load(original)['files'])
    if '--freeze' in sys.argv:
        assert not list((o/'reasoning/qwen3').rglob('*.json'))
        assert not list((o/'scores/qwen3').rglob('*.json'))
        files=[ROOT/'src/formalcrrc/code_five_q3_meter_v2.py',pathlib.Path(__file__),original]
        cp.immutable(freeze,dict(created_at=cp.now(),files=cp.manifest(ROOT,files),supersedes_sha256=cp.sha(original.read_bytes()),
            reason='803692 failed during meter initialization before any reasoning request because callable RPC requires disabled pickle fallback. Named worker_extension_cls and method strings are supported by the inspected installed source; keep serialization safeguards enabled.',
            scientific_protocol_unchanged=True,main_reasoning_requests_before_amendment=0,smoke_reasoning_requests_before_amendment=0))
        print('NAMED_RPC_METER_FROZEN',cp.sha(freeze.read_bytes()));return
    cp.assert_manifest(ROOT,cp.load(freeze)['files']);f.verify_frozen(ROOT,'qwen3')
    from importlib.metadata import distribution
    dist=distribution('vllm')
    for name,expected in cp.load(original)['runtime_source_hashes'].items():assert cp.sha(pathlib.Path(dist.locate_file(name)).read_bytes())==expected
    import vllm
    from formalcrrc.code_five_q3_meter_v2 import named_rpc_metered_class
    vllm.LLM=named_rpc_metered_class(vllm.LLM,ROOT)
    from formalcrrc.code_five_inference import run
    run(ROOT,'qwen3')
if __name__=='__main__':main()
