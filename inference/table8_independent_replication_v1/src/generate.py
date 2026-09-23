"""Use the exact original candidate generator; redirect only output and freeze gates."""
import pathlib,sys
O=pathlib.Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(O/'preregistration/reused_source'))
from formalcrrc import code_extension as cp, code_extension_inference as old
from protocol import load,sha
def verify(_):
 p=O/'preregistration/stage1_design_freeze.json';f=load(p)
 for rel,h in f['files'].items():assert sha((O/rel).read_bytes())==h,rel
 # The unmodified generator records freeze_A_sha256, preserved as an exact alias.
 assert (O/'freeze_A.json').read_bytes()==p.read_bytes()
 return f
def main():
 cp.outdir=lambda _:O
 cp.assert_freeze_a=verify
 old.generate(O,'/REDACTED_LOCAL_PATH')
if __name__=='__main__':main()
