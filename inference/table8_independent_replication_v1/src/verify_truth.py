"""Exact original sandbox + independent direct equality audit; no judge data."""
import pathlib,sys
O=pathlib.Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(O/'preregistration/reused_source'))
from formalcrrc import code_pilot as pilot,code_extension as cp,code_extension_validation as direct
from generate import verify
def main():
 pilot.outdir=cp.outdir=lambda _:O
 pilot.assert_freeze_a=cp.assert_freeze_a=verify
 cp.verify_truth(O,workers=4)
 direct.verify(O,workers=4)
if __name__=='__main__':main()
