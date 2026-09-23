import argparse,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_five as f

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','immediate','qwen3']);p.add_argument('--model',choices=f.NEW);a=p.parse_args()
    if a.stage=='preflight': f.preflight(ROOT,a.model)
    else:
        from formalcrrc import code_five_inference as inf
        inf.run(ROOT,a.stage)
if __name__=='__main__': main()
