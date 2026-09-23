#!/usr/bin/env python3
"""Independent code extension v2 phases; all commands are resumable."""
import argparse
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from formalcrrc import code_extension as cp

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['init-development','diagnose','summarize-development','prepare','sandbox-smoke','inventory','freeze-a','generate','prepare-prompts','verify-truth','freeze','smoke','run','analyze','package'])
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1])
    p.add_argument('--cache-dir',default='/REDACTED_LOCAL_PATH')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--model',choices=['all','mistral','qwen'],default='all')
    a=p.parse_args(); s=a.stage
    if s=='init-development': cp.init_development(a.root)
    elif s=='prepare': cp.prepare(a.root)
    elif s=='sandbox-smoke': cp.sandbox_smoke(a.root)
    elif s=='freeze-a': cp.freeze_a(a.root)
    elif s=='verify-truth': cp.verify_truth(a.root,a.workers)
    elif s=='freeze': cp.freeze_b(a.root)
    elif s in ['summarize-development','analyze','package']:
        from formalcrrc import code_extension_analysis as an
        getattr(an,s.replace('-','_'))(a.root)
    else:
        from formalcrrc import code_extension_inference as inf
        if s=='inventory': inf.inventory(a.root,a.cache_dir)
        elif s=='generate': inf.generate(a.root,a.cache_dir)
        elif s=='prepare-prompts': inf.prepare_prompts(a.root,a.cache_dir); inf.continue_allocation(a.root,a.cache_dir)
        elif s=='diagnose': inf.run(a.root,a.cache_dir,model_key=a.model,development=True)
        else: inf.run(a.root,a.cache_dir,'smoke' if s=='smoke' else 'main',a.model)
    return 0
if __name__=='__main__': raise SystemExit(main())
