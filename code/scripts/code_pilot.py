#!/usr/bin/env python3
"""Independent code_pilot_v1 CLI. Run --help for phase dependencies."""
import argparse
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from formalcrrc import code_pilot as cp

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','sandbox-smoke','inventory','freeze-a','generate','verify-truth',
        'prepare-prompts','freeze','smoke','run','check-integrity','analyze','package'])
    parser.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument('--cache-dir',default='/REDACTED_LOCAL_PATH')
    parser.add_argument('--split',choices=['all','main','smoke'],default='all')
    parser.add_argument('--model',choices=['all','mistral','qwen'],default='all')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(); stage=args.stage
    if stage=='prepare': cp.prepare(args.root,args.workers)
    elif stage=='sandbox-smoke': cp.sandbox_smoke(args.root)
    elif stage=='verify-truth': cp.verify_truth(args.root,args.split,args.workers)
    elif stage=='freeze-a': cp.freeze_a(args.root)
    elif stage=='freeze': cp.freeze_b(args.root)
    elif stage=='check-integrity':
        result=cp.integrity(args.root); cp.snapshot(cp.outdir(args.root)/'integrity.json',result)
        print(cp.canonical(result)); return 0 if result['passed'] else 2
    elif stage in ['analyze','package']:
        from formalcrrc import code_pilot_analysis as analysis
        getattr(analysis,stage)(args.root)
    else:
        from formalcrrc import code_pilot_inference as inference
        if stage=='inventory': inference.inventory(args.root,args.cache_dir)
        elif stage=='generate': inference.generate(args.root,args.cache_dir,args.split)
        elif stage=='prepare-prompts':
            inference.prepare_prompts(args.root,args.cache_dir)
            inference.continue_allocation(args.root,args.cache_dir)
        else: inference.run(args.root,args.cache_dir,'smoke' if stage=='smoke' else 'main',args.model)
    return 0

if __name__=='__main__': raise SystemExit(main())
