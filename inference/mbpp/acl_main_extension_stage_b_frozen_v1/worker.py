"""Authorized one-slot formal judge worker; no model import on module import."""
import argparse
import os
import signal
import time
from common import *
from gates import claim
from ledger import Store,Checkpoint
from pipeline import load_rows,Runner
STOP=False
def stop(*args):
    global STOP;STOP=True
def main():
    p=argparse.ArgumentParser();p.add_argument('--slot',type=int,required=True);p.add_argument('--chunk',required=True);a=p.parse_args()
    config,token,remaining,permits=claim(a.slot,a.chunk)
    directory=WORK/'slots'/str(a.slot).zfill(2);deadline=time.monotonic()+remaining
    for s in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(s,stop)
    try:
        chunks={r['chunk_id']:r for r in read(ROOT/config['resources']['chunks'])};chunk=chunks[a.chunk];judge=chunk['judge']
        rows=load_rows(ROOT,judge,chunk);store=Store(WORK,token['stage_B_sha256'],token['job_id'],a.slot,
            can_start=lambda mode:not STOP and deadline-time.monotonic()>(300 if mode=='generation' else 45),permits=permits)
        pending=[r for r in rows if store.completed(r['result_id'],r['binding']) is None]
        if not pending:raise RuntimeError('Chunk already complete; refuse unnecessary model load')
        if STOP or deadline-time.monotonic()<600:raise Checkpoint('Insufficient time before model initialization')
        for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','VLLM_NO_USAGE_STATS','DO_NOT_TRACK']:os.environ[key]='1'
        os.environ['TOKENIZERS_PARALLELISM']='false'
        from backend import Backend
        backend=Backend(ROOT,judge,config,rows,token);save(directory/'model_loaded.json',dict(backend.load_record,stage_B_sha256=token['stage_B_sha256']))
        runner=Runner(config,judge,backend,store)
        def progress(row):print(json.dumps({'result_id':row['result_id'],'status':row['status'],'completed_utc':row['completed_utc']}),flush=True)
        try:runner.run(rows,progress)
        except Checkpoint:pass
        complete=[r['result_id'] for r in rows if store.completed(r['result_id'],r['binding']) is not None]
        save(directory/'checkpoint.json',{'job_id':token['job_id'],'slot':a.slot,'chunk_id':a.chunk,'completed_utc':now(),
           'reason':'chunk_complete' if len(complete)==len(rows) else 'clean_time_checkpoint','completed_results':complete,
           'pending_results':[r['result_id'] for r in rows if r['result_id'] not in set(complete)],'stage_B_sha256':token['stage_B_sha256']})
    except Checkpoint as exc:
        save(directory/'checkpoint.json',{'job_id':token['job_id'],'slot':a.slot,'chunk_id':a.chunk,'completed_utc':now(),'reason':'clean_preload_checkpoint','detail':str(exc),'stage_B_sha256':token['stage_B_sha256']})
    except BaseException as exc:
        save(directory/'failure.json',{'job_id':token['job_id'],'slot':a.slot,'chunk_id':a.chunk,'failed_utc':now(),'exception':type(exc).__name__,'message':str(exc),
             'stage_B_sha256':token['stage_B_sha256'],'automatic_retry':False});raise
if __name__=='__main__':main()
