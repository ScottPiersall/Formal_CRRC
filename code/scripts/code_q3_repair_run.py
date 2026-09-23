import pathlib,sys,traceback,os,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_q3_repair as q,code_extension as cp
if __name__=='__main__':
    try:
        from formalcrrc.code_q3_repair_inference import run
        run(ROOT,sys.argv[1])
    except Exception:
        cp.immutable(q.out(ROOT)/'failures'/f'top_{os.environ.get("SLURM_JOB_ID","local")}_{time.time_ns()}.json',dict(created_at=cp.now(),error=traceback.format_exc()))
        raise
