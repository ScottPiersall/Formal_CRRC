import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from formalcrrc import code_extension_answer_forms as forms
from formalcrrc import code_extension as cp
if not (cp.outdir(pathlib.Path(__file__).resolve().parents[1])/'main_judge_complete.json').exists():
    from code_extension_allocation import main as recover_baseline
    recover_baseline()
forms.run(pathlib.Path(__file__).resolve().parents[1],'/REDACTED_LOCAL_PATH')
from formalcrrc import code_extension_answer_precision as precision
precision.run(pathlib.Path(__file__).resolve().parents[1],'/REDACTED_LOCAL_PATH')
