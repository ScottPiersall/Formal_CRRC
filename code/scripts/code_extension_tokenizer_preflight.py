"""Read-only pinned inventory and all development prompt checks; no model scoring."""
import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from formalcrrc import code_extension_inference as inf
root=pathlib.Path(__file__).resolve().parents[1]
cache='/REDACTED_LOCAL_PATH'
inf.inventory(root,cache)
inf.prepare_prompts(root,cache,development=True)
print('DEVELOPMENT_TOKENIZER_PREFLIGHT_PASS',flush=True)
