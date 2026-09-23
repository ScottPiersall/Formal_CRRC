import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from formalcrrc import code_extension_answer_tokens as tokens
tokens.prepare(pathlib.Path(__file__).resolve().parents[1],development=False)
