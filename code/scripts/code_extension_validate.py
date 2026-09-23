"""Run mandatory independent direct-equality validation before freeze B."""
import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from formalcrrc import code_extension_validation as validation
validation.verify(pathlib.Path(__file__).resolve().parents[1],4)
