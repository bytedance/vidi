"""
Copyright 2025 Intelligent Editing Team.
"""
from .lmm.dattn.gemma import DattnGemma2ForCausalLM

def get_dattn_cls(model_name_or_path):
    return DattnGemma2ForCausalLM