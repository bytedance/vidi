from model.lmm.dattn.mistral import DattnMistralForCausalLM

def get_dattn_cls(model_name_or_path):
    return DattnMistralForCausalLM