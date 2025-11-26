from model.lmm.dattn.mistral import DattnMistralForCausalLM, DattnMistralConfig

def get_dattn_cls(model_name_or_path):
    if "mistral" in model_name_or_path.lower():
        return DattnMistralForCausalLM
    else:
        raise NotImplementedError(f"Unsupported model type: {model_name_or_path}")