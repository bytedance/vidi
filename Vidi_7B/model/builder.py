
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2025 Bytedance.
# ------------------------------------------------------------------------
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
from transformers import BitsAndBytesConfig
from model import *

def load_pretrained_model(model_name_or_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=True, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    num_try, max_try = 0, 5
    while True:
        try:
            LMM_CLS = get_dattn_cls(model_name_or_path)       
            model = LMM_CLS.from_pretrained(
                model_name_or_path, low_cpu_mem_usage=True, **kwargs
            )
            break
        except Exception as e:
            print(repr(e))
            num_try += 1
            if num_try == max_try: raise ConnectionError("Failed to download/reload model weights.")

    text_tokenizer = model.get_model().text_tokenizer
    image_processor = model.get_model().image_processor
    audio_processor = model.get_model().audio_processor
    model.generation_config.eos_token_id = model.config.eos_token_id

    return model, text_tokenizer, image_processor, audio_processor
