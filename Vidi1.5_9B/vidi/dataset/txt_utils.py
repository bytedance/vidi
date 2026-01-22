"""
Copyright 2025 Intelligent Editing Team.
"""
from typing import Sequence, Dict, List, Tuple
import langid
import re
from collections import Counter

import torch
import transformers

from vidi.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def preprocess_mm(source: Sequence[str], data_args) -> Dict:
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence['value']:
            sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
            sentence['value'] = sentence['value'].strip()

    return source


def tokenize(
    conversation:str,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool
):
    if has_image:
        input_ids = tokenizer_image_token(conversation, tokenizer, return_tensors="pt")
    else:
        input_ids = tokenizer(
            conversation,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]
    
    return input_ids


def chat_template(
    source: List[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    roles_chat: Tuple[str] = ("user", "assistant"),
    roles_data: Tuple[str] = ("human", "gpt")
) -> str:
    messages = []
    for i, sentence in enumerate(source):
        assert sentence["from"] == roles_data[i%2]
        messages.append({
            "role": roles_chat[i%2], "content": sentence["value"]
        })
    conversation = tokenizer.apply_chat_template(messages, tokenize=False)
    if tokenizer.bos_token:
        conversation = conversation.replace(tokenizer.bos_token, "")

    return conversation


def chat_template_gemma2(
    source: List[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    roles_chat: Tuple[str] = ("user", "model"),
    roles_data: Tuple[str] = ("human", "gpt"),
    generation: bool = False
) -> str:
    conversation = chat_template(source, tokenizer, roles_chat, roles_data)
    if generation:
        conversation += "<start_of_turn>model\n"

    return conversation


def targets_gemma2(
    conversation: str,
    input_ids: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> torch.Tensor:

    targets = input_ids.clone()
    cur_len = 1  # bos token
    targets[:cur_len] = IGNORE_INDEX

    sep_round = f"<start_of_turn>user\n"
    sep_part = f"<start_of_turn>model\n"
    
    for rou in conversation.split(sep_round):
        if rou == "": continue

        parts = rou.split(sep_part)
        assert len(parts) == 2

        if has_image:
            round_len = len(tokenizer_image_token(rou, tokenizer)) + 2
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) + 5
        else:
            round_len = len(tokenizer(rou).input_ids) + 2
            instruction_len = len(tokenizer(parts[0]).input_ids) + 5

        targets[cur_len-1 : cur_len+instruction_len] = IGNORE_INDEX

        cur_len += round_len

    if cur_len < tokenizer.model_max_length and cur_len != len(targets):
        targets[:] = IGNORE_INDEX
        print(f"WARNING: tokenization mismatch: {cur_len} vs. {len(targets)}. (ignored)")
    
    return targets


def preprocess_conv(
    source: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conversation = chat_template_gemma2(source, tokenizer)
    input_ids = tokenize(conversation, tokenizer, has_image)
    targets = targets_gemma2(conversation, input_ids, tokenizer, has_image)
    
    return dict(input_ids=input_ids, labels=targets)


def preprocess_chat(
    source: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer
) -> str:
    conversation = chat_template_gemma2(source, tokenizer, generation=True)
    
    return conversation
