"""
Copyright 2025 Intelligent Editing Team.
"""
# coding=utf-8
# Copyright 2024 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional
import torch
from torch import distributed as dist

from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal
assert is_flash_attn_2_available()
assert is_flash_attn_greater_or_equal("2.6.0")

from transformers.modeling_flash_attention_utils import _get_unpad_data, fa_peft_integration_check
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
from flash_attn import flash_attn_func, flash_attn_varlen_func

from vidi.model.lmm.dattn.sequence_parallel.ulysses_attn import UlyssesAttention
from vidi.model.lmm.dattn.sequence_parallel.globals import get_ulysses_sp_pg, get_ulysses_sp_size


def _unpad_xattn_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    q_attention_mask: torch.Tensor,
    kv_attention_mask: torch.Tensor,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens even though they belong to different batches.

    This function is used instead of `flash_attn.bert_padding.unpad_input` in order to avoid the recomputation of the same intermediary
    tensors for query, key, value tensors.

    Arguments:
        query_layer (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        q_attention_mask (`torch.Tensor`):
            Attention mask for query state. Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        kv_attention_mask (`torch.Tensor`):
            Attention mask for key and value states. Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
    Return:
        query_layer (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(kv_attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )

    query_length = query_layer.shape[1]
    if query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        assert q_attention_mask.shape[1] == query_length
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, q_attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def _flash_attn_varlen_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    q_attention_mask: torch.Tensor,
    kv_attention_mask: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    **flash_kwargs
):
    batch_size, query_length = query_states.shape[:2]
    query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _unpad_xattn_input(
        query_states, key_states, value_states, q_attention_mask, kv_attention_mask
    )
    cu_seqlens_q, cu_seqlens_k = cu_seq_lens
    max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

    attn_output_unpad = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_in_batch_q,
        max_seqlen_k=max_seqlen_in_batch_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=False,
        **flash_kwargs,
    )
    attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    
    return attn_output


def flash_cross_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    q_attention_mask: torch.Tensor,
    kv_attention_mask: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    softcap: Optional[float] = None,
    deterministic: bool = None,
):    
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        q_attention_mask (`torch.Tensor`):
            The padding mask for query - corresponds to a tensor of size `(batch_size, seq_len_q)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        kv_attention_mask (`torch.Tensor`):
            The padding mask for key and values - corresponds to a tensor of size `(batch_size, seq_len_kv)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout_p (`float`):
            Attention dropout_p
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    # FA2 uses non-transposed inputs
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query_states.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype
    
    # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )

    flash_kwargs = {}

    if deterministic is None:
        deterministic = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
    flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap
    
    assert (q_attention_mask is not None) == (kv_attention_mask is not None)
    use_varlen = (q_attention_mask is not None and kv_attention_mask is not None) and \
        (0 in q_attention_mask or 0 in kv_attention_mask)

    # Contains at least one padding token in the sequence
    if use_varlen:
        if dist.is_initialized() and get_ulysses_sp_size() > 1:
            attn_output = UlyssesAttention(_flash_attn_varlen_func, get_ulysses_sp_pg())(
                query_states,
                key_states,
                value_states,
                q_attention_mask=q_attention_mask,
                kv_attention_mask=kv_attention_mask,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                **flash_kwargs
            )
        else:
            attn_output = _flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                q_attention_mask=q_attention_mask,
                kv_attention_mask=kv_attention_mask,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                **flash_kwargs
            )
    else:
        if dist.is_initialized() and get_ulysses_sp_size() > 1:
            attn_output = UlyssesAttention(flash_attn_func, get_ulysses_sp_pg())(
                query_states,
                key_states,
                value_states,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=False,
                **flash_kwargs
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=False,
                **flash_kwargs
            )

    return attn_output
