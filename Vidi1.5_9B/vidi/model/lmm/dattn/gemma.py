"""
Copyright 2025 Intelligent Editing Team.
"""
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

from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from liger_kernel.transformers.monkey_patch import *
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2ForCausalLM, Gemma2Model, Gemma2DecoderLayer, Gemma2Config,
    Gemma2RMSNorm, Gemma2MLP, Gemma2Attention, Gemma2RotaryEmbedding,
    repeat_kv
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.cache_utils import Cache, DynamicCache, HybridCache
from transformers.utils import logging, ModelOutput
from transformers.generation.utils import GenerateOutput
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from .multimodal import DattnMMModel, DattnMMMixin
from .outputs import DattnCausalLMOutputWithPast, DattnBaseModelOutputWithPast
from .xattn import flash_cross_attention_forward
from .ctx_fn import make_context_fn
from vidi.constants import IGNORE_INDEX
from vidi.model.lmm.dattn.split import splitted_call

logger = logging.get_logger(__name__)

ALL_LAYERNORM_LAYERS.append(Gemma2RMSNorm)


class DattnGemma2Attention(Gemma2Attention):
    def forward_xattn(
        self,
        hidden_states_q: torch.Tensor,
        attention_mask_q: Optional[torch.LongTensor] = None,
        hidden_states_kv: Optional[torch.Tensor] = None,
        attention_mask_kv: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
    ):
        query_states = self.q_proj(hidden_states_q)
        if past_key_value is not None:
            if len(past_key_value) <= self.layer_idx:
                key_states = splitted_call(self.k_proj, hidden_states_kv, self.config.mm_splits, dim_split=1)
                value_states = splitted_call(self.v_proj, hidden_states_kv, self.config.mm_splits, dim_split=1)
                past_key_value.update(key_states, value_states, self.layer_idx)
            else:
                key_states, value_states = past_key_value[self.layer_idx]
        else:
            key_states = splitted_call(self.k_proj, hidden_states_kv, self.config.mm_splits, dim_split=1)
            value_states = splitted_call(self.v_proj, hidden_states_kv, self.config.mm_splits, dim_split=1)

        bsz, q_len, _ = query_states.size()
        _, kv_len, _ = key_states.size()

        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        attn_output = flash_cross_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask_q,
            attention_mask_kv,
            dropout_p=dropout_rate,
            softmax_scale=self.scaling,
            softcap=self.config.attn_logit_softcapping,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, value_states.transpose(1, 2)


class DattnGemma2DecoderLayer(Gemma2DecoderLayer):
    def __init__(self, config: Gemma2Config, layer_idx: int):
        super(Gemma2DecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.is_sliding = not bool(layer_idx % 2)
        self.self_attn = DattnGemma2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma2MLP(config)
        self.input_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.pre_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window

        self.layer_idx = layer_idx
    
    def feed_foward(self, embeds):
        residual = embeds
        embeds = self.pre_feedforward_layernorm(embeds)
        embeds = self.mlp(embeds)
        embeds = self.post_feedforward_layernorm(embeds)
        embeds = residual + embeds

        return embeds

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        past_image_key_value: Optional[Tuple[torch.Tensor]] = None,
        past_audio_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: int = 0,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if self.training:
            assert image_embeds is not None and audio_embeds is not None
        elif image_embeds is None and audio_embeds is None:
            outputs = super().forward(
                hidden_states, position_embeddings, attention_mask, position_ids, past_key_value,
                output_attentions, use_cache, cache_position, last_cache_position, **kwargs
            )
            return outputs, image_embeds, audio_embeds
        
        if self.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
            # In prefill, we may be larger than sliding window
            effective_seq_len = max(cache_position.shape[0], self.sliding_window)
            # For FA2, the mask is 2D and is of shape [bs, processed_tokens] (not [bs, max_cache_len]),
            # thus we must slice from the right (at most `effective_seq_len` elements)
            attention_mask = attention_mask[:, -effective_seq_len:]

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # T2T Self Attention
        hidden_states_text, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        if image_embeds is not None:
            # T2V Cross Attention
            use_image_cache = (past_image_key_value is not None and len(past_image_key_value) > self.layer_idx)
            image_mask = torch.sum(image_attention_mask, dim=-1)
            _image_attention_mask = image_attention_mask.clone().detach()
            _image_attention_mask[image_mask == 0] = True
            _image_embeds = image_embeds if use_image_cache else \
                splitted_call(self.input_layernorm, image_embeds, self.config.mm_splits, dim_split=1)
            hidden_states_image, image_value_states = self.self_attn.forward_xattn(
                hidden_states_q=hidden_states,
                attention_mask_q=attention_mask,
                hidden_states_kv=_image_embeds,
                attention_mask_kv=_image_attention_mask,
                past_key_value=past_image_key_value
            )
            hidden_states_image = hidden_states_image * (image_mask != 0)[:, None, None]

            # Diagonal V2V Attn
            if not use_image_cache:
                image_value_states = image_value_states.flatten(2, 3)
                image_value_states = self.self_attn.o_proj(image_value_states)
                image_value_states = splitted_call(
                    self.post_attention_layernorm, image_value_states, self.config.mm_splits, dim_split=1
                )
                image_embeds = image_embeds + image_value_states
                image_embeds = splitted_call(self.feed_foward, image_embeds, self.config.mm_splits, dim_split=1)
        else:
            hidden_states_image = 0.0
        
        if audio_embeds is not None:
            # T2A Cross Attention
            use_audio_cache = (past_audio_key_value is not None and len(past_audio_key_value) > self.layer_idx)
            audio_mask = torch.sum(audio_attention_mask, dim=-1)
            _audio_attention_mask = audio_attention_mask.clone().detach()
            _audio_attention_mask[audio_mask == 0] = True
            _audio_embeds = audio_embeds if use_audio_cache else \
                splitted_call(self.input_layernorm, audio_embeds, self.config.mm_splits, dim_split=1)
            hidden_states_audio, audio_value_states = self.self_attn.forward_xattn(
                hidden_states_q=hidden_states,
                attention_mask_q=attention_mask,
                hidden_states_kv=_audio_embeds,
                attention_mask_kv=_audio_attention_mask,
                past_key_value=past_audio_key_value
            )
            hidden_states_audio = hidden_states_audio * (audio_mask != 0)[:, None, None]

            # Diagonal A2A Attn
            if not use_audio_cache:
                audio_value_states = audio_value_states.flatten(2, 3)
                audio_value_states = self.self_attn.o_proj(audio_value_states)
                audio_value_states = splitted_call(
                    self.post_attention_layernorm, audio_value_states, self.config.mm_splits, dim_split=1
                )
                audio_embeds = audio_embeds + audio_value_states
                audio_embeds = splitted_call(self.feed_foward, audio_embeds, self.config.mm_splits, dim_split=1)
        else:
            hidden_states_audio = 0.0

        # Residual connection for T2T attention
        hidden_states = hidden_states_text + hidden_states_image + hidden_states_audio
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_foward(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        return outputs, image_embeds, audio_embeds


Gemma2Model._no_split_modules.append("DattnGemma2DecoderLayer")
Gemma2Model._skip_keys_device_placement.extend(["past_image_key_values", "past_audio_key_values"])
class DattnGemma2Model(Gemma2Model):
    def __init__(self, config: Gemma2Config):
        super(Gemma2Model, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        assert config._attn_implementation == "flash_attention_2", "Only support FlashAttention 2 for now."

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DattnGemma2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_image_key_values: Optional[List[torch.FloatTensor]] = None,
        past_audio_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: Optional[int] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, DattnBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if use_cache and not self.training:
            if past_key_values is None:
                batch_size, seq_len, _ = inputs_embeds.shape
                past_key_values = HybridCache(
                    self.config,
                    max_batch_size=batch_size,
                    max_cache_len=seq_len,
                    dtype=inputs_embeds.dtype,
                )
            if past_image_key_values is None and image_embeds is not None:
                past_image_key_values = DynamicCache()
            if past_audio_key_values is None and audio_embeds is not None:
                past_audio_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        # This is needed to correctly slice the mask without data-dependent slicing later on if using dynamo tracing
        # (retrieving the same value from `cache_position` later on would crash dynamo)
        if last_cache_position is None:
            last_cache_position = 0
            if attention_mask is not None:
                # In case a 4d mask is passed directly without using `generate`, we have to rely on cache_position
                # It will break dynamo tracing but there are no way around it (and it should never happen in practice)
                last_cache_position = (
                    attention_mask.shape[-1] if attention_mask.dim() == 2 else cache_position[-1].item()
                )
        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # normalized
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        if image_embeds is not None: image_embeds = image_embeds * normalizer
        if audio_embeds is not None: audio_embeds = audio_embeds * normalizer

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs, image_embeds, audio_embeds = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    position_ids,
                    image_embeds,
                    image_attention_mask,
                    audio_embeds,
                    audio_attention_mask,
                    past_key_values,
                    past_image_key_values,
                    past_audio_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    last_cache_position,
                    context_fn=make_context_fn(decoder_layer)
                )
            else:
                layer_outputs, image_embeds, audio_embeds = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    image_embeds=image_embeds,
                    image_attention_mask=image_attention_mask,
                    audio_embeds=audio_embeds,
                    audio_attention_mask=audio_attention_mask,
                    past_key_value=past_key_values,
                    past_image_key_value=past_image_key_values,
                    past_audio_key_value=past_audio_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    last_cache_position=last_cache_position,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        assert return_dict
        return DattnBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            past_image_key_values=past_image_key_values if use_cache else None,
            past_audio_key_values=past_audio_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DattnGemma2Config(Gemma2Config):
    model_type = "dattn_gemma2"
    
    mm_projector_type = "mlp2x_gelu"
    mm_vision_tower = "openai/clip-vit-large-patch14"
    mm_vision_select_layer = -2
    mm_image_pool_size = None
    mm_image_aspect_ratio = "resize"
    mm_input_type = "image"
    mm_image_grid_points = [
        [1, 2], [2, 1], [2, 2],
        [1, 3], [3, 1], [1, 4], [4, 1]
    ]

    mm_audio_tower = "openai/whisper-large-v3"
    mm_audio_pool_size = None

    mm_splits = 1
    mm_std = None
    mm_time_interval = None

    loss_thres = None


class DattnGemma2MMModel(DattnMMModel, DattnGemma2Model):
    config_class = DattnGemma2Config

    def __init__(self, config: DattnGemma2Config):
        super().__init__(config)
    
    def build_text_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.name_or_path, model_max_length=4096, padding_side="right"
        )
        eos_token_id = 107
        self.config.eos_token_id = eos_token_id

        return tokenizer


class DattnGemma2ForCausalLM(Gemma2ForCausalLM, DattnMMMixin):
    config_class = DattnGemma2Config
    _keys_to_ignore_on_load_missing = [".*mm_vis.*", ".*mm_aud.*", ".*mm_rand.*"]
    accepts_loss_kwargs = False

    def __init__(self, config: DattnGemma2Config):
        super(Gemma2ForCausalLM, self).__init__(config)
        self.model = DattnGemma2MMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_image_key_values: Optional[List[torch.FloatTensor]] = None,
        past_audio_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        image_embeds: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        audios: Optional[torch.FloatTensor] = None,
        audio_sizes: Optional[List[int]] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[Tuple, DattnCausalLMOutputWithPast]:

        if images is not None or audios is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_embeds,
                image_attention_mask,
                audio_embeds,
                audio_attention_mask
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                audios,
                audio_sizes
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_embeds=image_embeds,
            image_attention_mask=image_attention_mask,
            audio_embeds=audio_embeds,
            audio_attention_mask=audio_attention_mask,
            past_key_values=past_key_values,
            past_image_key_values=past_image_key_values,
            past_audio_key_values=past_audio_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **loss_kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            labels = F.pad(labels, (0, 1), value=IGNORE_INDEX)
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            logits = logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(logits.device)

            if self.config.loss_thres is not None:
                loss = F.cross_entropy(logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="none")
                thres = 0.0 if torch.all(loss < self.config.loss_thres) else self.config.loss_thres
                loss = torch.mean(loss[loss > thres])
            else:
                loss = F.cross_entropy(logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="mean")

        assert return_dict
        return DattnCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            past_image_key_values=outputs.past_image_key_values,
            past_audio_key_values=outputs.past_audio_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        audios: Optional[torch.Tensor] = None,
        audio_sizes: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or audios is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                image_embeds,
                image_attention_mask,
                audio_embeds,
                audio_attention_mask
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes,
                audios,
                audio_sizes
            )
        else:
            image_embeds, image_attention_mask = None, None
            audio_embeds, audio_attention_mask = None, None
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            image_attention_mask=image_attention_mask,
            audio_embeds=audio_embeds,
            audio_attention_mask=audio_attention_mask,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )

        inputs['image_embeds'] = kwargs.pop("image_embeds", None)
        inputs['image_attention_mask'] = kwargs.pop("image_attention_mask", None)
        inputs['past_image_key_values'] = kwargs.pop("past_image_key_values", None)
        
        inputs['audio_embeds'] = kwargs.pop("audio_embeds", None)
        inputs['audio_attention_mask'] = kwargs.pop("audio_attention_mask", None)
        inputs['past_audio_key_values'] = kwargs.pop("past_audio_key_values", None)

        return inputs
    
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs['past_image_key_values'] = outputs.past_image_key_values
        model_kwargs['past_audio_key_values'] = outputs.past_audio_key_values

        return model_kwargs


AutoConfig.register("dattn_gemma2", DattnGemma2Config)
AutoModelForCausalLM.register(DattnGemma2Config, DattnGemma2ForCausalLM)
apply_liger_kernel_to_gemma2(fused_linear_cross_entropy=False, cross_entropy=False)
if LigerRMSNorm not in ALL_LAYERNORM_LAYERS:
    ALL_LAYERNORM_LAYERS.append(LigerRMSNorm)
