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
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM, MistralModel, MistralDecoderLayer, MistralConfig,
    MistralRMSNorm, MistralMLP, MistralFlashAttention2,
    repeat_kv
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging, ModelOutput
from transformers.generation.utils import GenerateOutput

from .multimodal import DattnMMModel, DattnMMMixin
from .outputs import DattnCausalLMOutputWithPast, DattnBaseModelOutputWithPast
from .xattn import flash_cross_attention_forward
from model.constants import IGNORE_INDEX
from model.lmm.dattn.split import splitted_call

logger = logging.get_logger(__name__)

ALL_LAYERNORM_LAYERS.append(MistralRMSNorm)


class DattnMistralFlashAttention2(MistralFlashAttention2):
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
                key_states = self.k_proj(hidden_states_kv)
                value_states = self.v_proj(hidden_states_kv)
                past_key_value.update(key_states, value_states, self.layer_idx)
            else:
                key_states, value_states = past_key_value[self.layer_idx]
        else:
            key_states = self.k_proj(hidden_states_kv)
            value_states = self.v_proj(hidden_states_kv)

        bsz, q_len, _ = query_states.size()
        _, kv_len, _ = key_states.size()

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reshape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = flash_cross_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask_q,
            attention_mask_kv,
            dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value, value_states


class DattnMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super(MistralDecoderLayer, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = DattnMistralFlashAttention2(config=config, layer_idx=layer_idx)
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def feed_foward(self, embeds):
        residual = embeds
        embeds = self.post_attention_layernorm(embeds)
        embeds = self.mlp(embeds)
        embeds = residual + embeds

        return embeds

    def forward(
        self,
        hidden_states: torch.Tensor,
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
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            image_embeds (`torch.FloatTensor`, *optional*): image input to the layer of shape `(batch, seq_len, embed_dim)`
            image_attention_mask (`torch.BoolTensor`, *optional*):
                attention mask for image embeds of size `(batch_size, sequence_length)`
            audio_embeds (`torch.FloatTensor`, *optional*): audio input to the layer of shape `(batch, seq_len, embed_dim)`
            audio_attention_mask (`torch.BoolTensor`, *optional*):
                attention mask for audio embeds of size `(batch_size, sequence_length)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        if self.training: assert not (image_embeds is None) ^ (audio_embeds is None)

        if image_embeds is None and audio_embeds is None:
            outputs = super().forward(
                hidden_states, attention_mask, position_ids, past_key_value,
                output_attentions, use_cache, cache_position, **kwargs
            )
            return outputs, image_embeds, audio_embeds

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # T2T Self Attention
        hidden_states_text, _, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position
        )

        if image_embeds is not None:
            # T2V Cross Attention
            use_image_cache = (past_image_key_value is not None and len(past_image_key_value) > self.layer_idx)
            image_mask = torch.sum(image_attention_mask, dim=-1)
            _image_attention_mask = image_attention_mask.clone().detach()
            _image_attention_mask[image_mask == 0] = True
            _image_embeds = image_embeds if use_image_cache else \
                splitted_call(self.input_layernorm, image_embeds, self.config.mm_splits, dim_split=1)
            hidden_states_image, present_image_key_value, image_value_states = self.self_attn.forward_xattn(
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
                image_embeds = image_embeds + image_value_states
                image_embeds = splitted_call(
                    self.feed_foward, image_embeds, self.config.mm_splits, dim_split=1
                )
        else:
            hidden_states_image = 0.0
            present_image_key_value = None

        if audio_embeds is not None:
            # T2A Cross Attention
            use_audio_cache = (past_audio_key_value is not None and len(past_audio_key_value) > self.layer_idx)
            audio_mask = torch.sum(audio_attention_mask, dim=-1)
            _audio_attention_mask = audio_attention_mask.clone().detach()
            _audio_attention_mask[audio_mask == 0] = True
            _audio_embeds = audio_embeds if use_audio_cache else \
                splitted_call(self.input_layernorm, audio_embeds, self.config.mm_splits, dim_split=1)
            hidden_states_audio, present_audio_key_value, audio_value_states = self.self_attn.forward_xattn(
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
                audio_embeds = audio_embeds + audio_value_states
                audio_embeds = splitted_call(
                    self.feed_foward, audio_embeds, self.config.mm_splits, dim_split=1
                )
        else:
            hidden_states_audio = 0.0
            present_audio_key_value = None

        # Residual connection for T2T attention
        hidden_states = residual + hidden_states_text + hidden_states_image + hidden_states_audio
        hidden_states = self.feed_foward(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (None,)

        if use_cache:
            outputs += (present_key_value, present_image_key_value, present_audio_key_value)
        
        return outputs, image_embeds, audio_embeds


MistralModel._no_split_modules.append("DattnMistralDecoderLayer")
class DattnMistralModel(MistralModel):
    def __init__(self, config: MistralConfig):
        super(MistralModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        assert config._attn_implementation == "flash_attention_2", "Only support FlashAttention 2 for now."

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DattnMistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
    ) -> Union[Tuple, DattnBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache) and not self.training:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            return_legacy_cache = True
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
        
        if use_cache and not self.training:
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

        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions
        # )
        if attention_mask is not None and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != inputs_embeds.size()[0]
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        next_decoder_image_cache = None
        next_decoder_audio_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs, image_embeds, audio_embeds = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
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
                )
            else:
                layer_outputs, image_embeds, audio_embeds = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    image_embeds=image_embeds,
                    image_attention_mask=image_attention_mask,
                    audio_embeds=audio_embeds,
                    audio_attention_mask=audio_attention_mask,
                    past_key_value=past_key_values,
                    past_image_key_value=past_image_key_values,
                    past_audio_key_values=past_audio_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                if image_embeds is not None:
                    next_decoder_image_cache = layer_outputs[3 if output_attentions else 2]
                if audio_embeds is not None:
                    next_decoder_audio_cache = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache, next_image_cache, next_audio_cache = None, None, None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if return_legacy_cache else next_decoder_cache
            if image_embeds is not None: next_image_cache = next_decoder_image_cache
            if audio_embeds is not None: next_audio_cache = next_decoder_audio_cache

        assert return_dict
        return DattnBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            past_image_key_values=next_image_cache,
            past_audio_key_values=next_audio_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DattnMistralConfig(MistralConfig):
    model_type = "dattn_mistral"
    
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


class DattnMistralMMModel(DattnMMModel, DattnMistralModel):
    config_class = DattnMistralConfig

    def __init__(self, config: DattnMistralConfig):
        super().__init__(config)
    
    def build_text_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.name_or_path, model_max_length=4096, padding_side="right"
        )
        tokenizer.pad_token = tokenizer.unk_token
        self.config.pad_token_id = tokenizer.pad_token_id

        return tokenizer


class DattnMistralForCausalLM(MistralForCausalLM, DattnMMMixin):
    config_class = DattnMistralConfig
    _keys_to_ignore_on_load_missing = [".*mm_vis.*", ".*mm_aud.*", ".*mm_rand.*"]

    def __init__(self, config: DattnMistralConfig):
        super(MistralForCausalLM, self).__init__(config)
        self.model = DattnMistralMMModel(config)
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
        )

        hidden_states = outputs[0]
        if labels is not None:  # for training
            logits = None  # might need to fix this in some special occasions
            # Shift so that tokens < n predict n
            shift_logits = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Discard tokens labeled as IGNORE_INDEX
            shift_logits = shift_logits[shift_labels != IGNORE_INDEX]
            shift_labels = shift_labels[shift_labels != IGNORE_INDEX]
            # Compute logits
            shift_logits = self.lm_head(shift_logits)
            shift_logits = shift_logits.float()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            
            if self.config.loss_thres is None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)
            else:
                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits, shift_labels)
                thres = 0.0 if torch.all(loss < self.config.loss_thres) else self.config.loss_thres
                loss = torch.mean(loss[loss > thres])
        else:
            loss = None
            logits = self.lm_head(hidden_states)
            logits = logits.float()

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


AutoConfig.register("dattn_mistral", DattnMistralConfig)
AutoModelForCausalLM.register(DattnMistralConfig, DattnMistralForCausalLM)
