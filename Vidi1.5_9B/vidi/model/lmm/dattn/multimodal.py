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

from abc import ABC, abstractmethod
import numpy as np

import torch
from torch import nn

from vidi.dataset.img_utils import get_anyres_image_grid_shape
from vidi.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from vidi.model.mm_vision import (
    CLIPVisionTower, SiglipVisionTower, LearnablePosEmbd, Conv2DPool
)
from vidi.model.mm_audio import WhisperAudioTower
from vidi.model.mm_layer import MLP, RMSNorm, rms_norm
from vidi.model.lmm.dattn.split import splitted_call
from vidi.utils import resize_by_tokens

class DattnMMModel(object):
    def __init__(self, config):
        super().__init__(config)
        
        if "clip" in config.mm_vision_tower.lower():
            vision_tower = CLIPVisionTower
        elif "siglip" in config.mm_vision_tower.lower():
            vision_tower = SiglipVisionTower
        else:
            raise NotImplementedError(f"Unsupported vision tower type: {config.mm_vision_tower}")
        self.mm_vis = vision_tower.from_pretrained(
            config.mm_vision_tower,
            select_layer=config.mm_vision_select_layer,
            attn_implementation="flash_attention_2"
        )

        if "whisper" in config.mm_audio_tower.lower():
            audio_tower = WhisperAudioTower
        else:
            raise NotImplementedError(f"Unsupported audio tower type: {config.mm_audio_tower}")
        self.mm_aud = audio_tower.from_pretrained(
            config.mm_audio_tower,
            attn_implementation="flash_attention_2"
        )

        self.image_processor = self.mm_vis.image_processor
        self.audio_processor = self.mm_aud.audio_processor
        self.text_tokenizer = self.build_text_tokenizer()

        self.mm_rand_llm_norm = RMSNorm(config.hidden_size, std=config.mm_std)

        if config.mm_input_type == "video":
            assert config.mm_image_pool_size is not None
            assert config.mm_image_pool_size <= self.mm_vis.num_patches_per_side            
            self.mm_rand_img_pool = Conv2DPool(
                d_in=self.mm_vis.hidden_size, d_out=self.mm_vis.hidden_size,
                s_in=self.mm_vis.num_patches_per_side, s_out=config.mm_image_pool_size, 
                mm_splits=config.mm_splits, mm_image_pool_size=config.mm_image_pool_size
            )
            self.mm_rand_img_projector = MLP(
                config.mm_projector_type, self.mm_vis.hidden_size * (config.mm_image_pool_size ** 2), config.hidden_size
            )
            self.mm_rand_img_norm = RMSNorm(config.hidden_size)
            self.mm_rand_pos_w = LearnablePosEmbd(
                config.hidden_size, config.mm_image_pool_size
            )
            self.mm_rand_pos_h = LearnablePosEmbd(
                config.hidden_size, config.mm_image_pool_size
            )
            
            assert config.mm_audio_pool_size is not None
            self.mm_rand_aud_pool = nn.Conv1d(
                self.mm_aud.hidden_size, config.hidden_size, bias=False,
                kernel_size=config.mm_audio_pool_size, stride=config.mm_audio_pool_size
            )
            self.mm_rand_aud_projector = MLP(
                config.mm_projector_type, config.hidden_size, config.hidden_size
            )
            self.mm_rand_aud_norm = RMSNorm(config.hidden_size)

            self.mm_rand_pos_t = LearnablePosEmbd(config.hidden_size, config.mm_time_interval)
        elif config.mm_input_type == "image":
            if isinstance(config.mm_image_grid_points, str):
                config.mm_image_grid_points = eval(config.mm_image_grid_points)
            mm_image_grid_points = np.array(config.mm_image_grid_points)
            mm_image_grid_res = mm_image_grid_points * self.mm_vis.config.image_size
            config.mm_image_grid_res = mm_image_grid_res.tolist()

            self.mm_rand_projector = MLP(
                config.mm_projector_type, self.mm_vis.hidden_size, config.hidden_size
            )
            self.mm_rand_norm = RMSNorm(config.hidden_size)
            if config.mm_image_aspect_ratio == 'anyres':
                self.mm_rand_pos_w = LearnablePosEmbd(
                    config.hidden_size, self.mm_vis.num_patches_per_side * mm_image_grid_points.max()
                )
                self.mm_rand_pos_h = LearnablePosEmbd(
                    config.hidden_size, self.mm_vis.num_patches_per_side * mm_image_grid_points.max()
                )
            else:
                self.mm_rand_pos_w = LearnablePosEmbd(
                    config.hidden_size, self.mm_vis.num_patches_per_side
                )
                self.mm_rand_pos_h = LearnablePosEmbd(
                    config.hidden_size, self.mm_vis.num_patches_per_side
                )
        else:
            raise NotImplementedError(f"Unsupported input type: {config.mm_input_type}")

    def build_text_tokenizer(self):
        raise NotImplementedError("Implement build_text_tokenizer() in your LLM.")


class MMMixin(ABC):
    def encode_multimodal_inputs(self, images, image_sizes, audios, audio_sizes):
        if self.config.mm_input_type == "video":
            return self.encode_videos(images, audios, audio_sizes)
        elif self.config.mm_input_type == "image":
            return self.encode_images(images, image_sizes)
        else:
            raise NotImplementedError(f"Unsupported multimodal input type: {self.config.mm_input_type}")

    @abstractmethod
    def get_model(self) -> DattnMMModel:
        pass

    @abstractmethod
    def encode_videos(self, images):
        pass
    
    @abstractmethod
    def encode_images(self, images, image_sizes):
        pass

    @abstractmethod
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes,
    ):
        pass


class DattnMMMixin(MMMixin):
    def encode_video_images(self, images):
        split_sizes = [image.shape[0] for image in images]
        concat_images = torch.cat([image for image in images], dim=0)
        if not self.config.train_vis and self.get_model().mm_vis.vision_model.encoder.gradient_checkpointing:
            self.get_model().mm_vis.gradient_checkpointing_disable()
        training = (self.config.train_vis and self.training)
        with torch.set_grad_enabled(training):
            _, image_features = splitted_call(
                func=self.get_model().mm_vis,
                inputs=concat_images,
                num_splits=self.config.mm_splits,
                dim_split=0,
                distributed=(not training or len(images) == 1)
            )

        height = width = self.get_model().mm_vis.num_patches_per_side
        image_features = image_features.reshape(len(image_features), height, width, -1)
        image_features = image_features.permute(0, 3, 1, 2)

        n_tokens = image_features.size(0)  * (image_features.size(-1)+1) * (image_features.size(-2)+1) 
        max_tokens = 60000 * self.config.mm_image_pool_size * self.config.mm_image_pool_size
        if n_tokens > max_tokens:
            new_h, new_W = resize_by_tokens(image_features, max_tokens)
        else:
            new_h, new_W = 28, 28

        image_features = splitted_call(
            func=self.get_model().mm_rand_img_pool,
            inputs=image_features,
            num_splits=self.config.mm_splits,
            dim_split=0,
            distributed=False,
            hw=(new_h, new_W)
        )
        image_features = image_features.permute(0, 2, 3, 1)

        image_features = self.get_model().mm_rand_img_projector(image_features)
        image_features = self.get_model().mm_rand_img_norm(image_features)
        image_features = image_features + rms_norm(self.get_model().mm_rand_pos_h(image_features, dim=1))
        image_features = image_features + rms_norm(self.get_model().mm_rand_pos_w(image_features, dim=2))
        image_features = torch.split(image_features, split_sizes, dim=0)
        image_features = [f + rms_norm(self.get_model().mm_rand_pos_t(f, dim=0)) for f in image_features]
        image_features = [f.flatten(0, 2) for f in image_features]
        image_features = torch.nn.utils.rnn.pad_sequence(image_features, batch_first=True)

        image_attention_mask = (torch.sum(torch.abs(image_features), dim=-1) != 0)
        image_masks = (torch.stack([torch.sum(torch.abs(x)) for x in images]) != 0)
        image_masks = image_masks.to(image_attention_mask.device)
        image_attention_mask = image_attention_mask * image_masks.unsqueeze(-1)
        image_features = self.get_model().mm_rand_llm_norm(image_features)
        image_features = image_features * image_attention_mask.unsqueeze(-1)

        return image_features, image_attention_mask
    
    def encode_video_audios(self, audios, audio_sizes):
        concat_audios = torch.cat([audio for audio in audios], dim=0)
        if not self.config.train_aud and self.get_model().mm_aud.encoder.gradient_checkpointing:
            self.get_model().mm_aud.gradient_checkpointing_disable()
        training = self.config.train_aud and self.training
        with torch.set_grad_enabled(training):
            audio_features = splitted_call(
                func=self.get_model().mm_aud,
                inputs=concat_audios,
                num_splits=self.config.mm_splits,
                dim_split=0,
                distributed=(not training or len(audios) == 1)
            )
        
        split_sizes = [len(audio) for audio in audios]
        audio_features = torch.split(audio_features, split_sizes, dim=0)
        pool_ratio = self.get_model().mm_aud.config.max_source_positions / self.get_model().audio_processor.nb_max_frames
        audio_sizes = np.floor(np.array(audio_sizes) * pool_ratio).astype(int)
        audio_features = [f.flatten(0, 1)[:s] for f, s in zip(audio_features, audio_sizes)]
        audio_features = torch.nn.utils.rnn.pad_sequence(audio_features, batch_first=True)

        audio_features = audio_features.permute(0, 2, 1)
        audio_features = self.get_model().mm_rand_aud_pool(audio_features)
        audio_features = audio_features.permute(0, 2, 1)
        audio_sizes = np.floor(audio_sizes / self.config.mm_audio_pool_size)
        audio_sizes = audio_sizes.astype(int).tolist()
        audio_features = [f[:s] for f, s in zip(audio_features, audio_sizes)]
        
        audio_features = torch.cat(audio_features, dim=0)
        audio_features = self.get_model().mm_rand_aud_projector(audio_features)
        audio_features = self.get_model().mm_rand_aud_norm(audio_features)
        audio_features = torch.split(audio_features, audio_sizes, dim=0)
        audio_features = [f + rms_norm(self.get_model().mm_rand_pos_t(f, dim=0)) for f in audio_features]
        audio_features = torch.nn.utils.rnn.pad_sequence(audio_features, batch_first=True)

        audio_attention_mask = (torch.sum(torch.abs(audio_features), dim=-1) != 0)
        audio_masks = (torch.stack([torch.sum(torch.abs(x)) for x in audios]) != 0)
        audio_masks = audio_masks.to(audio_attention_mask.device)
        audio_attention_mask = audio_attention_mask * audio_masks.unsqueeze(-1)
        audio_features = self.get_model().mm_rand_llm_norm(audio_features)
        audio_features = audio_features * audio_attention_mask.unsqueeze(-1)

        return audio_features, audio_attention_mask
    
    def encode_videos(self, images, audios, audio_sizes):
        if images is not None:
            image_features, image_attention_mask = self.encode_video_images(images)
        else:
            image_features = image_attention_mask = None
        
        if audios is not None:
            audio_features, audio_attention_mask = self.encode_video_audios(audios, audio_sizes)
        else:
            audio_features = audio_attention_mask = None

        return image_features, image_attention_mask, audio_features, audio_attention_mask

    def encode_images(self, images, image_sizes):
        if type(images) is list or images.ndim == 5:
            assert self.config.mm_image_aspect_ratio == 'anyres'

            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            if not self.config.train_vis and self.get_model().mm_vis.vision_model.encoder.gradient_checkpointing:
                self.get_model().mm_vis.gradient_checkpointing_disable()
            training = (self.config.train_vis and self.training)
            with torch.set_grad_enabled(training):
                _, image_features = splitted_call(
                    func=self.get_model().mm_vis,
                    inputs=concat_images,
                    num_splits=self.config.mm_splits,
                    dim_split=0,
                    distributed=(not training or len(images) == 1)
                )
            image_features = self.get_model().mm_rand_projector(image_features)

            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                height = width = self.get_model().mm_vis.num_patches_per_side
                assert image_feature.shape[0] > 1
                assert height * width == image_feature[0].shape[0]

                base_image_feature = image_feature[0].reshape(height, width, -1)
                base_image_feature = base_image_feature + rms_norm(self.get_model().mm_rand_pos_h(base_image_feature, dim=0))
                base_image_feature = base_image_feature + rms_norm(self.get_model().mm_rand_pos_w(base_image_feature, dim=1))

                anyres_image_feature = image_feature[1:]
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.mm_image_grid_res,
                    self.get_model().mm_vis.config.image_size
                )
                anyres_image_feature = anyres_image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                anyres_image_feature = anyres_image_feature.permute(0, 2, 1, 3, 4)
                anyres_image_feature = anyres_image_feature.flatten(0, 1).flatten(1, 2)
                anyres_image_feature = anyres_image_feature + rms_norm(self.get_model().mm_rand_pos_h(anyres_image_feature, dim=0))
                anyres_image_feature = anyres_image_feature + rms_norm(self.get_model().mm_rand_pos_w(anyres_image_feature, dim=1))

                image_feature = torch.cat([base_image_feature.flatten(0, 1), anyres_image_feature.flatten(0, 1)])
                new_image_features.append(image_feature)

            image_features = torch.nn.utils.rnn.pad_sequence(new_image_features, batch_first=True)
            image_attention_mask = (torch.sum(torch.abs(image_features), dim=-1) != 0)
            image_attention_mask = image_attention_mask.detach()
        else:
            _, image_features = self.get_model().mm_vis(images)

            height = width = self.get_model().mm_vis.num_patches_per_side
            image_features = image_features.reshape(len(image_features), height, width, -1)
            image_features = self.get_model().mm_rand_projector(image_features)
            image_features = self.get_model().mm_rand_norm(image_features)
            image_features = image_features + rms_norm(self.get_model().mm_rand_pos_h(image_features, dim=1))
            image_features = image_features + rms_norm(self.get_model().mm_rand_pos_w(image_features, dim=2))
            image_features = image_features.flatten(1, 2)

            image_attention_mask = torch.ones(
                image_features.shape[:2], dtype=torch.bool, device=image_features.device
            )
        
        image_masks = (torch.stack([torch.sum(torch.abs(x)) for x in images]) != 0)
        image_attention_mask = image_attention_mask * image_masks.unsqueeze(-1)
        image_features = self.get_model().mm_rand_llm_norm(image_features)

        return image_features, image_attention_mask, None, None
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes, audios, audio_sizes
    ):
        if input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None
                
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            assert num_images <= 1, "only support at most one image for now."

            if num_images == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if self.get_model().text_tokenizer.padding_side == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        image_features, image_attention_mask, audio_features, audio_attention_mask = \
            self.encode_multimodal_inputs(images, image_sizes, audios, audio_sizes)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, \
            image_features, image_attention_mask, audio_features, audio_attention_mask
