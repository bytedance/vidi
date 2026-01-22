"""
Copyright 2025 Intelligent Editing Team.
"""
import torch
from torch import nn
from torch import distributed as dist
from transformers.activations import ACT2FN
from transformers import (
    SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
)
from transformers.models.siglip import modeling_siglip
from vidi.model.lmm.dattn.split import splitted_call_local
from vidi.model.lmm.dattn.sequence_parallel.globals import get_ulysses_sp_size


class SiglipVisionTowerConfig(SiglipVisionConfig):
    select_layer = -2


class SiglipVisionTower(SiglipVisionModel):
    config_class = SiglipVisionTowerConfig

    def __init__(self, config: SiglipVisionTowerConfig):
        super().__init__(config)

        self.image_processor = SiglipImageProcessor.from_pretrained(self.name_or_path)
        self.image_processor.output_size = self.image_processor.size["height"]

    def forward(self, images):
        image_forward_outs = super().forward(images, output_hidden_states=True)
        image_features_cls = image_forward_outs.pooler_output
        image_features_pch = image_forward_outs.hidden_states[self.config.select_layer]

        return image_features_cls, image_features_pch
    
    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward_step(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_splits = get_ulysses_sp_size() if dist.is_initialized() else 1
        return splitted_call_local(
            self.forward_step, hidden_states, num_splits=num_splits, dim_split=1
        )

modeling_siglip.SiglipMLP = SiglipMLP
