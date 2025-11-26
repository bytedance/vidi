from transformers import (
    SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
)


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