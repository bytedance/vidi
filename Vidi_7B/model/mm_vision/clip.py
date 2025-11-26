from transformers import (
    CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
)


class CLIPVisionTowerConfig(CLIPVisionConfig):
    select_layer = -2


class CLIPVisionTower(CLIPVisionModel):
    config_class = CLIPVisionTowerConfig

    def __init__(self, config: CLIPVisionTowerConfig):
        super().__init__(config)

        self.image_processor = CLIPImageProcessor.from_pretrained(self.name_or_path)
        self.image_processor.output_size = self.image_processor.size["shortest_edge"]

    def forward(self, images):
        image_forward_outs = super().forward(images, output_hidden_states=True)
        image_features = image_forward_outs.hidden_states[self.config.select_layer]
        image_features_cls = image_features[:, 0]
        image_features_pch = image_features[:, 1:]

        return image_features_cls, image_features_pch

    def encode(self, inputs_embeds, attention_mask):
        inputs_embeds = self.vision_model.pre_layrnorm(inputs_embeds)
        inputs_embeds = self.vision_model.encoder(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            return_dict=True, output_hidden_states=True
        ).hidden_states[self.config.select_layer]

        return inputs_embeds

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
