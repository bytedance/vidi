from transformers.models.whisper.modeling_whisper import WhisperModel, WhisperEncoder
from transformers import WhisperConfig, WhisperFeatureExtractor


class WhisperAudioTowerConfig(WhisperConfig):
    pass


class WhisperAudioTower(WhisperModel):
    config_class = WhisperAudioTowerConfig

    def __init__(self, config: WhisperAudioTowerConfig):
        super(WhisperModel, self).__init__(config)

        self.encoder = WhisperEncoder(config)
        self.post_init()

        self.audio_processor = WhisperFeatureExtractor.from_pretrained(self.name_or_path)

    def get_input_embeddings(self):
        return self.encoder.conv1
    
    def forward(self, audios):
        return self.encoder(audios)[0]
    
    @property
    def hidden_size(self):
        return self.config.d_model
