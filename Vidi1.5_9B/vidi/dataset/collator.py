"""
Copyright 2025 Intelligent Editing Team.
"""
from typing import Dict, Sequence

import torch
import transformers

from vidi.constants import IGNORE_INDEX


class VideoTrainingCollator(object):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, image_processor, audio_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.audio_processor = audio_processor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        input_ids = input_ids[:, :self.tokenizer.model_max_length]

        labels = [instance["labels"] for instance in instances]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        labels = labels[:, :self.tokenizer.model_max_length]

        images, image_sizes = [], []
        for instance in instances:
            if instance.get("image", None) is None:
                image_size = (self.image_processor.output_size, self.image_processor.output_size)
                image = torch.zeros((2, 3, *image_size))
            else:
                image = instance['image']
                image_size = instance['image_size']
            images.append(image)
            image_sizes.append(image_size)
        
        audios, audio_sizes = [], []
        for instance in instances:
            if instance.get("audio", None) is None:
                audio = torch.zeros(
                    (1, self.audio_processor.feature_size, self.audio_processor.nb_max_frames)
                )
                audio_size = self.audio_processor.nb_max_frames
            else:
                audio = instance['audio']
                audio_size = instance['audio_size']
            audios.append(audio)
            audio_sizes.append(audio_size)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            images=images,
            image_sizes=image_sizes,
            audios=audios,
            audio_sizes=audio_sizes
        )

        return batch


class VideoInferenceCollator(object):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, image_processor, audio_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.audio_processor = audio_processor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        input_ids = input_ids[:, :self.tokenizer.model_max_length]

        images = [instance.get("image", None) for instance in instances]
        image_sizes = [instance.get("image_size", None) for instance in instances]
        if all([x is None for x in images]):
            images = image_sizes = None
        else:
            assert all([x is not None for x in images])
            assert all([x is not None for x in image_sizes])
        
        audios = [instance.get("audio", None) for instance in instances]
        audio_sizes = [instance.get("audio_size", None) for instance in instances]
        if all([x is None for x in audios]):
            audios = audio_sizes = None
        else:
            assert all([x is not None for x in audios])
            assert all([x is not None for x in audio_sizes])

        batch = dict(
            input_ids=input_ids,
            images=images,
            image_sizes=image_sizes,
            audios=audios,
            audio_sizes=audio_sizes
        )

        return batch
    