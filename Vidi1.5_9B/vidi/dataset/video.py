"""
Copyright 2025 Intelligent Editing Team.
"""
from pathlib import Path
import copy
from typing import Dict
import random
from PIL import Image
import orjson

import torch
from torch import distributed as dist

from vidi.dataset.img_utils import process_images, process_slideshow_image
from vidi.dataset.txt_utils import preprocess_conv, preprocess_mm
from vidi.dataset.vid_utils import load_video, load_audio, process_audio, get_media_length
from vidi.dataset.image import ImageConvDataset
from vidi.model.lmm.dattn.sequence_parallel.globals import get_ulysses_sp_rank


class VideoConvDataset(ImageConvDataset):
    def __init__(self, data_args, image_processor, audio_processor, tokenizer):
        super(ImageConvDataset, self).__init__()
        self.data_args = data_args
        with open(data_args.data_path, "rb") as f:
            self.list_data_dict = orjson.loads(f.read())
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer

    @property
    def lengths(self):
        return [data['length'] for data in self.list_data_dict]

    @property
    def dummy(self):
        data = [
            {'from': 'human', 'value': '<image>\nDummy query.'},
            {'from': 'gpt', 'value': 'Dummy answer.'}
        ]
        image = torch.zeros((2, 3, self.image_processor.output_size, self.image_processor.output_size))
        image_size = (self.image_processor.output_size, self.image_processor.output_size)
        audio = torch.zeros((1, 128, self.audio_processor.nb_max_frames))
        audio_size = self.audio_processor.nb_max_frames
        has_image = True

        return data, image, image_size, audio, audio_size, has_image
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if dist.is_initialized() and get_ulysses_sp_rank() != 0:
            data, image, image_size, audio, audio_size, has_image = self.dummy
        else:
            num_try, max_try = 0, 5
            while True:
                try:
                    data = copy.deepcopy(self.list_data_dict[i])
                    if "image" in data:
                        has_image = True
                        image_file = Path(self.data_args.image_folder) / data['image']
                        image = Image.open(image_file).convert('RGB')
                        image_size = image.size
                        image = process_slideshow_image(image, self.image_processor, self.data_args.mm_image_grid_res)
                        audio = audio_size = None
                        data = preprocess_mm(data["conversations"], self.data_args)
                    elif "video" in data:
                        assert self.data_args.mm_image_aspect_ratio == 'resize'
                        has_image = True
                        video_file = Path(self.data_args.video_folder) / data['video']
                        video_length = get_media_length(video_file)
                        assert abs(video_length - data['length']) < 1, \
                            f"Video duration mismatch, got {video_length} vs {data['length']}"

                        image = load_video(video_file, self.data_args.video_fps)
                        assert len(image) > 1, "Input video should have more than one frame."
                        image_size = image[0].size
                        image = process_images(image, self.image_processor, self.data_args)

                        audio = load_audio(video_file, self.audio_processor.sampling_rate)
                        audio, audio_size = process_audio(audio, self.audio_processor)

                        data = preprocess_mm(data["conversations"], self.data_args)
                    else:
                        has_image = False
                        data = data["conversations"]
                    break
                except Exception as e:
                    print(repr(e))
                    num_try += 1
                    if num_try == max_try:
                        raise IOError(f"Error reading data.")
                    else:
                        i = random.randint(0, len(self.list_data_dict) - 1)
        
        data_dict = preprocess_conv(data, self.tokenizer, has_image=has_image)
        if has_image:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
            data_dict['audio'] = audio
            data_dict['audio_size'] = audio_size

        
        return data_dict


