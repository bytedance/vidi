"""
Copyright 2025 Intelligent Editing Team.
"""
from pathlib import Path
import copy
import json
from typing import Dict
import random
from PIL import Image

import torch
from torch.utils.data import Dataset

from .img_utils import process_images
from .txt_utils import preprocess_conv, preprocess_mm


class ImageConvDataset(Dataset):
    def __init__(self, data_args, processor, tokenizer):
        super().__init__()
        self.data_args = data_args
        self.list_data_dict = json.load(open(data_args.data_path, "r"))
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 512 if "image" in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_try, max_try = 0, 5
        while True:
            try:
                data = copy.deepcopy(self.list_data_dict[i])
                if "image" in data:
                    has_image = True
                    image_file = Path(self.data_args.image_folder) / data['image']
                    image = Image.open(image_file).convert('RGB')
                    image_size = image.size
                    image = process_images([image, ], self.processor, self.data_args)[0]
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
        else:
            height = width = self.processor.output_size
            if self.data_args.mm_image_aspect_ratio == "anyres":
                data_dict['image'] = torch.zeros(3, 3, height, width)
                data_dict['image_size'] = (height*2, width)
            else:
                data_dict['image'] = torch.zeros(3, height, width)
                data_dict['image_size'] = (height, width)
        
        return data_dict
