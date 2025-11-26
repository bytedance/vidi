from PIL import Image
from io import BytesIO
import base64
import math
import ast
import itertools
import numpy as np

import torch
from transformers import StoppingCriteria


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height), resample=Image.BICUBIC)

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    # image_padded = resize_and_pad_image(image, best_resolution)
    image_resized = image.resize(best_resolution, resample=Image.BICUBIC)

    patches = divide_to_patches(image_resized, processor.output_size)

    image_original_resize = image.resize(
        (processor.output_size, processor.output_size), resample=Image.BICUBIC
    )

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "mm_image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == "resize":
        for image in images:
            image = image.resize(
                (image_processor.output_size, image_processor.output_size), resample=Image.BICUBIC
            )
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.mm_image_grid_res)
            new_images.append(image)
    elif image_aspect_ratio == "crop":
        return image_processor(images, return_tensors='pt')['pixel_values']
    else:
        raise NotImplementedError(f"Unsupported image aspect ratio: {image_aspect_ratio}")
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def divide_to_slides(image, patch_size, min_interval, max_interval, rng: np.random.Generator):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches, boxes = [], []
    width, height = image.size

    interval_h = rng.uniform(min_interval, max_interval)
    interval_w = rng.uniform(min_interval, max_interval)
    starting_point = int(rng.integers(0, 3, endpoint=True))
    if starting_point == 0 or starting_point == 1:  # top left to right or top right to left
        reverse_flag = (starting_point == 1)
        for i in range(0, height - patch_size + 1, int(patch_size/interval_h)):
            patches_row, boxes_row = [], []
            for j in range(0, width - patch_size + 1, int(patch_size/interval_w)):
                box = (j, i, j + patch_size, i + patch_size)
                boxes_row.append(box)
                patch = image.crop(box)
                patches_row.append(patch)
            
            if reverse_flag:
                boxes.append(boxes_row[::-1])
                patches.append(patches_row[::-1])
                reverse_flag = False
            else:
                boxes.append(boxes_row)
                patches.append(patches_row)
                reverse_flag = True
    elif starting_point == 2 or starting_point == 3:  # left top to bottom or left bottom to top
        reverse_flag = (starting_point == 3)
        for j in range(0, width - patch_size + 1, int(patch_size/interval_w)):
            patches_col, boxes_col = [], []
            for i in range(0, height - patch_size + 1, int(patch_size/interval_h)):
                box = (j, i, j + patch_size, i + patch_size)
                boxes_col.append(box)
                patch = image.crop(box)
                patches_col.append(patch)
            
            if reverse_flag:
                boxes.append(boxes_col[::-1])
                patches.append(patches_col[::-1])
                reverse_flag = False
            else:
                boxes.append(boxes_col)
                patches.append(patches_col)
                reverse_flag = True
    else:
        raise ValueError
    
    reverse = int(rng.integers(0, 1, endpoint=True))
    if reverse == 1:
        patches.reverse()
        boxes.reverse()

    patches = list(itertools.chain.from_iterable(patches))
    boxes = list(itertools.chain.from_iterable(boxes))

    return patches, boxes


def process_slideshow_image(
        image, processor,
        min_scale=2., max_scale=4.,
        min_interval=2., max_interval=6.,
        rng: np.random.Generator = None
    ):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if rng is None: rng = np.random.default_rng()

    w, h = image.size
    if w < h:
        _h = math.ceil(rng.uniform(min_scale, max_scale) * processor.output_size)
        _w = math.ceil(w * _h / h)
        if _w < processor.output_size:
            _w = processor.output_size
            _h = math.ceil(h * _w / w)
    else:
        _w = math.ceil(rng.uniform(min_scale, max_scale) * processor.output_size)
        _h = math.ceil(h * _w / w)
        if _h < processor.output_size:
            _h = processor.output_size
            _w = math.ceil(w * _h / h)
    image_resized = image.resize((_w, _h), resample=Image.BICUBIC)

    image_patches, image_boxes = divide_to_slides(
        image_resized, processor.output_size, min_interval, max_interval, rng
    )
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    image_patches = torch.stack(image_patches, dim=0)
    
    image_boxes = np.array(image_boxes, dtype=float)
    image_boxes[:, 0] /= _w
    image_boxes[:, 1] /= _h
    image_boxes[:, 2] /= _w
    image_boxes[:, 3] /= _h
    
    return image_patches, image_boxes


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
