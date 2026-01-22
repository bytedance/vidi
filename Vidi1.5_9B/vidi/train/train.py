"""
Copyright 2025 Intelligent Editing Team.
"""
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import dataclasses
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from itertools import chain

import torch
import transformers

from vidi.train.vidi_trainer import VidiTrainer
from vidi.dataset import (
    ImageConvDataset, VideoConvDataset, VideoTrainingCollator
)
from vidi.model import get_dattn_cls
from vidi.model.lmm.dattn.sequence_parallel import set_pg_manager


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    llm_attn: Optional[str] = field(default="dattn")
    mm_vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-2)
    mm_image_pool_size: Optional[int] = field(default=None)
    mm_image_aspect_ratio: Optional[str] = field(default="resize")
    mm_input_type: Optional[str] = field(default="image")
    mm_image_grid_points: Optional[str] = field(default=None)
    mm_audio_tower: Optional[str] = field(default=None)
    mm_audio_pool_size: Optional[int] = field(default=None)
    mm_splits: Optional[int] = field(default=1)
    mm_std: float = field(default=None)
    mm_time_interval: int = field(default=10000)
    model_max_length: int = field(default=4096)
    loss_thres: float = field(default=None)

    def __str__(self):
        self_as_dict = dataclasses.asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[float] = field(default=1.)
    dataset_type: Optional[str] = field(default=None)

    def __str__(self):
        self_as_dict = dataclasses.asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    mm_rand_lr: Optional[float] = None
    mm_vis_lr: Optional[float] = None
    mm_aud_lr: Optional[float] = None
    attn_implementation: str = "flash_attention_2"
    train_rand: bool = field(default=True)
    train_vis: bool = field(default=False)
    train_aud: bool = field(default=False)
    train_llm: bool = field(default=False)
    seq_parallel_size: int = field(
        default=-1,
        metadata={"help": "The degree of sequence parallelism (SP). SP is disabled by default (value: -1). "},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    transformers.set_seed(training_args.seed)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    sp_degree = training_args.seq_parallel_size
    set_pg_manager(sp_degree)
    print(f"Sequence parallelism is enabled, SP = {sp_degree}")

    num_try, max_try = 0, 5
    while True:
        try:
            if model_args.llm_attn == "dattn":
                LMM_CLS = get_dattn_cls(model_args.model_name_or_path)
                model = LMM_CLS.from_pretrained(
                    model_args.model_name_or_path,
                    mm_vision_tower=model_args.mm_vision_tower,
                    mm_vision_select_layer=model_args.mm_vision_select_layer,
                    mm_image_pool_size=model_args.mm_image_pool_size,
                    mm_image_aspect_ratio=model_args.mm_image_aspect_ratio,
                    mm_image_grid_points=model_args.mm_image_grid_points,
                    mm_audio_tower=model_args.mm_audio_tower,
                    mm_audio_pool_size=model_args.mm_audio_pool_size,
                    mm_input_type=model_args.mm_input_type,
                    mm_splits=model_args.mm_splits,
                    mm_std=model_args.mm_std,
                    mm_time_interval=model_args.mm_time_interval,
                    loss_thres=model_args.loss_thres,
                    attn_implementation="flash_attention_2",
                    _attn_implementation="flash_attention_2",
                    torch_dtype=compute_dtype,
                )
            else:
                raise NotImplementedError(f"Unsupported attention type: {model_args.llm_attn}")
            break
        except Exception as e:
            print(repr(e))
            num_try += 1
            if num_try == max_try: raise ConnectionError("Failed to download/reload model weights.")

    model.config.use_cache = False
    model.generation_config.do_sample = True
    model.requires_grad_(False)
    model.enable_input_require_grads()

    model.config.train_rand = training_args.train_rand
    if training_args.train_rand:
        for n, m in model.get_model().named_parameters():
            if "mm_rand" in n:
                m.requires_grad_(True)
    
    model.config.train_vis = training_args.train_vis
    if training_args.train_vis:
        model.get_model().mm_vis.requires_grad_(True)
        model.get_model().mm_vis.enable_input_require_grads()
    
    if hasattr(model.get_model(), "mm_aud"):
        model.config.train_aud = training_args.train_aud
        if training_args.train_aud:
            model.get_model().mm_aud.requires_grad_(True)
            model.get_model().mm_aud.enable_input_require_grads()
        
    model.config.train_llm = training_args.train_llm
    if training_args.train_llm:
        model.lm_head.requires_grad_(True)
        model.model.requires_grad_(True)
    
    data_args.mm_image_aspect_ratio = model.config.mm_image_aspect_ratio
    data_collator = VideoTrainingCollator(
        tokenizer=model.get_model().text_tokenizer,
        image_processor=model.get_model().image_processor,
        audio_processor=model.get_model().audio_processor
    )
    if data_args.dataset_type == "image-conv":
        data_args.mm_image_grid_res = model.config.mm_image_grid_res
        train_dataset = ImageConvDataset(
            data_args, model.get_model().image_processor, model.get_model().text_tokenizer
        )
    elif data_args.dataset_type == "video-conv":
        train_dataset = VideoConvDataset(
            data_args=data_args,
            image_processor=model.get_model().image_processor,
            audio_processor=model.get_model().audio_processor,
            tokenizer=model.get_model().text_tokenizer
        )
    else:
        raise NotImplementedError(f"Unsupported dataset type: {data_args.dataset_type}")
    
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    trainer = VidiTrainer(
        model=model, processing_class=model.get_model().text_tokenizer, args=training_args,
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )
    if training_args.local_rank == 0:
        print(model_args, data_args, training_args)

    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
