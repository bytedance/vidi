"""
Copyright 2025 Intelligent Editing Team.
"""
from typing import List, Dict, Optional, Sized, Iterator, Mapping
from pprint import pprint
import itertools

import torch
import torch.nn as nn
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import *
from transformers.trainer_pt_utils import get_length_grouped_indices

from vidi.model.lmm.dattn.sequence_parallel.globals import (
    get_ulysses_sp_size, get_ulysses_sp_rank, get_ulysses_sp_pg,
    get_data_parallel_size, get_data_parallel_rank
)


def get_sp_data_idx(data_idx: list, bs_local: int):
    sp_size = get_ulysses_sp_size()
    dp_size = get_data_parallel_size()
    world_size = dist.get_world_size()
    assert sp_size * dp_size == world_size

    bs_global = world_size * bs_local
    assert bs_global % sp_size == 0
    bs_global = bs_global // sp_size

    dp_ranks = []
    for dp in range(dp_size):
        dp_ranks.extend([dp, ]*sp_size)
    assert len(dp_ranks) == world_size

    num_batches = math.ceil(len(data_idx) / bs_global)
    data_idx_sp = []
    for bi in range(num_batches):
        idx_batch = data_idx[bi*bs_global:(bi+1)*bs_global]
        for r in range(world_size):
            idx_rank = idx_batch[dp_ranks[r]*bs_local:(dp_ranks[r]+1)*bs_local]
            data_idx_sp.extend(idx_rank)
    assert len(data_idx_sp) == len(data_idx) * sp_size
    
    return data_idx_sp


def get_mm_length_grouped_indices(lengths, batch_size, generator=None):
    assert all(l != 0 for l in lengths), "Should not have zero length."

    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        grouped_indices = get_length_grouped_indices(lengths, batch_size, generator=generator)
    else:
        mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
        lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

        mega_batch_mult_mm = max(min(len(mm_lengths) // (batch_size * 4), 50), 1)
        megabatch_size_mm = mega_batch_mult_mm * batch_size
        mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, mega_batch_mult_mm, generator)]
        mm_megabatches = [mm_shuffle[i : i + megabatch_size_mm] for i in range(0, len(mm_shuffle), megabatch_size_mm)]

        mega_batch_mult_lang = max(min(len(lang_lengths) // (batch_size * 4), 50), 1)
        megabatch_size_lang = mega_batch_mult_lang * batch_size
        lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, mega_batch_mult_lang, generator)]
        lang_megabatches = [lang_shuffle[i : i + megabatch_size_lang] for i in range(0, len(lang_shuffle), megabatch_size_lang)]

        additional_batch = mm_megabatches[-1] + lang_megabatches[-1]
        megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
        megabatch_indices = torch.randperm(len(megabatches), generator=generator)
        megabatches = [megabatches[i] for i in megabatch_indices]
        if len(additional_batch) > 0:
            megabatches.append(additional_batch)
        
        grouped_indices = [i for megabatch in megabatches for i in megabatch]
    
    mega_grouped_indices = [grouped_indices[i : i + batch_size] for i in range(0, len(grouped_indices), batch_size)]
    mega_grouped_indices_perm = torch.randperm(len(mega_grouped_indices), generator=generator)
    mega_grouped_indices = [mega_grouped_indices[i] for i in mega_grouped_indices_perm]

    return [i for mega_grouped_index in mega_grouped_indices for i in mega_grouped_index]


class SPLengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """
    def __init__(
        self,
        bs_local: int,
        world_size: int,
        grad_accum: int,
        lengths: List[int],
    ):
        self.bs_local = bs_local
        self.bs_global = bs_local * world_size * grad_accum // get_ulysses_sp_size()
        self.lengths = lengths

    def __iter__(self):
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

        data_idx = get_mm_length_grouped_indices(self.lengths, self.bs_global, generator=generator)
        data_idx = get_sp_data_idx(data_idx, self.bs_local)
        yield from data_idx
    
    def __len__(self):
        return len(self.lengths) * get_ulysses_sp_size()


class SPRandomSampler(Sampler[int]):
    def __init__(self, data_source: Sized, bs_local: int) -> None:
        self.dset_len = len(data_source)
        self.bs_local = bs_local

    def __iter__(self) -> Iterator[int]:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

        data_idx = torch.randperm(self.dset_len, generator=generator).tolist()
        data_idx = get_sp_data_idx(data_idx, self.bs_local)
        yield from data_idx
    
    def __len__(self) -> int:
        return self.dset_len * get_ulysses_sp_size()


class VidiTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            return SPLengthGroupedSampler(
                bs_local=self.args.train_batch_size,
                world_size=self.args.world_size,
                grad_accum=self.args.gradient_accumulation_steps,
                lengths=self.train_dataset.lengths
            )
        else:
            return SPRandomSampler(self.train_dataset, bs_local=self.args.train_batch_size)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            mm_modules = ("mm_rand", "mm_vis", "mm_aud")
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (
                            p.requires_grad and n in decay_parameters and \
                            all([mm not in n for mm in mm_modules])
                        )
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (
                            p.requires_grad and n not in decay_parameters and \
                            all([mm not in n for mm in mm_modules])
                        )
                    ],
                    "weight_decay": 0.0,
                }
            ]
            for mm in mm_modules:
                lr = getattr(self.args, f"{mm}_lr", None)
                lr = lr if lr is not None else self.args.learning_rate

                optimizer_grouped_parameters.append({
                    "params": [
                        p for n, p in opt_model.named_parameters() if (
                            p.requires_grad and n in decay_parameters and mm in n
                        )
                    ],
                    "weight_decay": self.args.weight_decay, "lr": lr
                })
                optimizer_grouped_parameters.append({
                    "params": [
                        p for n, p in opt_model.named_parameters() if (
                            p.requires_grad and n not in decay_parameters and mm in n
                        )
                    ],
                    "weight_decay": 0.0, "lr": lr
                })

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def broadcast_inputs(self, inputs):
        src = get_data_parallel_rank() * get_ulysses_sp_size()
        if isinstance(inputs, Mapping):
            return type(inputs)({k: self.broadcast_inputs(v) for k, v in inputs.items()})
        elif isinstance(inputs, (tuple, list)):
            return type(inputs)(self.broadcast_inputs(v) for v in inputs)
        elif isinstance(inputs, torch.Tensor):
            if get_ulysses_sp_rank() == 0:
                dist.broadcast_object_list([inputs.shape, ], src=src, group=get_ulysses_sp_pg())
                dist.broadcast(inputs, src=src, group=get_ulysses_sp_pg())
            else:
                inputs_shape = [None, ]
                dist.broadcast_object_list(inputs_shape, src=src, group=get_ulysses_sp_pg())
                inputs = torch.empty(inputs_shape[0], dtype=inputs.dtype, device=inputs.device)
                dist.broadcast(inputs, src=src, group=get_ulysses_sp_pg())
            return inputs
        else:  # int, float, str, ..., other generic python types
            if get_ulysses_sp_rank() == 0:
                dist.broadcast_object_list([inputs, ], src=src, group=get_ulysses_sp_pg())
            else:
                inputs = [None, ]
                dist.broadcast_object_list(inputs, src=src, group=get_ulysses_sp_pg())
                inputs = inputs[0]
            return inputs
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        if get_ulysses_sp_size() > 1:
            # ensure identical data within the same sp group
            inputs = self.broadcast_inputs(inputs)

        return super().training_step(model, inputs, num_items_in_batch)
