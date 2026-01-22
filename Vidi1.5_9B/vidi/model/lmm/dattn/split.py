"""
Copyright 2025 Intelligent Editing Team.
"""
from typing import Callable, Sequence
import math
import torch
from torch import distributed as dist
from .sequence_parallel.globals import get_ulysses_sp_size, get_ulysses_sp_pg
from .sequence_parallel.all_to_all import Slice, Gather, pad_tensor, unpad_tensor


def split_data(x, num_splits, dim_split):
    original_size = x.shape[dim_split]
    
    if x.shape[dim_split] <= num_splits:
        size = torch.ones(x.ndim, dtype=torch.int)
        size[dim_split] = math.ceil(num_splits / x.shape[dim_split])
        x = x.repeat(*size)

    splitted_data = torch.tensor_split(x, num_splits, dim_split)
    assert len(splitted_data) == num_splits
    
    return splitted_data, original_size


def merge_data(outputs, dim_split, original_size):
    if isinstance(outputs[0], Sequence):  # multiple returns
        outputs = [
            torch.narrow(
                torch.cat(o, dim=dim_split),
                dim=dim_split, start=0, length=original_size
            )
            for o in zip(*outputs)
        ]
    else:
        outputs = torch.narrow(
            torch.cat(outputs, dim=dim_split),
            dim=dim_split, start=0, length=original_size
        )

    return outputs


def splitted_call_local(func: Callable, inputs: torch.Tensor, num_splits: int = 1, dim_split: int = 0, grad_ckpt: bool = False, hw: tuple = None):
    # no need to split if num_splits is 1
    if num_splits == 1:
        return func(inputs)
    
    # split data into num_splits chunks along the dim_split dimension
    splitted_inputs, original_size = split_data(inputs, num_splits, dim_split)

    # splitted function call
    out = []
    for x in splitted_inputs:
        if grad_ckpt:
            o = torch.utils.checkpoint.checkpoint(
                func, x, use_reentrant=False
            )
        else:
            if hw is not None:
                o = func(x, hw)
            else:
                o = func(x)
        out.append(o)
    
    # merge output chunks along the dim_split dimension
    out = merge_data(out, dim_split, original_size)

    return out


def splitted_call(func: Callable, inputs: torch.Tensor, num_splits: int = 1, dim_split: int = 0, distributed: bool = True, grad_ckpt: bool = False, hw: tuple = None):
    if dist.is_initialized() and distributed and get_ulysses_sp_size() > 1:
        sp_size = get_ulysses_sp_size()
        spg = get_ulysses_sp_pg()
        dim_size = inputs.shape[dim_split]
        padding_size = 0
        padded_inputs = inputs
        if dim_size < sp_size:
            padding_size = sp_size - dim_size
            padded_inputs = pad_tensor(inputs, dim_split, padding_size)
        x = Slice.apply(spg, padded_inputs, dim_split, True)
        
        y = splitted_call_local(func, x, num_splits, dim_split, grad_ckpt)
        if isinstance(y, Sequence):  # multiple returns
            outputs = [
                unpad_tensor(Gather.apply(spg, o, dim_split, True), dim_split, padding_size)
                for o in y
            ]
        else:
            outputs = unpad_tensor(Gather.apply(spg, y, dim_split, True), dim_split, padding_size)

        return outputs
    else:
        if dist.is_initialized():
            num_splits = num_splits * get_ulysses_sp_size()
        return splitted_call_local(func, inputs, num_splits, dim_split, grad_ckpt, hw)