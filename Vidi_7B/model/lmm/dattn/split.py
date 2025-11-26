from typing import Callable
import math
import torch


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
    if isinstance(outputs[0], tuple):  # multiple returns
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


def splitted_call(func: Callable, inputs: torch.Tensor, num_splits: int = 1, dim_split: int = 0):
    # no need to split if num_splits is 1
    if num_splits == 1:
        return func(inputs)
    
    # split data into num_splits chunks along the dim_split dimension
    splitted_inputs, original_size = split_data(inputs, num_splits, dim_split)

    # splitted function call
    out = []
    for x in splitted_inputs:
        o = func(x)
        out.append(o)
    
    # merge output chunks along the dim_split dimension
    out = merge_data(out, dim_split, original_size)

    return out
