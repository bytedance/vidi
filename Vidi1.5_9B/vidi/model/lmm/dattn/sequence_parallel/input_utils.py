"""
Copyright 2025 Intelligent Editing Team.
"""

"""
@author: luyanzuo
@email:  luyanzuo@bytedance.com
"""

import math
import os
from collections import defaultdict
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from .globals import (
    get_ulysses_sp_pg,
    get_ulysses_sp_rank,
    get_ulysses_sp_size,
)

_SEQ_DATA_BUF = defaultdict(lambda: [None, None, None])
_SEQ_DATA_META_SHAPES = defaultdict()
_SEQ_DATA_META_DTYPES = defaultdict()
_SEQ_DATA_ASYNC_COMMS = defaultdict(list)

def slice_tensor(tensor, dim, start, end):
    indices = slice(start, end)
    return tensor[(slice(None),) * dim + (indices,)]

def pad_tensor(x: Tensor, dim: int, padding_size: int):
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)

def unpad_tensor(x: Tensor, dim: int, padding_size: int):
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[slc]

def unpadding_tensor_for_seqeunce_parallel(x: Tensor, dim: int, unpadded_dim_size: int):
    """
    A func to remove the padding part of the tensor based on its original shape
    """
    group = get_unified_parallel_group()
    if group is None:
        return x
    sp_world = get_unified_parallel_world_size()
    if unpadded_dim_size % sp_world == 0:
        return x
    padding_size = sp_world - (unpadded_dim_size % sp_world)
    assert (padding_size + unpadded_dim_size) % sp_world == 0
    return unpad_tensor(x, dim=dim, padding_size=padding_size)

def all_to_all_split(input_list, output_list):
    # 初始化一个全0的矩阵
    num_ranks = len(input_list)
    input_split = np.zeros((num_ranks, num_ranks), dtype=int)

    # 用来记录每个rank处理的偏移量
    input_offset = [0] * num_ranks
    output_offset = [0] * num_ranks

    for i in range(num_ranks):
        # 当前rank i的input数据总量
        remaining_input = input_list[i]

        for j in range(num_ranks):
            # rank i发给rank j的数据量
            send_amount = min(remaining_input, output_list[j] - output_offset[j])
            input_split[i][j] = send_amount
            # 更新remaining_input和output_offset
            remaining_input -= send_amount
            output_offset[j] += send_amount

            if remaining_input == 0:
                break

    return input_split

class MetaTuple(tuple):
    def __new__(cls, values):
        return super().__new__(cls, values)

    def __init__(self, values):
        self._use_scatter = False
        self._scatter_dim = None
        self.values = values


def mark_sp(x: Tensor, scatter_dim: int):
    """
    Mark tensor to be used in sequence parallel
    """
    x._use_scatter = True
    x._scatter_dim = scatter_dim
    return x


def _construct_sync_buffer(shapes, dtypes, device):
    if isinstance(shapes, (torch.Size, MetaTuple)):
        if getattr(shapes, "_use_scatter", False):
            scatter_dim = shapes._scatter_dim
            sp_size = get_unified_parallel_world_size()
            shapes = list(shapes)
            shapes[scatter_dim] = math.ceil(shapes[scatter_dim] / sp_size)
            return mark_sp(torch.empty(shapes, dtype=dtypes, device=device), scatter_dim)
        return torch.empty(shapes, dtype=dtypes, device=device)

    if isinstance(shapes, list):
        buffer = [_construct_sync_buffer(sub_shape, dtypes[i], device) for i, sub_shape in enumerate(shapes)]
    elif isinstance(shapes, tuple):
        buffer = tuple(_construct_sync_buffer(sub_shape, dtypes[i], device) for i, sub_shape in enumerate(shapes))
    elif isinstance(shapes, Mapping):
        buffer = shapes.__class__(
            {key: _construct_sync_buffer(sub_shape, dtypes[key], device) for key, sub_shape in shapes.items()}
        )
    else:
        return shapes
    return buffer


def _traverse(data: Any, op: Callable) -> Union[None, List, Mapping, Any]:
    if isinstance(data, list):
        return [_traverse(sub_data, op) for sub_data in data]
    if isinstance(data, tuple):
        return tuple([_traverse(sub_data, op) for sub_data in data])
    elif isinstance(data, Mapping):
        return data.__class__({key: _traverse(sub_data, op) for key, sub_data in data.items()})
    elif isinstance(data, Tensor):
        return op(data)
    else:
        return data


def shape_with_meta(data: Tensor):
    shape = data.shape
    if getattr(data, "_use_scatter", False):
        shape = MetaTuple(shape)
        shape._use_scatter = True
        shape._scatter_dim = data._scatter_dim
    return shape


def _get_shapes(data):
    return _traverse(data, op=shape_with_meta)


def _get_dtypes(data):
    return _traverse(data, op=lambda x: x.dtype)


def _sync_data_in_group(data, shape, dtype, src, is_src, group, async_op, return_storage=False):
    comms = []
    storages = []
    if isinstance(data, (list, tuple)):
        for i, sub_shape in enumerate(shape):
            sub_comms, sub_storages = _sync_data_in_group(
                data[i], sub_shape, dtype[i], src, is_src, group, async_op, True
            )
            comms.extend(sub_comms)
            storages.extend(sub_storages)
    elif isinstance(data, Mapping):
        for key, sub_data in data.items():
            sub_comms, sub_storages = _sync_data_in_group(
                sub_data, shape[key], dtype[key], src, is_src, group, async_op, True
            )
            comms.extend(sub_comms)
            storages.extend(sub_storages)
    elif isinstance(data, Tensor):
        if getattr(shape, "_use_scatter", False):
            sp_size = get_ulysses_sp_size()
            if is_src:
                scatter_dim = shape._scatter_dim
                # scatter will just use the tensor storage, so contiguous() is a must
                scatter_list = [
                    t.contiguous() for t in pad_tensor(data, scatter_dim, sp_size).chunk(sp_size, scatter_dim)
                ]
                data.set_(torch.empty_like(scatter_list[get_ulysses_sp_rank()]))
            else:
                scatter_list = None
            data._unpad_shape = torch.Size(shape)
            if int(os.environ.get("DIST_ATTN_SYNC_SCATTER", 0)):
                torch.distributed.scatter(data, scatter_list, src=src, group=group, async_op=False)
                del scatter_list
            else:
                comms.append(torch.distributed.scatter(data, scatter_list, src=src, group=group, async_op=async_op))
                if is_src:
                    for r, data in enumerate(scatter_list):
                        if r != get_ulysses_sp_rank():
                            storages.append(data.untyped_storage())
        else:
            data = data.contiguous()
            comms.append(torch.distributed.broadcast(data, src=src, group=group, async_op=async_op))
    if return_storage:
        return comms, storages
    return comms


class SPDistForward:
    """A forward tool to sync different result across sp group

    Args:
        module: a function or module to process users input
        sp_step: current training step to judge which rank to broadcast its result to all
        name: a distinct str to save meta and async comm
        comm_shape: if different ranks have different shape, mark this arg to True
        start_rank: which sp rank we start to loop
        device: the device for current rank, can be empty
    """

    def __init__(
        self,
        name: str,
        comm_shape: bool,
        start_rank: int = 0,
        device: torch.device = None,
    ):
        self.name = name
        self.comm_shape = comm_shape
        self.start_rank = start_rank
        if device:
            self.device = device
        else:
            self.device = torch.cuda.current_device()

    def __call__(self, inputs) -> Any:
        group = get_ulysses_sp_pg()
        if not group:
            yield inputs
        else:
            device = self.device
            sp_world = get_ulysses_sp_size()
            sp_rank = get_ulysses_sp_rank()
            for local_step in range(sp_world):
                local_step = (local_step + self.start_rank) % sp_world
                src_rank = dist.get_global_rank(group, local_step)
                is_src = sp_rank == local_step
                local_shapes = []
                local_dtypes = []
                if local_step == self.start_rank:
                    # we sync shape and dtype inside the group in the first step
                    local_result = inputs
                    _SEQ_DATA_BUF[self.name][-1] = local_result
                    local_shapes = _get_shapes(local_result)
                    local_dtypes = _get_dtypes(local_result)
                    if self.comm_shape:
                        group_shapes_lists = [None] * sp_world
                        # dist.all_gather_object(group_shapes_lists, local_shapes, group=group)
                        sp_cpu_group = get_ulysses_sp_pg()
                        dist.all_gather_object(group_shapes_lists, local_shapes, group=sp_cpu_group)
                        _SEQ_DATA_META_SHAPES[self.name] = group_shapes_lists
                    else:
                        _SEQ_DATA_META_SHAPES[self.name] = [local_shapes] * sp_world
                    _SEQ_DATA_META_DTYPES[self.name] = local_dtypes

                shapes = _SEQ_DATA_META_SHAPES[self.name][local_step]
                dtypes = _SEQ_DATA_META_DTYPES[self.name]
                buf_id = local_step % 2
                if local_step == self.start_rank:
                    # sync data in the first step, async in other steps
                    sync_sp_data = local_result if is_src else _construct_sync_buffer(shapes, dtypes, device)
                    _sync_data_in_group(sync_sp_data, shapes, dtypes, src_rank, is_src, group, False)
                    _SEQ_DATA_BUF[self.name][buf_id] = sync_sp_data

                # wait for async comm ops
                if _SEQ_DATA_ASYNC_COMMS[self.name]:
                    for comm in _SEQ_DATA_ASYNC_COMMS[self.name]:
                        comm.wait()

                # before return the sync result, do async broadcast for next batch
                if local_step != (self.start_rank - 1) % sp_world:
                    next_buf_id = 1 - buf_id
                    shapes = _SEQ_DATA_META_SHAPES[self.name][local_step + 1]
                    src_rank = dist.get_global_rank(group, local_step + 1)
                    is_src = sp_rank == local_step + 1
                    next_sync_data = (
                        _SEQ_DATA_BUF[self.name][-1] if is_src else _construct_sync_buffer(shapes, dtypes, device)
                    )
                    _SEQ_DATA_ASYNC_COMMS[self.name] = _sync_data_in_group(
                        next_sync_data, shapes, dtypes, src_rank, is_src, group, True
                    )
                    _SEQ_DATA_BUF[self.name][next_buf_id] = next_sync_data
                yield _SEQ_DATA_BUF[self.name][buf_id]