"""
Copyright 2025 Intelligent Editing Team.
"""
import contextlib
from torch import nn


def make_context_fn(module: nn.Module):
    # This context manager is only used during the recompute pass.
    class RecomputeContext:
        def __enter__(self):
            pass
        
        def __exit__(self, exc_type, exc_value, traceback):
            # Correct the forward counter of DeepSpeed Zero3
            for m in module.modules():
                if hasattr(m, 'ds_grads_remaining'):
                    m.ds_grads_remaining = m.ds_grads_remaining // 2

    def context_fn():
        return contextlib.nullcontext(), RecomputeContext()
    
    return context_fn