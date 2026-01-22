"""
Copyright 2025 Intelligent Editing Team.
"""
import torch
import torch.nn as nn
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


def rms_norm(hidden_states, eps=1e-5):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    
    return hidden_states.to(input_dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, std=1.0, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)*std)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return self.weight * rms_norm(hidden_states, self.variance_epsilon)


ALL_LAYERNORM_LAYERS.append(RMSNorm)
