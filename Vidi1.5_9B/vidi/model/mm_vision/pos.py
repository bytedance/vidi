"""
Copyright 2025 Intelligent Editing Team.
"""
import math
import torch
from torch import nn, Tensor

from vidi.model.mm_layer import Linear


class FractionalSinusoidalEmbedding:
    def __init__(self, d):
        assert d % 2 == 0
        self.d = d
        self.div_term = torch.exp((torch.arange(0, d, 2, dtype=torch.float) * -(math.log(10000.0) / d)))

    @torch.no_grad()
    def __call__(self, position: Tensor):
        N = len(position)
        pe = torch.zeros(N, self.d, dtype=torch.float, device=position.device)
        position = position.float().unsqueeze(1)
        self.div_term = self.div_term.to(position.device)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)

        return pe


class LearnablePosEmbd(nn.Module):
    def __init__(self, d, N, add_noise=True):
        super().__init__()
        self.d = d
        self.N = N
        self.add_noise = add_noise

        self.embd_weights = FractionalSinusoidalEmbedding(d)
        self.mlp = nn.Sequential(
            Linear(d, d, dtype=torch.float32), nn.GELU(), Linear(d, d, dtype=torch.float32)
        )

    def forward(self, x, dim, l=None):
        assert x.shape[dim] > 1
        if l is None:
            l = x.shape[dim]
        else:
            assert l > 1 and l <= x.shape[dim]
        
        with torch.no_grad():
            p = torch.arange(l, dtype=torch.float, device=x.device)
            if self.training and self.add_noise:
                n = torch.clamp(torch.randn_like(p) * 0.45, min=-0.45, max=0.45)
                p = torch.clamp(p+n, min=0, max=l-1)
            p = p / (l-1) * (self.N-1)

        pe = self.embd_weights(p)
        pe = self.mlp(pe).to(x.dtype)
        if l < x.shape[dim]:
            pe = torch.cat([pe, x.new_zeros(x.shape[dim]-l, self.d)])
            l = x.shape[dim]
            
        shape = [1 if d != dim else l for d in range(x.ndim - 1)]
        shape.append(self.d)
        pe = pe.reshape(shape)

        return pe
