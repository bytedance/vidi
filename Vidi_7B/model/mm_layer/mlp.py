import re
from torch import nn, Tensor
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, arch: str, d_mm: int, d_llm: int):
        super().__init__()

        if arch == 'linear':
            self.model = nn.Linear(d_mm, d_llm)
        elif arch.startswith('mlp'):
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', arch)
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(d_mm, d_llm)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(d_llm, d_llm))
            self.model = nn.Sequential(*modules)
        
        else:
            raise NotImplementedError(f'Unknown projector arch: {arch}')
    
    def forward(self, x):
        return self.model(x)


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        self.dtype = dtype
    
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(
            input.to(self.dtype), self.weight.to(self.dtype),
            self.bias.to(self.dtype) if self.bias is not None else None
        )