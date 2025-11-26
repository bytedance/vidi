import math
from torch import nn
from torch.nn import functional as F


class Conv2DPool(nn.Module):
    def __init__(self, d_in, d_out, s_in, s_out):
        super().__init__()

        assert s_in >= s_out

        self.d_in = d_in
        self.d_out = d_out
        self.s_in = s_in
        self.s_out = s_out
        self.conv = nn.Conv2d(
            d_in, d_out, bias=False, kernel_size=math.ceil(s_in / s_out)
        )
    
    def forward(self, x):
        x = self.conv(x)
        assert x.shape[-1] >= self.s_out
        x = F.interpolate(
            x, size=self.s_out, mode='bilinear', align_corners=True
        )

        return x