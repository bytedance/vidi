"""
Copyright 2025 Intelligent Editing Team.
"""
import math
from torch import nn
from torch.nn import functional as F
from vidi.utils import space_to_depth


class Conv2DPool(nn.Module):
    def __init__(self, d_in, d_out, s_in, s_out, mm_splits, mm_image_pool_size):
        super().__init__()

        assert s_in >= s_out

        self.d_in = d_in
        self.d_out = d_out
        self.s_in = s_in
        self.s_out = s_out
        self.mm_splits = mm_splits
        self.merge_size = mm_image_pool_size
    
    def forward(self, x, hw, mode='bilinear'):
        x = F.pad(x, (0, 1, 0, 1), mode='constant', value=0) # shape is [b, C, 28, 28] , B = b * mm_splits
       
        if hw[0] != 28:
            x = F.interpolate(x, size=hw, mode=mode, align_corners=False if mode in ['bilinear', 'bicubic'] else None)

        x = space_to_depth(x, m_size=self.merge_size)

        # print("Visual Feature:", x.size())
        return x

# 1. zero padding
# optional inter + concat
# < 60K and H*W>=25
# audio 5 ——> 1