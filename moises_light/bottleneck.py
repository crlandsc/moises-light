"""Dual-path RoPE bottleneck for Moises-Light.

References:
- SCNet separation.py:58-78 (DualPathRNN pattern)
- Moises-Light paper (RoPE replaces LSTM in dual-path)
"""

import torch
import torch.nn as nn

from .rope_transformer import RoPETransformer
from .modules import SplitAndMergeModule


class DualPathRoPEBlock(nn.Module):
    """Single dual-path block: freq transformer + time transformer.
    Order: freq-path first, then time-path (SCNet convention).
    """

    def __init__(self, dim, transformer_params):
        super().__init__()
        self.freq_transformer = RoPETransformer(dim=dim, depth=1, **transformer_params)
        self.time_transformer = RoPETransformer(dim=dim, depth=1, **transformer_params)

    def forward(self, x):  # [B, C, F, T]
        B, C, F, T = x.shape

        # Freq path: process F for each time step
        residual = x
        xf = x.permute(0, 3, 2, 1).reshape(B * T, F, C)  # [B*T, F, C]
        xf = self.freq_transformer(xf)
        xf = xf.reshape(B, T, F, C).permute(0, 3, 2, 1)      # [B, C, F, T]
        x = xf + residual

        # Time path: process T for each frequency bin
        residual = x
        xt = x.permute(0, 2, 3, 1).reshape(B * F, T, C)   # [B*F, T, C]
        xt = self.time_transformer(xt)
        xt = xt.reshape(B, F, T, C).permute(0, 3, 1, 2)       # [B, C, F, T]
        x = xt + residual

        return x


class DualPathRoPEBottleneck(nn.Module):
    """Full bottleneck: 1 SplitAndMerge + N_RoPE dual-path RoPE blocks."""

    def __init__(self, channels, n_bands, n_split, freq_dim, bn_factor,
                 n_rope, transformer_params):
        super().__init__()
        self.split_merge = SplitAndMergeModule(
            channels, n_bands, n_split, freq_dim, bn_factor
        )
        self.rope_blocks = nn.ModuleList([
            DualPathRoPEBlock(channels, transformer_params)
            for _ in range(n_rope)
        ])

    def forward(self, x):
        x = self.split_merge(x)
        for block in self.rope_blocks:
            x = block(x)
        return x
