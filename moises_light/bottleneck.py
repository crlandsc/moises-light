"""Dual-path RoPE bottleneck for Moises-Light.

References:
- SCNet separation.py:58-78 (DualPathRNN pattern)
- Moises-Light paper (RoPE replaces LSTM in dual-path)
"""

import torch
import torch.nn as nn

from .rope_transformer import RoPETransformer, RMSNorm
from .modules import SplitAndMergeModule


class DualPathRoPEBlock(nn.Module):
    """Single dual-path block: time transformer + freq transformer.
    Order: time-path first, then freq-path (BS-RoFormer convention).
    """

    def __init__(self, dim, transformer_params):
        super().__init__()
        self.time_transformer = RoPETransformer(dim=dim, depth=1, **transformer_params)
        self.freq_transformer = RoPETransformer(dim=dim, depth=1, **transformer_params)

    def forward(self, x):  # [B, C, F, T]
        B, C, F, T = x.shape

        # Time path: process T for each frequency bin
        # (inner residuals in RoPETransformer handle skip connection)
        x = x.permute(0, 2, 3, 1).reshape(B * F, T, C)    # [B*F, T, C]
        x = self.time_transformer(x)
        x = x.reshape(B, F, T, C).permute(0, 3, 1, 2)     # [B, C, F, T]

        # Freq path: process F for each time step
        x = x.permute(0, 3, 2, 1).reshape(B * T, F, C)    # [B*T, F, C]
        x = self.freq_transformer(x)
        x = x.reshape(B, T, F, C).permute(0, 3, 2, 1)     # [B, C, F, T]

        return x


class DualPathRoPEBottleneck(nn.Module):
    """Full bottleneck: 1 SplitAndMerge + N_RoPE dual-path RoPE blocks."""

    def __init__(self, channels, n_bands, n_split, freq_dim, bn_factor,
                 n_rope, transformer_params, norm, act):
        super().__init__()
        self.split_merge = SplitAndMergeModule(
            channels, n_bands, n_split, freq_dim, bn_factor, norm, act
        )
        self.rope_blocks = nn.ModuleList([
            DualPathRoPEBlock(channels, transformer_params)
            for _ in range(n_rope)
        ])
        self.final_norm = RMSNorm(channels)

    def forward(self, x):
        x = self.split_merge(x)
        for block in self.rope_blocks:
            x = block(x)
        x = x.permute(0, 2, 3, 1)   # [B, F, T, C]
        x = self.final_norm(x)
        x = x.permute(0, 3, 1, 2)   # [B, C, F, T]
        return x
