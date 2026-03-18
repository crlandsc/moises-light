"""RoPE Transformer for Moises-Light.

Uses local Attend with SDPA backend selection. No windowed/sink attention.
"""

import torch
import torch.nn as nn
from torch.nn import Module, ModuleList
from einops import rearrange
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from .attend import Attend


def exists(val):
    return val is not None


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (F.normalize(x, dim=-1) * self.scale * self.gamma).to(x.dtype)


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, attn_dropout=0., proj_dropout=0.,
                 rotary_embed=None, flash=True):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.attend = Attend(flash=flash, dropout=attn_dropout, scale=self.scale)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(proj_dropout),
        )
        nn.init.zeros_(self.to_out[0].weight)

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class RoPETransformer(Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8,
                 attn_dropout=0.1, proj_dropout=0.1, ff_dropout=0.1,
                 ff_mult=4, norm_output=False, flash_attn=True):
        super().__init__()
        self.layers = ModuleList([])
        rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            attn = Attention(
                dim=dim, dim_head=dim_head, heads=heads,
                attn_dropout=attn_dropout, proj_dropout=proj_dropout,
                rotary_embed=rotary_embed, flash=flash_attn,
            )
            self.layers.append(ModuleList([
                attn,
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
