"""Flash attention with SDPA backend selection for Moises-Light."""

from collections import namedtuple

import os
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# constants

FlashAttentionConfig = namedtuple(
    'FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient']
)

# helpers

def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


class Once:
    def __init__(self, fn):
        self.fn = fn
        self.called = False

    def __call__(self, *args, **kwargs):
        if self.called:
            return
        self.called = True
        return self.fn(*args, **kwargs)

    def reset(self):
        self.called = False


print_once = Once(print)


class Attend(nn.Module):
    """Flash attention with SDPA backend selection."""

    def __init__(self, dropout=0., flash=True, scale=None):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash

        # determine efficient attention configs for cuda and cpu
        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        if (device_properties.major, device_properties.minor) >= (8, 0):
            if os.name == 'nt':
                print_once('Windows detected, using math or mem efficient attention')
                self.cuda_config = FlashAttentionConfig(False, True, True)
            else:
                print_once('GPU >= 8.0, using flash attention')
                self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            print_once('GPU < 8.0, using math or mem efficient attention')
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        original_dtype = q.dtype

        if original_dtype in (torch.float16, torch.bfloat16):
            flash_dtype = original_dtype
        elif torch.cuda.is_available():
            flash_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available():
            flash_dtype = torch.bfloat16
        else:
            flash_dtype = torch.float32

        E = q.shape[-1]
        scale = self.scale if self.scale is not None else E ** -0.5

        config = self.cuda_config if q.is_cuda else self.cpu_config
        backends = []
        if config.enable_flash:
            backends += [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]
        if config.enable_mem_efficient:
            backends.append(SDPBackend.EFFICIENT_ATTENTION)
        if config.enable_math:
            backends.append(SDPBackend.MATH)

        def _sdpa(q, k, v):
            return F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=scale,
            )

        try:
            device_type = 'cuda' if q.is_cuda else ('mps' if device.type == 'mps' else 'cpu')
            with sdpa_kernel(backends):
                with torch.amp.autocast(device_type=device_type, dtype=flash_dtype):
                    out = _sdpa(q, k, v)
        except RuntimeError as e:
            if "No available kernel" not in str(e):
                raise
            print_once("Primary backends failed, falling back to MATH")
            try:
                with sdpa_kernel([SDPBackend.MATH]):
                    out = _sdpa(q, k, v)
            except RuntimeError as e:
                if "No available kernel" not in str(e):
                    raise
                print_once("All backends failed, falling back to einsum attention")
                sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
                attn = sim.softmax(dim=-1)
                attn = self.attn_dropout(attn)
                out = einsum("b h i j, b h j d -> b h i d", attn, v)

        if out.dtype != original_dtype:
            out = out.to(original_dtype)

        return out

    def forward(self, q, k, v):
        device = q.device
        k = k.to(device)
        v = v.to(device)

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v)

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out
