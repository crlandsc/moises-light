"""Core modules for Moises-Light: SplitModule, TDF, SplitAndMergeModule, TimeDown/Up.

References:
- DTTNet modules.py (TFC_TDF_Res2 pattern)
- DTTNet dp_tdf_net.py (encoder/decoder stride)
- Moises-Light paper (group conv band splitting)

Note: All modules call .contiguous() before BatchNorm2d. This is required because
BatchNorm2d backward uses .view() internally, which fails on non-contiguous tensors
produced by group convs, ConvTranspose2d, and transposes under bf16-mixed autocast on MPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitModule(nn.Module):
    """Group conv with n_bands groups. K=1 for projections, K=3 for processing."""

    def __init__(self, c_in, c_out, n_bands, norm, act, kernel_size=1):
        super().__init__()
        assert c_in % n_bands == 0 and c_out % n_bands == 0, \
            f"Channels must be divisible by n_bands={n_bands}: c_in={c_in}, c_out={c_out}"

        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride=1,
                              padding=kernel_size // 2, groups=n_bands)
        self.bn = norm(c_out)
        self.act = act()
    
    def forward(self, x):  # [B, C, F_band, T] -> [B, C_out, F_band, T]
        return self.act(self.bn(self.conv(x).contiguous()))


class TDF(nn.Module):
    """Time-Distributed Frequency FC from DTTNet.
    Linear bottleneck on frequency axis: freq -> freq//bn -> freq.
    """

    def __init__(self, channels, freq_dim, bn_factor, norm, act, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(freq_dim, freq_dim // bn_factor, bias=bias)
        self.bn1 = norm(channels)
        self.fc2 = nn.Linear(freq_dim // bn_factor, freq_dim, bias=bias)
        self.bn2 = norm(channels)
        self.act = act()

    def forward(self, x):  # [B, C, F_band, T]
        # Transpose F<->T so Linear acts on freq (last dim)
        x = x.transpose(-1, -2).contiguous()  # [B, C, T, F]
        x = self.act(self.bn1(self.fc1(x).contiguous()))
        x = self.act(self.bn2(self.fc2(x).contiguous()))
        return x.transpose(-1, -2).contiguous()  # [B, C, F_band, T]


class SplitAndMergeModule(nn.Module):
    """TFC_TDF_Res2 with TFC replaced by SplitModule group convs.

    res = SplitModule(K=3)(x)
    x = N_split x SplitModule(K=3)(x)   # "TFC1"
    x = x + TDF(x)                      # frequency FC (residual)
    x = N_split x SplitModule(K=3)(x)   # "TFC2"
    x = x + res                          # outer residual
    """

    def __init__(self, channels, n_bands, n_split, freq_dim, bn_factor, norm, act, bias=True):
        super().__init__()
        self.res = SplitModule(channels, channels, n_bands, norm, act, kernel_size=3)
        self.split1 = nn.Sequential(
            *[SplitModule(channels, channels, n_bands, norm, act, kernel_size=3) for _ in range(n_split)]
        )
        self.tdf = TDF(channels, freq_dim, bn_factor, norm, act, bias)
        self.split2 = nn.Sequential(
            *[SplitModule(channels, channels, n_bands, norm, act, kernel_size=3) for _ in range(n_split)]
        )

    def forward(self, x):  # [B, C, F_band, T] -> [B, C, F_band, T]
        res = self.res(x)
        x = self.split1(x)
        x = x + self.tdf(x)
        x = self.split2(x)
        x = x + res
        return x


class TimeDownsample(nn.Module):
    """Halves T, changes channels. Stride only on time axis."""

    def __init__(self, c_in, c_out, norm, act):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 2), stride=(1, 2))
        self.bn = norm(c_out)
        self.act = act()

    def forward(self, x):  # [B, C_in, F, T] -> [B, C_out, F, T//2]
        return self.act(self.bn(self.conv(x).contiguous()))


class TimeUpsample(nn.Module):
    """Doubles T, changes channels. Stride only on time axis."""

    def __init__(self, c_in, c_out, norm, act):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c_in, c_out, kernel_size=(1, 2), stride=(1, 2))
        self.bn = norm(c_out)
        self.act = act()

    def forward(self, x):  # [B, C_in, F, T] -> [B, C_out, F, T*2]
        return self.act(self.bn(self.conv(x).contiguous()))
