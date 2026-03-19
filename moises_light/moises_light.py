"""Moises-Light: Resource-efficient Band-split U-Net for Music Source Separation.

Per-stem spectral-only U-Net with internal STFT/iSTFT, equal-width band splitting
via group convolutions, and dual-path RoPE bottleneck.

References:
- Moises-Light paper (Resource-efficient Band-split U-Net)
- DTTNet (TFC_TDF pattern, encoder/decoder loop, freq truncation)
- SCNet (asymmetric decoder, multi-source output, dual-path pattern)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import SplitModule, SplitAndMergeModule, TimeDownsample, TimeUpsample
from .bottleneck import DualPathRoPEBottleneck


class MoisesLight(nn.Module):
    def __init__(
        self,
        sources=None,
        audio_channels=2,
        n_fft=6144,
        hop_size=1024,
        win_size=6144,
        freq_dim=2048,
        n_bands=4,
        G=56,
        n_enc=3,
        n_dec=1,
        n_split_enc=3,
        n_split_dec=1,
        n_rope=5,
        bn_factor=4,
        transformer_params=None,
        normalized=True,
        use_mask=True,
    ):
        super().__init__()

        if G % n_bands != 0:
            raise ValueError(
                f"G={G} must be divisible by n_bands={n_bands}. "
                f"For {n_bands} bands, valid G values include: "
                f"{', '.join(str(n_bands * i) for i in range(1, 20))}"
            )

        if sources is None:
            sources = ['vocals', 'drums', 'bass', 'other']
        self.sources = sources
        self.use_mask = use_mask
        self.audio_channels = audio_channels
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.freq_dim = freq_dim
        self.n_bands = n_bands
        self.n_enc = n_enc
        self.n_dec = n_dec

        n_stft_channels = audio_channels * 2  # stereo x real/imag = 4
        n_bins = n_fft // 2 + 1               # 3073 for n_fft=6144
        freq_band = freq_dim // n_bands        # 512

        # Frequency truncation tradeoff:
        # STFT produces n_bins bins, but only freq_dim are kept (rest zero-padded
        # for iSTFT). E.g. paper presets: 2048/3073 bins ~ 14.7 kHz cutoff;
        # fullband presets: 3072/3073 bins ~ 22 kHz (near-lossless).
        # Truncation saves compute through the entire U-Net and gives clean
        # band-split math (freq_dim / n_bands bins per band).

        # STFT config
        self.stft_config = {
            'n_fft': n_fft,
            'hop_length': hop_size,
            'win_length': win_size,
            'normalized': normalized,
            'center': True,
        }
        self.register_buffer('stft_window', torch.hann_window(win_size))

        # Frequency zero-padding for iSTFT reconstruction
        self.register_buffer('freq_pad', torch.zeros(1, n_stft_channels, n_bins - freq_dim, 1))

        # --- First conv: SplitModule K=1 (expand channels) ---
        self.first_conv = SplitModule(n_stft_channels * n_bands, G, n_bands, kernel_size=1)

        # --- Encoder: N_enc blocks of SplitAndMerge + TimeDown ---
        self.encoder_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()
        c = G
        for i in range(n_enc):
            self.encoder_blocks.append(
                SplitAndMergeModule(c, n_bands, n_split_enc, freq_band, bn_factor)
            )
            self.ds.append(TimeDownsample(c, c + G))
            c += G

        # --- Bottleneck ---
        default_tp = {
            'heads': 4, 'dim_head': 32, 'ff_mult': 2,
            'attn_dropout': 0.0, 'proj_dropout': 0.0, 'ff_dropout': 0.0,
            'flash_attn': True,
        }
        tp = transformer_params if transformer_params is not None else default_tp
        self.bottleneck = DualPathRoPEBottleneck(
            c, n_bands, n_split_enc, freq_band, bn_factor, n_rope, tp
        )

        # --- Decoder (asymmetric): n_dec heavy + (n_enc - n_dec) light ---
        # Heavy stages (deepest, with SplitAndMerge processing)
        self.dec_heavy_us = nn.ModuleList()
        self.dec_heavy_blocks = nn.ModuleList()
        for i in range(n_dec):
            self.dec_heavy_us.append(TimeUpsample(c, c - G))
            c -= G
            self.dec_heavy_blocks.append(
                SplitAndMergeModule(c, n_bands, n_split_dec, freq_band, bn_factor)
            )

        # Light stages (just upsample + skip, no SplitAndMerge)
        self.dec_light_us = nn.ModuleList()
        for i in range(n_enc - n_dec):
            self.dec_light_us.append(TimeUpsample(c, c - G))
            c -= G

        # --- Final conv: SplitModule K=1 (reduce channels) ---
        self.final_conv = SplitModule(c, n_stft_channels * n_bands, n_bands, kernel_size=1)

        # --- Source head: expand to multi-source masks (after band merge) ---
        self.source_head = nn.Conv2d(n_stft_channels, len(sources) * n_stft_channels, 1)

    def _band_split(self, x):
        """[B, C, F, T] -> [B, C*N_band, F/N_band, T]

        Splits F into N_band equal subbands, moves band index into channels.
        Channel ordering: [band0_ch0, band0_ch1, ..., band1_ch0, ...]
        so that group conv with groups=N_band processes each band independently.
        """
        B, C, F, T = x.shape
        F_band = F // self.n_bands
        # [B, C, N_band, F_band, T] -> [B, N_band, C, F_band, T] -> [B, N_band*C, F_band, T]
        x = x.reshape(B, C, self.n_bands, F_band, T)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.reshape(B, self.n_bands * C, F_band, T)
        return x

    def _band_merge(self, x):
        """[B, C*N_band, F/N_band, T] -> [B, C, F, T]

        Inverse of _band_split.
        """
        B, CN, F_band, T = x.shape
        C = CN // self.n_bands
        # [B, N_band, C, F_band, T] -> [B, C, N_band, F_band, T] -> [B, C, F, T]
        x = x.reshape(B, self.n_bands, C, F_band, T)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.reshape(B, C, self.n_bands * F_band, T)
        return x

    def forward(self, x, return_auxiliary_outputs=False):
        """
        Args:
            x: [B, C, L] audio waveform
        Returns:
            [B, S, C, L] separated sources, or ([B, S, C, L], {}) if return_auxiliary_outputs
        """
        B, C, L = x.shape

        # --- Pre-padding: ensure STFT frames T divisible by 2^n_enc ---
        time_factor = 2 ** self.n_enc  # 8
        padding = self.hop_size - L % self.hop_size
        while ((L + padding) // self.hop_size + 1) % time_factor != 0:
            padding += self.hop_size
        x = F.pad(x, (0, padding))

        # --- STFT ---
        L_padded = x.shape[-1]
        x = x.reshape(B * self.audio_channels, L_padded)
        x = torch.stft(x, **self.stft_config, window=self.stft_window,
                        return_complex=True)                          # [B*C, F_full, T]
        x = torch.view_as_real(x)                                     # [B*C, F_full, T, 2]
        x = x.permute(0, 3, 1, 2)                                     # [B*C, 2, F_full, T]
        x = x.reshape(B, self.audio_channels * 2, -1, x.shape[-1])   # [B, 4, F_full, T]
        x = x[:, :, :self.freq_dim, :]                                 # [B, n_stft_ch, freq_dim, T]
        # Detach so mask path doesn't backprop through the original STFT
        x_orig = x.detach()

        # --- Normalize ---
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        # --- Band split ---
        x = self._band_split(x)        # [B, n_stft_ch*n_bands, freq_band, T]

        # --- First conv ---
        x = self.first_conv(x)         # [B, G, freq_band, T]

        # --- Encoder ---
        skips = []
        for i in range(self.n_enc):
            x = self.encoder_blocks[i](x)
            skips.append(x)             # save BEFORE downsample
            x = self.ds[i](x)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # --- Decoder (heavy stages) ---
        for i in range(self.n_dec):
            x = self.dec_heavy_us[i](x)
            x = x * skips.pop()         # element-wise multiply (DTTNet pattern)
            x = self.dec_heavy_blocks[i](x)

        # --- Decoder (light stages) ---
        for i in range(len(self.dec_light_us)):
            x = self.dec_light_us[i](x)
            x = x * skips.pop()

        # --- Final conv ---
        x = self.final_conv(x)         # [B, n_stft_ch*n_bands, freq_band, T]

        # --- Band merge ---
        x = self._band_merge(x)        # [B, n_stft_ch, freq_dim, T]

        # --- Source head ---
        x = self.source_head(x)        # [B, S*n_stft_ch, freq_dim, T]

        # --- Output: mask or denorm, then iSTFT ---
        S = len(self.sources)
        n = self.audio_channels * 2     # 4

        # Reshape to per-source
        x = x.reshape(B, S, n, self.freq_dim, -1)    # [B, S, n_stft_ch, freq_dim, T]
        if self.use_mask:
            # Masking: multiply network output by unnormalized original STFT
            x = x * x_orig.unsqueeze(1)
        else:
            # Direct generation: denormalize the network output
            x = x * std.unsqueeze(1) + mean.unsqueeze(1)

        # Freq zero-pad back to n_fft//2+1 for iSTFT
        T_stft = x.shape[-1]
        x = x.reshape(B * S, n, self.freq_dim, T_stft)            # [B*S, n_stft_ch, freq_dim, T]
        freq_pad = self.freq_pad.expand(B * S, -1, -1, T_stft)
        x = torch.cat([x, freq_pad], dim=2)                     # [B*S, 4, 3073, T]

        # Convert to complex for iSTFT
        n_bins = self.n_fft // 2 + 1
        x = x.reshape(B * S * self.audio_channels, 2, n_bins, T_stft)  # [B*S*C, 2, 3073, T]
        x = x.permute(0, 2, 3, 1).contiguous()                       # [B*S*C, 3073, T, 2]
        x = torch.view_as_complex(x)                                  # [B*S*C, 3073, T] complex

        # iSTFT
        if x.device.type == 'mps':
            x_device = x.device
            stft_cfg = {k: v.to('cpu') if torch.is_tensor(v) else v
                        for k, v in self.stft_config.items()}
            x = torch.istft(x.to('cpu'), **stft_cfg,
                            window=self.stft_window.to('cpu'), length=L_padded)
            x = x.to(x_device)
        else:
            x = torch.istft(x, **self.stft_config,
                            window=self.stft_window, length=L_padded)

        # Reshape to [B, S, C, L] and trim padding
        x = x.reshape(B, S, self.audio_channels, -1)
        x = x[:, :, :, :L]

        if return_auxiliary_outputs:
            return x, {}
        return x
