# (WORK IN PROGRESS)
# Moises-Light

[![PyPI version](https://img.shields.io/pypi/v/moises-light.svg)](https://pypi.org/project/moises-light/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Resource-efficient Band-split U-Net for Music Source Separation**

PyTorch implementation of the Moises-Light architecture from ["Resource-efficient Separation Transformer"](https://arxiv.org/abs/2412.11132) (Rigatelli et al., 2024). The paper does not release code; this is an independent implementation based on the paper's description.

## Key Features

- Spectral-only U-Net with internal STFT/iSTFT
- Equal-width band splitting via group convolutions
- Dual-path RoPE transformer bottleneck
- Asymmetric encoder/decoder (3 heavy encoder stages, 1 heavy + 2 light decoder stages)
- Multi-stem output: all stems in a single forward pass (unlike the paper's per-stem approach)
- 6 preset configurations from 2.5M to 8.4M parameters

## Installation

```bash
pip install moises-light
```

## Quick Start

```python
import torch
from moises_light import MoisesLight, configs

# Use a preset
model = MoisesLight(**configs['paper_large'])

# Forward pass
x = torch.randn(1, 2, 264600)  # [batch, channels, samples] 6s @ 44.1kHz
y = model(x)                     # [1, 4, 2, 264600] = [batch, stems, channels, samples]

# With auxiliary outputs interface (for training framework compatibility)
y, aux = model(x, return_auxiliary_outputs=True)
```

## Preset Configurations

All presets use `n_fft=6144`, `hop_size=1024`, stereo input, and 4-stem output (vocals, drums, bass, other).

### Paper-Faithful (truncated spectrum, 0-14.7 kHz)

Faithful to the paper's architecture. Frequencies above ~14.7 kHz are zeroed.

| Preset | G | Bands | Per-group ch | Freq coverage | Params |
|--------|---|-------|-------------|---------------|--------|
| `paper_large` | 56 | 4 | 14 | 0-14.7 kHz | 4,660,592 |
| `paper_small` | 32 | 4 | 8 | 0-14.7 kHz | 2,520,592 |

### Fullband Matched-Param (full spectrum, 0-22 kHz, similar param budget)

Full spectrum via 6 bands of 512 bins (`freq_dim=3072`). G adjusted to keep param count close to paper variants. Trades per-band capacity for full spectrum coverage.

| Preset | G | Bands | Per-group ch | Freq coverage | Params |
|--------|---|-------|-------------|---------------|--------|
| `fullband_large` | 60 | 6 | 10 | 0-22 kHz | 4,948,612 |
| `fullband_small` | 36 | 6 | 6 | 0-22 kHz | 2,824,244 |

### Fullband Wide (full spectrum, 0-22 kHz, matched per-group capacity)

Full spectrum with the same per-group channel capacity as the paper models. More total params but no per-band capacity compromise.

| Preset | G | Bands | Per-group ch | Freq coverage | Params |
|--------|---|-------|-------------|---------------|--------|
| `fullband_large_wide` | 84 | 6 | 14 | 0-22 kHz | 8,354,908 |
| `fullband_small_wide` | 48 | 6 | 8 | 0-22 kHz | 4,102,712 |

### Why Three Tiers?

**G (base channel width)** must be divisible by `n_bands` for group convolutions. With 6 bands, G cannot be 56 (not divisible by 6). Two strategies:

1. **Matched-param**: Pick the nearest divisible G that keeps total params similar (G=60 or 36). Each group conv processes fewer channels per band, but total model size stays close to the paper.

2. **Matched per-group (wide)**: Pick G so that `G/n_bands` equals the paper's per-group count (84/6=14, matching 56/4=14). Each band gets identical capacity, but total params increase ~1.8x.

## Architecture

```
Input [B, C, L]
    |
    v
STFT -> [B, 4, F, T]         (stereo * real/imag = 4 channels)
    |
    v
Freq truncation -> [B, 4, freq_dim, T]
    |
    v
Z-score normalize
    |
    v
Band split -> [B, 4*n_bands, freq_dim/n_bands, T]
    |
    v
First conv (group K=1) -> [B, G, F_band, T]
    |
    v
Encoder (3x): SplitAndMerge + TimeDownsample
    |
    v
Bottleneck: SplitAndMerge + N_rope x DualPath(FreqRoPE + TimeRoPE)
    |
    v
Decoder: 1 heavy (upsample + skip * SplitAndMerge) + 2 light (upsample + skip)
    |
    v
Final conv (group K=1) -> [B, 4*n_bands, F_band, T]
    |
    v
Band merge -> [B, 4, freq_dim, T]
    |
    v
Source head -> [B, S*4, freq_dim, T]
    |
    v
Multiplicative mask on original STFT
    |
    v
Freq zero-pad + iSTFT -> [B, S, C, L]
```

## Key Parameters

| Parameter | Description | Constraint |
|-----------|-------------|------------|
| `G` | Base channel width. Channels at encoder stage i = G*(i+1) | Must be divisible by `n_bands` |
| `n_bands` | Number of equal-width frequency bands for group conv | `freq_dim` must be divisible by `n_bands` |
| `freq_dim` | Number of STFT bins to process (rest zero-padded) | Paper: 2048 (~14.7 kHz). Fullband: 3072 (~22 kHz) |
| `n_rope` | Number of dual-path RoPE transformer blocks in bottleneck | Paper large: 5, paper small: 6 |
| `n_enc` / `n_dec` | Encoder stages / heavy decoder stages | Asymmetric: `n_dec < n_enc` saves params |
| `n_split_enc` / `n_split_dec` | Number of group conv layers per SplitAndMerge block | Controls depth within each stage |
| `bn_factor` | TDF bottleneck factor (freq_dim -> freq_dim/bn_factor -> freq_dim) | Higher = more compression |

## Multi-Stem vs Per-Stem

This implementation outputs all stems simultaneously. The paper trains separate per-stem models. To use per-stem:

```python
model = MoisesLight(**{**configs['paper_large'], 'sources': ['vocals']})
y = model(x)  # [B, 1, C, L]
```

## Frequency Truncation

Paper presets use `freq_dim=2048`, keeping 2048 of 3073 STFT bins (~14.7 kHz at 44.1 kHz sample rate). Everything above is zeroed — hi-hats, vocal air, cymbal shimmer above ~15 kHz are not recovered. This saves ~33% compute through the U-Net.

Fullband presets use `freq_dim=3072`, covering 0-22 kHz (nearly the full spectrum).

## Integration

### ZFTurbo Music-Source-Separation-Training

```python
from moises_light import MoisesLight

model = MoisesLight(
    sources=['vocals', 'drums', 'bass', 'other'],
    audio_channels=2, n_fft=6144, hop_size=1024, win_size=6144,
    freq_dim=2048, n_bands=4, G=56, n_enc=3, n_dec=1,
    n_split_enc=3, n_split_dec=1, n_rope=5, bn_factor=4,
)
```

### Custom Training Loop

```python
model = MoisesLight(**configs['paper_large'])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

for batch in dataloader:
    mix = batch['mix']          # [B, 2, L]
    targets = batch['targets']  # [B, 4, 2, L]
    pred = model(mix)           # [B, 4, 2, L]
    loss = criterion(pred, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Known Limitations

- **MPS (Apple Silicon)**: `torch.istft` does not support MPS. The model automatically falls back to CPU for iSTFT, which adds overhead. This is a PyTorch limitation, not a model issue.
- **Frequency truncation**: Paper presets zero frequencies above ~14.7 kHz. Use fullband presets if high-frequency content matters.

## Citation

```bibtex
@article{rigatelli2024resource,
  title={Resource-efficient Separation Transformer},
  author={Rigatelli, Luca and Condorelli, Stefano and Scardapane, Simone},
  journal={arXiv preprint arXiv:2412.11132},
  year={2024}
}
```

## License

MIT
