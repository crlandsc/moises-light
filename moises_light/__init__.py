from importlib.metadata import version, PackageNotFoundError

from .moises_light import MoisesLight

try:
    __version__ = version('moises-light')
except PackageNotFoundError:
    __version__ = 'unknown'

_TRANSFORMER_PARAMS = {
    'heads': 4,
    'dim_head': 32,
    'ff_mult': 2,
    'attn_dropout': 0.0,
    'proj_dropout': 0.0,
    'ff_dropout': 0.0,
    'flash_attn': True,
}

# Shared defaults. Per-preset params (G, n_bands, freq_dim, n_rope) are merged
# in the configs dict below, so each final preset is fully explicit.
_BASE = dict(
    sources=['vocals', 'drums', 'bass', 'other'],
    audio_channels=2,
    n_fft=6144,
    hop_size=1024,
    win_size=6144,
    n_enc=3,
    n_dec=1,
    n_split_enc=3,
    n_split_dec=1,
    bn_factor=8,
    normalized=True,
    use_mask=True,
    transformer_params=_TRANSFORMER_PARAMS,
)

configs = {
    # Paper-faithful: truncated spectrum (0-14.7 kHz)
    'paper_large':         {**_BASE, 'G': 56, 'n_bands': 4, 'freq_dim': 2048, 'n_rope': 5},  # 5,451,216 params
    'paper_small':         {**_BASE, 'G': 32, 'n_bands': 4, 'freq_dim': 2048, 'n_rope': 6},  # 2,558,768 params

    # Fullband matched-param: full spectrum (0-22 kHz), similar param budget
    'fullband_large':      {**_BASE, 'G': 60, 'n_bands': 6, 'freq_dim': 3072, 'n_rope': 5},  # 5,477,844 params
    'fullband_small':      {**_BASE, 'G': 36, 'n_bands': 6, 'freq_dim': 3072, 'n_rope': 6},  # 2,805,796 params

    # Fullband wide: full spectrum, matched per-group capacity to paper
    'fullband_large_wide': {**_BASE, 'G': 84, 'n_bands': 6, 'freq_dim': 3072, 'n_rope': 5},  # 9,704,844 params
    'fullband_small_wide': {**_BASE, 'G': 48, 'n_bands': 6, 'freq_dim': 3072, 'n_rope': 6},  # 4,323,976 params
}
