import pytest
import torch
from moises_light import MoisesLight, configs


# Expected exact param counts for all presets
EXPECTED_PARAMS = {
    'paper_large':         4_660_592,
    'paper_small':         2_520_592,
    'fullband_large':      4_948_612,
    'fullband_small':      2_824_244,
    'fullband_large_wide': 8_354_908,
    'fullband_small_wide': 4_102_712,
}


@pytest.fixture
def model_paper_large():
    return MoisesLight(**configs['paper_large'])


@pytest.fixture
def model_fullband_large():
    return MoisesLight(**configs['fullband_large'])


# --- Shape tests ---

def test_forward_shape(model_paper_large):
    x = torch.randn(1, 2, 264600)  # 6s @ 44.1kHz
    with torch.no_grad():
        y = model_paper_large(x)
    assert y.shape == (1, 4, 2, 264600)  # [B, S=4, C=2, L]


def test_forward_return_auxiliary(model_paper_large):
    x = torch.randn(1, 2, 264600)
    with torch.no_grad():
        y, aux = model_paper_large(x, return_auxiliary_outputs=True)
    assert y.shape == (1, 4, 2, 264600)
    assert isinstance(aux, dict)


def test_forward_shape_fullband(model_fullband_large):
    x = torch.randn(1, 2, 264600)
    with torch.no_grad():
        y = model_fullband_large(x)
    assert y.shape == (1, 4, 2, 264600)


# --- Param count tests (exact) ---

@pytest.mark.parametrize("preset_name,expected_count", list(EXPECTED_PARAMS.items()))
def test_param_count(preset_name, expected_count):
    model = MoisesLight(**configs[preset_name])
    total = sum(p.numel() for p in model.parameters())
    assert total == expected_count, f"{preset_name}: expected {expected_count:,}, got {total:,}"


# --- Configs import ---

def test_configs_has_all_presets():
    expected = {'paper_large', 'paper_small', 'fullband_large', 'fullband_small',
                'fullband_large_wide', 'fullband_small_wide'}
    assert set(configs.keys()) == expected


def test_configs_version():
    from moises_light import __version__
    assert isinstance(__version__, str)
    assert __version__ != 'unknown'


# --- Gradient flow ---

def test_gradient_flow(model_paper_large):
    x = torch.randn(1, 2, 44100, requires_grad=False)  # 1s
    model_paper_large.train()
    y = model_paper_large(x)
    loss = y.sum()
    loss.backward()
    grad_params = [p for p in model_paper_large.parameters() if p.grad is not None]
    assert len(grad_params) > 0


# --- Band split/merge roundtrip ---

def test_band_split_roundtrip_4band(model_paper_large):
    """4-band config: F=2048, n_bands=4."""
    x = torch.randn(2, 4, 2048, 32)
    split = model_paper_large._band_split(x)
    merged = model_paper_large._band_merge(split)
    assert torch.allclose(x, merged)


def test_band_split_roundtrip_6band(model_fullband_large):
    """6-band config: F=3072, n_bands=6."""
    x = torch.randn(2, 4, 3072, 32)
    split = model_fullband_large._band_split(x)
    merged = model_fullband_large._band_merge(split)
    assert torch.allclose(x, merged)


# --- Validation ---

def test_g_divisibility_validation():
    """G must be divisible by n_bands."""
    with pytest.raises(ValueError, match="G=55 must be divisible by n_bands=4"):
        MoisesLight(G=55, n_bands=4)

    with pytest.raises(ValueError, match="G=50 must be divisible by n_bands=6"):
        MoisesLight(G=50, n_bands=6)


def test_use_mask_default_true():
    model = MoisesLight(**configs['paper_large'])
    assert model.use_mask is True


def test_forward_direct_generation():
    """use_mask=False: direct spectrogram generation with denormalization."""
    cfg = {**configs['paper_large'], 'use_mask': False}
    model = MoisesLight(**cfg)
    x = torch.randn(1, 2, 264600)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 4, 2, 264600)


def test_direct_generation_param_count_unchanged():
    """use_mask adds no parameters."""
    cfg = {**configs['paper_large'], 'use_mask': False}
    model = MoisesLight(**cfg)
    total = sum(p.numel() for p in model.parameters())
    assert total == EXPECTED_PARAMS['paper_large']


def test_channel_divisibility():
    """All channel dims must be divisible by n_bands."""
    for G in [32, 48, 56, 64]:
        for n_enc in [2, 3]:
            for i in range(n_enc + 2):
                assert (G * (i + 1)) % 4 == 0, f"G={G}, stage {i}: {G*(i+1)} not div by 4"
