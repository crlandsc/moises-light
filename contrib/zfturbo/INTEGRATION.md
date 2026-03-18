# ZFTurbo Integration

Files to add/modify in [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training):

## Files to Add

1. `models/moises_light_model.py` - Model wrapper (per-stem output for ZFTurbo's loss interface)
2. `configs/config_musdb18_moises_light.yaml` - Training config for MUSDB18

## Files to Modify

### `utils/settings.py`

Add this elif block in the model selection chain:

```python
elif model_type == 'moises_light':
    from models.moises_light_model import MoisesLightModel
    model = MoisesLightModel(**dict(config.model))
```

### `requirements.txt`

Add:

```
moises-light
```
