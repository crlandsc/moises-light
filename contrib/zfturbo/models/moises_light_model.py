from moises_light import MoisesLight


class MoisesLightModel(MoisesLight):
    """MoisesLight adapted for ZFTurbo Music-Source-Separation-Training.

    Operates in per-stem mode: forward(x) returns [B, C, L] for the target stem.
    Set target_instrument in the config to select which stem to train.
    """
    def __init__(self, target_instrument=None, **kwargs):
        super().__init__(**kwargs)
        self.target_instrument = target_instrument
        if target_instrument and target_instrument in self.sources:
            self._target_idx = self.sources.index(target_instrument)
        else:
            self._target_idx = None

    def forward(self, x):
        # Full multi-stem forward
        y = super().forward(x)  # [B, S, C, L]

        if self._target_idx is not None:
            # Per-stem mode: return only the target stem
            return y[:, self._target_idx]  # [B, C, L]

        # Multi-stem mode: flatten stems into channels for ZFTurbo loss compat
        B, S, C, L = y.shape
        return y.reshape(B, S * C, L)  # [B, S*C, L]
