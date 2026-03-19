from moises_light import MoisesLight


class MoisesLightModel(MoisesLight):
    """MoisesLight adapted for ZFTurbo Music-Source-Separation-Training.

    With target_instrument set: forward(x) returns [B, C, L] for that stem.
    Without target_instrument: forward(x) returns [B, S*C, L] (stems flattened into channels).
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
