"""Inhibition shells — generic guard pattern for layer-level suppression.

paper §3.2.4: "Stronger connections have higher inhibition ratios (24% → 29%)."
Brain's trunk circuits have stronger braking than branch circuits.

paper §3.2.5 / §3.2.9: Whole-brain inhibition ratio 23.9%, conserved 20-26%.
Currently Tamashii has I≈1% (only `taboo` at L2). Need more inhibitory shells
at multiple layers to reach paper's 20% target.

This module defines a generic `InhibitionShell` that:
  - Reads a configurable set of slots (activity signal)
  - If combined activity exceeds threshold, emits a dampening delta
  - shell_sign = -1 → core.py flips delta direction (suppressive)

Pre-configured variants (L1, L3, L5) cover the hierarchy's key joints.
"""
from __future__ import annotations

import numpy as np

from tamashii.shell_base import Shell


class InhibitionShell(Shell):
    """Generic layer-level inhibitory shell.

    Monitors input slots; when combined activity crosses threshold, emits
    a dampening delta to target slots. Config drives everything.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # Ensure inhibitory
        self.shell_sign = -1  # override config default

        # Slots to monitor (activity in = how hot is this layer)
        mon = self.params.get("monitor_slots", {"start": 16, "end": 35})
        self.mon_start = int(mon.get("start", 16))
        self.mon_end = int(mon.get("end", 35))

        # Slots to dampen (where suppression writes)
        damp = self.params.get("dampen_slots", {"start": 16, "end": 35})
        self.damp_start = int(damp.get("start", 16))
        self.damp_end = int(damp.get("end", 35))

        # Threshold above which suppression fires
        self.activation_threshold = float(
            self.params.get("activation_threshold", 0.7))

        # Suppression strength (fraction of current value to subtract)
        self.suppression_frac = float(
            self.params.get("suppression_frac", 0.2))

        # Smoothing (inhibition has latency, biologically)
        self.ema_alpha = float(self.params.get("ema_alpha", 0.5))
        self._prev_mean = 0.0

    def step(self, S_snapshot: np.ndarray, external=None) -> np.ndarray:
        delta = np.zeros_like(S_snapshot)

        # Monitor layer activity
        mon_values = S_snapshot[self.mon_start:self.mon_end]
        if len(mon_values) == 0:
            return delta
        activity = float(np.mean(np.abs(mon_values)))
        # EMA smoothing
        activity = (self.ema_alpha * activity
                    + (1 - self.ema_alpha) * self._prev_mean)
        self._prev_mean = activity

        # Only act if overactivation
        if activity < self.activation_threshold:
            return delta

        # Dampening: target = current * (1 - suppression_frac)
        # delta = target - current = -suppression_frac * current
        # BUT shell_sign=-1 in core.py flips this → we emit POSITIVE delta
        # that gets applied as NEGATIVE via the sign flip.
        dampen_current = S_snapshot[self.damp_start:self.damp_end]
        excess = max(0.0, activity - self.activation_threshold)
        scale = min(1.0, excess / 0.3)  # ramp up quickly over 0.3 above thresh
        # Positive delta (will be flipped to negative by shell_sign=-1)
        delta[self.damp_start:self.damp_end] = (
            self.suppression_frac * scale * dampen_current)
        return delta

    def reset(self):
        super().reset()
        self._prev_mean = 0.0
