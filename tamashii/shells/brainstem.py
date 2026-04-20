"""Brainstem shell — homeostasis / vital regulation.

Biologically inspired: the brainstem maintains baseline vital function
regardless of what higher structures do. If S drifts outside safe range,
it pulls it back via additive corrective delta.

No training needed — parameters are structural (clip thresholds).
Runs on fast tick (default 10ms) as background regulator.
"""
from __future__ import annotations

import numpy as np

from tamashii.shell_base import Shell


class Brainstem(Shell):
    """Keep S within homeostatic range via additive corrective force.

    Params:
      s_min, s_max: target range for every dim (default -3, 3)
      pull_strength: how hard to pull toward range per tick (default 0.1)
      vital_indices: S slice used for vital signs (exposed for metrics)
    """

    def step(self, S_snapshot: np.ndarray, external=None) -> np.ndarray:
        s_min = float(self.params.get("s_min", -3.0))
        s_max = float(self.params.get("s_max", 3.0))
        pull = float(self.params.get("pull_strength", 0.1))

        delta = np.zeros_like(S_snapshot)

        if self.output_indices.size > 0:
            view = S_snapshot[self.output_indices]
            # Pull toward valid range; 0 if already in range
            below = np.minimum(view - s_min, 0.0)  # negative if below s_min
            above = np.maximum(view - s_max, 0.0)  # positive if above s_max
            out_slice = self.output_indices
            delta[out_slice] = -pull * (below + above)

        # Track vitals: mean S energy (for external monitoring)
        vitals_idx = np.asarray(self.params.get("vital_indices", []), dtype=int)
        if vitals_idx.size > 0:
            energy = float(np.mean(np.abs(S_snapshot[vitals_idx])))
            self._state["energy"] = energy

        return delta
