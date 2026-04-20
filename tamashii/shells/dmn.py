"""DMN shell — Default Mode Network: offline simulation / replay.

Biologically: DMN activates during rest / low external engagement,
performing self-referential simulation and memory consolidation.

Here we implement:
1. Monitor shell activity (novelty / coupling proxy via S delta magnitude)
2. When "engagement" is LOW (rest), replay hippocampus traces to
   scratchpad S[scratchpad_slice] = imagined sensory input
3. When engagement is HIGH (active), DMN suppressed (small delta)

This gives the agent an offline simulation mode that doesn't interfere
with active processing (biological anti-correlation between DMN and
task-positive networks).
"""
from __future__ import annotations

import numpy as np

from tamashii.shell_base import Shell


class DMN(Shell):
    """Rest-state replay / simulation."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.hippo_slice = slice(
            int(self.params.get("hippo_start", 83)),
            int(self.params.get("hippo_end", 147)))
        self.scratch_slice = slice(
            int(self.params.get("scratch_start", 176)),
            int(self.params.get("scratch_end", 192)))
        self.motor_dims = list(self.params.get("motor_dims", [16, 17, 18]))
        self.engagement_threshold = float(self.params.get("engagement_threshold", 0.3))
        self.replay_strength = float(self.params.get("replay_strength", 0.15))
        self.decay = float(self.params.get("decay", 0.9))  # scratch decay
        self.slot_dim = int(self.params.get("slot_dim", 8))

        # Rolling estimate of engagement (|motor| magnitude + delta)
        self._engagement_ema = 0.0
        self._engagement_alpha = 0.1
        self._replay_idx = 0

    def step(self, S_snapshot: np.ndarray, external=None) -> np.ndarray:
        # Engagement proxy: recent motor magnitude + sensor variation
        motor = S_snapshot[self.motor_dims]
        motor_mag = float(np.linalg.norm(motor))
        self._engagement_ema = (
            (1 - self._engagement_alpha) * self._engagement_ema
            + self._engagement_alpha * motor_mag
        )

        delta = np.zeros_like(S_snapshot)

        # Decay scratchpad (always)
        sc_start = self.scratch_slice.start
        sc_end = self.scratch_slice.stop
        for i in range(sc_start, sc_end):
            delta[i] = (self.decay - 1.0) * S_snapshot[i]  # shrink toward 0

        # If engagement is low (rest), replay hippocampus trace
        if self._engagement_ema < self.engagement_threshold:
            hp_traces = S_snapshot[self.hippo_slice]
            n_slots = max(1, hp_traces.size // self.slot_dim)
            trace_offset = (self._replay_idx % n_slots) * self.slot_dim
            trace = hp_traces[trace_offset : trace_offset + self.slot_dim]

            scratch_width = sc_end - sc_start
            # Write trace (truncated) into scratchpad
            n_write = min(trace.size, scratch_width)
            for i in range(n_write):
                delta[sc_start + i] += self.replay_strength * float(trace[i])
            self._replay_idx += 1

        return delta

    def reset(self):
        super().reset()
        self._engagement_ema = 0.0
        self._replay_idx = 0

    def engagement(self) -> float:
        return float(self._engagement_ema)
