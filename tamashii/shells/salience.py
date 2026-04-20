"""Salience shell — attention / novelty gain.

Biologically: amygdala-like salience network flags "important" dimensions
and modulates how strongly they influence downstream. Here we:

1. Track a rolling mean of selected S dims (=baseline)
2. Compute deviation = |S - baseline| (novelty signal)
3. Write an attention map to salience workspace
4. Apply multiplicative gain boost (via additive delta) to novel dims in
   the sensor region, amplifying signal where unexpected
"""
from __future__ import annotations

import numpy as np

from tamashii.shell_base import Shell


class Salience(Shell):
    """Novelty-driven attention modulator."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.watched_slice = slice(
            int(self.params.get("watched_start", 0)),
            int(self.params.get("watched_end", 35)))
        self.attention_slice = slice(
            int(self.params.get("attention_start", 51)),
            int(self.params.get("attention_end", 83)))
        self.baseline_alpha = float(self.params.get("baseline_alpha", 0.05))
        self.novelty_threshold = float(self.params.get("novelty_threshold", 0.15))
        self.boost_strength = float(self.params.get("boost_strength", 0.1))
        self._baseline: np.ndarray | None = None

    def step(self, S_snapshot: np.ndarray, external=None) -> np.ndarray:
        watched = S_snapshot[self.watched_slice].astype(np.float64)

        if self._baseline is None:
            self._baseline = watched.copy()
        else:
            self._baseline = (
                (1 - self.baseline_alpha) * self._baseline
                + self.baseline_alpha * watched
            )

        # Novelty: how far current watched is from rolling baseline
        novelty = np.abs(watched - self._baseline)

        # Attention map (normalized to [0, 1] by divide-by-max)
        max_n = float(novelty.max()) if novelty.size else 0.0
        if max_n > 1e-6:
            attention = novelty / max_n
        else:
            attention = np.zeros_like(novelty)

        delta = np.zeros_like(S_snapshot)

        # Write attention map to workspace (absolute-write via subtraction)
        att_width = self.attention_slice.stop - self.attention_slice.start
        att_to_write = attention[:att_width] if attention.size > att_width else attention
        a_start = self.attention_slice.start
        for i, val in enumerate(att_to_write):
            delta[a_start + i] = val - S_snapshot[a_start + i]

        # Boost novel dims in watched region (additive gain)
        # (above threshold only → sparse attention)
        boost_mask = novelty > self.novelty_threshold
        w_start = self.watched_slice.start
        for i, is_novel in enumerate(boost_mask):
            if is_novel:
                sign = 1.0 if watched[i] >= self._baseline[i] else -1.0
                delta[w_start + i] = self.boost_strength * sign * novelty[i]

        return delta

    def reset(self):
        super().reset()
        self._baseline = None
