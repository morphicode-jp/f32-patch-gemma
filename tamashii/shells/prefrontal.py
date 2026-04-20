"""Prefrontal shell — working memory + top-down goal bias.

Biologically: prefrontal cortex maintains task-relevant info in working
memory across seconds, biasing sensory and motor processing toward goals.

Here we implement:
1. Watch the salience attention map (S[51:83])
2. Identify top-K attended dims
3. "Lock in" those dims as goal slots (with decay)
4. Each tick, push a bias toward the lock-in value (top-down amplification)

Reads:  S[attention_slice] (attention map from salience)
        S[watched_slice]   (monitored feature region)
Writes: S[wm_slot]          (working memory workspace, visible to other shells)
        additive bias to    S[watched_slice] dims in goal set
"""
from __future__ import annotations

import numpy as np

from tamashii.shell_base import Shell


class Prefrontal(Shell):
    """Working memory with top-down goal maintenance bias."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.attention_slice = slice(
            int(self.params.get("attention_start", 51)),
            int(self.params.get("attention_end", 83)))
        self.watched_slice = slice(
            int(self.params.get("watched_start", 0)),
            int(self.params.get("watched_end", 35)))
        self.wm_slice = slice(
            int(self.params.get("wm_start", 160)),
            int(self.params.get("wm_end", 176)))
        self.n_slots = int(self.params.get("n_slots", 3))
        self.decay_rate = float(self.params.get("decay_rate", 0.05))
        self.lock_threshold = float(self.params.get("lock_threshold", 0.5))
        self.bias_strength = float(self.params.get("bias_strength", 0.1))

        # slots: list of dicts {source_idx, target_value, activation}
        self._slots: list[dict] = []

    def step(self, S_snapshot: np.ndarray, external=None) -> np.ndarray:
        attention = S_snapshot[self.attention_slice]
        watched = S_snapshot[self.watched_slice]
        w_start = self.watched_slice.start
        w_width = self.watched_slice.stop - w_start
        att_width = attention.size

        delta = np.zeros_like(S_snapshot)

        # Decay existing slot activations
        for slot in self._slots:
            slot["activation"] *= (1.0 - self.decay_rate)

        # Identify newly-attended dims (attention > threshold)
        if att_width > 0:
            att_trim = attention[:w_width] if att_width > w_width else attention
            lock_candidates = np.where(att_trim > self.lock_threshold)[0]
            for c_idx in lock_candidates:
                # Check if already in slots
                exists = any(s["source_idx"] == int(c_idx) for s in self._slots)
                if not exists and len(self._slots) < self.n_slots:
                    # Lock in: store target value = current watched value
                    self._slots.append({
                        "source_idx": int(c_idx),
                        "target_value": float(watched[c_idx]),
                        "activation": 1.0,
                    })

        # Remove slots whose activation fell below a floor
        self._slots = [s for s in self._slots if s["activation"] > 0.05]

        # Sort by activation descending, keep top n_slots
        self._slots.sort(key=lambda s: -s["activation"])
        self._slots = self._slots[: self.n_slots]

        # Push top-down bias: for each slot, delta pushes watched toward target
        for slot in self._slots:
            idx = slot["source_idx"]
            cur = watched[idx]
            tgt = slot["target_value"]
            gain = self.bias_strength * slot["activation"]
            delta[w_start + idx] += gain * (tgt - cur)

        # Write slot summary to wm_slice (for visibility / diagnostics)
        # Layout: [activations ... | source_indices ... ]
        wm_start = self.wm_slice.start
        wm_width = self.wm_slice.stop - wm_start
        if wm_width >= 2 * self.n_slots:
            for i in range(self.n_slots):
                if i < len(self._slots):
                    act = self._slots[i]["activation"]
                    src = float(self._slots[i]["source_idx"])
                else:
                    act, src = 0.0, 0.0
                delta[wm_start + i] = act - S_snapshot[wm_start + i]
                delta[wm_start + self.n_slots + i] = src - S_snapshot[wm_start + self.n_slots + i]

        return delta

    def reset(self):
        super().reset()
        self._slots = []

    def slot_stats(self) -> dict:
        return {
            "n_active_slots": len(self._slots),
            "mean_activation": (
                float(np.mean([s["activation"] for s in self._slots]))
                if self._slots else 0.0),
        }
