"""Mimir shell — Layer 5 meta-dispatcher.

Top-level shell in the M1 hierarchy (paper §4.1). Mimir's role in the
twelve stack is a context-aware dispatcher that decides which lower-layer
shells get emphasis given the current state S.

Design (minimal POC):
  - Reads the full S snapshot
  - Classifies the "situation" into a few coarse regimes
      threat    : peer_proximity high, or energy low
      opportunity: olfactory high, energy mid
      idle      : nothing salient
      goal      : prefrontal_workspace shows non-zero goal
  - For each regime, emits a "context hint" delta to specific slots that
    other shells read as context boost/attenuation.

What it does NOT do yet (future work):
  - Direct modulation of other shell gains (requires per-shell gain API)
  - LLM oracle query (M5+ / future phase)
  - Goal formation (prefrontal owns that)

This is the architectural seat; the logic is intentionally simple so that
Cardinal/ISS evolution can populate it over time.

Slot convention:
  S[190] : taboo violation signal (existing)
  S[191] : NEW — mimir context classification (0=idle, 0.25=goal,
           0.5=opportunity, 0.75=threat, 1.0=emergency)

Inhibitory? No — mimir is an integrator (like dorsolateral prefrontal cortex).
"""
from __future__ import annotations

import numpy as np

from tamashii.shell_base import Shell


REGIME_IDLE = 0.00
REGIME_GOAL = 0.25
REGIME_OPPORTUNITY = 0.50
REGIME_THREAT = 0.75
REGIME_EMERGENCY = 1.00


class MimirShell(Shell):
    """Layer 5 context dispatcher."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.context_slot = int(self.params.get("context_slot", 191))
        self.peer_prox_slot = int(self.params.get("peer_prox_slot", 14))
        self.olf_slot = int(self.params.get("olf_slot", 10))
        self.violation_slot = int(self.params.get("violation_slot", 190))
        # Salience slot range (where Layer 2 writes attention)
        self.salience_start = int(self.params.get("salience_start", 100))
        self.salience_end = int(self.params.get("salience_end", 120))
        # Prefrontal goal slot (Layer 4)
        self.goal_start = int(self.params.get("goal_start", 160))
        self.goal_end = int(self.params.get("goal_end", 180))

        # Thresholds
        self.threat_peer_prox = float(
            self.params.get("threat_peer_prox", 0.5))
        self.opportunity_olf = float(
            self.params.get("opportunity_olf", 0.8))
        self.emergency_violation = float(
            self.params.get("emergency_violation", 0.7))

        # Smoothing (mimir has latency — this is prefrontal, not reflex)
        self.context_ema_alpha = float(
            self.params.get("context_ema_alpha", 0.3))
        self._prev_context = 0.0

    def step(self, S_snapshot: np.ndarray, external=None) -> np.ndarray:
        delta = np.zeros_like(S_snapshot)

        peer_prox = float(S_snapshot[self.peer_prox_slot])
        olf = float(S_snapshot[self.olf_slot])
        violation = float(S_snapshot[self.violation_slot])
        salience_mean = float(np.mean(
            S_snapshot[self.salience_start:self.salience_end]))
        goal_strength = float(np.linalg.norm(
            S_snapshot[self.goal_start:self.goal_end]))

        # Classify regime (highest priority wins)
        if violation > self.emergency_violation:
            regime = REGIME_EMERGENCY
        elif peer_prox > self.threat_peer_prox:
            regime = REGIME_THREAT
        elif olf > self.opportunity_olf:
            regime = REGIME_OPPORTUNITY
        elif goal_strength > 0.1:
            regime = REGIME_GOAL
        else:
            regime = REGIME_IDLE

        # EMA smoothing (prefrontal-like deliberation)
        smoothed = (self.context_ema_alpha * regime
                    + (1 - self.context_ema_alpha) * self._prev_context)
        self._prev_context = smoothed

        # Write classification to context slot
        current = float(S_snapshot[self.context_slot])
        delta[self.context_slot] = smoothed - current

        return delta

    def reset(self):
        super().reset()
        self._prev_context = 0.0
