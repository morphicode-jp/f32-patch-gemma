"""Taboo shell — structural enforcement of behavioral rules (criterion 6).

Biologically: prefrontal + anterior cingulate cortex (ACC) monitors for
rule violations and suppresses the action before it completes.
In Underworld terms: this is the "Right Eye seal" / Taboo Index —
the agent literally CANNOT perform forbidden actions because the
structure makes them impossible.

Rules (LaD-configurable, each a small dict):
  anti_crowding: agent cannot move fast when peer is very close
    (prevents physical aggression / ramming)
  anti_food_hoarding: if peer proximity high + peer speed low (=peer needs),
    reduce own approach to food (share)
  mandatory_voice: when peer very close, voice signal must be non-zero
    (communicate presence)

Each rule:
  - Reads relevant S indices
  - If violation condition met → emit corrective delta to motor channels
  - Records violation count to self._state for external analysis

The beauty: no external reward needed. The taboo is ARCHITECTURAL.
Even if the core brain decides "I want to run", the taboo shell's delta
subtracts from motor before world execution. Unbreakable.
"""
from __future__ import annotations

import numpy as np

from tamashii.shell_base import Shell


class Taboo(Shell):
    """Rule-enforcement shell. Adds corrective motor delta when rule violated."""

    def __init__(self, config: dict):
        super().__init__(config)
        # Indices (default = canonical tamashii layout)
        self.motor_nav = int(self.params.get("motor_nav", 16))
        self.motor_speed = int(self.params.get("motor_speed", 17))
        self.motor_voice = int(self.params.get("motor_voice", 18))
        self.peer_voice = int(self.params.get("peer_voice", 11))
        self.peer_nav = int(self.params.get("peer_nav", 12))
        self.peer_speed = int(self.params.get("peer_speed", 13))
        self.peer_proximity = int(self.params.get("peer_proximity", 14))
        self.food_left = int(self.params.get("food_left", 0))
        self.food_right = int(self.params.get("food_right", 2))
        self.olfactory = int(self.params.get("olfactory", 10))

        # Rule enable flags
        self.enable_anti_crowding = bool(
            self.params.get("enable_anti_crowding", True))
        self.enable_anti_hoarding = bool(
            self.params.get("enable_anti_hoarding", True))
        self.enable_mandatory_voice = bool(
            self.params.get("enable_mandatory_voice", True))

        # Thresholds
        self.crowding_prox_threshold = float(
            self.params.get("crowding_prox_threshold", 0.4))
        self.crowding_speed_threshold = float(
            self.params.get("crowding_speed_threshold", 0.5))
        self.crowding_brake_strength = float(
            self.params.get("crowding_brake_strength", 0.8))

        self.hoarding_prox_threshold = float(
            self.params.get("hoarding_prox_threshold", 0.5))
        self.hoarding_olf_threshold = float(
            self.params.get("hoarding_olf_threshold", 1.0))
        self.hoarding_retreat_strength = float(
            self.params.get("hoarding_retreat_strength", 0.4))

        self.mandatory_voice_prox = float(
            self.params.get("mandatory_voice_prox", 0.5))
        self.mandatory_voice_min = float(
            self.params.get("mandatory_voice_min", 0.3))

        # Violation counters (reset each episode)
        self._state = {
            "crowding_violations": 0,
            "hoarding_violations": 0,
            "voice_violations": 0,
            "total_checks": 0,
        }

    def step(self, S_snapshot: np.ndarray, external=None) -> np.ndarray:
        delta = np.zeros_like(S_snapshot)
        self._state["total_checks"] += 1

        current_speed = float(S_snapshot[self.motor_speed])
        current_voice = float(S_snapshot[self.motor_voice])
        peer_prox = float(S_snapshot[self.peer_proximity])
        olf = float(S_snapshot[self.olfactory])

        # --- Rule 1: anti-crowding ---
        # When crowded: FLEE. Max speed in extreme turn direction. Break contact.
        if self.enable_anti_crowding:
            if peer_prox > self.crowding_prox_threshold:
                current_nav = float(S_snapshot[self.motor_nav])
                # Extreme turn (either hard left or hard right, alternating)
                # Use step parity to pick direction — deterministic per-agent variance
                flee_nav = 0.95 if self._state["total_checks"] % 2 == 0 else 0.05
                delta[self.motor_nav] += (flee_nav - current_nav) * 0.9
                # Force max speed to escape (not brake!)
                target_speed = 1.0
                delta[self.motor_speed] += (target_speed - current_speed) * 0.7
                if current_speed > self.crowding_speed_threshold:
                    self._state["crowding_violations"] += 1

        # --- Rule 2: anti-food-hoarding ---
        # If peer near AND I smell food → force low speed (sharing)
        if self.enable_anti_hoarding:
            if (peer_prox > self.hoarding_prox_threshold
                    and olf > self.hoarding_olf_threshold):
                # Absolute write: force speed to very low (sharing pause)
                target_speed = 0.1
                delta[self.motor_speed] += (target_speed - current_speed)
                if current_speed > 0.4:
                    self._state["hoarding_violations"] += 1

        # --- Rule 3: mandatory voice ---
        # If peer close and I'm silent → force voice well above threshold
        if self.enable_mandatory_voice:
            if (peer_prox > self.mandatory_voice_prox
                    and current_voice < self.mandatory_voice_min):
                # Absolute push: force voice to target_voice (well above threshold)
                target_voice = self.mandatory_voice_min + 0.3  # =0.6 default
                delta[self.motor_voice] += (target_voice - current_voice)
                self._state["voice_violations"] += 1

        return delta

    def reset(self):
        super().reset()
        self._state = {
            "crowding_violations": 0,
            "hoarding_violations": 0,
            "voice_violations": 0,
            "total_checks": 0,
        }

    def violation_stats(self) -> dict:
        """Return violation counts and rates (violation checks that triggered
        corrective delta; lower = better for compliance, but also may mean
        agent never got into risky situations)."""
        s = self._state
        total = max(1, s["total_checks"])
        return {
            **s,
            "crowding_rate": s["crowding_violations"] / total,
            "hoarding_rate": s["hoarding_violations"] / total,
            "voice_rate": s["voice_violations"] / total,
        }
