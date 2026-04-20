"""Core brain shell — Kathara 16N reactive nucleus.

Wraps kathara16_brain.simulate_step_16 as the "neural core" of Tamashii.
Reads sensors from S[sensor_slot], pushes motor intent into S[motor_slot],
exposes firing pattern in S[firing_slot] for other shells to read.

Internal state (16D kathara state) lives in shell instance (not in S),
reset between episodes via reset().

Slot convention (D=192):
  S[0:16]    sensor_input    ← world writes
  S[16:19]   motor_intent    ← core pushes; world reads
  S[19:35]   firing (16D)    ← core writes (other shells can read)

Biological analog: direct sensory → motor pathway (like spinal reflex + M1).
Fast tick (50ms).
"""
from __future__ import annotations

import numpy as np

from tamashii.shell_base import Shell


class CoreBrain(Shell):
    """Kathara 16N brain as reactive core shell."""

    def __init__(self, config: dict):
        super().__init__(config)
        # Lazy imports so tamashii module doesn't crash if kathara not importable
        from kathara16_brain import (
            PARAM_RANGES_16, TOTAL_PARAMS_16, DEFAULT_INHIBIT_16, make_inhibit_sign_16,
            N_NODES_16,
        )
        self.N_NODES = N_NODES_16
        self.TOTAL_PARAMS = TOTAL_PARAMS_16

        kp = self.params.get("kathara_params")
        if kp is None:
            # Midpoint default (deterministic baseline)
            kp = [(lo + hi) / 2 for lo, hi in PARAM_RANGES_16]
        kp = np.asarray(kp, dtype=np.float64)
        assert kp.shape == (TOTAL_PARAMS_16,), (
            f"kathara_params shape {kp.shape} != ({TOTAL_PARAMS_16},)")
        self.kathara_params = kp

        # Dale's law inhibitory sign (16D)
        if self.params.get("dale_law", True):
            self.inhibit_sign = make_inhibit_sign_16()
        else:
            self.inhibit_sign = np.ones(N_NODES_16, dtype=np.float64)

        # Slot convention
        self.sensor_slot = slice(
            int(self.params.get("sensor_start", 0)),
            int(self.params.get("sensor_end", 16)))
        self.motor_nav = int(self.params.get("motor_nav", 16))
        self.motor_speed = int(self.params.get("motor_speed", 17))
        self.motor_voice = int(self.params.get("motor_voice", 18))
        self.firing_slot = slice(
            int(self.params.get("firing_start", 19)),
            int(self.params.get("firing_end", 35)))

        # Brain state (16D, NOT in S — kept per-shell, reset between episodes)
        self.brain_state = np.zeros(N_NODES_16, dtype=np.float64)

        # Motor node indices within the 16-node brain (firing[5]=nav, [11]=speed, [1]=voice)
        self.motor_node_nav = int(self.params.get("motor_node_nav", 5))
        self.motor_node_speed = int(self.params.get("motor_node_speed", 11))
        self.motor_node_voice = int(self.params.get("motor_node_voice", 1))

    def step(self, S_snapshot: np.ndarray, external=None) -> np.ndarray:
        from kathara16_brain import simulate_step_16

        sensor = S_snapshot[self.sensor_slot].astype(np.float64)
        new_state, firing = simulate_step_16(
            self.kathara_params, sensor, self.brain_state,
            inhibit_sign=self.inhibit_sign,
        )
        self.brain_state = new_state

        delta = np.zeros_like(S_snapshot)

        # Motor push: firing values at motor nodes minus current S (absolute-write pattern)
        # With brainstem decay, this creates steady-state S ≈ firing / decay
        delta[self.motor_nav] = float(firing[self.motor_node_nav]) - S_snapshot[self.motor_nav]
        delta[self.motor_speed] = float(firing[self.motor_node_speed]) - S_snapshot[self.motor_speed]
        delta[self.motor_voice] = float(firing[self.motor_node_voice]) - S_snapshot[self.motor_voice]

        # Expose firing to firing_slot (absolute-write so other shells see real activity)
        f_start = self.firing_slot.start
        f_end = self.firing_slot.stop
        width = f_end - f_start
        if width >= self.N_NODES:
            delta[f_start:f_start + self.N_NODES] = (
                firing - S_snapshot[f_start:f_start + self.N_NODES])

        return delta

    def reset(self):
        super().reset()
        self.brain_state = np.zeros(self.N_NODES, dtype=np.float64)

    def get_firing(self) -> np.ndarray:
        """Current firing pattern (16D), for diagnostics."""
        return self.brain_state.copy()
