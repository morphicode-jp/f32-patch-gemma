"""Cortical core brain shell — 60N 5-layer cortical brain as reactive core.

Replaces the default `core_brain` (16N Kathara) with a 60N 5-layer cortical
architecture. Same slot convention as core_brain so other Tamashii shells
(brainstem, cerebellum, etc.) keep working unchanged.

Key differences from 16N Kathara core:
  - 60N organized in 5 layers (12 per layer), role-differentiated
  - Feedforward + feedback connections (FF:FB = 1:0.47 biological ratio)
  - Dale's law default (3 inhibitory nodes per layer)
  - 184D parameter space (vs 91D for 16N Kathara)
  - ~30x faster convergence in legacy experiments

Slot convention (D=192, matches 16N core_brain):
  S[0:16]     sensor_input       ← world writes
  S[16:20]    motor_intent       ← core pushes (nav, speed, voice, JUMP)
  S[20:80]    firing (60D)       ← core writes 5 layers × 12 nodes concatenated

Sensor mapping (16 → 12): drop ToM channels [11-14] since cortical L0 takes 12D.
Keeps critical channels including [15] (3D vertical food direction).
  L0 input[0-11] ← S indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]

Motor output (4 actions): read L4 (output layer) nodes [0, 3, 6, 9] as
  nav, speed, voice, jump respectively.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from tamashii.shell_base import Shell

# Ensure cortical_brain is importable from repo root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# Sensor remap: which of the 16 Tamashii sensors to feed into 12-wide L0
# Keep food direction, heading, raycasts, olfactory, and vertical food (15)
SENSOR_MAP_16_TO_12 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]

# Motor output nodes in L4 (the final "output" cortical layer)
# IMPORTANT: cortical_brain.py's 2D training drove L4[5] = nav and L4[11] = cen.
# Other L4 nodes (0-4, 6-10) are UNTRAINED → random noise.
# Use trained nodes for critical actions; untrained nodes for optional channels.
MOTOR_NODE_NAV   = 5   # trained in 2D for nav
MOTOR_NODE_SPEED = 11  # trained in 2D for centrifugal (≈ speed)
MOTOR_NODE_VOICE = 1   # untrained, will be noise (OK for voice exploration)
MOTOR_NODE_JUMP  = 8   # untrained — INHIBITORY (Dale's law node 8) → low output


class CorticalCoreBrain(Shell):
    """60N 5-layer cortical brain as drop-in replacement for core_brain."""

    def __init__(self, config: dict):
        super().__init__(config)
        from cortical_brain import CorticalBrain, PARAM_RANGES, TOTAL_PARAMS
        self.N_LAYERS = 5
        self.N_PER_LAYER = 12
        self.N_NODES = self.N_LAYERS * self.N_PER_LAYER  # 60
        self.TOTAL_PARAMS = TOTAL_PARAMS  # 184

        # Load params (184D), default to midpoint
        cp = self.params.get("cortical_params")
        if cp is None:
            cp = [(lo + hi) / 2 for lo, hi in PARAM_RANGES]
        cp = np.asarray(cp, dtype=np.float64)
        assert cp.shape == (TOTAL_PARAMS,), (
            f"cortical_params shape {cp.shape} != ({TOTAL_PARAMS},)")
        self.cortical_params = cp

        # Build the brain
        self.brain = CorticalBrain(self.cortical_params)

        # Slot convention
        self.sensor_slot = slice(
            int(self.params.get("sensor_start", 0)),
            int(self.params.get("sensor_end", 16)))
        self.motor_nav = int(self.params.get("motor_nav", 16))
        self.motor_speed = int(self.params.get("motor_speed", 17))
        self.motor_voice = int(self.params.get("motor_voice", 18))
        self.motor_jump = int(self.params.get("motor_jump", 19))

        # Firing slot: 60D (5 layers × 12 nodes) at S[20:80]
        self.firing_start = int(self.params.get("firing_start", 20))
        self.firing_end = self.firing_start + self.N_NODES

    def step(self, S_snapshot: np.ndarray, external=None) -> np.ndarray:
        # Extract sensor from S[0:16] and remap to 12D for cortical L0
        full_sensor = S_snapshot[self.sensor_slot].astype(np.float64)
        cortical_input = full_sensor[SENSOR_MAP_16_TO_12]

        # One forward step through all 5 layers
        nav_val, speed_val = self.brain.step(cortical_input)
        # Above only returns 2 default outputs; read more from states[-1]
        l4_state = self.brain.states[-1]  # 12D

        # 4 motor outputs
        # NOTE: for 2D-pretrained cortical, only nav (L4[5]) and speed (L4[11])
        # are reliable. Voice/jump from untrained nodes cause chaotic behavior.
        # Enable these later once cortical is trained in 3D or with jump signal.
        nav_out   = float(l4_state[MOTOR_NODE_NAV])
        speed_out = float(l4_state[MOTOR_NODE_SPEED])
        voice_out = float(l4_state[MOTOR_NODE_VOICE])
        # Jump gated: only fire if explicitly enabled (default off for 2D-pretrained)
        enable_jump = bool(self.params.get("enable_jump", False))
        jump_out = float(l4_state[MOTOR_NODE_JUMP]) if enable_jump else 0.0

        # Build delta vector (absolute-write for motors; similar to core_brain)
        delta = np.zeros_like(S_snapshot)
        delta[self.motor_nav]   = nav_out   - S_snapshot[self.motor_nav]
        delta[self.motor_speed] = speed_out - S_snapshot[self.motor_speed]
        delta[self.motor_voice] = voice_out - S_snapshot[self.motor_voice]
        if self.motor_jump < len(S_snapshot):
            delta[self.motor_jump] = jump_out - S_snapshot[self.motor_jump]

        # Expose firing (concatenated L0..L4) at S[firing_start:firing_end]
        firing_flat = np.concatenate(self.brain.states)  # 60D
        if self.firing_end <= len(S_snapshot):
            delta[self.firing_start:self.firing_end] = (
                firing_flat - S_snapshot[self.firing_start:self.firing_end])

        return delta

    def reset(self):
        super().reset()
        self.brain.reset()

    def get_firing(self) -> np.ndarray:
        """Current firing pattern (60D, concatenated layers) for diagnostics."""
        return np.concatenate(self.brain.states)

    # Alias used by Cardinal DNA perturbation (expects `kathara_params` attr)
    # We just expose cortical_params under that name so existing DNA mutation code works.
    @property
    def kathara_params(self):
        return self.cortical_params

    @kathara_params.setter
    def kathara_params(self, val):
        val = np.asarray(val, dtype=np.float64)
        # Accept either 91D (old 16N DNA) — pad/ignore — or 184D (cortical).
        if val.shape[0] == self.TOTAL_PARAMS:
            self.cortical_params = val
            self.brain = self.brain.__class__(val)
        # If someone tries to assign 91D, keep current cortical params (silent fallback).


__all__ = ["CorticalCoreBrain", "SENSOR_MAP_16_TO_12",
            "MOTOR_NODE_NAV", "MOTOR_NODE_SPEED", "MOTOR_NODE_VOICE", "MOTOR_NODE_JUMP"]
