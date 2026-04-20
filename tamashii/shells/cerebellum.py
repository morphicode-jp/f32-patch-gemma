"""Cerebellum shell — motor prediction + smoothing.

Biologically: cerebellum predicts sensory consequences of motor commands
and reduces prediction error. Here we implement a simplified version:

1. Keep a rolling history of motor outputs
2. Predict next motor as smoothed/extrapolated from history
3. Push delta toward the prediction (stabilization: reduce motor jitter)

Reads:  S[motor_slot] (own motor output)
Writes: delta at S[cerebellum_ws] (working space) + small delta at motor
        to smooth sudden changes
"""
from __future__ import annotations

from collections import deque

import numpy as np

from tamashii.shell_base import Shell


class Cerebellum(Shell):
    """Predictive motor smoother with EMA + history blend."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.motor_dims = list(self.params.get("motor_dims", [16, 17, 18]))
        self.ws_start = int(self.params.get("ws_start", 35))
        self.ws_end = int(self.params.get("ws_end", 51))
        # Workspace width covers motor prediction history (at least 3 × len(motor_dims))
        self.history_len = int(self.params.get("history_len", 5))
        self.ema_alpha = float(self.params.get("ema_alpha", 0.6))
        self.smoothing_strength = float(
            self.params.get("smoothing_strength", 0.3))
        self._motor_history: deque = deque(maxlen=self.history_len)
        self._ema: np.ndarray | None = None

    def step(self, S_snapshot: np.ndarray, external=None) -> np.ndarray:
        motor = S_snapshot[self.motor_dims].astype(np.float64)

        # Update history + EMA
        self._motor_history.append(motor.copy())
        if self._ema is None:
            self._ema = motor.copy()
        else:
            self._ema = (self.ema_alpha * motor
                         + (1 - self.ema_alpha) * self._ema)

        # Predict next motor: simple linear extrapolation if we have 2+ samples
        if len(self._motor_history) >= 2:
            prev = self._motor_history[-2]
            delta_motor = motor - prev
            prediction = motor + 0.5 * delta_motor
        else:
            prediction = self._ema

        # Blend EMA with prediction (biological: cerebellum predicts using priors)
        blend_weight = float(self.params.get("prediction_weight", 0.5))
        pred_blended = (blend_weight * prediction
                        + (1 - blend_weight) * self._ema)

        # Prediction error (cerebellum's hallmark signal: |actual - predicted|)
        # Use PREVIOUS tick's prediction vs current motor
        pred_error = float(np.mean(np.abs(motor - self._ema)))
        self._state["last_prediction_error"] = pred_error

        delta = np.zeros_like(S_snapshot)

        # Write prediction_error to a dedicated slot (always ws_end - 1)
        # This gives a single scalar that other shells can read as "curiosity signal"
        pred_err_idx = int(self.params.get("prediction_error_idx", self.ws_end - 1))
        delta[pred_err_idx] = pred_error - S_snapshot[pred_err_idx]

        # Smoothing: push motor toward EMA (damp oscillation)
        for idx, m in zip(self.motor_dims, motor):
            ema_val = self._ema[self.motor_dims.index(idx)]
            delta[idx] = self.smoothing_strength * (ema_val - m)

        # Write prediction to workspace (other shells can read)
        width = self.ws_end - self.ws_start
        if width >= len(self.motor_dims):
            # Store: [current_motor, ema, prediction] if width allows
            slots = width // len(self.motor_dims)
            offset = self.ws_start
            if slots >= 1:
                for i, val in enumerate(motor):
                    delta[offset + i] = val - S_snapshot[offset + i]
                offset += len(self.motor_dims)
            if slots >= 2:
                for i, val in enumerate(self._ema):
                    delta[offset + i] = val - S_snapshot[offset + i]
                offset += len(self.motor_dims)
            if slots >= 3:
                for i, val in enumerate(pred_blended):
                    delta[offset + i] = val - S_snapshot[offset + i]

        return delta

    def reset(self):
        super().reset()
        self._motor_history.clear()
        self._ema = None
