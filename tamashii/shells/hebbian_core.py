"""HebbianCoreBrain — core brain with RUNTIME Hebbian self-learning.

Unlike CoreBrain (static kathara_params), this shell updates its edge
weights on every tick via Hebbian learning. Reward comes from INTRINSIC
signals (salience attention + sensor novelty), not external task reward.

Biological analog: spike-timing-dependent plasticity driven by attention/
salience signals (dopaminergic modulation of plasticity in cortex).

Self-learning dynamics:
  1. Every tick: read sensor + salience attention map + prev sensor
  2. Compute intrinsic reward = mean(salience) + weight * sensor_delta
  3. Run simulate_step_16_hebbian → updates w_adapt (edge perturbation)
  4. New edges = base kathara_params + w_adapt
  5. w_adapt persists within episode (and optionally across episodes)

Params (beyond CoreBrain):
  hebbian_lr         : Hebbian learning rate (default 0.05)
  hebbian_decay      : weight decay per step (default 0.005)
  hebbian_k          : top-K winners in soft k-WTA (default 3)
  w_clip             : max |w_adapt| (default 2.0)
  reward_scale       : multiplier on reward signal (default 5.0)
  salience_slice_start/end : S slice from which to read salience reward
  sensor_novelty_weight : how much sensor-delta contributes to reward
  w_adapt_carry_over : if True, w_adapt carries across episodes (=long-term learning)
"""
from __future__ import annotations

import numpy as np

from tamashii.shells.core_brain import CoreBrain


class HebbianCoreBrain(CoreBrain):
    """Core brain with runtime Hebbian plasticity driven by intrinsic reward."""

    def __init__(self, config: dict):
        super().__init__(config)
        from kathara16_brain import KATHARA16_EDGES
        self.n_edges = len(KATHARA16_EDGES)

        self.hebbian_lr = float(self.params.get("hebbian_lr", 0.05))
        self.hebbian_decay = float(self.params.get("hebbian_decay", 0.005))
        self.hebbian_k = int(self.params.get("hebbian_k", 3))
        self.w_clip = float(self.params.get("w_clip", 2.0))
        self.reward_scale = float(self.params.get("reward_scale", 5.0))

        # Intrinsic reward sources
        self.salience_slice = slice(
            int(self.params.get("salience_start", 51)),
            int(self.params.get("salience_end", 83)))
        self.sensor_novelty_weight = float(
            self.params.get("sensor_novelty_weight", 0.3))
        self.salience_weight = float(self.params.get("salience_weight", 0.7))
        # Prediction error reward (from cerebellum)
        self.pred_error_idx = int(self.params.get("pred_error_idx", 50))
        self.pred_error_weight = float(self.params.get("pred_error_weight", 0.0))
        # Taboo violation penalty (from taboo shell, ACC-like)
        self.violation_slot = int(self.params.get("violation_slot", 190))
        self.violation_penalty_weight = float(
            self.params.get("violation_penalty_weight", 5.0))

        self.w_adapt_carry_over = bool(self.params.get("w_adapt_carry_over", False))

        # Exploration noise (biological neural noise)
        # High when reward is low (explore), low when reward is high (exploit)
        self.noise_base = float(self.params.get("noise_base", 0.15))
        self.noise_reward_gate = float(self.params.get("noise_reward_gate", 2.0))
        self._noise_rng = np.random.default_rng(
            int(self.params.get("noise_seed", 42)))

        # Self-learning state
        self.w_adapt = np.zeros(self.n_edges, dtype=np.float64)
        self._prev_sensor: np.ndarray | None = None
        self._cumulative_reward = 0.0
        self._step_count = 0
        self._recent_reward_ema = 0.0

    def step(self, S_snapshot: np.ndarray, external=None) -> np.ndarray:
        from kathara16_brain import simulate_step_16_hebbian

        sensor = S_snapshot[self.sensor_slot].astype(np.float64)

        # --- Intrinsic reward computation ---
        # 1. Salience: mean attention in salience workspace (high = novel world)
        sal_mean = float(
            np.mean(np.abs(S_snapshot[self.salience_slice])))
        # 2. Sensor novelty: L1 change since last step
        if self._prev_sensor is None:
            sensor_nov = 0.0
        else:
            sensor_nov = float(np.mean(np.abs(sensor - self._prev_sensor)))
        self._prev_sensor = sensor.copy()

        # 3. Prediction error (from cerebellum, scalar at pred_error_idx)
        pred_err = float(abs(S_snapshot[self.pred_error_idx]))
        # 4. Taboo violation penalty (from taboo shell, ACC-like error signal)
        violation = float(S_snapshot[self.violation_slot])

        reward = (self.salience_weight * sal_mean
                  + self.sensor_novelty_weight * sensor_nov
                  + self.pred_error_weight * pred_err
                  - self.violation_penalty_weight * violation)
        self._cumulative_reward += reward
        self._step_count += 1
        # Track recent reward (EMA over last ~20 steps)
        self._recent_reward_ema = 0.9 * self._recent_reward_ema + 0.1 * reward

        # Exploration noise: scale INVERSELY to recent reward
        # Low reward → high noise (explore); high reward → low noise (exploit)
        # Clamp denominator to avoid negative scale when reward_ema goes negative
        denom = max(0.1, 1.0 + self.noise_reward_gate *
                    max(0.0, self._recent_reward_ema))
        noise_scale = max(0.0, self.noise_base / denom)
        sensor_with_noise = sensor + self._noise_rng.normal(
            0, noise_scale, size=sensor.shape)

        # --- Hebbian simulation step ---
        new_state, firing, self.w_adapt = simulate_step_16_hebbian(
            self.kathara_params, sensor_with_noise, self.brain_state, self.w_adapt,
            reward=reward, k=self.hebbian_k, lr=self.hebbian_lr,
            decay=self.hebbian_decay, w_clip=self.w_clip,
            reward_scale=self.reward_scale,
        )
        self.brain_state = new_state

        delta = np.zeros_like(S_snapshot)
        delta[self.motor_nav] = float(firing[self.motor_node_nav]) - S_snapshot[self.motor_nav]
        delta[self.motor_speed] = float(firing[self.motor_node_speed]) - S_snapshot[self.motor_speed]
        delta[self.motor_voice] = float(firing[self.motor_node_voice]) - S_snapshot[self.motor_voice]
        f_start = self.firing_slot.start
        width = self.firing_slot.stop - f_start
        if width >= self.N_NODES:
            delta[f_start:f_start + self.N_NODES] = (
                firing - S_snapshot[f_start:f_start + self.N_NODES])
        return delta

    def reset(self):
        super().reset()
        if not self.w_adapt_carry_over:
            self.w_adapt[:] = 0.0
        self._prev_sensor = None
        self._cumulative_reward = 0.0
        self._step_count = 0
        self._recent_reward_ema = 0.0

    def hebbian_stats(self) -> dict:
        return {
            "w_adapt_mean_abs": float(np.abs(self.w_adapt).mean()),
            "w_adapt_max_abs": float(np.abs(self.w_adapt).max()),
            "n_changed_edges": int((np.abs(self.w_adapt) > 1e-4).sum()),
            "total_edges": int(self.n_edges),
            "cumulative_reward": float(self._cumulative_reward),
            "mean_reward": (float(self._cumulative_reward / self._step_count)
                            if self._step_count else 0.0),
        }
