"""gpu_runner.py — batched multi-agent tamashii tick with GPU core_brain.

Usage pattern:
    from tamashii.gpu_runner import GPUBatchRunner
    runner = GPUBatchRunner(agents, device="cuda")
    runner.tick_once(sensor_matrix)  # (N, 16)
    # other shells (brainstem, cerebellum, ...) still run per-agent on CPU
    # but are much cheaper than core_brain

Design:
  - Extract all agents' core_brain shells
  - Stack their kathara_params into (N, 129) GPU tensor (once, at init)
  - Per tick:
      1. Gather all S snapshots from agents' shells[0] perspective
      2. Run simulate_step_16_batched GPU
      3. Write new_states + firing back to each shell's _state and S[motor/firing slots]
      4. Run OTHER shells (brainstem, cerebellum, etc.) on CPU as normal
  - Maintains numerical parity with sequential CPU tick
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tamashii.core import Tamashii
from tamashii.shells.core_brain import CoreBrain

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class GPUBatchRunner:
    """Runs N Tamashii agents with core_brain batched on GPU.

    Other shells continue per-agent CPU (they're cheap vs core_brain).
    """

    def __init__(self, agents: list[Tamashii], device: str = "cuda"):
        if not HAVE_TORCH:
            raise ImportError("PyTorch required for GPUBatchRunner")
        # Ensure cuda available if requested
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.agents = agents
        self.N = len(agents)

        # Extract all core_brain shells (must be first shell)
        self.core_shells: list[CoreBrain] = []
        for ag in agents:
            assert len(ag.shells) > 0 and isinstance(ag.shells[0], CoreBrain), (
                "agent.shells[0] must be a CoreBrain instance")
            self.core_shells.append(ag.shells[0])

        # Stack kathara_params as (N, 129) tensor on device
        params_list = [np.asarray(cs.kathara_params, dtype=np.float32)
                       for cs in self.core_shells]
        self.params_tensor = torch.from_numpy(np.stack(params_list)).to(device)

        # Stack inhibit_signs (all same usually, but respect per-agent)
        inh_list = [np.asarray(cs.inhibit_sign, dtype=np.float32)
                    for cs in self.core_shells]
        self.inhibit_tensor = torch.from_numpy(np.stack(inh_list)).to(device)

        # Brain states (N, 16) — persist across ticks
        self.brain_states = torch.zeros(self.N, 16, dtype=torch.float32,
                                         device=device)

    def sync_params_from_shells(self):
        """Re-read kathara_params from shells (needed if Hebbian updated them)."""
        params_list = [np.asarray(cs.kathara_params, dtype=np.float32)
                       for cs in self.core_shells]
        self.params_tensor = torch.from_numpy(
            np.stack(params_list)).to(self.device)

    def add_agent(self, tamashii_agent):
        """Dynamically append an agent (e.g. new child from reproduction).

        Grows params_tensor, brain_states, inhibit_tensor via torch.cat.
        """
        assert len(tamashii_agent.shells) > 0 and isinstance(
            tamashii_agent.shells[0], CoreBrain), (
            "agent.shells[0] must be CoreBrain")
        cs = tamashii_agent.shells[0]
        self.agents.append(tamashii_agent)
        self.core_shells.append(cs)
        # Append new row to params_tensor
        new_params = torch.from_numpy(np.asarray(cs.kathara_params, dtype=np.float32)
                                       ).unsqueeze(0).to(self.device)
        self.params_tensor = torch.cat([self.params_tensor, new_params], dim=0)
        # Append inhibit_sign row
        new_inh = torch.from_numpy(np.asarray(cs.inhibit_sign, dtype=np.float32)
                                    ).unsqueeze(0).to(self.device)
        self.inhibit_tensor = torch.cat([self.inhibit_tensor, new_inh], dim=0)
        # Append zero brain state
        new_state = torch.zeros(1, 16, dtype=torch.float32, device=self.device)
        self.brain_states = torch.cat([self.brain_states, new_state], dim=0)
        self.N = len(self.agents)

    def reset(self):
        """Zero brain states, re-sync params from shells."""
        self.brain_states.zero_()
        for cs in self.core_shells:
            cs.brain_state = np.zeros(cs.N_NODES, dtype=np.float64)
        self.sync_params_from_shells()

    def tick_all(self, sensors_matrix: np.ndarray | Any):
        """One tick for all N agents. Batched core_brain, sequential other shells.

        sensors_matrix: (N, 16) numpy array — current sensor input for each agent.
        """
        from kathara16_brain_gpu import simulate_step_16_batched

        # 1. Write sensors into each agent's S[0:16]
        for i, ag in enumerate(self.agents):
            with ag._lock:
                ag.S[0:16] = np.asarray(sensors_matrix[i], dtype=np.float64)

        # 2. Batched core_brain step on GPU
        inputs_np = np.asarray(sensors_matrix, dtype=np.float32)
        inputs_t = torch.from_numpy(inputs_np).to(self.device)

        new_states, firing = simulate_step_16_batched(
            self.params_tensor, inputs_t, self.brain_states,
            self.inhibit_tensor,
        )
        # Update persistent brain states
        self.brain_states = new_states
        firing_np = firing.detach().cpu().numpy()  # (N, 16)
        new_states_np = new_states.detach().cpu().numpy()

        # 3. For each agent: apply core_brain delta to S, then run OTHER shells
        for i, ag in enumerate(self.agents):
            cs = self.core_shells[i]
            # Update core_brain internal state
            cs.brain_state = new_states_np[i].astype(np.float64)

            # Compute core_brain delta manually (mirror CoreBrain.step logic)
            snapshot = ag.read_state()
            firing_i = firing_np[i]
            delta = np.zeros_like(snapshot)
            delta[cs.motor_nav] = float(firing_i[cs.motor_node_nav]) - snapshot[cs.motor_nav]
            delta[cs.motor_speed] = float(firing_i[cs.motor_node_speed]) - snapshot[cs.motor_speed]
            delta[cs.motor_voice] = float(firing_i[cs.motor_node_voice]) - snapshot[cs.motor_voice]
            f_start = cs.firing_slot.start
            width = cs.firing_slot.stop - f_start
            if width >= cs.N_NODES:
                delta[f_start:f_start + cs.N_NODES] = (
                    firing_i - snapshot[f_start:f_start + cs.N_NODES])

            # Apply core_brain delta to agent's S
            ag.S = ag.S + cs.gain * delta
            ag._last_delta_norms[cs.name] = float(np.linalg.norm(delta))

            # Fallthrough — per-agent other shell runner handled outside.
        # Now run other shells — try BATCHED impl first, fallback to per-agent
        self._run_other_shells_batched()

    def _run_other_shells_batched(self):
        """Batch-process non-core shells across all agents where possible."""
        from tamashii.shell_batched import has_batched_impl, apply_batched

        # Determine ordering of shells (assume all agents share same shell order)
        shell_names = [sh.name for sh in self.agents[0].shells[1:]]
        for offset_idx, shell_name in enumerate(shell_names):
            i_shell = offset_idx + 1  # index in agents[a].shells
            # Take snapshot of S for all agents
            S_batch = np.stack([ag.S for ag in self.agents])  # (N, D)
            shells_of_kind = [ag.shells[i_shell] for ag in self.agents]

            if has_batched_impl(shell_name):
                # Batched numpy version
                delta_batch = apply_batched(shell_name, S_batch, shells_of_kind)
            else:
                # Per-agent fallback for complex shells (hippocampus, prefrontal, dmn)
                delta_batch = np.zeros_like(S_batch)
                for i, ag in enumerate(self.agents):
                    sh = ag.shells[i_shell]
                    delta_batch[i] = sh.step(ag.S.copy(), ag.external)

            # Apply deltas
            for i, ag in enumerate(self.agents):
                d = delta_batch[i]
                ag.S = ag.S + shells_of_kind[i].gain * d
                ag._last_delta_norms[shells_of_kind[i].name] = float(np.linalg.norm(d))
