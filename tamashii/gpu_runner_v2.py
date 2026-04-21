"""gpu_runner_v2.py — optimized batched multi-agent runner.

Key optimizations vs v1 (gpu_runner.py):

1. **Single S_batch materialization per tick** (not per-shell)
   v1: np.stack inside each shell iteration → 7 stacks per tick
   v2: stack once, mutate in-place, unstack once

2. **Vectorized core_brain delta computation**
   v1: Python for loop over N agents with scalar ops
   v2: single (N, 192) delta matrix built with vector ops

3. **In-place S update via slice assignment**
   v1: `ag.S = ag.S + delta` reassigns → creates new array
   v2: S_batch operates in-place; final `ag.S = S_batch[i]` is a view assignment

Expected: 3-5x speedup on top of existing v1 GPU batching.
Combined with v1's 2x, target is 6-10x end-to-end.
"""
from __future__ import annotations

import os
import sys

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


class GPUBatchRunnerV2:
    """Optimized batched multi-agent runner.

    Maintains a master (N, D) numpy matrix; shells operate on it vectorized.
    Agent.S attributes are views into this matrix (no copies).
    """

    def __init__(self, agents: list[Tamashii], device: str = "cuda"):
        if not HAVE_TORCH:
            raise ImportError("PyTorch required")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.agents = agents
        self.N = len(agents)
        self.D = agents[0].D if self.N > 0 else 192

        # Extract core_brain shells
        self.core_shells: list[CoreBrain] = []
        for ag in agents:
            assert len(ag.shells) > 0 and isinstance(ag.shells[0], CoreBrain)
            self.core_shells.append(ag.shells[0])

        # Master (N, D) state matrix — this is the ONLY copy of S
        # Initialize from agents' current S
        S_init = np.stack([ag.S for ag in agents]).astype(np.float64)
        self.S_batch = S_init  # shape (N, D)

        # Replace each agent's S attribute with a VIEW into S_batch
        # (so agent.S reads reflect batch state; writes will go through matrix)
        for i, ag in enumerate(agents):
            ag.S = self.S_batch[i]

        # GPU tensors for core_brain params
        params_list = [np.asarray(cs.kathara_params, dtype=np.float32)
                       for cs in self.core_shells]
        self.params_tensor = torch.from_numpy(np.stack(params_list)).to(device)
        inh_list = [np.asarray(cs.inhibit_sign, dtype=np.float32)
                    for cs in self.core_shells]
        self.inhibit_tensor = torch.from_numpy(np.stack(inh_list)).to(device)
        self.brain_states = torch.zeros(self.N, 16, dtype=torch.float32,
                                         device=device)

        # Cache the shared motor indices (assume all core_shells same config)
        cs0 = self.core_shells[0]
        self._motor_nav = cs0.motor_nav
        self._motor_speed = cs0.motor_speed
        self._motor_voice = cs0.motor_voice
        self._motor_node_nav = cs0.motor_node_nav
        self._motor_node_speed = cs0.motor_node_speed
        self._motor_node_voice = cs0.motor_node_voice
        self._firing_start = cs0.firing_slot.start
        self._firing_end = cs0.firing_slot.stop
        self._firing_width = self._firing_end - self._firing_start
        self._N_NODES = cs0.N_NODES
        self._core_gain = cs0.gain

    def add_agent(self, agent: Tamashii):
        """Add new agent (e.g. child from reproduction)."""
        cs = agent.shells[0]
        assert isinstance(cs, CoreBrain)
        self.agents.append(agent)
        self.core_shells.append(cs)
        # Expand S_batch
        new_row = agent.S.reshape(1, -1).copy() if agent.S.ndim == 1 else agent.S.copy()
        new_row = new_row.astype(np.float64)
        self.S_batch = np.concatenate([self.S_batch, new_row], axis=0)
        # Rebind all agents' .S to point at new matrix rows
        for j, ag in enumerate(self.agents):
            ag.S = self.S_batch[j]
        # Expand GPU tensors
        new_p = torch.from_numpy(np.asarray(cs.kathara_params, dtype=np.float32)
                                  ).unsqueeze(0).to(self.device)
        self.params_tensor = torch.cat([self.params_tensor, new_p], dim=0)
        new_inh = torch.from_numpy(np.asarray(cs.inhibit_sign, dtype=np.float32)
                                    ).unsqueeze(0).to(self.device)
        self.inhibit_tensor = torch.cat([self.inhibit_tensor, new_inh], dim=0)
        new_state = torch.zeros(1, 16, dtype=torch.float32, device=self.device)
        self.brain_states = torch.cat([self.brain_states, new_state], dim=0)
        self.N = len(self.agents)

    def reset(self):
        self.brain_states.zero_()
        for cs in self.core_shells:
            cs.brain_state = np.zeros(cs.N_NODES, dtype=np.float64)
        # Re-sync params
        params_list = [np.asarray(cs.kathara_params, dtype=np.float32)
                       for cs in self.core_shells]
        self.params_tensor = torch.from_numpy(np.stack(params_list)).to(self.device)

    def sync_params_from_shells(self):
        """Rebuild params_tensor if shells' params changed (Hebbian)."""
        params_list = [np.asarray(cs.kathara_params, dtype=np.float32)
                       for cs in self.core_shells]
        self.params_tensor = torch.from_numpy(np.stack(params_list)).to(self.device)

    def tick_all(self, sensors_matrix: np.ndarray):
        """One tick for all N agents — OPTIMIZED batched path.

        sensors_matrix: (N, 16) — current sensor inputs.
        """
        from kathara16_brain_gpu import simulate_step_16_batched
        from tamashii.shell_batched import has_batched_impl, apply_batched

        # 1. Write sensors into S_batch[:, 0:16] (in-place)
        sensors = np.asarray(sensors_matrix, dtype=np.float64)
        self.S_batch[:, 0:16] = sensors

        # 2. GPU batched core_brain
        inputs_t = torch.from_numpy(sensors.astype(np.float32)).to(self.device)
        new_states, firing = simulate_step_16_batched(
            self.params_tensor, inputs_t, self.brain_states, self.inhibit_tensor)
        self.brain_states = new_states
        firing_np = firing.detach().cpu().numpy().astype(np.float64)  # (N, 16)
        new_states_np = new_states.detach().cpu().numpy().astype(np.float64)

        # 3. VECTORIZED core_brain delta (all agents at once)
        snapshots = self.S_batch  # already a view
        # Motor channel writes: set target = firing at node, write diff
        nav_target   = firing_np[:, self._motor_node_nav]
        speed_target = firing_np[:, self._motor_node_speed]
        voice_target = firing_np[:, self._motor_node_voice]
        # Write motors directly (additive delta absorbed into target assignment)
        self.S_batch[:, self._motor_nav]   += self._core_gain * (nav_target   - snapshots[:, self._motor_nav])
        self.S_batch[:, self._motor_speed] += self._core_gain * (speed_target - snapshots[:, self._motor_speed])
        self.S_batch[:, self._motor_voice] += self._core_gain * (voice_target - snapshots[:, self._motor_voice])

        # Firing slot: S[firing_start:firing_end] ← firing (full N_NODES)
        if self._firing_width >= self._N_NODES:
            self.S_batch[:, self._firing_start:self._firing_start + self._N_NODES] += \
                self._core_gain * (firing_np - snapshots[:, self._firing_start:self._firing_start + self._N_NODES])

        # Propagate brain_state to each shell (per-agent, unavoidable)
        for i, cs in enumerate(self.core_shells):
            cs.brain_state = new_states_np[i]

        # 4. Run other shells — reuse S_batch throughout
        shell_names = [sh.name for sh in self.agents[0].shells[1:]]
        for offset_idx, shell_name in enumerate(shell_names):
            i_shell = offset_idx + 1
            shells_of_kind = [ag.shells[i_shell] for ag in self.agents]

            if has_batched_impl(shell_name):
                # batched numpy impl (reads S_batch, returns (N, D) delta)
                delta_batch = apply_batched(shell_name, self.S_batch, shells_of_kind)
            else:
                # per-agent fallback
                delta_batch = np.zeros_like(self.S_batch)
                for i, ag in enumerate(self.agents):
                    sh = ag.shells[i_shell]
                    delta_batch[i] = sh.step(self.S_batch[i].copy(), ag.external)

            # Vectorized in-place update (all agents at once)
            gain = shells_of_kind[0].gain  # assume same gain
            self.S_batch += gain * delta_batch

            # Update delta norm per agent (diagnostic, O(N) but trivial)
            norms = np.linalg.norm(delta_batch, axis=1)
            for i, ag in enumerate(self.agents):
                ag._last_delta_norms[shells_of_kind[i].name] = float(norms[i])
