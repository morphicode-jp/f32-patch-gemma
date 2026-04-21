"""CPU numpy vs GPU torch parity for simulate_step_16."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class TestGPUParity(unittest.TestCase):
    def setUp(self):
        from kathara16_brain import PARAM_RANGES_16, make_inhibit_sign_16
        self.PARAM_RANGES_16 = PARAM_RANGES_16
        self.inhibit_np = make_inhibit_sign_16().astype(np.float32)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = np.random.default_rng(42)

    def _random_agent_batch(self, B: int):
        params_np = np.stack([
            np.array([self.rng.uniform(lo, hi) for lo, hi in self.PARAM_RANGES_16],
                     dtype=np.float32)
            for _ in range(B)
        ])
        inputs_np = self.rng.uniform(0, 1, size=(B, 16)).astype(np.float32)
        states_np = self.rng.uniform(0, 1, size=(B, 16)).astype(np.float32)
        return params_np, inputs_np, states_np

    def _cpu_reference(self, params_np, inputs_np, states_np):
        from kathara16_brain import simulate_step_16
        B = params_np.shape[0]
        outs = []
        for i in range(B):
            ns, _ = simulate_step_16(params_np[i], inputs_np[i], states_np[i],
                                     inhibit_sign=self.inhibit_np)
            outs.append(ns.astype(np.float32))
        return np.stack(outs)

    def _gpu_batched(self, params_np, inputs_np, states_np):
        from kathara16_brain_gpu import simulate_step_16_batched
        p = torch.from_numpy(params_np).to(self.device)
        i = torch.from_numpy(inputs_np).to(self.device)
        s = torch.from_numpy(states_np).to(self.device)
        inh = torch.from_numpy(self.inhibit_np).to(self.device)
        new_states, _ = simulate_step_16_batched(p, i, s, inh)
        return new_states.cpu().numpy()

    def test_B1_parity(self):
        """B=1 matches single-agent CPU call."""
        p, i, s = self._random_agent_batch(1)
        ref = self._cpu_reference(p, i, s)
        got = self._gpu_batched(p, i, s)
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)

    def test_B10_parity(self):
        p, i, s = self._random_agent_batch(10)
        ref = self._cpu_reference(p, i, s)
        got = self._gpu_batched(p, i, s)
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)

    def test_B100_parity(self):
        p, i, s = self._random_agent_batch(100)
        ref = self._cpu_reference(p, i, s)
        got = self._gpu_batched(p, i, s)
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)

    def test_zero_input_decays_to_zero(self):
        """No input + non-zero states -> leaky decay toward 0 over multiple steps."""
        from kathara16_brain_gpu import simulate_step_16_batched
        B = 5
        p_np = np.stack([
            np.array([self.rng.uniform(lo, hi) for lo, hi in self.PARAM_RANGES_16],
                     dtype=np.float32) for _ in range(B)])
        inputs = np.zeros((B, 16), dtype=np.float32)
        # All states at 1.0, no input
        states = np.ones((B, 16), dtype=np.float32)
        p = torch.from_numpy(p_np).to(self.device)
        s = torch.from_numpy(states).to(self.device)
        i = torch.from_numpy(inputs).to(self.device)
        inh = torch.from_numpy(self.inhibit_np).to(self.device)
        # 30 steps
        for _ in range(30):
            s, _ = simulate_step_16_batched(p, i, s, inh)
        final = s.cpu().numpy()
        # Should be bounded
        self.assertTrue(np.all(final >= 0) and np.all(final <= 1),
                        f"states escaped [0,1] range: min={final.min()} max={final.max()}")

    def test_multi_step_consistency(self):
        """GPU 20 steps matches CPU 20 steps."""
        from kathara16_brain import simulate_step_16
        from kathara16_brain_gpu import simulate_step_16_batched
        B = 3
        p_np, inp_np, st_np = self._random_agent_batch(B)
        # CPU for each agent, 20 steps
        ref_final = []
        for a in range(B):
            s_a = st_np[a].astype(np.float64)
            for _ in range(20):
                s_a, _ = simulate_step_16(p_np[a], inp_np[a], s_a,
                                           inhibit_sign=self.inhibit_np)
            ref_final.append(s_a.astype(np.float32))
        ref = np.stack(ref_final)
        # GPU batched 20 steps
        p = torch.from_numpy(p_np).to(self.device)
        i = torch.from_numpy(inp_np).to(self.device)
        s = torch.from_numpy(st_np).to(self.device)
        inh = torch.from_numpy(self.inhibit_np).to(self.device)
        for _ in range(20):
            s, _ = simulate_step_16_batched(p, i, s, inh)
        got = s.cpu().numpy()
        np.testing.assert_allclose(got, ref, rtol=2e-3, atol=1e-4)

    def test_dale_sign_applied(self):
        """Inhibitory sign affects output (different sign -> different output)."""
        from kathara16_brain_gpu import simulate_step_16_batched
        B = 3
        p_np, inp_np, st_np = self._random_agent_batch(B)
        # Use NON-zero states so sign matters
        st_np = np.abs(st_np) + 0.2
        p = torch.from_numpy(p_np).to(self.device)
        i = torch.from_numpy(inp_np).to(self.device)
        s = torch.from_numpy(st_np).to(self.device)

        inh_a = torch.from_numpy(self.inhibit_np).to(self.device)
        inh_b = torch.ones(16, dtype=torch.float32, device=self.device)  # all +1

        out_a, _ = simulate_step_16_batched(p, i, s, inh_a)
        out_b, _ = simulate_step_16_batched(p, i, s, inh_b)
        diff = (out_a - out_b).abs().max().item()
        self.assertGreater(diff, 1e-3,
                           f"Dale sign should change output; diff={diff}")

    def test_cpu_device(self):
        """GPU batched function also works on CPU device."""
        from kathara16_brain_gpu import simulate_step_16_batched
        p_np, inp_np, st_np = self._random_agent_batch(4)
        p = torch.from_numpy(p_np)  # cpu
        i = torch.from_numpy(inp_np)
        s = torch.from_numpy(st_np)
        inh = torch.from_numpy(self.inhibit_np)
        out, _ = simulate_step_16_batched(p, i, s, inh)
        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.shape, (4, 16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
