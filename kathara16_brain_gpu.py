"""kathara16_brain_gpu.py — PyTorch batched Kathara(16, {3,6,7}) brain.

GPU-parallel version of simulate_step_16, enabling N-agent brain steps in
a single CUDA kernel launch.

Usage:
    import torch
    from kathara16_brain_gpu import simulate_step_16_batched, ADJ_MASK_TORCH

    B = 100  # batch of 100 agents
    params = torch.rand(B, 129, device='cuda') * 6 - 3
    inputs = torch.rand(B, 16, device='cuda')
    states = torch.zeros(B, 16, device='cuda')
    inhibit_sign = get_inhibit_sign_tensor(device='cuda')  # (16,)

    new_states, firing = simulate_step_16_batched(
        params, inputs, states, inhibit_sign)
    # both (B, 16)

Guarantees:
  - numerically equivalent to numpy simulate_step_16 at B=1 (within float32 precision)
  - Works on CPU device (fallback)
  - Preserves Dale's law sign convention

PyTorch import is lazy — kathara16_brain_gpu can be imported even if torch
is missing (functions will raise informative error on first call).
"""
from __future__ import annotations

import numpy as np

from kathara16_brain import (
    KATHARA16_EDGES, N_NODES_16, TOTAL_PARAMS_16, PARAMS_PER_NODE,
    _INHIBIT_NODES_16, make_inhibit_sign_16,
)

N_EDGES_16 = len(KATHARA16_EDGES)  # = 48


# Precompute adjacency mask as plain numpy first (serialized, shareable)
# ADJ_EDGE_IDX[i, j] = edge_index if i-j connected, else -1
_ADJ_EDGE_IDX = np.full((N_NODES_16, N_NODES_16), -1, dtype=np.int64)
for eidx, (a, b) in enumerate(KATHARA16_EDGES):
    _ADJ_EDGE_IDX[a, b] = eidx
    _ADJ_EDGE_IDX[b, a] = eidx

_ADJ_MASK_NP = (_ADJ_EDGE_IDX >= 0).astype(np.float32)  # (16, 16)


# Lazy torch / tensor cache
_TORCH = None
_TENSOR_CACHE: dict = {}


def _require_torch():
    global _TORCH
    if _TORCH is None:
        try:
            import torch as _t
        except ImportError as e:
            raise ImportError(
                "kathara16_brain_gpu requires PyTorch. "
                "Install with: pip install torch") from e
        _TORCH = _t
    return _TORCH


def _get_adj_tensors(device: str):
    """Return (edge_idx, adj_mask) torch tensors on given device, cached."""
    torch = _require_torch()
    key = ("adj", device)
    if key not in _TENSOR_CACHE:
        edge_idx = torch.from_numpy(_ADJ_EDGE_IDX).to(device)
        mask = torch.from_numpy(_ADJ_MASK_NP).to(device)
        _TENSOR_CACHE[key] = (edge_idx, mask)
    return _TENSOR_CACHE[key]


def get_inhibit_sign_tensor(device: str = "cuda"):
    """Return the default (16,) Dale's law sign tensor on given device."""
    torch = _require_torch()
    key = ("inhibit", device)
    if key not in _TENSOR_CACHE:
        sign_np = make_inhibit_sign_16().astype(np.float32)
        _TENSOR_CACHE[key] = torch.from_numpy(sign_np).to(device)
    return _TENSOR_CACHE[key]


def simulate_step_16_batched(params, inputs, states, inhibit_sign=None):
    """Batched Kathara(16) step on GPU.

    Args:
      params: (B, 129) tensor of parameters
      inputs: (B, 16) tensor of sensor inputs
      states: (B, 16) tensor of previous states
      inhibit_sign: (16,) or (B, 16) Dale's law sign (+1 exc, -1 inh).
                    If None, uses default with nodes [0, 4, 8] inhibitory.

    Returns:
      (new_states, firing): both (B, 16) tensors.

    Preserves simulate_step_16's equations exactly:
      syn_input[b, i] = sum_j states_effective[b, j] * edge_w[b, adj_edge_idx[i,j]]
      total = syn_input + input_w * inputs + bias
      activation = sigmoid(clip(gain * total, -10, 10))
      activation *= 0.1 if activation < threshold else 1.0
      new_state = states * (1 - leak) + activation * leak
      clip(new_state, 0, 1)
    """
    torch = _require_torch()
    assert params.dim() == 2 and params.shape[1] == TOTAL_PARAMS_16, (
        f"params shape {params.shape} != (B, {TOTAL_PARAMS_16})")
    assert inputs.shape == states.shape == (params.shape[0], N_NODES_16)
    B = params.shape[0]
    device = params.device

    # Default inhibit_sign
    if inhibit_sign is None:
        inhibit_sign = get_inhibit_sign_tensor(device=str(device))

    # Ensure inhibit_sign is on correct device
    if inhibit_sign.device != device:
        inhibit_sign = inhibit_sign.to(device)

    # Slice params
    edge_w = params[:, :N_EDGES_16]  # (B, 48)
    node_params = params[:, N_EDGES_16:N_EDGES_16 + N_NODES_16 * PARAMS_PER_NODE].view(
        B, N_NODES_16, PARAMS_PER_NODE)
    gains = node_params[:, :, 0]    # (B, 16)
    biases = node_params[:, :, 1]
    input_w = node_params[:, :, 2]
    leaks = node_params[:, :, 3]
    thres = node_params[:, :, 4]

    # Dale's law
    if inhibit_sign.dim() == 1:
        eff_states = states * inhibit_sign.unsqueeze(0)  # broadcast (1,16) -> (B,16)
    else:
        eff_states = states * inhibit_sign

    # Adjacency tensors (cached per-device)
    edge_idx_tensor, adj_mask = _get_adj_tensors(str(device))

    # Build (B, 16, 16) pair weight tensor via gather.
    # edge_idx_tensor: (16, 16), with -1 for non-connected.
    # For each (i, j), pair_w[b, i, j] = edge_w[b, edge_idx[i,j]] if connected else 0
    # Use clamp to avoid -1 index: pair_w = gather(edge_w, edge_idx.clamp(min=0)) * adj_mask
    safe_idx = edge_idx_tensor.clamp(min=0)  # (16, 16), replace -1 with 0
    # expand edge_w to (B, 16*16) then gather using safe_idx flattened
    safe_idx_flat = safe_idx.reshape(-1).unsqueeze(0).expand(B, -1)  # (B, 256)
    pair_w_flat = torch.gather(edge_w, dim=1, index=safe_idx_flat)   # (B, 256)
    pair_w = pair_w_flat.view(B, N_NODES_16, N_NODES_16)             # (B, 16, 16)
    pair_w = pair_w * adj_mask.unsqueeze(0)                          # zero out invalid

    # Synaptic aggregation: syn[b, i] = sum_j eff_states[b, j] * pair_w[b, i, j]
    # eff_states shape (B, 16), broadcast to (B, 1, 16) and multiply (B, 16, 16) then sum dim 2
    syn_input = (eff_states.unsqueeze(1) * pair_w).sum(dim=2)  # (B, 16)

    # Total + activation
    total = syn_input + input_w * inputs + biases  # (B, 16)
    x = gains * total
    x_clamped = torch.clamp(x, -10.0, 10.0)
    activation = torch.sigmoid(x_clamped)
    # Threshold suppression: where activation < thres, multiply by 0.1
    activation = torch.where(activation < thres, activation * 0.1, activation)

    # Leaky integration
    new_states = states * (1.0 - leaks) + activation * leaks
    new_states = torch.clamp(new_states, 0.0, 1.0)

    # firing = new_states.clone() (matches numpy API which returns copy)
    return new_states, new_states.clone()


# Convenience: convert numpy params/inputs/states to batched torch tensor
def to_batched_tensors(
    params_list, inputs_list, states_list, device: str = "cuda",
    dtype=None,
):
    """Convert lists of numpy arrays to stacked torch tensors on device.

    params_list: list of (129,) arrays
    inputs_list: list of (16,) arrays
    states_list: list of (16,) arrays
    """
    torch = _require_torch()
    if dtype is None:
        dtype = torch.float32
    p = torch.from_numpy(np.stack(params_list).astype(np.float32)).to(device=device, dtype=dtype)
    i = torch.from_numpy(np.stack(inputs_list).astype(np.float32)).to(device=device, dtype=dtype)
    s = torch.from_numpy(np.stack(states_list).astype(np.float32)).to(device=device, dtype=dtype)
    return p, i, s


# Quick self-check
def _self_check():
    """Compare batched GPU vs single-agent numpy, expect agreement at float32."""
    torch = _require_torch()
    from kathara16_brain import simulate_step_16, PARAM_RANGES_16

    rng = np.random.default_rng(42)
    # Build 5 random agents
    agents_params = []
    for _ in range(5):
        p = np.array([rng.uniform(lo, hi) for lo, hi in PARAM_RANGES_16],
                     dtype=np.float32)
        agents_params.append(p)
    inputs = rng.uniform(0, 1, size=(5, 16)).astype(np.float32)
    states = rng.uniform(0, 1, size=(5, 16)).astype(np.float32)
    inhibit = make_inhibit_sign_16().astype(np.float32)

    # Numpy per-agent reference
    ref_new_states = []
    for i in range(5):
        ns, _ = simulate_step_16(agents_params[i], inputs[i], states[i],
                                  inhibit_sign=inhibit)
        ref_new_states.append(ns.astype(np.float32))
    ref = np.stack(ref_new_states)

    # GPU batched
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p_t = torch.from_numpy(np.stack(agents_params)).to(device)
    i_t = torch.from_numpy(inputs).to(device)
    s_t = torch.from_numpy(states).to(device)
    inhibit_t = torch.from_numpy(inhibit).to(device)

    new_states, firing = simulate_step_16_batched(p_t, i_t, s_t, inhibit_t)
    got = new_states.cpu().numpy()

    diff = np.abs(ref - got)
    print(f"Self-check: device={device}, max |diff| = {diff.max():.2e}")
    print(f"  rtol=1e-4 tolerance: {'PASS' if diff.max() < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    _self_check()
