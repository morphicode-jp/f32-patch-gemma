"""Offline shell-param training via mimir().

For each shell (except brainstem which is structural):
  1. Define the shell's tunable params and ranges
  2. Define a shell-specific fitness eval_fn that runs a mini Tamashii episode
     and returns a score matching the shell's biological role
  3. Call mimir() — auto-dispatches owl / Reigen based on eval cost
  4. Save best params to tamashii/configs/<shell>_trained.json

Usage:
  python -m tamashii.train_shells --shell core_brain --budget 600
  python -m tamashii.train_shells --all --budget 120
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tamashii.core import Tamashii
from tamashii.shell_base import load_shell_from_config
from tamashii.shells.brainstem import Brainstem
from tamashii.shells.cerebellum import Cerebellum
from tamashii.shells.core_brain import CoreBrain
from tamashii.shells.dmn import DMN
from tamashii.shells.hippocampus import Hippocampus
from tamashii.shells.prefrontal import Prefrontal
from tamashii.shells.salience import Salience


CONFIGS = os.path.join(THIS_DIR, "configs")


def _load_base_config(shell_name: str) -> dict:
    with open(os.path.join(CONFIGS, f"{shell_name}.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _build_tamashii_with_override(
    shell_classes: list[tuple[type, str]],  # [(cls, shell_name), ...]
    override_shell: str,
    override_params: dict,
) -> Tamashii:
    """Build a Tamashii with one shell's params overridden (others default)."""
    shells = []
    for cls, name in shell_classes:
        cfg = _load_base_config(name)
        if name == override_shell:
            cfg["params"].update(override_params)
        shells.append(cls(cfg))
    return Tamashii(shells=shells, D=192)


def _run_mini_episode(agent: Tamashii, world_cls, seed: int,
                     n_steps: int = 30, ticks_per_step: int = 2) -> dict:
    """One agent (team of N) run. Returns motor, S traj, reach, delta norms."""
    N_AGENTS = 1
    world = world_cls(n_agents=N_AGENTS, seed=seed)
    sensors_list = world.reset()
    initial_dist = world.get_food_dist(0)
    min_dist = initial_dist
    motor_series = []
    S_traj = []
    delta_log = []

    for step in range(n_steps):
        # Inject sensor
        with agent._lock:
            agent.S[0:16] = np.asarray(sensors_list[0], dtype=np.float64)
        # Tick shells
        for _ in range(ticks_per_step):
            agent.tick_once()
            delta_log.append(agent.shell_delta_norms())
            S_traj.append(agent.read_state())
        # Extract action
        S = agent.read_state()
        nav = float(np.clip(S[16], 0.0, 1.0))
        speed = float(np.clip(S[17], 0.0, 1.0))
        voice = float(np.clip(S[18], 0.0, 1.0))
        motor_series.append((nav, speed, voice))
        # Step world
        sensors_list, all_reached, reached_flags = world.step(
            [(nav, speed, voice)])
        d = world.get_food_dist(0)
        if d < min_dist:
            min_dist = d
        if all_reached:
            break

    reached = bool(world.reached[0])
    approach = max(0.0, initial_dist - min_dist) / (initial_dist + 1e-6)
    return {
        "motor": np.array(motor_series),
        "S_traj": np.array(S_traj),
        "reached": reached,
        "approach": approach,
        "delta_log": delta_log,
    }


def _compute_autocorr(motor_series: np.ndarray) -> float:
    """Lag-1 autocorrelation averaged over motor dims. 0 = random, 1 = perfect."""
    if motor_series.shape[0] < 3:
        return 0.0
    vals = []
    for d in range(motor_series.shape[1]):
        x = motor_series[:, d]
        m, v = x.mean(), x.var()
        if v < 1e-10:
            continue
        ac = float(np.mean((x[:-1] - m) * (x[1:] - m)) / v)
        vals.append(ac)
    return float(np.mean(vals)) if vals else 0.0


def _coupling_from_log(delta_log: list[dict]) -> float:
    if not delta_log:
        return 0.0
    names = sorted(delta_log[0].keys())
    series = np.array([[d.get(n, 0.0) for d in delta_log] for n in names])
    n = len(names)
    if n < 2 or series.shape[1] < 2:
        return 0.0
    stds = series.std(axis=1)
    if (stds < 1e-10).any():
        return 0.0
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                corr[i, j] = float(np.corrcoef(series[i], series[j])[0, 1])
    return float(np.abs(corr[~np.eye(n, dtype=bool)]).mean())


def _novelty_response(S_traj: np.ndarray, sensor_input: np.ndarray) -> float:
    if len(S_traj) < 3 or len(sensor_input) < 3:
        return 0.0
    s_var = float(np.var(np.diff(S_traj, axis=0).flatten()))
    in_var = float(np.var(np.diff(sensor_input, axis=0).flatten()))
    if in_var < 1e-10:
        return 0.0
    return s_var / in_var


# ============================================================
# Per-shell trainers
# ============================================================

ALL_SHELL_CLASSES = [
    (CoreBrain, "core_brain"),
    (Brainstem, "brainstem"),
    (Cerebellum, "cerebellum"),
    (Salience, "salience"),
    (Hippocampus, "hippocampus"),
    (Prefrontal, "prefrontal"),
    (DMN, "dmn"),
]


def _save_trained(shell_name: str, params: dict, meta: dict):
    path = os.path.join(CONFIGS, f"{shell_name}_trained.json")
    out = {"params": params, "_meta": meta}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved trained params: {path}", flush=True)


# ---------------------------------------------------------------
# core_brain: 129D Kathara params
# ---------------------------------------------------------------
def train_core_brain(budget: float):
    from kathara16_brain import PARAM_RANGES_16, TOTAL_PARAMS_16
    from twelve.agent.mimir import mimir
    from multi_agent_world_N import MultiAgentCoopWorldN

    param_ranges = list(PARAM_RANGES_16)

    def fitness(params_list: list[float]) -> float:
        # Build 1-agent Tamashii with core + brainstem + cerebellum
        agent = _build_tamashii_with_override(
            [(CoreBrain, "core_brain"), (Brainstem, "brainstem"),
             (Cerebellum, "cerebellum")],
            "core_brain",
            {"kathara_params": list(params_list)},
        )
        # Run 3 episodes
        scores = []
        for ep in range(3):
            ep_result = _run_mini_episode(agent, MultiAgentCoopWorldN,
                                          seed=ep * 7 + 3, n_steps=30)
            autocorr = _compute_autocorr(ep_result["motor"])
            approach = ep_result["approach"]
            reach = 1.0 if ep_result["reached"] else 0.0
            score = autocorr * 50 + approach * 30 + reach * 20
            scores.append(score)
        return float(np.mean(scores))

    print(f"\n[train core_brain] 129D, budget={budget}s", flush=True)
    result = mimir(
        eval_fn=fitness,
        param_ranges=param_ranges,
        experience_id="tamashii_core_brain",
        time_budget=budget,
        verbose=True,
    )
    best_params = list(result["best_params"])
    meta = {
        "best_score": float(result["best_score"]),
        "tool_used": result.get("tool_used"),
        "eval_cost_s": result.get("eval_cost_s"),
        "proxy_r2": result.get("proxy_r2"),
        "dead_dims": result.get("dead_dims", [])[:10],
    }
    _save_trained("core_brain", {"kathara_params": best_params}, meta)
    return meta


# ---------------------------------------------------------------
# cerebellum: 3D (ema_alpha, smoothing_strength, prediction_weight)
# ---------------------------------------------------------------
def train_cerebellum(budget: float):
    from twelve.agent.mimir import mimir
    from multi_agent_world_N import MultiAgentCoopWorldN

    ranges = [(0.2, 0.95), (0.05, 0.5), (0.2, 0.8)]

    def fitness(p):
        agent = _build_tamashii_with_override(
            [(CoreBrain, "core_brain"), (Brainstem, "brainstem"),
             (Cerebellum, "cerebellum")],
            "cerebellum",
            {"ema_alpha": float(p[0]), "smoothing_strength": float(p[1]),
             "prediction_weight": float(p[2])},
        )
        scores = []
        for ep in range(2):
            ep_result = _run_mini_episode(
                agent, MultiAgentCoopWorldN, seed=ep * 11 + 5, n_steps=30)
            autocorr = _compute_autocorr(ep_result["motor"])
            motor_std = float(ep_result["motor"].std())
            # Reward smoothness (autocorr high AND motor not frozen)
            score = autocorr * 50 + min(motor_std, 0.3) * 30
            scores.append(score)
        return float(np.mean(scores))

    print(f"\n[train cerebellum] 3D, budget={budget}s", flush=True)
    result = mimir(eval_fn=fitness, param_ranges=ranges,
                   experience_id="tamashii_cerebellum",
                   time_budget=budget, verbose=True)
    bp = list(result["best_params"])
    _save_trained("cerebellum",
                  {"ema_alpha": bp[0], "smoothing_strength": bp[1],
                   "prediction_weight": bp[2]},
                  {"best_score": float(result["best_score"]),
                   "tool_used": result.get("tool_used")})
    return {"best_score": float(result["best_score"])}


# ---------------------------------------------------------------
# salience: 3D (baseline_alpha, novelty_threshold, boost_strength)
# ---------------------------------------------------------------
def train_salience(budget: float):
    from twelve.agent.mimir import mimir
    from multi_agent_world_N import MultiAgentCoopWorldN

    ranges = [(0.01, 0.3), (0.05, 0.4), (0.03, 0.3)]

    def fitness(p):
        agent = _build_tamashii_with_override(
            [(CoreBrain, "core_brain"), (Brainstem, "brainstem"),
             (Salience, "salience")],
            "salience",
            {"baseline_alpha": float(p[0]), "novelty_threshold": float(p[1]),
             "boost_strength": float(p[2])},
        )
        scores = []
        for ep in range(2):
            ep_result = _run_mini_episode(
                agent, MultiAgentCoopWorldN, seed=ep * 13 + 1, n_steps=30)
            # Attention entropy (more selective = better)
            S = ep_result["S_traj"]
            att = S[:, 51:83]
            att_norm = att / (att.sum(axis=1, keepdims=True) + 1e-6)
            entropy = -float(np.mean(
                np.sum(att_norm * np.log(att_norm + 1e-10), axis=1)))
            # But also want response (not stuck)
            nov = _novelty_response(S, S[:, 0:16])
            score = entropy * 10 + min(nov, 5.0) * 5
            scores.append(score)
        return float(np.mean(scores))

    print(f"\n[train salience] 3D, budget={budget}s", flush=True)
    result = mimir(eval_fn=fitness, param_ranges=ranges,
                   experience_id="tamashii_salience",
                   time_budget=budget, verbose=True)
    bp = list(result["best_params"])
    _save_trained("salience",
                  {"baseline_alpha": bp[0], "novelty_threshold": bp[1],
                   "boost_strength": bp[2]},
                  {"best_score": float(result["best_score"])})
    return {"best_score": float(result["best_score"])}


# ---------------------------------------------------------------
# hippocampus: 3D (recall_threshold, store_threshold, recall_strength)
# ---------------------------------------------------------------
def train_hippocampus(budget: float):
    from twelve.agent.mimir import mimir
    from multi_agent_world_N import MultiAgentCoopWorldN

    ranges = [(0.5, 0.95), (0.3, 0.85), (0.05, 0.5)]

    def fitness(p):
        recall_thr, store_thr, recall_str = p
        # Ensure store < recall (otherwise never stores)
        store_thr = min(store_thr, recall_thr - 0.05)
        agent = _build_tamashii_with_override(
            [(CoreBrain, "core_brain"), (Brainstem, "brainstem"),
             (Hippocampus, "hippocampus")],
            "hippocampus",
            {"recall_threshold": float(recall_thr),
             "store_threshold": float(store_thr),
             "recall_strength": float(recall_str)},
        )
        scores = []
        for ep in range(2):
            ep_result = _run_mini_episode(
                agent, MultiAgentCoopWorldN, seed=ep * 17 + 9, n_steps=40)
            hp = agent.shells[2]  # hippocampus
            stats = hp.memory_stats()
            n_stored = stats["n_stored"]
            # Reward utilizing memory (n_stored > 2, not filling all slots)
            util_score = min(n_stored, 6) * 10
            # Penalize if zero (not learning)
            if n_stored == 0:
                util_score = -20
            # Plus diversity of S trajectory
            S_diff_std = float(np.std(np.diff(ep_result["S_traj"], axis=0)))
            score = util_score + min(S_diff_std, 1.0) * 20
            scores.append(score)
        return float(np.mean(scores))

    print(f"\n[train hippocampus] 3D, budget={budget}s", flush=True)
    result = mimir(eval_fn=fitness, param_ranges=ranges,
                   experience_id="tamashii_hippocampus",
                   time_budget=budget, verbose=True)
    bp = list(result["best_params"])
    _save_trained("hippocampus",
                  {"recall_threshold": bp[0],
                   "store_threshold": min(bp[1], bp[0] - 0.05),
                   "recall_strength": bp[2]},
                  {"best_score": float(result["best_score"])})
    return {"best_score": float(result["best_score"])}


# ---------------------------------------------------------------
# prefrontal: 3D (decay_rate, lock_threshold, bias_strength)
# ---------------------------------------------------------------
def train_prefrontal(budget: float):
    from twelve.agent.mimir import mimir
    from multi_agent_world_N import MultiAgentCoopWorldN

    ranges = [(0.01, 0.2), (0.2, 0.8), (0.02, 0.2)]

    def fitness(p):
        agent = _build_tamashii_with_override(
            [(CoreBrain, "core_brain"), (Brainstem, "brainstem"),
             (Salience, "salience"), (Prefrontal, "prefrontal")],
            "prefrontal",
            {"decay_rate": float(p[0]), "lock_threshold": float(p[1]),
             "bias_strength": float(p[2])},
        )
        scores = []
        for ep in range(2):
            ep_result = _run_mini_episode(
                agent, MultiAgentCoopWorldN, seed=ep * 19 + 3, n_steps=40)
            pf = agent.shells[-1]  # prefrontal is last
            stats = pf.slot_stats()
            autocorr = _compute_autocorr(ep_result["motor"])
            # Reward: motor autocorr improvement + goal activity
            score = autocorr * 40 + stats["n_active_slots"] * 5 + stats["mean_activation"] * 20
            scores.append(score)
        return float(np.mean(scores))

    print(f"\n[train prefrontal] 3D, budget={budget}s", flush=True)
    result = mimir(eval_fn=fitness, param_ranges=ranges,
                   experience_id="tamashii_prefrontal",
                   time_budget=budget, verbose=True)
    bp = list(result["best_params"])
    _save_trained("prefrontal",
                  {"decay_rate": bp[0], "lock_threshold": bp[1],
                   "bias_strength": bp[2]},
                  {"best_score": float(result["best_score"])})
    return {"best_score": float(result["best_score"])}


# ---------------------------------------------------------------
# dmn: 3D (engagement_threshold, replay_strength, decay)
# ---------------------------------------------------------------
def train_dmn(budget: float):
    from twelve.agent.mimir import mimir
    from multi_agent_world_N import MultiAgentCoopWorldN

    ranges = [(0.1, 0.8), (0.05, 0.4), (0.5, 0.98)]

    def fitness(p):
        agent = _build_tamashii_with_override(
            [(CoreBrain, "core_brain"), (Brainstem, "brainstem"),
             (Hippocampus, "hippocampus"), (DMN, "dmn")],
            "dmn",
            {"engagement_threshold": float(p[0]),
             "replay_strength": float(p[1]), "decay": float(p[2])},
        )
        scores = []
        for ep in range(2):
            ep_result = _run_mini_episode(
                agent, MultiAgentCoopWorldN, seed=ep * 23 + 7, n_steps=40)
            dmn = agent.shells[-1]
            engagement = dmn.engagement()
            # DMN should be active when engagement is low, passive when high
            # Reward anti-correlation: low engagement → higher S scratchpad activity
            S = ep_result["S_traj"]
            scratch_act = float(np.mean(np.abs(S[:, 176:192])))
            # Good if scratch activity exists (DMN working)
            score = scratch_act * 30 + (1.0 - min(engagement, 1.0)) * 10
            scores.append(score)
        return float(np.mean(scores))

    print(f"\n[train dmn] 3D, budget={budget}s", flush=True)
    result = mimir(eval_fn=fitness, param_ranges=ranges,
                   experience_id="tamashii_dmn",
                   time_budget=budget, verbose=True)
    bp = list(result["best_params"])
    _save_trained("dmn",
                  {"engagement_threshold": bp[0], "replay_strength": bp[1],
                   "decay": bp[2]},
                  {"best_score": float(result["best_score"])})
    return {"best_score": float(result["best_score"])}


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
TRAINERS = {
    "core_brain": train_core_brain,
    "cerebellum": train_cerebellum,
    "salience": train_salience,
    "hippocampus": train_hippocampus,
    "prefrontal": train_prefrontal,
    "dmn": train_dmn,
}

# Reasonable budgets per shell
DEFAULT_BUDGETS = {
    "core_brain": 600,  # 129D, needs lots of time
    "cerebellum": 60,
    "salience": 60,
    "hippocampus": 60,
    "prefrontal": 60,
    "dmn": 60,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", type=str, default=None,
                    choices=list(TRAINERS.keys()),
                    help="train single shell (default: all)")
    ap.add_argument("--all", action="store_true", help="train all shells")
    ap.add_argument("--budget", type=float, default=None,
                    help="override budget per shell (s); default = per-shell")
    ap.add_argument("--skip_core_brain", action="store_true",
                    help="skip 129D core_brain (use existing trained)")
    args = ap.parse_args()

    to_train = []
    if args.shell:
        to_train = [args.shell]
    elif args.all or args.shell is None:
        to_train = list(TRAINERS.keys())
    if args.skip_core_brain:
        to_train = [s for s in to_train if s != "core_brain"]

    print("=" * 70, flush=True)
    print(f"  TAMASHII offline shell training via mimir()", flush=True)
    print(f"  shells to train: {to_train}", flush=True)
    print("=" * 70, flush=True)

    summary = {}
    t_all = time.time()
    for shell_name in to_train:
        budget = args.budget if args.budget is not None else DEFAULT_BUDGETS[shell_name]
        t0 = time.time()
        try:
            result = TRAINERS[shell_name](budget)
            elapsed = time.time() - t0
            summary[shell_name] = {**result, "elapsed_s": round(elapsed, 1)}
            print(f"  [{shell_name}] done in {elapsed:.0f}s "
                  f"score={result.get('best_score', 0):.2f}", flush=True)
        except Exception as e:
            import traceback
            print(f"  [{shell_name}] FAILED: {e}", flush=True)
            traceback.print_exc()
            summary[shell_name] = {"error": str(e)[:200]}

    total = time.time() - t_all
    print(f"\n{'='*70}", flush=True)
    print(f"  TRAINING COMPLETE ({total:.0f}s = {total/60:.1f}min)", flush=True)
    print(f"{'='*70}", flush=True)
    for shell, info in summary.items():
        if "error" in info:
            print(f"  {shell:12s}: FAILED ({info['error'][:60]})", flush=True)
        else:
            print(f"  {shell:12s}: score={info.get('best_score', 0):.2f}, "
                  f"{info.get('elapsed_s', 0):.0f}s", flush=True)

    with open(os.path.join(REPO_ROOT, "tamashii_training_summary.json"), "w") as f:
        json.dump({"total_elapsed_s": round(total, 1), "summary": summary}, f,
                  indent=2, default=str)


if __name__ == "__main__":
    main()
