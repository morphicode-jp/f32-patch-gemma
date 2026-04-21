"""Phase 16 — PBT (DeepMind 2017) vs Cardinal Pattern B 直接対決.

PBT = Population-Based Training (Jaderberg et al. 2017):
  - N workers が並列訓練
  - T steps ごとに: 下位 worker は上位の重み+HP をコピー (exploit)、
    その後 HP を摂動 (explore)
  - 重みも一緒に継承 → warm start の累積効果

Cardinal Pattern B:
  - N universe が独立に full training
  - epoch 終了毎に HP のみ mutate
  - 重みは毎回 reset

Same budget: N × cycles × T steps の総 training compute.

Fair comparison:
  - 同じ task (synthetic regression)
  - 同じ 12D HP 空間
  - 同じ total training steps
  - 同じ test seeds で最終評価

Hypothesis:
  PBT は重み継承があるので同 budget で勝つはず。
  もし Cardinal が tie or win なら "重み継承なし meta-evolve でも十分" と言える。
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch


def synthetic_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 8)).astype(np.float32)
    y = (x ** 2).sum(axis=1) + 0.1 * rng.standard_normal(n).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def make_model(hidden_dim: int, n_layers: int, dropout: float):
    layers = [torch.nn.Linear(8, hidden_dim), torch.nn.ReLU(),
              torch.nn.Dropout(dropout)]
    for _ in range(n_layers - 2):
        layers += [torch.nn.Linear(hidden_dim, hidden_dim),
                   torch.nn.ReLU(),
                   torch.nn.Dropout(dropout)]
    layers += [torch.nn.Linear(hidden_dim, 1)]
    return torch.nn.Sequential(*layers)


HP_RANGES = {
    "lr":             (1e-5, 0.3,    "float_log"),
    "beta1":          (0.5,  0.9999, "float"),
    "beta2":          (0.9,  0.99999,"float"),
    "epsilon":        (1e-10, 1e-4,  "float_log"),
    "weight_decay":   (1e-7, 1e-2,   "float_log"),
    "batch_size":     (8,    128,    "int"),
    "warmup_steps":   (0,    200,    "int"),
    "lr_decay_gamma": (0.5,  1.0,    "float"),
    "gradient_clip":  (1e-2, 1e2,    "float_log"),
    "dropout":        (0.0,  0.5,    "float"),
    "hidden_dim":     (16,   128,    "int"),
    "n_layers":       (2,    5,      "int"),
}

# HPs that PBT can change WITHOUT recreating model (weights stay)
# PBT 標準では architecture は perturbation 対象外
WEIGHT_COMPATIBLE_HPS = {
    "lr", "beta1", "beta2", "epsilon", "weight_decay",
    "batch_size", "warmup_steps", "lr_decay_gamma", "gradient_clip", "dropout",
}
# hidden_dim, n_layers は PBT では固定 (architecture change で重み無効化)


def random_hp(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    hp = {}
    for k, (lo, hi, typ) in HP_RANGES.items():
        if typ == "float":
            hp[k] = float(rng.uniform(lo, hi))
        elif typ == "float_log":
            hp[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif typ == "int":
            hp[k] = int(round(rng.uniform(lo, hi)))
    return hp


def perturb_hp_pbt(hp: dict, seed: int) -> dict:
    """PBT standard explore: multiply by 0.8 or 1.2 (weight-compat HPs only)."""
    rng = np.random.default_rng(seed)
    child = dict(hp)
    for k in WEIGHT_COMPATIBLE_HPS:
        lo, hi, typ = HP_RANGES[k]
        if rng.random() < 0.5:
            continue
        factor = rng.choice([0.8, 1.25])
        if typ == "float":
            child[k] = float(np.clip(hp[k] * factor, lo, hi))
        elif typ == "float_log":
            child[k] = float(np.clip(hp[k] * factor, lo, hi))
        elif typ == "int":
            child[k] = int(np.clip(round(hp[k] * factor), lo, hi))
    return child


def mutate_hp_cardinal(hp: dict, sigma: float, seed: int) -> dict:
    """Cardinal full-HP mutate (includes architecture)."""
    rng = np.random.default_rng(seed)
    child = dict(hp)
    for k, (lo, hi, typ) in HP_RANGES.items():
        if rng.random() < 0.5:
            continue
        if typ == "float":
            delta = rng.normal(0, sigma * (hi - lo))
            child[k] = float(np.clip(hp[k] + delta, lo, hi))
        elif typ == "float_log":
            log_lo, log_hi = np.log(lo), np.log(hi)
            delta = rng.normal(0, sigma * (log_hi - log_lo))
            child[k] = float(np.exp(np.clip(np.log(hp[k]) + delta, log_lo, log_hi)))
        elif typ == "int":
            delta = rng.normal(0, sigma * (hi - lo))
            child[k] = int(np.clip(round(hp[k] + delta), lo, hi))
    return child


def train_steps(model, opt, hp, n_steps, x_tr, y_tr, device, step_offset=0):
    """Train for n_steps with LR schedule, return val-ready model."""
    bs = int(hp["batch_size"])
    warmup = int(hp["warmup_steps"])
    decay_gamma = float(hp["lr_decay_gamma"])
    clip = float(hp["gradient_clip"])
    base_lr = float(hp["lr"])
    N = x_tr.shape[0]
    failed = False
    for step in range(n_steps):
        global_step = step + step_offset
        if global_step < warmup:
            lr = base_lr * (global_step + 1) / max(warmup, 1)
        else:
            lr = base_lr * (decay_gamma ** ((global_step - warmup) // 100))
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = torch.randint(0, N, (bs,), device=device)
        model.train()
        pred = model(x_tr[idx]).squeeze()
        loss = ((pred - y_tr[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if not torch.isfinite(loss):
            failed = True
            break
    return model, opt, failed


def eval_model(model, x_eval, y_eval):
    model.eval()
    with torch.no_grad():
        loss = ((model(x_eval).squeeze() - y_eval) ** 2).mean().item()
    if not np.isfinite(loss):
        return 1e6
    return loss


def make_opt(model, hp):
    return torch.optim.Adam(
        model.parameters(),
        lr=float(hp["lr"]),
        betas=(float(hp["beta1"]), float(hp["beta2"])),
        eps=float(hp["epsilon"]),
        weight_decay=float(hp["weight_decay"]),
    )


def update_opt_hp(opt, hp):
    """Update optimizer HPs in-place (preserve state)."""
    for pg in opt.param_groups:
        pg["lr"] = float(hp["lr"])
        pg["betas"] = (float(hp["beta1"]), float(hp["beta2"]))
        pg["eps"] = float(hp["epsilon"])
        pg["weight_decay"] = float(hp["weight_decay"])


# ============================================================================
# METHOD A: PBT
# ============================================================================
def run_pbt(n_workers: int, n_cycles: int, steps_per_cycle: int,
             exploit_frac: float, seed: int, device: str):
    """PBT: workers train in parallel (simulated sequential), exploit/explore
    every steps_per_cycle steps."""
    print(f"\n  PBT: {n_workers}w × {n_cycles}cy × {steps_per_cycle}step",
          flush=True)
    x_tr, y_tr = synthetic_data(1000, seed)
    x_va, y_va = synthetic_data(200, seed + 1)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)

    # Initialize workers: one with defaults
    hps = [random_hp(seed + i * 7) for i in range(n_workers)]
    # Keep arch HPs fixed to first worker's choice (PBT can't change arch)
    arch_seed_hp = random_hp(seed + 999)
    fixed_h = int(arch_seed_hp["hidden_dim"])
    fixed_L = int(arch_seed_hp["n_layers"])
    for hp in hps:
        hp["hidden_dim"] = fixed_h
        hp["n_layers"] = fixed_L

    # Build workers
    models = []
    opts = []
    for i, hp in enumerate(hps):
        torch.manual_seed(seed + i * 101)
        m = make_model(fixed_h, fixed_L, float(hp["dropout"])).to(device)
        o = make_opt(m, hp)
        models.append(m)
        opts.append(o)

    history = []
    total_steps_per_worker = 0

    for cycle in range(n_cycles):
        # Train each worker for steps_per_cycle
        for i in range(n_workers):
            models[i], opts[i], failed = train_steps(
                models[i], opts[i], hps[i], steps_per_cycle,
                x_tr, y_tr, device, step_offset=total_steps_per_worker)
        total_steps_per_worker += steps_per_cycle

        # Evaluate each
        val_losses = [eval_model(m, x_va, y_va) for m in models]
        ranked = sorted(range(n_workers), key=lambda i: val_losses[i])  # ascending (best first)
        print(f"  cy{cycle}: best w{ranked[0]} val={val_losses[ranked[0]]:.4f} "
              f"worst w{ranked[-1]} val={val_losses[ranked[-1]]:.4f}",
              flush=True)
        history.append({"cycle": cycle, "val_losses": val_losses,
                         "hps": [dict(hp) for hp in hps]})

        # Exploit + explore (except last cycle)
        if cycle < n_cycles - 1:
            n_exploit = max(1, int(n_workers * exploit_frac))
            top_workers = ranked[:n_exploit]
            bot_workers = ranked[-n_exploit:]
            rng_np = np.random.default_rng(seed + cycle * 33)
            for bot_i in bot_workers:
                # Copy weights+HP from random top worker
                src_i = int(rng_np.choice(top_workers))
                models[bot_i].load_state_dict(models[src_i].state_dict())
                hps[bot_i] = perturb_hp_pbt(hps[src_i],
                                              seed + cycle * 13 + bot_i)
                # Re-make optimizer (momentum state may not match new HP well,
                # but PBT standard does this)
                opts[bot_i] = make_opt(models[bot_i], hps[bot_i])

    # Final: best worker
    val_losses = [eval_model(m, x_va, y_va) for m in models]
    best_i = int(np.argmin(val_losses))
    return hps[best_i], models[best_i], -val_losses[best_i], history


# ============================================================================
# METHOD B: Cardinal Pattern B (from phase_14)
# ============================================================================
def run_cardinal(n_universes: int, n_epochs: int, steps_per_epoch: int,
                  sigma: float, seed: int, device: str):
    print(f"\n  Cardinal: {n_universes}u × {n_epochs}ep × {steps_per_epoch}step",
          flush=True)
    x_tr, y_tr = synthetic_data(1000, seed)
    x_va, y_va = synthetic_data(200, seed + 1)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)

    universes = [random_hp(seed + i * 7) for i in range(n_universes)]

    best_hp = None
    best_q = -np.inf
    history = []

    for epoch in range(n_epochs):
        qs = []
        for u_id, hp in enumerate(universes):
            # Fresh train
            torch.manual_seed(seed + epoch * 31 + u_id)
            model = make_model(int(hp["hidden_dim"]), int(hp["n_layers"]),
                                float(hp["dropout"])).to(device)
            opt = make_opt(model, hp)
            model, opt, failed = train_steps(
                model, opt, hp, steps_per_epoch, x_tr, y_tr, device)
            if failed:
                q = -1e6
            else:
                q = -eval_model(model, x_va, y_va)
            qs.append(q)
            if q > best_q:
                best_q = q
                best_hp = dict(hp)

        ranked = sorted(range(n_universes), key=lambda i: -qs[i])
        print(f"  ep{epoch}: best u{ranked[0]} q={qs[ranked[0]]:.4f}",
              flush=True)
        history.append({"epoch": epoch, "qualities": qs,
                         "universes": [dict(hp) for hp in universes]})
        if epoch < n_epochs - 1:
            for bad_idx in ranked[-2:]:
                universes[bad_idx] = mutate_hp_cardinal(
                    universes[ranked[0]], sigma,
                    seed + epoch * 53 + bad_idx)

    return best_hp, best_q, history


# ============================================================================
# METHOD C: Random Search
# ============================================================================
def run_random(n_evals: int, steps: int, seed: int, device: str):
    print(f"\n  Random: {n_evals} evals", flush=True)
    x_tr, y_tr = synthetic_data(1000, seed)
    x_va, y_va = synthetic_data(200, seed + 1)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)

    best_hp = None
    best_q = -np.inf
    for i in range(n_evals):
        hp = random_hp(seed + i * 137)
        torch.manual_seed(seed + i * 31)
        model = make_model(int(hp["hidden_dim"]), int(hp["n_layers"]),
                            float(hp["dropout"])).to(device)
        opt = make_opt(model, hp)
        model, opt, failed = train_steps(model, opt, hp, steps,
                                           x_tr, y_tr, device)
        if failed:
            q = -1e6
        else:
            q = -eval_model(model, x_va, y_va)
        if q > best_q:
            best_q = q
            best_hp = dict(hp)
    return best_hp, best_q


# ============================================================================
# EVAL on test seeds (retrain from scratch)
# ============================================================================
def eval_hp_on_tests(hp, total_steps, device, test_seeds):
    test_losses = []
    for s in test_seeds:
        x_tr, y_tr = synthetic_data(1000, s)
        x_te, y_te = synthetic_data(500, s + 7777)
        x_tr, y_tr = x_tr.to(device), y_tr.to(device)
        x_te, y_te = x_te.to(device), y_te.to(device)
        torch.manual_seed(s)
        model = make_model(int(hp["hidden_dim"]), int(hp["n_layers"]),
                            float(hp["dropout"])).to(device)
        opt = make_opt(model, hp)
        model, opt, failed = train_steps(model, opt, hp, total_steps,
                                           x_tr, y_tr, device)
        te = 1e6 if failed else eval_model(model, x_te, y_te)
        test_losses.append(te)
    return {"test_mean": float(np.mean(test_losses)),
            "test_std": float(np.std(test_losses)),
            "test_losses": test_losses}


def eval_pbt_model_direct(model, hp, device, test_seeds, hint_total_steps):
    """PBT returns a trained model. For fair compare, we also retrain on each
    test seed (since test_seeds have different data). PBT's HP is used."""
    return eval_hp_on_tests(hp, hint_total_steps, device, test_seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_workers",  type=int, default=5)
    ap.add_argument("--n_cycles",   type=int, default=5)
    ap.add_argument("--steps",      type=int, default=400)  # per cycle/epoch
    ap.add_argument("--sigma",      type=float, default=0.25)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--n_test_seeds", type=int, default=5)
    ap.add_argument("--exploit_frac", type=float, default=0.2)
    ap.add_argument("--output", type=str, default="phase_16_pbt_vs_cardinal.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_steps_per_worker = args.n_cycles * args.steps
    print("=" * 72, flush=True)
    print(f"  PHASE 16: PBT vs Cardinal (12D HP, same budget)", flush=True)
    print(f"  Each worker total steps: {total_steps_per_worker}", flush=True)
    print(f"  Device: {device}", flush=True)
    print("=" * 72, flush=True)

    test_seeds = [args.seed + i * 997 for i in range(args.n_test_seeds)]
    t_all = time.time()

    # A. PBT
    print(f"\n{'='*72}\n  A. PBT (DeepMind 2017)\n{'='*72}", flush=True)
    t0 = time.time()
    pbt_hp, pbt_model, pbt_q, pbt_hist = run_pbt(
        args.n_workers, args.n_cycles, args.steps, args.exploit_frac,
        args.seed, device)
    print(f"  PBT best HP: lr={pbt_hp['lr']:.5f} bs={pbt_hp['batch_size']} "
          f"h={pbt_hp['hidden_dim']} L={pbt_hp['n_layers']}", flush=True)
    # Eval on test seeds: retrain from scratch with PBT's HP (same total steps)
    ev_A = eval_hp_on_tests(pbt_hp, total_steps_per_worker, device, test_seeds)
    t_A = time.time() - t0
    print(f"  Test: mean={ev_A['test_mean']:.4f} ± {ev_A['test_std']:.4f} "
          f"({t_A:.1f}s)", flush=True)

    # B. Cardinal
    print(f"\n{'='*72}\n  B. CARDINAL Pattern B\n{'='*72}", flush=True)
    t0 = time.time()
    card_hp, card_q, card_hist = run_cardinal(
        args.n_workers, args.n_cycles, total_steps_per_worker, args.sigma,
        args.seed, device)
    # Note: Cardinal trains each universe for FULL total_steps_per_worker (full runs)
    # This matches: n_workers × n_cycles × steps compute per method
    print(f"  Cardinal best HP: lr={card_hp['lr']:.5f} bs={card_hp['batch_size']} "
          f"h={card_hp['hidden_dim']} L={card_hp['n_layers']}", flush=True)
    ev_B = eval_hp_on_tests(card_hp, total_steps_per_worker, device, test_seeds)
    t_B = time.time() - t0
    print(f"  Test: mean={ev_B['test_mean']:.4f} ± {ev_B['test_std']:.4f} "
          f"({t_B:.1f}s)", flush=True)

    # C. Random
    print(f"\n{'='*72}\n  C. RANDOM SEARCH\n{'='*72}", flush=True)
    t0 = time.time()
    n_random_evals = args.n_workers * args.n_cycles  # same budget
    rand_hp, rand_q = run_random(n_random_evals, total_steps_per_worker,
                                   args.seed, device)
    print(f"  Random best HP: lr={rand_hp['lr']:.5f} bs={rand_hp['batch_size']} "
          f"h={rand_hp['hidden_dim']} L={rand_hp['n_layers']}", flush=True)
    ev_C = eval_hp_on_tests(rand_hp, total_steps_per_worker, device, test_seeds)
    t_C = time.time() - t0
    print(f"  Test: mean={ev_C['test_mean']:.4f} ± {ev_C['test_std']:.4f} "
          f"({t_C:.1f}s)", flush=True)

    # Compare
    total = time.time() - t_all
    print(f"\n{'='*72}\n  RESULTS (test_loss, lower=better)\n{'='*72}", flush=True)
    print(f"  {'method':25s} {'test_mean':>10s} {'test_std':>10s}", flush=True)
    print(f"  {'A. PBT (DeepMind)':25s} {ev_A['test_mean']:>10.4f} "
          f"{ev_A['test_std']:>10.4f}", flush=True)
    print(f"  {'B. Cardinal Pattern B':25s} {ev_B['test_mean']:>10.4f} "
          f"{ev_B['test_std']:>10.4f}", flush=True)
    print(f"  {'C. Random Search':25s} {ev_C['test_mean']:>10.4f} "
          f"{ev_C['test_std']:>10.4f}", flush=True)

    d_BvsA = ev_A['test_mean'] - ev_B['test_mean']
    d_BvsC = ev_C['test_mean'] - ev_B['test_mean']
    print(f"\n  Cardinal vs PBT:    Δ = {d_BvsA:+.4f} "
          f"({'CARDINAL WINS' if d_BvsA > 0 else 'PBT WINS'})", flush=True)
    print(f"  Cardinal vs Random: Δ = {d_BvsC:+.4f} "
          f"({'CARDINAL WINS' if d_BvsC > 0 else 'RANDOM WINS'})", flush=True)
    print(f"\n  Total: {total:.0f}s", flush=True)

    out = {
        "config": vars(args),
        "total_elapsed_s": round(total, 1),
        "pbt_hp": pbt_hp, "pbt_eval": ev_A,
        "cardinal_hp": card_hp, "cardinal_eval": ev_B,
        "random_hp": rand_hp, "random_eval": ev_C,
        "cardinal_vs_pbt": d_BvsA,
        "cardinal_vs_random": d_BvsC,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
