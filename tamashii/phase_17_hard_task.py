"""Phase 17 — Hard task で Cardinal の真価を測る (Phase 16 リベンジ).

Phase 16 の問題: synthetic regression (y=Σx²) は簡単すぎた。
  - HP の sweet spot が broad → random search でも当たる
  - モデル容量が task 複雑度を超えている → 大半の HP で OK

Hard task の特徴を入れる:
  1. **深い net (6 層)** → 初期化・lr に敏感、勾配消失可能
  2. **小データ (300 train / 100 val / 500 test)** → 過学習必須、wd/dropout critical
  3. **分類 (10 class)** → cross-entropy、連続回帰より HP sensitive
  4. **30D 入力 + 非線形 target** → HP 間 interaction 必要
  5. **ステップ数倍** (1000/cycle) → 累積的な差が出る
  6. **狭い HP sweet spot**: lr 高すぎ → diverge、lr 低すぎ → 進まない

同じ 3 手法比較:
  A. PBT (重み継承あり)
  B. Cardinal Pattern B (重み継承なし)
  C. Random Search

Hypothesis:
  Task の narrow sweet spot + 深 net → random 弱化、structured search (PBT/Cardinal) 有利
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch


N_CLASSES = 5
INPUT_DIM = 20

# Fixed target: shared across all seeds so task has learnable signal
_TARGET_W = np.random.default_rng(131).standard_normal(
    (N_CLASSES, INPUT_DIM)).astype(np.float32)


def generate_hard_data(n: int, seed: int):
    """20D input, 5-class target. Target = nonlinear but learnable combination."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, INPUT_DIM)).astype(np.float32)
    # Nonlinear target: tanh-transformed x dot fixed W
    # Harder than linear but clearly learnable with good HP
    x_nl = np.tanh(x) + 0.3 * np.sin(2 * x)
    logits = x_nl @ _TARGET_W.T
    y = logits.argmax(axis=1).astype(np.int64)
    # Add label noise (3%)
    n_noise = int(n * 0.03)
    noise_idx = rng.choice(n, n_noise, replace=False)
    y[noise_idx] = rng.integers(0, N_CLASSES, n_noise).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def make_model(hidden_dim: int, n_layers: int, dropout: float):
    layers = [torch.nn.Linear(INPUT_DIM, hidden_dim), torch.nn.ReLU(),
              torch.nn.Dropout(dropout)]
    for _ in range(n_layers - 2):
        layers += [torch.nn.Linear(hidden_dim, hidden_dim),
                   torch.nn.ReLU(),
                   torch.nn.Dropout(dropout)]
    layers += [torch.nn.Linear(hidden_dim, N_CLASSES)]
    return torch.nn.Sequential(*layers)


HP_RANGES = {
    "lr":             (1e-5, 0.3,    "float_log"),
    "beta1":          (0.5,  0.9999, "float"),
    "beta2":          (0.9,  0.99999,"float"),
    "epsilon":        (1e-10, 1e-4,  "float_log"),
    "weight_decay":   (1e-7, 1e-2,   "float_log"),
    "batch_size":     (8,    128,    "int"),
    "warmup_steps":   (0,    500,    "int"),
    "lr_decay_gamma": (0.5,  1.0,    "float"),
    "gradient_clip":  (1e-2, 1e2,    "float_log"),
    "dropout":        (0.0,  0.5,    "float"),
    "hidden_dim":     (32,   256,    "int"),  # wider for 10-class
    "n_layers":       (3,    8,      "int"),  # deeper — critical for hardness
}


WEIGHT_COMPATIBLE_HPS = {
    "lr", "beta1", "beta2", "epsilon", "weight_decay",
    "batch_size", "warmup_steps", "lr_decay_gamma", "gradient_clip", "dropout",
}


DEFAULT_HP = {
    "lr":             1e-3,
    "beta1":          0.9,
    "beta2":          0.999,
    "epsilon":        1e-8,
    "weight_decay":   1e-4,
    "batch_size":     32,
    "warmup_steps":   0,
    "lr_decay_gamma": 1.0,
    "gradient_clip":  1.0,
    "dropout":        0.1,
    "hidden_dim":     64,
    "n_layers":       4,
}


def random_hp(seed):
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


def perturb_hp_pbt(hp, seed):
    rng = np.random.default_rng(seed)
    child = dict(hp)
    for k in WEIGHT_COMPATIBLE_HPS:
        lo, hi, typ = HP_RANGES[k]
        if rng.random() < 0.5:
            continue
        factor = rng.choice([0.8, 1.25])
        child[k] = float(np.clip(hp[k] * factor, lo, hi)) if typ != "int" \
                    else int(np.clip(round(hp[k] * factor), lo, hi))
    return child


def mutate_hp_cardinal(hp, sigma, seed):
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


def make_opt(model, hp):
    return torch.optim.Adam(
        model.parameters(),
        lr=float(hp["lr"]),
        betas=(float(hp["beta1"]), float(hp["beta2"])),
        eps=float(hp["epsilon"]),
        weight_decay=float(hp["weight_decay"]),
    )


def train_steps(model, opt, hp, n_steps, x_tr, y_tr, device, step_offset=0):
    bs = int(hp["batch_size"])
    warmup = int(hp["warmup_steps"])
    decay_gamma = float(hp["lr_decay_gamma"])
    clip = float(hp["gradient_clip"])
    base_lr = float(hp["lr"])
    N = x_tr.shape[0]
    failed = False
    ce = torch.nn.CrossEntropyLoss()
    for step in range(n_steps):
        gs = step + step_offset
        if gs < warmup:
            lr = base_lr * (gs + 1) / max(warmup, 1)
        else:
            lr = base_lr * (decay_gamma ** ((gs - warmup) // 200))
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = torch.randint(0, N, (bs,), device=device)
        model.train()
        logits = model(x_tr[idx])
        loss = ce(logits, y_tr[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if not torch.isfinite(loss):
            failed = True
            break
    return model, opt, failed


def eval_accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        acc = (pred == y).float().mean().item()
    return acc if np.isfinite(acc) else 0.0


# ============================================================================
# A: PBT
# ============================================================================
def run_pbt(n_workers, n_cycles, steps_per_cycle, exploit_frac, seed, device):
    print(f"\n  PBT: {n_workers}w × {n_cycles}cy × {steps_per_cycle}step", flush=True)
    x_tr, y_tr = generate_hard_data(800, seed)
    x_va, y_va = generate_hard_data(200, seed + 1)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)

    hps = [random_hp(seed + i * 7) for i in range(n_workers)]
    arch_hp = random_hp(seed + 999)
    fixed_h, fixed_L = int(arch_hp["hidden_dim"]), int(arch_hp["n_layers"])
    for hp in hps:
        hp["hidden_dim"] = fixed_h
        hp["n_layers"] = fixed_L

    models, opts = [], []
    for i, hp in enumerate(hps):
        torch.manual_seed(seed + i * 101)
        m = make_model(fixed_h, fixed_L, float(hp["dropout"])).to(device)
        o = make_opt(m, hp)
        models.append(m); opts.append(o)

    total_per_worker = 0
    for cycle in range(n_cycles):
        for i in range(n_workers):
            models[i], opts[i], _ = train_steps(
                models[i], opts[i], hps[i], steps_per_cycle,
                x_tr, y_tr, device, step_offset=total_per_worker)
        total_per_worker += steps_per_cycle
        vaccs = [eval_accuracy(m, x_va, y_va) for m in models]
        ranked = sorted(range(n_workers), key=lambda i: -vaccs[i])
        print(f"  cy{cycle}: best w{ranked[0]} acc={vaccs[ranked[0]]:.3f} "
              f"worst={vaccs[ranked[-1]]:.3f}", flush=True)
        if cycle < n_cycles - 1:
            n_exploit = max(1, int(n_workers * exploit_frac))
            top = ranked[:n_exploit]; bot = ranked[-n_exploit:]
            rng = np.random.default_rng(seed + cycle * 33)
            for bi in bot:
                src = int(rng.choice(top))
                models[bi].load_state_dict(models[src].state_dict())
                hps[bi] = perturb_hp_pbt(hps[src], seed + cycle * 13 + bi)
                opts[bi] = make_opt(models[bi], hps[bi])

    vaccs = [eval_accuracy(m, x_va, y_va) for m in models]
    best_i = int(np.argmax(vaccs))
    return hps[best_i], vaccs[best_i]


# ============================================================================
# B: Cardinal
# ============================================================================
def run_cardinal(n_u, n_ep, steps_per_ep, sigma, seed, device):
    print(f"\n  Cardinal: {n_u}u × {n_ep}ep × {steps_per_ep}step", flush=True)
    x_tr, y_tr = generate_hard_data(800, seed)
    x_va, y_va = generate_hard_data(200, seed + 1)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)

    universes = [random_hp(seed + i * 7) for i in range(n_u)]
    best_hp, best_q = None, -np.inf
    for epoch in range(n_ep):
        qs = []
        for u_id, hp in enumerate(universes):
            torch.manual_seed(seed + epoch * 31 + u_id)
            m = make_model(int(hp["hidden_dim"]), int(hp["n_layers"]),
                            float(hp["dropout"])).to(device)
            o = make_opt(m, hp)
            m, o, _ = train_steps(m, o, hp, steps_per_ep, x_tr, y_tr, device)
            q = eval_accuracy(m, x_va, y_va)
            qs.append(q)
            if q > best_q:
                best_q = q; best_hp = dict(hp)
        ranked = sorted(range(n_u), key=lambda i: -qs[i])
        print(f"  ep{epoch}: best u{ranked[0]} acc={qs[ranked[0]]:.3f}",
              flush=True)
        if epoch < n_ep - 1:
            for bad in ranked[-2:]:
                universes[bad] = mutate_hp_cardinal(
                    universes[ranked[0]], sigma, seed + epoch * 53 + bad)
    return best_hp, best_q


# ============================================================================
# C: Random
# ============================================================================
def run_random(n_evals, steps, seed, device):
    print(f"\n  Random: {n_evals} evals × {steps}step", flush=True)
    x_tr, y_tr = generate_hard_data(800, seed)
    x_va, y_va = generate_hard_data(200, seed + 1)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)

    best_hp, best_q = None, -np.inf
    for i in range(n_evals):
        hp = random_hp(seed + i * 137)
        torch.manual_seed(seed + i * 31)
        m = make_model(int(hp["hidden_dim"]), int(hp["n_layers"]),
                        float(hp["dropout"])).to(device)
        o = make_opt(m, hp)
        m, o, _ = train_steps(m, o, hp, steps, x_tr, y_tr, device)
        q = eval_accuracy(m, x_va, y_va)
        if q > best_q:
            best_q = q; best_hp = dict(hp)
    return best_hp, best_q


def eval_on_test_seeds(hp, total_steps, device, test_seeds):
    test_accs = []
    for s in test_seeds:
        x_tr, y_tr = generate_hard_data(300, s)
        x_te, y_te = generate_hard_data(500, s + 7777)
        x_tr, y_tr = x_tr.to(device), y_tr.to(device)
        x_te, y_te = x_te.to(device), y_te.to(device)
        torch.manual_seed(s)
        m = make_model(int(hp["hidden_dim"]), int(hp["n_layers"]),
                        float(hp["dropout"])).to(device)
        o = make_opt(m, hp)
        m, o, failed = train_steps(m, o, hp, total_steps, x_tr, y_tr, device)
        acc = 0.0 if failed else eval_accuracy(m, x_te, y_te)
        test_accs.append(acc)
    return {"test_mean_acc": float(np.mean(test_accs)),
            "test_std_acc": float(np.std(test_accs)),
            "test_accs": test_accs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_workers", type=int, default=6)
    ap.add_argument("--n_cycles",  type=int, default=6)
    ap.add_argument("--steps",     type=int, default=1000)
    ap.add_argument("--sigma",     type=float, default=0.25)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--n_test_seeds", type=int, default=5)
    ap.add_argument("--exploit_frac", type=float, default=0.25)
    ap.add_argument("--output", type=str, default="phase_17_hard_task.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_per_worker = args.n_cycles * args.steps
    print("=" * 72, flush=True)
    print(f"  PHASE 17: HARD TASK (10-class, 300 samples, 6+ layers, {total_per_worker} steps)", flush=True)
    print(f"  Budget: {args.n_workers * args.n_cycles} evals", flush=True)
    print("=" * 72, flush=True)

    test_seeds = [args.seed + i * 997 for i in range(args.n_test_seeds)]

    # Baseline: default HP
    print(f"\n  A0. DEFAULT HP baseline", flush=True)
    ev_def = eval_on_test_seeds(DEFAULT_HP, total_per_worker, device, test_seeds)
    print(f"  Defaults test acc: {ev_def['test_mean_acc']:.3f} ± {ev_def['test_std_acc']:.3f}",
          flush=True)

    # A. PBT
    print(f"\n{'='*72}\n  A. PBT\n{'='*72}", flush=True)
    t0 = time.time()
    pbt_hp, pbt_q = run_pbt(args.n_workers, args.n_cycles, args.steps,
                              args.exploit_frac, args.seed, device)
    ev_A = eval_on_test_seeds(pbt_hp, total_per_worker, device, test_seeds)
    print(f"  PBT test acc: {ev_A['test_mean_acc']:.3f} ± {ev_A['test_std_acc']:.3f} ({time.time()-t0:.0f}s)",
          flush=True)

    # B. Cardinal
    print(f"\n{'='*72}\n  B. CARDINAL\n{'='*72}", flush=True)
    t0 = time.time()
    card_hp, card_q = run_cardinal(args.n_workers, args.n_cycles, total_per_worker,
                                     args.sigma, args.seed, device)
    ev_B = eval_on_test_seeds(card_hp, total_per_worker, device, test_seeds)
    print(f"  Cardinal test acc: {ev_B['test_mean_acc']:.3f} ± {ev_B['test_std_acc']:.3f} ({time.time()-t0:.0f}s)",
          flush=True)

    # C. Random
    print(f"\n{'='*72}\n  C. RANDOM\n{'='*72}", flush=True)
    t0 = time.time()
    n_rand = args.n_workers * args.n_cycles
    rand_hp, rand_q = run_random(n_rand, total_per_worker, args.seed, device)
    ev_C = eval_on_test_seeds(rand_hp, total_per_worker, device, test_seeds)
    print(f"  Random test acc: {ev_C['test_mean_acc']:.3f} ± {ev_C['test_std_acc']:.3f} ({time.time()-t0:.0f}s)",
          flush=True)

    print(f"\n{'='*72}\n  RESULTS (test accuracy, higher=better)\n{'='*72}", flush=True)
    print(f"  {'method':25s} {'test_acc':>10s} {'std':>8s}", flush=True)
    print(f"  {'A0. Defaults':25s} {ev_def['test_mean_acc']:>10.3f} {ev_def['test_std_acc']:>8.3f}", flush=True)
    print(f"  {'A.  PBT':25s} {ev_A['test_mean_acc']:>10.3f} {ev_A['test_std_acc']:>8.3f}", flush=True)
    print(f"  {'B.  Cardinal':25s} {ev_B['test_mean_acc']:>10.3f} {ev_B['test_std_acc']:>8.3f}", flush=True)
    print(f"  {'C.  Random':25s} {ev_C['test_mean_acc']:>10.3f} {ev_C['test_std_acc']:>8.3f}", flush=True)

    d_BA = ev_B['test_mean_acc'] - ev_A['test_mean_acc']
    d_BC = ev_B['test_mean_acc'] - ev_C['test_mean_acc']
    print(f"\n  Cardinal vs PBT:    Δ = {d_BA:+.3f} "
          f"({'CARD WINS' if d_BA > 0 else 'PBT WINS'})", flush=True)
    print(f"  Cardinal vs Random: Δ = {d_BC:+.3f} "
          f"({'CARD WINS' if d_BC > 0 else 'RAND WINS'})", flush=True)

    out = {
        "config": vars(args),
        "default_eval": ev_def,
        "pbt_hp": pbt_hp, "pbt_eval": ev_A,
        "cardinal_hp": card_hp, "cardinal_eval": ev_B,
        "random_hp": rand_hp, "random_eval": ev_C,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
