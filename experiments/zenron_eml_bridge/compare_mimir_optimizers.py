"""mimir_odin_stable / structure_policy で EML symbolic regression を試みる比較.

bridge (zenron_eml_bridge smoke) との数値比較用。同じ 12 target に対して、
bridge を経由せず純 mimir 最適化 (連続 EML tree decoder) で fit を試みる。

期待 (仮説):
  - mimir_odin_stable: 連続 8d 最適化、leaf type の整数ジャンプを proxy が
    扱えず大半 fail
  - structure_policy: curated seed 経路で多少救済、ただし不安定
  - bridge (kathara_mined): 9 target で 5/5、新 sin/cos/tanh で 0/5
  → 結論「bridge 必須、mimir 単独は EML symbolic 探索不可」を立証
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from twelve.agent.mimir_odin_stable import mimir_odin_stable
from twelve.agent.mimir_odin_structure_policy import mimir_odin_structure_policy


# ----- EML primitives (run_eml_zenron_smoke と同じ意味論) -----
def _safe_exp(x):
    return math.exp(min(50.0, max(-50.0, x)))


def _safe_log(y):
    return math.log(max(1e-12, abs(y)))


def eml(x, y):
    return _safe_exp(x) - _safe_log(y)


# ----- depth 3 EML tree (8 leaf) - run_eml_symbolic.py のデコーダを拡張 -----
# 8 leaf、3 階層の binary tree (full)
#   root = eml(L01, L23)
#   L01 = eml(L0, L1)、L23 = eml(L2, L3)
# 上記 2 段目 -> さらに 1 段増やすと leaf 8 個になる:
#   root = eml(eml(eml(l0,l1), eml(l2,l3)), eml(eml(l4,l5), eml(l6,l7)))
# 各 leaf type ∈ {1, x, +c, -c} で encode。

def leaf_value(leaf_type, c, x):
    if leaf_type == 0:
        return 1.0
    if leaf_type == 1:
        return x
    if leaf_type == 2:
        return c
    return -c


def decode_tree_depth3(p, x):
    types = [int(np.clip((v + 1.0) * 2.0, 0, 3.999)) for v in p[:8]]
    cs = [float(v) for v in p[8:16]]
    leaves = [leaf_value(t, c, x) for t, c in zip(types, cs)]
    n01 = eml(leaves[0], leaves[1])
    n23 = eml(leaves[2], leaves[3])
    n45 = eml(leaves[4], leaves[5])
    n67 = eml(leaves[6], leaves[7])
    n0123 = eml(n01, n23)
    n4567 = eml(n45, n67)
    return eml(n0123, n4567)


def decode_tree_for_policy(p):
    """structure_policy 用: x なしで policy 表現."""
    types = [int(np.clip((v + 1.0) * 2.0, 0, 3.999)) for v in p[:8]]
    cs = [float(v) for v in p[8:16]]
    return {"leaf_types": types, "cs": cs}


# ----- targets (run_eml_zenron_smoke の make_benchmarks と一致) -----
TARGETS = [
    ("exp(x)", np.exp, (0.25, 3.0)),
    ("log(x)", np.log, (0.25, 3.0)),
    ("log(log(x))", lambda x: np.log(np.log(x)), (1.25, 3.0)),
    ("x^2", lambda x: x ** 2, (0.25, 3.0)),
    ("x*log(x)", lambda x: x * np.log(x), (1.25, 3.0)),
    ("exp(log(x))", lambda x: np.exp(np.log(x)), (0.25, 3.0)),
    ("log(exp(x))", lambda x: np.log(np.exp(x)), (0.25, 3.0)),
    ("exp(x)+log(x)", lambda x: np.exp(x) + np.log(x), (1.25, 3.0)),
    ("x^2+log(x)", lambda x: x ** 2 + np.log(x), (1.25, 3.0)),
    ("sin(x)", np.sin, (0.25, 1.5)),
    ("cos(x)", np.cos, (0.25, 1.5)),
    ("tanh(x)", np.tanh, (0.25, 2.0)),
]


def train_test_xs(domain):
    lo, hi = domain
    train = np.linspace(lo, hi, 16)
    test = np.linspace(lo + 0.05, hi - 0.05, 16)
    return train, test


SUCCESS_LOSS = 1.0e-8  # smoke と同じ


def make_eval_fn(target_fn, domain):
    train_xs, _ = train_test_xs(domain)
    train_target = target_fn(train_xs)

    def eval_fn(p):
        try:
            ys_pred = np.array([decode_tree_depth3(p, x) for x in train_xs])
            if not np.all(np.isfinite(ys_pred)):
                return -1e6
            mse = float(np.mean((ys_pred - train_target) ** 2))
        except Exception:
            return -1e6
        return -mse  # higher is better
    return eval_fn


def make_test_loss(target_fn, domain, params):
    _, test_xs = train_test_xs(domain)
    test_target = target_fn(test_xs)
    try:
        ys_pred = np.array([decode_tree_depth3(params, x) for x in test_xs])
        if not np.all(np.isfinite(ys_pred)):
            return float("inf")
        return float(np.mean((ys_pred - test_target) ** 2))
    except Exception:
        return float("inf")


def detail_fn_factory(target_fn, domain):
    eval_fn = make_eval_fn(target_fn, domain)

    def detail(p):
        return {
            "score": eval_fn(p),
            "policy": decode_tree_for_policy(p),
        }
    return detail


# ----- ranges -----
BOUNDS = [(-1.0, 1.0)] * 8 + [(-3.0, 3.0)] * 8  # 16 dim


def run_one(name, target_fn, domain, mode, time_budget=30.0, n_seeds=3):
    """同じ target で n_seeds 回実行、最良 test_loss と success rate を返す."""
    eval_fn = make_eval_fn(target_fn, domain)

    rng = np.random.default_rng(42)
    # Windows path-safe name (* / ( ) etc. を _ に sanitize)
    import re as _re
    safe_name = _re.sub(r"[^A-Za-z0-9_]", "_", name)

    best_test_losses = []
    successes = 0
    elapsed_total = 0.0
    for seed in range(n_seeds):
        t0 = time.time()
        if mode == "stable":
            r = mimir_odin_stable(
                eval_fn, BOUNDS,
                time_budget=time_budget,
                executor="thread",
                auto_check=False,
                verbose=False,
                experience_id=f"compare_{safe_name}_stable_s{seed}",
            )
            params = r["best_params"]
        elif mode == "structure_policy":
            seed_policies = []
            for _ in range(8):
                types_p = list(rng.uniform(-1, 1, size=8))
                cs_p = list(rng.uniform(-2, 2, size=8))
                seed_policies.append([float(v) for v in types_p + cs_p])
            r = mimir_odin_structure_policy(
                eval_fn, BOUNDS,
                policy_points=seed_policies,
                param_decoder=decode_tree_for_policy,
                detail_fn=detail_fn_factory(target_fn, domain),
                time_budget=time_budget,
                executor="thread",
                verbose=False,
                experience_id=f"compare_{safe_name}_sp_s{seed}",
            )
            if r.get("aborted"):
                best_test_losses.append(float("inf"))
                elapsed_total += time.time() - t0
                continue
            params = r["best_params"]
        else:
            raise ValueError(f"unknown mode: {mode}")

        elapsed_total += time.time() - t0
        test_loss = make_test_loss(target_fn, domain, params)
        best_test_losses.append(test_loss)
        if test_loss < SUCCESS_LOSS * 1e3:  # bridge は 1e-8、mimir では緩めて 1e-5
            successes += 1

    finite = [v for v in best_test_losses if np.isfinite(v)]
    return {
        "target": name,
        "mode": mode,
        "n_seeds": n_seeds,
        "successes": successes,
        "best_test_loss": float(min(finite)) if finite else float("inf"),
        "mean_test_loss": float(np.mean(finite)) if finite else float("inf"),
        "elapsed_s": elapsed_total,
    }


def main():
    print("=" * 78, flush=True)
    print("mimir_odin_stable / structure_policy で EML symbolic regression", flush=True)
    print("(zenron_eml_bridge smoke の 12 target を mimir 単独で解けるか測定)", flush=True)
    print("=" * 78, flush=True)
    print(f"depth 3 EML tree、16 連続 dim、time_budget=30s、3 seeds", flush=True)
    print(f"success threshold: test_loss < 1e-5 (bridge の 1e-8 より緩い)", flush=True)

    results = []
    for name, fn, dom in TARGETS:
        for mode in ("stable", "structure_policy"):
            print(f"\n--- {name} / {mode} ---", flush=True)
            r = run_one(name, fn, dom, mode, time_budget=30.0, n_seeds=3)
            print(f"  succ={r['successes']}/{r['n_seeds']}  "
                  f"test_loss(best/mean)={r['best_test_loss']:.3e}/{r['mean_test_loss']:.3e}  "
                  f"t={r['elapsed_s']:.0f}s", flush=True)
            results.append(r)

    # SUMMARY
    print("\n" + "=" * 78, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 78, flush=True)
    print(f"{'target':<18} {'mode':<18} {'succ':>5}  {'best':>12}  {'mean':>12}",
          flush=True)
    for r in results:
        print(f"{r['target']:<18} {r['mode']:<18} {r['successes']:>3}/3  "
              f"{r['best_test_loss']:>12.3e}  {r['mean_test_loss']:>12.3e}",
              flush=True)

    # save
    out_path = os.path.join(_HERE, "mimir_compare_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
