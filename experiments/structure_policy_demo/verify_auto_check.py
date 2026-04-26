"""mimir_odin_stable(auto_check=True) の実用検証.

7 ケースで auto_check が「呼び忘れ防止 / 本番救済 / 余計な誤発動なし」を
満たすか実測。各ケースで期待挙動と実測を比較、合計成功率を出す。
"""
from __future__ import annotations

import math
import os
import sys
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from twelve.agent.mimir_odin_stable import mimir_odin_stable


# ====================================================================
# eval_fns (各ケース)
# ====================================================================
def f_continuous(p):
    """純連続 smooth Gaussian peak."""
    p = np.asarray(p)
    return float(math.exp(-0.5 * np.linalg.norm(p - 0.3) ** 2))


def f_constant(p):
    """壊れた eval_fn = 定数 (apply漏れ模倣)."""
    return 0.42


def f_nan(p):
    """常に NaN 返す eval_fn."""
    return float("nan")


_STOCH_RNG = np.random.default_rng(0)
def f_stochastic(p):
    """LLM eval 模倣 (CV > 0.1 を確実に超える noise level)."""
    p = np.asarray(p)
    base = math.exp(-0.5 * np.linalg.norm(p - 0.3) ** 2)
    return float(base + _STOCH_RNG.normal(0, 1.0))


_CITIES_8 = np.random.default_rng(2026).uniform(0, 1, size=(8, 2))
def f_tsp(p):
    """TSP argsort (純離散構造)."""
    order = np.argsort(p)
    dist = sum(np.linalg.norm(_CITIES_8[order[i]] - _CITIES_8[order[(i+1) % 8]])
               for i in range(8))
    return float(-dist)


def f_integer_mask(p):
    """整数 + 連続 混合 (probe では見えない)."""
    p = np.asarray(p)
    abs_p = np.abs(p)
    k = max(2, min(8, int(round(p[0] * 3 + 5))))
    top_idx = np.argsort(abs_p)[::-1][:k]
    return float(sum(p[i] for i in top_idx))


# ====================================================================
# 検証 case 定義
# ====================================================================
CASES = [
    {
        "name": "1. 純連続 — auto_check pass + 本番完走",
        "fn": f_continuous,
        "ranges": [(-1.0, 1.5)] * 5,
        "kwargs": {"time_budget": 12.0},
        "expect_raise": False,
        "expect_recommended": "mimir_odin_stable",
        "expect_severity": "ok",
    },
    {
        "name": "2. constant — fatal raise (1h 救済)",
        "fn": f_constant,
        "ranges": [(-1.0, 1.5)] * 5,
        "kwargs": {"time_budget": 12.0},
        "expect_raise": True,
        "expect_recommended": None,
        "expect_severity": "fatal",
    },
    {
        "name": "3. NaN return — fatal raise",
        "fn": f_nan,
        "ranges": [(-1.0, 1.5)] * 5,
        "kwargs": {"time_budget": 12.0},
        "expect_raise": True,
        "expect_recommended": None,
        "expect_severity": "fatal",
    },
    {
        "name": "4. stochastic — pass + lad 推奨 warning",
        "fn": f_stochastic,
        "ranges": [(-1.0, 1.5)] * 5,
        "kwargs": {"time_budget": 12.0},
        "expect_raise": False,
        "expect_recommended": "mimir_odin_stable",
        "expect_stochastic_detected": True,
    },
    {
        "name": "5. TSP (純離散) — fatal raise (stable で TSP 誤用を阻止)",
        "fn": f_tsp,
        "ranges": [(-1.0, 1.0)] * 8,
        "kwargs": {"time_budget": 12.0},
        "expect_raise": True,
        "expect_raise_msg_contains": "fatal",
        # TSP の対称軸 tie で 2 点 constant 検出 → fatal raise
        # → ユーザーが適切な structure_policy に誘導される正しい挙動
    },
    {
        "name": "6. auto_check=False — fatal でも raise しない",
        "fn": f_constant,
        "ranges": [(-1.0, 1.5)] * 3,
        "kwargs": {"time_budget": 8.0, "auto_check": False},
        "expect_raise": False,
        "expect_auto_check_field": None,
    },
    {
        "name": "7. declared_structural=True + 混合 — structure_policy 推奨確定",
        "fn": f_integer_mask,
        "ranges": [(-1.5, 1.5)] * 8,
        "kwargs": {"time_budget": 12.0, "auto_check_declared_structural": True},
        "expect_raise": False,
        "expect_recommended": "mimir_odin_structure_policy",
        "expect_stable_continued": True,
    },
]


# ====================================================================
# 検証実行
# ====================================================================
def run_case(case):
    print(f"\n--- {case['name']} ---", flush=True)
    print(f"  期待: raise={case['expect_raise']}  "
          f"recommended={case.get('expect_recommended')}", flush=True)

    t_start = time.time()
    raised = False
    raised_msg = None
    result = None
    try:
        result = mimir_odin_stable(
            case["fn"], case["ranges"],
            executor="thread",
            verbose=False,
            **case["kwargs"],
        )
    except ValueError as e:
        raised = True
        raised_msg = str(e)
    except Exception as e:
        raised = True
        raised_msg = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t_start

    # 検証
    checks = []

    # raise の有無
    if case["expect_raise"]:
        if raised:
            checks.append(("raise", True, f"raised as expected: {raised_msg[:60]}"))
        else:
            checks.append(("raise", False, "expected raise but completed"))
    else:
        if raised:
            checks.append(("no_raise", False, f"unexpected raise: {raised_msg[:60]}"))
        else:
            checks.append(("no_raise", True, "completed"))

    # auto_check field の有無
    if not raised and result is not None:
        diag = result.get("auto_check")
        expected_field = case.get("expect_auto_check_field", "expect_present")
        if expected_field is None:
            if diag is None:
                checks.append(("auto_check_skipped", True, "skipped as expected"))
            else:
                checks.append(("auto_check_skipped", False, f"unexpected diag: {diag.get('severity')}"))
        else:
            if diag is None:
                checks.append(("auto_check_present", False, "expected diag, got None"))
            else:
                checks.append(("auto_check_present", True, f"severity={diag.get('severity')}"))

                # severity 期待値
                if case.get("expect_severity"):
                    actual_sev = diag.get("severity")
                    ok = actual_sev == case["expect_severity"]
                    checks.append(("severity_match", ok,
                                   f"expected={case['expect_severity']} actual={actual_sev}"))

                # recommended_optimizer 期待値
                if case.get("expect_recommended"):
                    actual_rec = diag.get("recommended_optimizer")
                    ok = actual_rec == case["expect_recommended"]
                    checks.append(("recommended_match", ok,
                                   f"expected={case['expect_recommended']} actual={actual_rec}"))

                # stochastic 検出期待
                if case.get("expect_stochastic_detected"):
                    detected = diag.get("stochastic_detected", False)
                    checks.append(("stochastic_detected", detected,
                                   f"detected={detected} cv={diag.get('stochastic_cv')}"))

                # 本番が走り抜けたかの確認 (best_params 存在)
                if case.get("expect_stable_continued"):
                    has_bp = result.get("best_params") is not None
                    checks.append(("stable_continued", has_bp,
                                   f"best_params present={has_bp}"))

    # check budget が overhead 範囲内 (期待 5-15s + 本番)
    budget_set = case["kwargs"].get("time_budget", 0)
    expected_max_overhead = 20.0  # check_budget 上限 + buffer
    if not case["kwargs"].get("auto_check") is False:
        # auto_check 走った想定
        # 本番実行時は budget + overhead で elapsed
        # raise 時は overhead だけ
        if case["expect_raise"]:
            ok = elapsed < expected_max_overhead
            checks.append(("fast_raise", ok,
                           f"elapsed={elapsed:.1f}s (raise should be < {expected_max_overhead}s)"))

    # まとめ
    all_ok = all(c[1] for c in checks)
    marker = "✓" if all_ok else "✗"
    print(f"  実測: elapsed={elapsed:.1f}s", flush=True)
    for name, ok, msg in checks:
        cm = "✓" if ok else "✗"
        print(f"    {cm} {name}: {msg}", flush=True)
    print(f"  {marker} {'PASS' if all_ok else 'FAIL'}", flush=True)
    return {"case": case["name"], "elapsed": elapsed, "raised": raised,
            "checks": [{"name": n, "ok": o, "msg": m} for n, o, m in checks],
            "all_ok": all_ok}


def main():
    print("=" * 80, flush=True)
    print("auto_check 実用検証: 7 ケースで機能確認", flush=True)
    print("=" * 80, flush=True)

    results = []
    for case in CASES:
        try:
            r = run_case(case)
            results.append(r)
        except Exception as e:
            print(f"  ERROR running case: {e}", flush=True)
            traceback.print_exc()
            results.append({"case": case["name"], "error": str(e), "all_ok": False})

    print("\n" + "=" * 80, flush=True)
    n_pass = sum(1 for r in results if r.get("all_ok"))
    n_total = len(results)
    print(f"SUMMARY: {n_pass}/{n_total} PASS ({100*n_pass/n_total:.0f}%)", flush=True)
    print("=" * 80, flush=True)
    for r in results:
        marker = "✓" if r.get("all_ok") else "✗"
        print(f"  {marker} {r['case']}", flush=True)

    import json
    out_path = os.path.join(_HERE, "verify_auto_check_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
