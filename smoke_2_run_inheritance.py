"""Smoke: 2 連続 reigen() 実行で 2 回目が 1 回目の best を継承するか (同一プロセス内)。

This uses the REAL reigen_meta_knowledge.json (not isolated tmp path).
Before running, back up the production file; restore after.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.reigen import reigen, Reigen
import twelve.agent.reigen as reigen_mod


MK_PATH = reigen_mod._MK_PATH
BAK = MK_PATH + ".smoke_bak"

# Clean slate
shutil.copy(MK_PATH, BAK)
try:
    # Reset to empty shell
    with open(MK_PATH, "w", encoding="utf-8") as f:
        json.dump({"_version": 1, "self_param_best": {}, "self_param_history": []}, f)
    # Reload module-level cache
    reigen_mod._MK = reigen_mod._load_meta_knowledge()
    print("[setup] meta_knowledge reset to empty")
    print(f"  _MK self_param_best: {reigen_mod._MK.get('self_param_best')}")
    print()

    # ---- Run 1 ----
    print("=== Run 1 (empty meta_knowledge; preset defaults used) ===")
    # Check defaults BEFORE run
    r1_before = Reigen(
        eval_fn=lambda p: 0, guard_fn=lambda p: 0,
        user_param_ranges=[(-1, 1)] * 2,
        self_dim_preset="minimal_4",
    )
    defaults_before = list(r1_before.self_param_defaults)
    print(f"  minimal_4 defaults at init: {defaults_before}")
    print(f"  (expected preset defaults: [0.30, 0.60, 0.50, 1.0])")

    # Actually execute a run to populate meta_knowledge
    r1 = reigen(
        lambda p: -sum((x - 1.0) ** 2 for x in p),
        lambda p: 0.0,
        user_param_ranges=[(-1, 1)] * 2,
        self_dim_preset="minimal_4",
        inner_time_budget=1,
        time_budget=6,
        experience_id="smoke2_run1",
        verbose=False,
    )
    print(f"  run 1 verdict: {r1['verdict']}")
    print(f"  run 1 user_best_score: {r1['user_best_score']:.4f}")
    print(f"  run 1 self_best: {r1['self_best_params']}")

    # Inspect meta_knowledge AFTER run 1
    mk_after = json.load(open(MK_PATH, encoding="utf-8"))
    print(f"  meta_knowledge['self_param_best']: {mk_after.get('self_param_best')}")
    print()

    # ---- Run 2 ----
    # IMPORTANT: in-memory _MK is refreshed by _mk_merge_run_result, so we
    # can immediately test inheritance in the same process.
    print("=== Run 2 (meta_knowledge populated; new Reigen should inherit) ===")
    r2_before = Reigen(
        eval_fn=lambda p: 0, guard_fn=lambda p: 0,
        user_param_ranges=[(-1, 1)] * 2,
        self_dim_preset="minimal_4",
    )
    defaults_after = list(r2_before.self_param_defaults)
    print(f"  minimal_4 defaults at init: {defaults_after}")
    print()

    # Verify inheritance happened
    r1_self_best = r1["self_best_params"]
    print("=== Verification ===")
    expected = [r1_self_best.get(n, d) for n, d in zip(r2_before.self_param_names,
                                                        defaults_before)]
    print(f"  expected (run 1 learned): {expected}")
    print(f"  actual   (run 2 init):    {defaults_after}")
    matches = all(abs(a - e) < 1e-9 for a, e in zip(defaults_after, expected))
    if matches:
        print("  ✓ INHERITANCE WORKS — Rule 9 now actually executes")
    else:
        print("  ✗ MISMATCH — inheritance broken")
        sys.exit(1)

finally:
    # Restore production
    shutil.copy(BAK, MK_PATH)
    os.remove(BAK)
    print("\n[teardown] meta_knowledge restored from backup")
