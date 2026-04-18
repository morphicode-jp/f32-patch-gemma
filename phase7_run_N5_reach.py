"""phase7_run_N5_reach.py - Phase 7 with REACH-CENTRIC reward.

Changes vs phase7_run_N5.py:
  - Reward now strongly rewards actual food reach (80pt) vs approach (15pt)
  - No warm-start (midpoint, preserves diversity for ToM)
  - No Hebbian (original setup that achieved 1/4 with ToM pass)
  - 3 cycles × 5 agents × 45s budget

Target: >=2/4 criteria (improve reach without sacrificing ToM).
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_coop_sentinel import train_phase7
from phase7_evaluation import run_full_evaluation


def main():
    print("=" * 70)
    print("  PHASE 7 with REACH-CENTRIC reward (N=5, midpoint, no Hebbian)")
    print("=" * 70)

    t0 = time.time()
    agent_params, history = train_phase7(
        N_AGENTS=5, n_cycles=3, budget_per_agent=45,
        use_hebbian=False, verbose=True,
    )
    train_time = time.time() - t0
    print(f"\n  Training: {train_time:.0f}s ({train_time/60:.1f} min)")

    with open("phase7_N5_reach_history.json", "w") as f:
        json.dump({
            "reward_version": "reach_centric_v2",
            "N_AGENTS": 5, "n_cycles": 3, "budget_per_agent_s": 45,
            "use_hebbian": False,
            "total_train_elapsed_s": round(train_time, 1),
            "history": history,
            "final_params": [[float(x) for x in p] for p in agent_params],
        }, f, indent=2)

    # Evaluate
    eval_result = run_full_evaluation(
        agent_params,
        save_path="phase7_N5_reach_eval.json"
    )
    total = time.time() - t0
    print(f"\n  TOTAL: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
