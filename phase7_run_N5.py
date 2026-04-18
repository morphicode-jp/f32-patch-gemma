"""phase7_run_N5.py - Full N=5 Phase 7 training and evaluation.

Trains N=5 agents with alternating co-evolution for 3 cycles, then runs
the full 4-criterion evaluation and saves results.
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
    print("  PHASE 7 FULL RUN: N=5, 3 cycles, no Hebbian (baseline)")
    print("=" * 70)

    t0 = time.time()

    # Training
    agent_params, history = train_phase7(
        N_AGENTS=5,
        n_cycles=3,
        budget_per_agent=30,  # 30s scheduled; ~90-150s actual per agent
        use_hebbian=False,
        verbose=True,
    )

    train_time = time.time() - t0
    print(f"\n  Total training: {train_time:.0f}s "
          f"({train_time/60:.1f} min, {len(history)} rounds)")

    # Save training history
    with open("phase7_N5_training_history.json", "w") as f:
        json.dump({
            "N_AGENTS": 5,
            "n_cycles": 3,
            "budget_per_agent_s": 30,
            "total_train_elapsed_s": round(train_time, 1),
            "history": history,
            "final_params": [[float(x) for x in p] for p in agent_params],
        }, f, indent=2)
    print(f"  Saved: phase7_N5_training_history.json")

    # Evaluation
    print(f"\n" + "=" * 70)
    print(f"  EVALUATION")
    print("=" * 70)
    eval_result = run_full_evaluation(
        agent_params,
        save_path="phase7_N5_eval_result.json"
    )

    total = time.time() - t0
    print(f"\n  TOTAL Phase 7 N=5 run: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
