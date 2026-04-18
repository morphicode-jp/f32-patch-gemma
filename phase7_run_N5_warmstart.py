"""phase7_run_N5_warmstart.py - Phase 7 Stage 2 with solo pre-train warm-start.

Loads solo-trained Kathara(16) brain from kathara16_solo_result.json,
replicates to N=5 agents as warm-start, runs 4 cycles of cooperative
training with Hebbian plasticity, then evaluates 4 criteria.
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
    print("  PHASE 7 STAGE 2: WARM-STARTED N=5 cooperative training")
    print("=" * 70)

    # Load solo-trained params
    solo_path = "kathara16_solo_result.json"
    if not os.path.exists(solo_path):
        print(f"\n  ERROR: {solo_path} not found. Run kathara16_solo_train.py first.")
        sys.exit(1)
    with open(solo_path) as f:
        solo = json.load(f)
    solo_params = solo["best_params"]
    solo_reach = solo["trained"]["reach_rate"]
    print(f"\n  Loaded solo params from {solo_path}:")
    print(f"    solo reach_rate = {solo_reach*100:.0f}%")
    print(f"    solo best_eval  = {solo['best_eval_score']:.2f}")

    # Replicate to N=5 (each agent starts identical; co-evolution diversifies)
    N = 5
    init_params_list = [list(solo_params) for _ in range(N)]

    # Training with Hebbian + warm-start + 4 cycles + 45s budget
    t0 = time.time()
    agent_params, history = train_phase7(
        N_AGENTS=N,
        n_cycles=4,
        budget_per_agent=45,
        use_hebbian=True,  # Phase 7 design spec
        verbose=True,
        init_params_list=init_params_list,
    )
    train_time = time.time() - t0
    print(f"\n  Training total: {train_time:.0f}s ({train_time/60:.1f} min)")

    # Save history
    with open("phase7_N5_warmstart_history.json", "w") as f:
        json.dump({
            "N_AGENTS": N,
            "n_cycles": 4,
            "budget_per_agent_s": 45,
            "use_hebbian": True,
            "warm_start_from": solo_path,
            "solo_reach_rate": solo_reach,
            "total_train_elapsed_s": round(train_time, 1),
            "history": history,
            "final_params": [[float(x) for x in p] for p in agent_params],
        }, f, indent=2)

    # Evaluate
    print("\n" + "=" * 70)
    print("  EVALUATION (4 criteria)")
    print("=" * 70)
    eval_result = run_full_evaluation(
        agent_params,
        save_path="phase7_N5_warmstart_eval.json",
    )

    total = time.time() - t0
    print(f"\n  TOTAL warmstart run: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
