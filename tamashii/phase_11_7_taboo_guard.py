"""Phase 11.7 — Taboo guard_fn integration (Phase 8 of Underworld plan).

Connect the existing `taboo` shell (which tracks violations) to Cardinal's
quality function so evolution learns to AVOID violations.

Condition A (baseline): quality = base (no penalty). Cardinal evolves freely.
Condition B (taboo-aware): quality = base - penalty_weight * avg_violation.
                            Cardinal must reduce violations while keeping quality.

Violations tracked (existing taboo shell):
  - crowding_violations: approached peer at high speed (collision risk)
  - hoarding_violations: alone at food while peer is hungry nearby
  - voice_violations: excessive voice output

Hypothesis: B ends with LOWER violation rates but similar quality. This
demonstrates Sentinel-style structural ethics (no external reward needed,
the constraint is built into the selection pressure).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tamashii.phase_11_cardinal_3d import (
    GravityUniverse, make_3d_universe_params, mutate_3d_universe_params,
)
from tamashii.phase_9_ecology import SHELLS_DEFAULT


class SensitiveTabooUniverse(GravityUniverse):
    """Like GravityUniverse but with taboo shell thresholds lowered so
    violations actually fire in sparse 3D environments.

    Overrides taboo shell config after build with 3D-appropriate thresholds.
    """

    taboo_penalty_weight: float = 0.0  # no penalty, just observe violations

    def initialize_agents(self):
        super().initialize_agents()
        # Lower taboo thresholds for more sensitive detection in 3D/sparse
        for a in self.agents:
            for shell in a.shells:
                if shell.name == "taboo":
                    shell.crowding_prox_threshold = 0.15  # was 0.4
                    shell.crowding_speed_threshold = 0.3  # was 0.5
                    shell.hoarding_prox_threshold = 0.15  # was 0.5
                    shell.hoarding_olf_threshold = 0.3    # was 1.0
                    shell.mandatory_voice_prox = 0.2      # was 0.5
                    break

    def _violation_counts(self) -> dict:
        """Sum violations across all alive agents' taboo shells."""
        totals = {"crowding": 0, "hoarding": 0, "voice": 0, "total": 0}
        n_with_taboo = 0
        for a in self.agents:
            for shell in a.shells:
                if shell.name == "taboo":
                    s = shell._state
                    totals["crowding"] += int(s.get("crowding_violations", 0))
                    totals["hoarding"] += int(s.get("hoarding_violations", 0))
                    totals["voice"]    += int(s.get("voice_violations", 0))
                    n_with_taboo += 1
                    break
        totals["total"] = totals["crowding"] + totals["hoarding"] + totals["voice"]
        totals["n_agents_counted"] = n_with_taboo
        return totals

    def quality(self) -> float:
        base = super().quality()
        vcounts = self._violation_counts()
        n_alive = max(self.world.stats()["n_alive"], 1)
        avg_violations = vcounts["total"] / n_alive
        return base - self.taboo_penalty_weight * avg_violations


class TabooAwareGravityUniverse(SensitiveTabooUniverse):
    """Sensitive taboo + penalty applied to quality (Phase 8 full integration)."""
    taboo_penalty_weight: float = 0.005  # per violation-per-agent


def run_condition(U_class, label, n_universes, n_epochs, epoch_steps,
                   agents, seed):
    print(f"\n{'=' * 72}\n  {label}\n{'=' * 72}", flush=True)
    shells = list(SHELLS_DEFAULT) + ["taboo"]  # explicitly add taboo
    cfg = "tamashii/configs"
    universes = []
    for u_id in range(n_universes):
        wp = make_3d_universe_params(seed + u_id * 17)
        u = U_class(u_id, wp, agents, seed + u_id * 101, shells, cfg, cfg)
        universes.append(u)

    history = []
    t0 = time.time()
    for epoch in range(n_epochs):
        for u in universes:
            u.run_epoch(epoch_steps, log_every=epoch_steps // 2)

        snaps = []
        for u in universes:
            s = u.world.stats()
            # Extract violations (only meaningful for TabooAware)
            if hasattr(u, "_violation_counts"):
                v = u._violation_counts()
            else:
                v = {"crowding": 0, "hoarding": 0, "voice": 0, "total": 0,
                      "n_agents_counted": 0}
            n_alive = max(s["n_alive"], 1)
            snaps.append({
                "id": u.id,
                "quality": float(u.quality()),
                "n_alive": s["n_alive"],
                "n_births": s["n_births"],
                "n_deaths": s["n_deaths"],
                "max_gen": s["max_generation"],
                "dna_div": float(s["dna_diversity"]),
                "mean_z": float(s.get("mean_agent_z", 0.5)),
                "gravity": float(u.world_params.get("gravity", 0)),
                "vert_food": float(u.world_params.get("vertical_food_frac", 0)),
                "mut_rate": float(u.world_params.get("mutation_rate", 0)),
                "violations_total": v["total"],
                "violations_crowding": v["crowding"],
                "violations_hoarding": v["hoarding"],
                "violations_voice": v["voice"],
                "violation_rate_per_agent": v["total"] / n_alive,
            })
        ranked = sorted(snaps, key=lambda x: -x["quality"])
        history.append({"epoch": epoch, "snapshots": snaps,
                         "best": ranked[0]["id"], "worst": ranked[-1]["id"]})

        print(f"  ep{epoch}: best u{ranked[0]['id']} q={ranked[0]['quality']:+.2f}",
              flush=True)
        for s in ranked:
            print(f"    u{s['id']}: q={s['quality']:+.2f} alive={s['n_alive']:2d} "
                  f"viol={s['violations_total']} (c{s['violations_crowding']}/"
                  f"h{s['violations_hoarding']}/v{s['violations_voice']}) "
                  f"DNA={s['dna_div']:.3f} mut={s['mut_rate']:.3f}",
                  flush=True)

        # Meta-evolve
        if epoch < n_epochs - 1:
            best_u = next(u for u in universes if u.id == ranked[0]["id"])
            new_params = mutate_3d_universe_params(
                best_u.world_params, seed=seed + epoch * 37)
            worst_idx = next(i for i, u in enumerate(universes)
                              if u.id == ranked[-1]["id"])
            old_id = universes[worst_idx].id
            universes[worst_idx] = U_class(
                old_id, new_params, agents,
                seed + old_id * 101 + epoch * 7,
                shells, cfg, cfg)

    total = time.time() - t0
    print(f"\n  {label} total: {total:.0f}s", flush=True)

    # Summary across final epoch
    final = history[-1]["snapshots"]
    alive_snaps = [s for s in final if s["n_alive"] > 0]
    return {
        "label": label,
        "total_elapsed_s": round(total, 1),
        "history": history,
        "final_snapshots": final,
        "final_mean_quality": float(np.mean([s["quality"] for s in final])),
        "final_max_quality": float(max(s["quality"] for s in final)),
        "final_mean_violation_rate": float(np.mean(
            [s["violation_rate_per_agent"] for s in alive_snaps])) if alive_snaps else 0.0,
        "final_total_violations": sum(s["violations_total"] for s in final),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=4)
    ap.add_argument("--agents",      type=int, default=8)
    ap.add_argument("--epoch_steps", type=int, default=1000)
    ap.add_argument("--n_epochs",    type=int, default=10)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--output",      type=str,
                     default="phase_11_7_taboo_guard.json")
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print(f"  PHASE 11.7: Taboo-aware Cardinal (Phase 8 integration)", flush=True)
    print("=" * 72, flush=True)

    result_A = run_condition(
        SensitiveTabooUniverse, "A. Sensitive taboo, NO penalty (baseline)",
        args.n_universes, args.n_epochs, args.epoch_steps, args.agents,
        args.seed)

    result_B = run_condition(
        TabooAwareGravityUniverse, "B. Sensitive taboo + penalty in quality",
        args.n_universes, args.n_epochs, args.epoch_steps, args.agents,
        args.seed)

    print(f"\n{'=' * 72}\n  COMPARISON\n{'=' * 72}", flush=True)
    print(f"  {'metric':35s} {'A baseline':>14s} {'B taboo-aware':>16s}", flush=True)
    print(f"  {'final_max_quality':35s} {result_A['final_max_quality']:>14.2f} "
          f"{result_B['final_max_quality']:>16.2f}", flush=True)
    print(f"  {'final_mean_quality':35s} {result_A['final_mean_quality']:>14.2f} "
          f"{result_B['final_mean_quality']:>16.2f}", flush=True)
    print(f"  {'final_total_violations':35s} {result_A['final_total_violations']:>14d} "
          f"{result_B['final_total_violations']:>16d}", flush=True)
    print(f"  {'final_mean_violation_rate':35s} {result_A['final_mean_violation_rate']:>14.3f} "
          f"{result_B['final_mean_violation_rate']:>16.3f}", flush=True)

    reduction_pct = 100.0 * (1.0 - result_B["final_total_violations"]
                              / max(result_A["final_total_violations"], 1))
    print(f"\n  Violation REDUCTION (B vs A): {reduction_pct:+.1f}%",
          flush=True)
    if reduction_pct > 15:
        print(f"  ★ Taboo guard EFFECTIVE: violations suppressed "
              f"structurally via selection", flush=True)
    elif reduction_pct > 0:
        print(f"  ~ Modest reduction", flush=True)
    else:
        print(f"  × No reduction (penalty too small or taboo shell not active)",
              flush=True)

    out = {"A_baseline": result_A, "B_taboo_aware": result_B,
            "reduction_pct": reduction_pct, "config": vars(args)}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
