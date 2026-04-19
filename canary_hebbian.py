"""Hebbian canary — reproducible 12-float signature for bit-identical verification.

Runs _hebbian_propagate with a deterministic pre-seeded history and input
self_p, prints 12 floats to stdout. Any change to hebbian math WILL shift
one of these bits.

Usage:
    python canary_hebbian.py > /tmp/baseline_hebbian.txt   # before changes
    python canary_hebbian.py > /tmp/after_hebbian.txt      # after changes
    diff /tmp/baseline_hebbian.txt /tmp/after_hebbian.txt  # must be empty
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.reigen import Reigen


def build_reigen():
    """Construct a Reigen instance that exercises _hebbian_propagate."""
    return Reigen(
        eval_fn=lambda p: -sum(x * x for x in p),
        guard_fn=lambda p: 0.0,
        user_param_ranges=[(-1.0, 1.0)] * 3,
        self_dim_preset="kathara_12",
        enable_hebbian=True,
        inner_time_budget=1,
        experience_id="canary_hebbian_fixed",
    )


def seed_history(r):
    """Seed _self_history deterministically (10 entries, required > 5 gate)."""
    # Fixed pseudo-history: 10 (self_p, score) pairs.
    # Values chosen to be representative, NOT random, so result is reproducible.
    hist = []
    for i in range(10):
        sp = tuple(0.1 + 0.05 * i + 0.01 * k for k in range(12))
        score = -((i - 5) ** 2) / 10.0       # parabola in i, max at i=5
        hist.append((sp, score))
    r._self_history = hist
    # also seed best-score fields so reward-gate can fire
    r._best_ever_score = -0.05
    return r


def main():
    r = build_reigen()
    seed_history(r)
    self_p = [0.5] * 12
    # _hebbian_propagate signature: (self, self_p). Reward gate uses history-internal best.
    out = r._hebbian_propagate(self_p)
    # Print with full precision (repr = round-trip safe)
    for i, v in enumerate(out):
        print(f"{i} {v!r}")


if __name__ == "__main__":
    main()
