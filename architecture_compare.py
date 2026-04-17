"""architecture_compare.py — Final comparison: 12N single-layer vs 5-layer cortical

Loads results from both architectures and generates side-by-side analysis.
Highlights which hypothesis (cortical trades peak for generalization) is supported.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "12N single": {
        "training": "brain_sentinel_biological_result.json",
        "generalization": "brain_generalization_result.json",
    },
    "5L cortical": {
        "training": "cortical_sentinel_result.json",
        "generalization": "cortical_generalization_result.json",
    },
}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    data = {}
    for arch, paths in FILES.items():
        data[arch] = {
            "training": load(paths["training"]),
            "generalization": load(paths["generalization"]),
        }

    print("=" * 78)
    print("  Architecture Comparison: 12N single-layer vs 5L cortical")
    print("=" * 78)

    # Summary table
    header = f"  {'Metric':<30s}  {'12N single':>15s}  {'5L cortical':>15s}"
    print(header)
    print("-" * 78)

    def row(label, v12, v5, fmt="{:.2f}"):
        v12s = fmt.format(v12) if v12 is not None else "-"
        v5s = fmt.format(v5) if v5 is not None else "-"
        print(f"  {label:<30s}  {v12s:>15s}  {v5s:>15s}")

    t12 = data["12N single"]["training"]
    t5 = data["5L cortical"]["training"]
    g12 = data["12N single"]["generalization"]
    g5 = data["5L cortical"]["generalization"]

    # Parameters
    row("Params (D)", 91, 184, "{:d}")
    row("Neurons", 12, 60, "{:d}")

    # Training metrics
    if t12 and t5:
        row("Training eval_score",
            t12.get("eval_score"), t5.get("eval_score"))
        row("Training best_ever",
            t12.get("best_ever_score", t12.get("eval_score")),
            t5.get("best_ever_score", t5.get("eval_score")))
        row("Training guard_score",
            t12.get("guard_score"), t5.get("guard_score"))
        row("Training elapsed (s)",
            t12.get("elapsed_s"), t5.get("elapsed_s"), "{:.0f}")

    # Generalization metrics
    if g12 and g5:
        row("Unseen seeds mean",
            g12.get("unseen_mean"), g5.get("unseen_mean"))
        row("Unseen seeds std",
            g12.get("unseen_std"), g5.get("unseen_std"))
        row("Unseen reach rate",
            g12.get("unseen_reach_rate", 0) * 100,
            g5.get("unseen_reach_rate", 0) * 100, "{:.0f}%")
        row("Generalization gap %",
            g12.get("gap_pct"), g5.get("gap_pct"), "{:+.1f}%")

    print("=" * 78)

    # Hypothesis check
    if g12 and g5:
        print("\n  HYPOTHESIS: Cortical trades peak eval for generalization")
        print("-" * 78)
        gap_12 = g12.get("gap_pct", 100)
        gap_5 = g5.get("gap_pct", 100)
        reach_12 = g12.get("unseen_reach_rate", 0)
        reach_5 = g5.get("unseen_reach_rate", 0)

        if gap_5 < gap_12:
            print(f"  [gap] Cortical gap {gap_5:.1f}% < 12N gap {gap_12:.1f}%  --> SUPPORTED")
        else:
            print(f"  [gap] Cortical gap {gap_5:.1f}% >= 12N gap {gap_12:.1f}%  --> NOT supported")

        if reach_5 > reach_12:
            print(f"  [reach] Cortical reach {reach_5*100:.0f}% > 12N reach {reach_12*100:.0f}%  --> SUPPORTED")
        else:
            print(f"  [reach] Cortical reach {reach_5*100:.0f}% <= 12N reach {reach_12*100:.0f}%  --> NOT supported")

        peak_12 = t12.get("best_ever_score", t12.get("eval_score", 0)) if t12 else 0
        peak_5 = t5.get("best_ever_score", t5.get("eval_score", 0)) if t5 else 0
        if peak_5 < peak_12:
            print(f"  [peak] Cortical peak {peak_5:.1f} < 12N peak {peak_12:.1f}  --> SUPPORTED (trade-off confirmed)")
        elif peak_5 > peak_12:
            print(f"  [peak] Cortical peak {peak_5:.1f} > 12N peak {peak_12:.1f}  --> BONUS (cortical also better at peak)")
        else:
            print(f"  [peak] Tied at {peak_5:.1f}")

    # Save consolidated
    with open("architecture_comparison_result.json", "w") as f:
        json.dump({
            "12N_single": {
                "params": 91,
                "neurons": 12,
                "training": t12,
                "generalization": g12,
            },
            "5L_cortical": {
                "params": 184,
                "neurons": 60,
                "training": t5,
                "generalization": g5,
            },
        }, f, indent=2, default=str)
    print("\n  Saved: architecture_comparison_result.json")


if __name__ == "__main__":
    main()
