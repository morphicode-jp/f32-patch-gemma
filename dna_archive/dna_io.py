"""dna_io.py — DNA pool persistence and lineage continuation.

Save/load kathara_params populations from Cardinal runs so that:
  - Evolved DNA is preserved across runs (not lost at process exit)
  - New Cardinal runs can CONTINUE from prior run's end-state
  - Multiple run lineages can be mixed (social learning)
  - Pristine baselines stay untouched

API:
  save_run(run_dir, universes, metadata)        — persist a finished run
  load_dna_from_run(run_dir, universe_id)       — one universe's final DNA
  load_dna_pool(sources, n_agents)              — merge from multiple sources
  list_runs()                                   — enumerate archived runs
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np

ARCHIVE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
)
RUNS_DIR = os.path.join(ARCHIVE_ROOT, "runs")
PRISTINE_DIR = os.path.join(ARCHIVE_ROOT, "pristine_baselines")


def ensure_runs_dir():
    os.makedirs(RUNS_DIR, exist_ok=True)


def new_run_dir(label: str = "run") -> str:
    """Create a fresh timestamped directory for a new Cardinal run."""
    ensure_runs_dir()
    ts = time.strftime("%Y%m%d_%H%M%S")
    dir_name = f"{ts}_{label}"
    path = os.path.join(RUNS_DIR, dir_name)
    os.makedirs(path, exist_ok=True)
    return path


def save_run(run_dir: str, universes: list, metadata: dict):
    """Persist the final state of a Cardinal run.

    For each universe, saves all alive agents' kathara_params + basic stats.
    """
    os.makedirs(run_dir, exist_ok=True)

    # Per-universe final DNA dump
    per_universe = {}
    for u in universes:
        alive_dna = []
        for i in range(u.world.n_agents):
            if u.world.agent_alive[i]:
                try:
                    dna = np.asarray(
                        u.world.agents_external[i].shells[0].kathara_params,
                        dtype=np.float64).tolist()
                    alive_dna.append({
                        "agent_idx": i,
                        "generation": int(u.world.agent_generation[i]),
                        "kathara_params": dna,
                    })
                except Exception:
                    pass
        # Write per-universe file
        u_file = os.path.join(run_dir, f"u{u.id}_final.json")
        u_data = {
            "universe_id": u.id,
            "world_params": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                              for k, v in u.world_params.items()},
            "stats": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                      for k, v in u.world.stats().items()},
            "quality": float(u.quality()),
            "n_alive_dna_saved": len(alive_dna),
            "alive_dna": alive_dna,
        }
        with open(u_file, "w", encoding="utf-8") as f:
            json.dump(u_data, f, indent=2, default=str)
        per_universe[u.id] = {
            "file": os.path.basename(u_file),
            "n_alive": len(alive_dna),
            "quality": float(u.quality()),
        }

    # Metadata
    meta_path = os.path.join(run_dir, "metadata.json")
    meta = dict(metadata)
    meta["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["per_universe"] = per_universe
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[dna_io] Saved {len(universes)} universes to {run_dir}",
          flush=True)
    return run_dir


def load_dna_from_run(run_dir: str, universe_id: int) -> list[np.ndarray]:
    """Load all alive DNA from a specific universe of a past run."""
    u_file = os.path.join(run_dir, f"u{universe_id}_final.json")
    if not os.path.exists(u_file):
        raise FileNotFoundError(f"No saved universe {universe_id} in {run_dir}")
    with open(u_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [np.asarray(d["kathara_params"], dtype=np.float64)
            for d in data.get("alive_dna", [])]


def load_dna_pool(sources: list[str | tuple[str, int]],
                    n_wanted: Optional[int] = None,
                    seed: int = 42) -> list[np.ndarray]:
    """Load DNA from one or more sources, return a pool of numpy arrays.

    sources: list of either
      - "pristine:16N_3d" → load pristine baseline's kathara_params
      - path_to_run_dir → load from that run's best universe
      - (run_dir, universe_id) → load from specific universe
    """
    pool = []
    for src in sources:
        if isinstance(src, tuple):
            run_dir, u_id = src
            pool.extend(load_dna_from_run(run_dir, u_id))
        elif isinstance(src, str) and src.startswith("pristine:"):
            tag = src[len("pristine:"):]
            # Find matching pristine file
            matches = [f for f in os.listdir(PRISTINE_DIR) if tag in f]
            if not matches:
                raise FileNotFoundError(f"No pristine baseline matching '{tag}' in {PRISTINE_DIR}")
            path = os.path.join(PRISTINE_DIR, sorted(matches)[-1])
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            p = d.get("params", {}).get("kathara_params")
            if p is None:
                p = d.get("params", {}).get("cortical_params")
            if p is None:
                p = d.get("best_params")  # Sentinel-format
            if p is None:
                raise ValueError(f"No kathara_params/cortical_params in {path}")
            pool.append(np.asarray(p, dtype=np.float64))
        elif isinstance(src, str) and os.path.isdir(src):
            # Load the BEST universe from this run (by quality)
            meta = os.path.join(src, "metadata.json")
            if os.path.exists(meta):
                with open(meta, "r", encoding="utf-8") as f:
                    m = json.load(f)
                per_u = m.get("per_universe", {})
                if per_u:
                    best_uid = max(per_u.items(), key=lambda x: x[1]["quality"])[0]
                    pool.extend(load_dna_from_run(src, int(best_uid)))
                    continue
            # Fallback: concat all u*.json
            for fname in sorted(os.listdir(src)):
                if fname.startswith("u") and fname.endswith("_final.json"):
                    uid = int(fname[1:fname.index("_")])
                    pool.extend(load_dna_from_run(src, uid))
        else:
            raise ValueError(f"Unknown source: {src}")

    # Cap or expand pool
    if n_wanted is not None and n_wanted > 0:
        rng = np.random.default_rng(seed)
        if len(pool) >= n_wanted:
            # Subsample
            idx = rng.choice(len(pool), n_wanted, replace=False)
            pool = [pool[i] for i in idx]
        elif len(pool) > 0:
            # Grow by sampling with replacement + noise
            while len(pool) < n_wanted:
                parent = pool[int(rng.integers(len(pool)))]
                child = parent + rng.normal(0, 0.05, parent.shape)
                pool.append(child)
    return pool


def list_runs() -> list[dict]:
    """Enumerate all archived runs."""
    if not os.path.exists(RUNS_DIR):
        return []
    runs = []
    for d in sorted(os.listdir(RUNS_DIR)):
        full = os.path.join(RUNS_DIR, d)
        meta = os.path.join(full, "metadata.json")
        if os.path.isdir(full) and os.path.exists(meta):
            with open(meta, "r", encoding="utf-8") as f:
                m = json.load(f)
            runs.append({
                "run_dir": full,
                "label": d,
                "timestamp": m.get("timestamp"),
                "per_universe": m.get("per_universe", {}),
                "metadata": {k: v for k, v in m.items() if k != "per_universe"},
            })
    return runs


if __name__ == "__main__":
    print("Existing runs:")
    runs = list_runs()
    if not runs:
        print("  (none yet)")
    for r in runs:
        print(f"  {r['label']} ({r['timestamp']})")
        for uid, info in r["per_universe"].items():
            print(f"    u{uid}: quality={info['quality']:.2f}, "
                  f"alive_saved={info['n_alive']}")
