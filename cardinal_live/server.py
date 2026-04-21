"""Cardinal live visualizer — runs original Cardinal (agent world) and serves
real-time state via HTTP so a browser can visualize the evolution.

Usage:
  python cardinal_live/server.py [--n_universes 4] [--agents 8] [--port 8765]

Then open http://localhost:8765/ in a browser.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tamashii"))

from tamashii.phase_10_3_cardinal import (
    Universe, make_universe_params, DEFAULT_WORLD_PARAMS, mutate_universe_params,
)
from tamashii.phase_9_ecology import SHELLS_DEFAULT


# Shared state (written by Cardinal thread, read by HTTP handler)
STATE_LOCK = threading.Lock()
STATE = {
    "step": 0,
    "epoch": 0,
    "status": "initializing",
    "universes": [],
    "history": [],
    "best_id": -1,
    "elapsed_s": 0.0,
    "started_at": 0.0,
}


def snapshot_universe(u: Universe) -> dict:
    """Extract live display state from a universe."""
    s = u.world.stats()
    alive_idx = [i for i in range(u.world.n_agents) if u.world.agent_alive[i]]
    # Agent positions (2D projection for viz)
    positions = []
    for i in alive_idx[:30]:  # cap for bandwidth
        p = u.world.agent_positions[i]
        energy = u.world.agent_energy[i]
        gen = u.world.agent_generation[i]
        positions.append({
            "x": float(p[0]), "y": float(p[1]),
            "e": float(energy), "g": int(gen),
        })
    # Food positions
    food = []
    for fp in u.world.food_positions[:40]:
        food.append({"x": float(fp[0]), "y": float(fp[1])})
    # Agent DNA samples (first 5 for display)
    dna_samples = []
    for i in alive_idx[:5]:
        try:
            dna = u.world.agents_external[i].shells[0].kathara_params
            dna_samples.append([float(x) for x in np.asarray(dna)[:8]])
        except Exception:
            pass

    return {
        "id": u.id,
        "quality": float(u.quality()),
        "world_params": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                          for k, v in u.world_params.items()},
        "stats": {
            "n_alive": s["n_alive"],
            "n_births": s["n_births"],
            "n_deaths": s["n_deaths"],
            "max_generation": s["max_generation"],
            "dna_diversity": float(s["dna_diversity"]),
            "mean_energy": float(s["mean_energy"]),
            "n_food_voxels": s["n_food_voxels"],
            "light_level": float(s["light_level"]),
        },
        "positions": positions,
        "food": food,
        "dna_samples": dna_samples,
        "world_size": int(u.world.size),
    }


def run_cardinal_loop(n_universes: int, agents_per_u: int, seed: int,
                      chunk_steps: int, epoch_steps: int, n_epochs: int):
    """Main Cardinal loop — runs in a background thread."""
    global STATE

    with STATE_LOCK:
        STATE["status"] = "building universes..."
    started = time.time()
    STATE["started_at"] = started

    shells = SHELLS_DEFAULT
    configs_dir = os.path.join(THIS_DIR, "..", "tamashii", "configs")
    trained_dir = "tamashii/configs"

    # Build universes
    try:
        universes = []
        for u_id in range(n_universes):
            wp = make_universe_params(seed + u_id * 17)
            u = Universe(u_id, wp, agents_per_u, seed + u_id * 101,
                          shells, configs_dir, trained_dir)
            universes.append(u)
            with STATE_LOCK:
                STATE["status"] = f"built universe {u_id+1}/{n_universes}"
    except Exception as e:
        with STATE_LOCK:
            STATE["status"] = f"ERROR building: {e}"
        traceback.print_exc()
        return

    history = []

    for epoch in range(n_epochs):
        with STATE_LOCK:
            STATE["epoch"] = epoch
            STATE["status"] = f"epoch {epoch}/{n_epochs} running..."

        # Run each universe in chunks, dumping state after each chunk
        for u in universes:
            steps_done = 0
            while steps_done < epoch_steps:
                chunk = min(chunk_steps, epoch_steps - steps_done)
                u.run_epoch(chunk, log_every=max(chunk // 2, 1))
                steps_done += chunk

                # Dump state snapshot
                snapshots = [snapshot_universe(uu) for uu in universes]
                best_q = max((s["quality"] for s in snapshots), default=0.0)
                best_id = -1
                for s in snapshots:
                    if abs(s["quality"] - best_q) < 1e-9:
                        best_id = s["id"]; break
                with STATE_LOCK:
                    STATE["universes"] = snapshots
                    STATE["step"] = epoch * epoch_steps + steps_done
                    STATE["best_id"] = best_id
                    STATE["elapsed_s"] = time.time() - started

        # End of epoch: record history + meta-evolve (Cardinal replacement)
        qualities = [(uu.id, uu.quality()) for uu in universes]
        qualities.sort(key=lambda x: -x[1])
        best_id = qualities[0][0]
        worst_id = qualities[-1][0]

        history.append({
            "epoch": epoch,
            "qualities": [{"id": q[0], "q": float(q[1])} for q in qualities],
            "best": best_id,
            "worst": worst_id,
        })

        with STATE_LOCK:
            STATE["history"] = history
            STATE["status"] = (f"epoch {epoch} done → meta-evolve: "
                                f"u{worst_id} ← variant of u{best_id}")

        # Cardinal meta-evolve: worst gets variant of best
        if epoch < n_epochs - 1:
            best_u = next(uu for uu in universes if uu.id == best_id)
            new_params = mutate_universe_params(best_u.world_params,
                                                  seed=seed + epoch * 37)
            worst_u_idx = next(i for i, uu in enumerate(universes)
                                if uu.id == worst_id)
            worst_u_old_id = universes[worst_u_idx].id
            new_u = Universe(worst_u_old_id, new_params, agents_per_u,
                              seed + worst_u_old_id * 101 + epoch * 7,
                              shells, configs_dir, trained_dir)
            universes[worst_u_idx] = new_u

    with STATE_LOCK:
        STATE["status"] = f"done (epoch {n_epochs} complete)"


# ============================================================================
# HTTP server
# ============================================================================
INDEX_HTML_PATH = os.path.join(THIS_DIR, "index.html")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence noisy logging
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open(INDEX_HTML_PATH, "rb") as f:
                    body = f.read()
            except Exception as e:
                self.send_error(404, str(e))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/state.json":
            with STATE_LOCK:
                body = json.dumps(STATE).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=4)
    ap.add_argument("--agents",      type=int, default=6)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--epoch_steps", type=int, default=800)
    ap.add_argument("--n_epochs",    type=int, default=5)
    ap.add_argument("--chunk_steps", type=int, default=40,
                     help="state dump interval (agent-steps)")
    ap.add_argument("--port",        type=int, default=8765)
    args = ap.parse_args()

    # Start Cardinal thread
    card_thread = threading.Thread(
        target=run_cardinal_loop,
        kwargs=dict(
            n_universes=args.n_universes,
            agents_per_u=args.agents,
            seed=args.seed,
            chunk_steps=args.chunk_steps,
            epoch_steps=args.epoch_steps,
            n_epochs=args.n_epochs,
        ),
        daemon=True,
    )
    card_thread.start()

    # Start HTTP server
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  *** Cardinal live server running at http://localhost:{args.port}/  ***\n",
          flush=True)
    print(f"  Parameters: {args.n_universes} universes × {args.agents} agents × "
          f"{args.n_epochs} epochs × {args.epoch_steps} steps", flush=True)
    print(f"  Open your browser. Ctrl-C to stop.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopping server...", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
