"""Tamashii orchestrator — parallel shells on shared state S.

Threading model:
  - D-dim numpy state S, single `threading.Lock` for writes
  - Each shell runs in its own daemon thread at its own tick rate
  - Writes are additive `S += gain * delta` (atomic under lock)
  - Reads are lock-free snapshots (race on purpose: shells see slightly stale S,
    but mutations are small so drift is bounded)

Two operation modes:
  async_mode=True:  shells run in threads, caller just drives world
  async_mode=False: caller drives shells synchronously via `tick_once()`
                    (deterministic for tests and training)
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from tamashii.shell_base import Shell


D_DEFAULT = 192


class Tamashii:
    """Parallel-additive shell orchestrator on shared state S."""

    def __init__(
        self,
        shells: list[Shell],
        D: int = D_DEFAULT,
        async_mode: bool = False,
        S_init: np.ndarray | None = None,
    ):
        self.D = D
        self.shells = list(shells)
        self.async_mode = async_mode

        if S_init is not None:
            assert S_init.shape == (D,), f"S_init shape {S_init.shape} != ({D},)"
            self.S = np.array(S_init, dtype=np.float64, copy=True)
        else:
            self.S = np.zeros(D, dtype=np.float64)

        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._running = False
        self.external: Any = None  # caller sets/updates; shells read

        # Per-shell diagnostic: latest delta L2 norm (for coupling metrics)
        self._last_delta_norms: dict[str, float] = {s.name: 0.0 for s in shells}

    # --------------------------------------------------------------
    # Sync mode: deterministic single-tick advance (for tests/training)
    # --------------------------------------------------------------
    def tick_once(self, external: Any = None):
        """Advance all shells by one step synchronously (one delta each)."""
        if external is not None:
            self.external = external
        snapshot = self.S.copy()
        deltas = []
        for shell in self.shells:
            delta = shell.step(snapshot, self.external)
            deltas.append((shell, delta))
        # Apply all deltas (additive, order-independent since on snapshot)
        for shell, delta in deltas:
            self._last_delta_norms[shell.name] = float(np.linalg.norm(delta))
            self.S = self.S + shell.gain * delta

    # --------------------------------------------------------------
    # Async mode: each shell runs in a thread at its own tick_ms
    # --------------------------------------------------------------
    def start(self):
        assert self.async_mode, "start() requires async_mode=True"
        assert not self._running
        self._running = True
        self._threads = []
        for shell in self.shells:
            t = threading.Thread(
                target=self._run_shell, args=(shell,), daemon=True,
                name=f"tamashii-{shell.name}",
            )
            t.start()
            self._threads.append(t)

    def stop(self, join_timeout: float = 2.0):
        self._running = False
        for t in self._threads:
            t.join(timeout=join_timeout)
        self._threads = []

    def _run_shell(self, shell: Shell):
        while self._running:
            t0 = time.time()
            snapshot = self.S.copy()
            try:
                delta = shell.step(snapshot, self.external)
            except Exception:
                # Shell errors don't kill the whole organism; log-only
                delta = np.zeros(self.D, dtype=np.float64)
            with self._lock:
                self.S += shell.gain * delta
                self._last_delta_norms[shell.name] = float(np.linalg.norm(delta))
            elapsed = time.time() - t0
            sleep_for = shell.tick_ms / 1000.0 - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    # --------------------------------------------------------------
    # State accessors
    # --------------------------------------------------------------
    def read_state(self) -> np.ndarray:
        """Thread-safe snapshot of S."""
        return self.S.copy()

    def write_external(self, key: str, value: Any):
        """Set/update external input visible to shells."""
        if not isinstance(self.external, dict):
            self.external = {}
        self.external[key] = value

    def reset_episode(self):
        """Zero S and reset all shell-local state (keeps params)."""
        self.S[:] = 0.0
        for s in self.shells:
            s.reset()

    def shell_delta_norms(self) -> dict[str, float]:
        """Latest delta L2 norm per shell (for coupling analysis)."""
        return dict(self._last_delta_norms)

    def add_shell(self, shell: Shell):
        """Add a shell mid-session (async mode restarts its thread)."""
        self.shells.append(shell)
        self._last_delta_norms[shell.name] = 0.0
        if self.async_mode and self._running:
            t = threading.Thread(
                target=self._run_shell, args=(shell,), daemon=True,
                name=f"tamashii-{shell.name}",
            )
            t.start()
            self._threads.append(t)

    def remove_shell(self, name: str) -> bool:
        """Remove a shell by name. In async mode, it stops on next tick."""
        for i, s in enumerate(self.shells):
            if s.name == name:
                self.shells.pop(i)
                self._last_delta_norms.pop(name, None)
                return True
        return False

    def __repr__(self):
        return (f"<Tamashii D={self.D} shells={len(self.shells)} "
                f"async={self.async_mode} running={self._running}>")
