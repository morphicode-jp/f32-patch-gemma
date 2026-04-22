"""Shell ABC — base class for all tamashii shells.

A shell is an independent module that reads from shared state S
and writes an additive delta back. Shells never call each other.
LaD (Logic-as-Data): all parameters loadable from JSON.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Shell(ABC):
    """Abstract shell. Subclass and implement `step`.

    Attributes set from config:
      name            : str, shell identifier
      tick_ms         : float, target period (ms)
      gain            : float, multiplier applied to delta when adding to S
      input_indices   : np.ndarray[int], which S dims this shell reads
      output_indices  : np.ndarray[int], which S dims this shell writes (via delta)
      params          : dict, shell-specific parameters (LaD)
    """

    def __init__(self, config: dict):
        self.name: str = config["shell_name"]
        self.tick_ms: float = float(config.get("tick_ms", 100.0))
        self.gain: float = float(config.get("gain", 1.0))
        self.input_indices = np.asarray(config.get("input_indices", []), dtype=int)
        self.output_indices = np.asarray(config.get("output_indices", []), dtype=int)
        self.params: dict = dict(config.get("params", {}))
        self._state: dict = {}
        # [M1] Hierarchy layer (0..5) + Dale's law sign for ISS inhibition metric
        # paper §4.1: H=6 deep hierarchy is required for abstract reasoning
        self.layer: int = int(config.get("layer", 0))
        # +1 excitatory, -1 inhibitory (applied to gain direction)
        self.shell_sign: int = int(config.get("shell_sign", +1))

    @classmethod
    def from_json(cls, path: str) -> "Shell":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(cfg)

    @abstractmethod
    def step(self, S_snapshot: np.ndarray, external: Any = None) -> np.ndarray:
        """Compute additive delta for S.

        Args:
            S_snapshot: read-only copy of current shared state (D,)
            external:   optional external input (sensors, world state, etc.)

        Returns:
            delta: np.ndarray (D,), mostly zeros except in output_indices.
                   Will be added as `S += gain * delta`.
        """
        ...

    def reset(self):
        """Reset internal shell state (between episodes)."""
        self._state = {}

    def __repr__(self):
        return f"<Shell {self.name} tick={self.tick_ms}ms gain={self.gain}>"


def load_shell_from_config(
    shell_cls: type, base_json: str, trained_json: str | None = None
) -> Shell:
    """Load a shell from base + optional trained-params JSON overlay.

    Precedence: trained > base. trained JSON can override `params`.
    """
    with open(base_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if trained_json and os.path.exists(trained_json):
        with open(trained_json, "r", encoding="utf-8") as f:
            trained = json.load(f)
        if "params" in trained:
            cfg.setdefault("params", {}).update(trained["params"])
    return shell_cls(cfg)
