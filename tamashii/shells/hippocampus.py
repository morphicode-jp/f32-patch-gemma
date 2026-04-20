"""Hippocampus shell — episodic trace memory + recall.

Biologically: hippocampus records event traces and reactivates similar
past states. Implementation: content-addressable memory with N slots.

1. Every tick, encode current S-slice into a query vector
2. Compare to stored slots → similarity scores
3. If MAX similarity < novel_threshold → store (displace LRU slot)
4. If any similarity > recall_threshold → reactivate that slot's
   payload (write to S workspace) = "this situation reminds me of..."
"""
from __future__ import annotations

import numpy as np

from tamashii.shell_base import Shell


class Hippocampus(Shell):
    """Episodic memory with similarity-based recall."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.query_slice = slice(
            int(self.params.get("query_start", 19)),
            int(self.params.get("query_end", 35)))
        self.trace_slice = slice(
            int(self.params.get("trace_start", 83)),
            int(self.params.get("trace_end", 147)))
        self.n_slots = int(self.params.get("n_slots", 8))
        self.slot_dim = int(self.params.get("slot_dim", 8))
        assert self.trace_slice.stop - self.trace_slice.start >= self.n_slots * self.slot_dim, \
            "trace_slice must hold n_slots * slot_dim dims"
        self.recall_threshold = float(self.params.get("recall_threshold", 0.8))
        self.store_threshold = float(self.params.get("store_threshold", 0.6))
        self.recall_strength = float(self.params.get("recall_strength", 0.2))
        # Autobiographical memory: if True, slots persist across reset_episode
        self.persistent_memory = bool(self.params.get("persistent_memory", False))

        # Per-shell memory (not in S): slot vectors (query space) + payloads (S-slice)
        self._slot_queries: list[np.ndarray] = []  # stored query vectors
        self._slot_payloads: list[np.ndarray] = []  # stored payloads (slot_dim each)
        self._usage: list[int] = []  # access count for LRU-ish
        self._step_count = 0
        # Recall event counter (how many times a stored slot was reactivated)
        self._recall_events = 0

    def step(self, S_snapshot: np.ndarray, external=None) -> np.ndarray:
        self._step_count += 1
        query = S_snapshot[self.query_slice].astype(np.float64)
        q_norm = np.linalg.norm(query) + 1e-6
        q_unit = query / q_norm

        delta = np.zeros_like(S_snapshot)

        # Compute similarities to stored slots (cosine)
        sims = []
        for stored in self._slot_queries:
            s_unit = stored / (np.linalg.norm(stored) + 1e-6)
            sims.append(float(np.dot(q_unit, s_unit)))
        max_sim = max(sims) if sims else 0.0
        best_idx = int(np.argmax(sims)) if sims else -1

        # Recall: if best match > recall_threshold, reactivate payload
        if max_sim > self.recall_threshold and best_idx >= 0:
            self._usage[best_idx] = self._step_count
            self._recall_events += 1
            payload = self._slot_payloads[best_idx]
            # Write payload to first slot of trace region (= currently active recall)
            t_start = self.trace_slice.start
            for i, val in enumerate(payload):
                target = t_start + i
                delta[target] = self.recall_strength * (val - S_snapshot[target])

        # Store: if novel enough (max sim below store_threshold) and slot budget remains
        if max_sim < self.store_threshold:
            payload = query[:self.slot_dim].copy()
            if len(self._slot_queries) < self.n_slots:
                # Append
                self._slot_queries.append(query.copy())
                self._slot_payloads.append(payload)
                self._usage.append(self._step_count)
            else:
                # LRU: replace least-recently-used
                lru = int(np.argmin(self._usage))
                self._slot_queries[lru] = query.copy()
                self._slot_payloads[lru] = payload
                self._usage[lru] = self._step_count

        # Mirror slot registry into trace_slice (for external visibility)
        # Layout: [slot0 payload | slot1 payload | ... | slot_{n-1} payload]
        t_start = self.trace_slice.start
        for s_i, payload in enumerate(self._slot_payloads):
            off = t_start + s_i * self.slot_dim
            for j, val in enumerate(payload):
                if off + j < self.trace_slice.stop:
                    delta[off + j] = 0.3 * (val - S_snapshot[off + j])  # gentle refresh

        return delta

    def reset(self):
        super().reset()
        # Autobiographical: preserve slot contents across episode reset
        if not self.persistent_memory:
            self._slot_queries = []
            self._slot_payloads = []
            self._usage = []
        # Step count always resets (within-episode reference)
        self._step_count = 0
        self._recall_events = 0

    def memory_stats(self) -> dict:
        return {
            "n_stored": len(self._slot_queries),
            "step_count": self._step_count,
            "recall_events": self._recall_events,
            "persistent": self.persistent_memory,
        }
