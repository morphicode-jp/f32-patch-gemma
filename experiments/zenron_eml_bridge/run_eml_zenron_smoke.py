"""Zenron x EML bridge v8 smoke benchmark.

Goal:
  Show the smallest useful version of the claim:
    EML gives a complete expression substrate, but raw search is heavy.
    LaD witness macros, mined macros, and Kathara sharing make the same
    substrate searchable.

This experiment is intentionally self-contained under experiments/. It does
not import or modify twelve.agent, Reigen, or production optimizers.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


RUN_SEED = 20260426
N_TRIALS = 5
SUCCESS_LOSS = 1.0e-8
SIZE_PENALTY = 1.0e-10
MAX_DEPTH = 18
RANDOM_TREE_DEPTH = 4
GENERATIONS = 240
POPULATION_SIZE = 12


KATHARA_NEIGHBORS = {
    i: sorted({(i - 1) % 12, (i + 1) % 12, (i - 4) % 12, (i + 4) % 12, (i + 6) % 12})
    for i in range(12)
}


@dataclass(frozen=True)
class Expr:
    op: str
    left: "Expr | None" = None
    right: "Expr | None" = None

    def is_leaf(self) -> bool:
        return self.op in {"one", "x"}


@dataclass(frozen=True)
class Benchmark:
    name: str
    fn: Callable[[np.ndarray], np.ndarray] | None
    domain: tuple[float, float]
    witness: Expr | None
    supported: bool = True
    unsupported_reason: str = ""


@dataclass(frozen=True)
class Mode:
    name: str
    label: str
    use_lad: bool
    use_kathara: bool
    memory_policy: str = "none"  # none | witness | mined
    updates_macro_bank: bool = False
    memory_weighted: bool = True
    seed_memory: bool = True
    learns_certificates: bool = True
    family: str = "main"


class MacroBank:
    """Tiny LaD-style macro memory mined from successful EML expressions."""

    def __init__(self) -> None:
        self._records: dict[str, dict] = {}

    def _sorted_records(self) -> list[dict]:
        return sorted(
            self._records.values(),
            key=lambda row: (row["depth"], row["size"], row["expr_str"]),
        )

    def _priority(self, row: dict) -> int:
        if row["expr_str"] == "1":
            return 1
        if row["expr_str"] == "x":
            return 2

        kinds = set(row.get("kinds", []))
        source_bonus = min(6, 2 * len(row["sources"]))
        learn_bonus = min(6, row["learn_count"])
        full_bonus = 8 if kinds & {"full", "certificate"} else 0
        certificate_bonus = 4 if "certificate" in kinds else 0
        shallow_bonus = max(0, 5 - row["depth"])
        composite_bonus = 3 if row["size"] > 3 else 0
        return min(20, 1 + source_bonus + learn_bonus + full_bonus + certificate_bonus + shallow_bonus + composite_bonus)

    def expressions(self) -> list[Expr]:
        records = self._sorted_records()
        exprs = [row["expr"] for row in records]
        return [X] + exprs if X not in exprs else exprs

    def weighted_expressions(self, limit: int = 128) -> list[Expr]:
        records = self._sorted_records()
        out = [X]

        for row in sorted(
            records,
            key=lambda item: (-self._priority(item), item["depth"], item["size"], item["expr_str"]),
        ):
            if row["expr"] == X:
                continue
            out.extend([row["expr"]] * self._priority(row))
            if len(out) >= limit:
                return out[:limit]

        seen = {expr_to_str(expr) for expr in out}
        for row in records:
            if row["expr_str"] not in seen:
                out.append(row["expr"])
                seen.add(row["expr_str"])
            if len(out) >= limit:
                break
        return out[:limit]

    def _add_expr(self, expr: Expr, source_target: str, *, kind: str) -> bool:
        key = expr_to_str(expr)
        added = False
        if key not in self._records:
            self._records[key] = {
                "expr": expr,
                "expr_str": key,
                "depth": expr_depth(expr),
                "size": expr_size(expr),
                "sources": [],
                "kinds": [],
                "learn_count": 0,
            }
            added = True
        record = self._records[key]
        if source_target not in record["sources"]:
            record["sources"].append(source_target)
        if kind not in record["kinds"]:
            record["kinds"].append(kind)
        record["learn_count"] += 1
        return added

    def learn_from(
        self,
        expr: Expr,
        source_target: str,
        *,
        max_depth: int = 5,
        full_kind: str = "full",
        part_kind: str = "part",
    ) -> int:
        added = 0
        # The full successful expression is itself a useful macro, even when
        # it is too deep to be mined as a small reusable part.
        if self._add_expr(expr, source_target, kind=full_kind):
            added += 1
        for sub in all_subtrees(expr):
            if expr_depth(sub) > max_depth:
                continue
            if self._add_expr(sub, source_target, kind=part_kind):
                added += 1
        return added

    def summary(self) -> list[dict]:
        return [
            {
                "expr": row["expr_str"],
                "depth": row["depth"],
                "size": row["size"],
                "sources": row["sources"],
                "kinds": row.get("kinds", []),
                "learn_count": row["learn_count"],
                "priority": self._priority(row),
            }
            for row in self._sorted_records()
        ]

    def __len__(self) -> int:
        return len(self._records)


ONE = Expr("one")
X = Expr("x")


def eml(left: Expr, right: Expr) -> Expr:
    return Expr("eml", left, right)


def eml_exp(a: Expr) -> Expr:
    # exp(a) = E(a, 1)
    return eml(a, ONE)


def eml_log(a: Expr) -> Expr:
    # log(a) = E(1, E(E(1, a), 1))
    return eml(ONE, eml(eml(ONE, a), ONE))


def eml_one_minus(a: Expr) -> Expr:
    # 1 - a = E(log(1), exp(a)). This avoids the log(0) singularity that
    # appears in the textbook 0-a construction during numeric evaluation.
    return eml(eml_log(ONE), eml_exp(a))


def eml_sub(a: Expr, b: Expr) -> Expr:
    # a - b = E(log(a), exp(b))
    return eml(eml_log(a), eml_exp(b))


def eml_neg(a: Expr) -> Expr:
    # -a = (1-a)-1, kept away from log(0) for typical non-unit inputs.
    return eml_sub(eml_one_minus(a), ONE)


def eml_add(a: Expr, b: Expr) -> Expr:
    return eml_sub(a, eml_neg(b))


def eml_mul(a: Expr, b: Expr) -> Expr:
    return eml_exp(eml_add(eml_log(a), eml_log(b)))


def eml_square(a: Expr) -> Expr:
    return eml_mul(a, a)


def expr_size(e: Expr) -> int:
    if e.is_leaf():
        return 1
    return 1 + expr_size(e.left) + expr_size(e.right)  # type: ignore[arg-type]


def expr_depth(e: Expr) -> int:
    if e.is_leaf():
        return 0
    return 1 + max(expr_depth(e.left), expr_depth(e.right))  # type: ignore[arg-type]


def expr_to_str(e: Expr) -> str:
    if e.op == "one":
        return "1"
    if e.op == "x":
        return "x"
    return f"E({expr_to_str(e.left)}, {expr_to_str(e.right)})"  # type: ignore[arg-type]


def eval_expr(e: Expr, x: np.ndarray) -> np.ndarray:
    z = x.astype(np.complex128)
    if e.op == "one":
        return np.ones_like(z)
    if e.op == "x":
        return z

    left = eval_expr(e.left, x)  # type: ignore[arg-type]
    right = eval_expr(e.right, x)  # type: ignore[arg-type]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.exp(left) - np.log(right)


def loss_parts(e: Expr, xs: np.ndarray, target: np.ndarray) -> dict:
    try:
        y = eval_expr(e, xs)
    except Exception as exc:
        return {
            "mse": float("inf"),
            "complex_mse": float("inf"),
            "size_penalty": SIZE_PENALTY * expr_size(e),
            "total": float("inf"),
            "error": type(exc).__name__,
        }

    if not np.all(np.isfinite(y.real)) or not np.all(np.isfinite(y.imag)):
        return {
            "mse": float("inf"),
            "complex_mse": float("inf"),
            "size_penalty": SIZE_PENALTY * expr_size(e),
            "total": float("inf"),
            "error": "non_finite",
        }

    with np.errstate(over="ignore", invalid="ignore"):
        mse = float(np.mean((y.real - target) ** 2))
        complex_mse = float(np.mean(y.imag ** 2))

    if not math.isfinite(mse) or not math.isfinite(complex_mse):
        return {
            "mse": float("inf"),
            "complex_mse": float("inf"),
            "size_penalty": SIZE_PENALTY * expr_size(e),
            "total": float("inf"),
            "error": "overflow",
        }

    size_penalty = SIZE_PENALTY * expr_size(e)
    return {
        "mse": mse,
        "complex_mse": complex_mse,
        "size_penalty": size_penalty,
        "total": mse + complex_mse + size_penalty,
        "error": None,
    }


def score_expr(e: Expr, xs: np.ndarray, target: np.ndarray) -> float:
    total = loss_parts(e, xs, target)["total"]
    if not math.isfinite(total):
        return -1.0e9
    return -math.log10(total + 1.0e-300)


def random_expr(rng: random.Random, max_depth: int) -> Expr:
    if max_depth <= 0 or rng.random() < 0.35:
        return X if rng.random() < 0.55 else ONE
    return eml(random_expr(rng, max_depth - 1), random_expr(rng, max_depth - 1))


def paths(e: Expr, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    out = [prefix]
    if not e.is_leaf():
        out.extend(paths(e.left, prefix + (0,)))  # type: ignore[arg-type]
        out.extend(paths(e.right, prefix + (1,)))  # type: ignore[arg-type]
    return out


def get_at(e: Expr, path: tuple[int, ...]) -> Expr:
    cur = e
    for step in path:
        cur = cur.left if step == 0 else cur.right  # type: ignore[assignment]
        if cur is None:
            raise ValueError("bad expression path")
    return cur


def replace_at(e: Expr, path: tuple[int, ...], new_subtree: Expr) -> Expr:
    if not path:
        return new_subtree
    if e.is_leaf():
        return e
    if path[0] == 0:
        return eml(replace_at(e.left, path[1:], new_subtree), e.right)  # type: ignore[arg-type]
    return eml(e.left, replace_at(e.right, path[1:], new_subtree))  # type: ignore[arg-type]


def all_subtrees(e: Expr) -> Iterable[Expr]:
    for path in paths(e):
        yield get_at(e, path)


def clamp_depth(e: Expr, rng: random.Random, max_depth: int = MAX_DEPTH) -> Expr:
    if expr_depth(e) <= max_depth:
        return e
    return random_expr(rng, RANDOM_TREE_DEPTH)


def witness_memory() -> list[Expr]:
    # All macros are still pure EML trees. LaD stores them as reusable witnesses.
    return [
        X,
        eml_exp(X),
        eml_log(X),
        eml_exp(eml_log(X)),
        eml_log(eml_exp(X)),
        eml_square(X),
    ]


def memory_for_mode(mode: Mode, macro_bank: MacroBank | None) -> list[Expr]:
    if mode.memory_policy == "witness":
        return witness_memory()
    if mode.memory_policy == "mined":
        if macro_bank is None:
            return [X]
        if mode.memory_weighted:
            return macro_bank.weighted_expressions()
        return macro_bank.expressions()
    return []


def dedupe_memory(memory: list[Expr]) -> list[Expr]:
    out = []
    seen = set()
    for expr in memory:
        key = expr_to_str(expr)
        if key in seen:
            continue
        out.append(expr)
        seen.add(key)
    return out


def refresh_memory(
    memory: list[Expr],
    elites: list[Expr],
    mode: Mode,
    macro_bank: MacroBank | None,
    limit: int = 160,
) -> list[Expr]:
    local = update_memory(dedupe_memory(memory), elites, limit=64)
    if mode.memory_policy != "mined" or macro_bank is None:
        return local

    # Keep the global mined bank policy while still letting the current run
    # inject fresh elite subtrees into the local LaD scratchpad.
    global_memory = macro_bank.weighted_expressions(limit=128) if mode.memory_weighted else macro_bank.expressions()
    seen = {expr_to_str(expr) for expr in global_memory}
    out = list(global_memory)
    for expr in local:
        key = expr_to_str(expr)
        if key not in seen:
            out.append(expr)
            seen.add(key)
        if len(out) >= limit:
            break
    return out[:limit]


def seed_population(rng: random.Random, mode: Mode, memory: list[Expr]) -> list[Expr]:
    population = [X, ONE]
    if mode.use_lad and mode.seed_memory and memory:
        for expr in dedupe_memory(memory):
            if expr in population:
                continue
            population.append(expr)
            if len(population) >= POPULATION_SIZE:
                return population[:POPULATION_SIZE]

    population.extend(random_expr(rng, 2) for _ in range(POPULATION_SIZE - len(population)))
    return population


def update_memory(memory: list[Expr], elites: list[Expr], limit: int = 32) -> list[Expr]:
    seen = {expr_to_str(e) for e in memory}
    for elite in elites:
        for sub in all_subtrees(elite):
            if expr_depth(sub) <= 4:
                key = expr_to_str(sub)
                if key not in seen:
                    seen.add(key)
                    memory.append(sub)
    memory.sort(key=lambda e: (expr_depth(e), expr_size(e), expr_to_str(e)))
    return memory[:limit]


def perturb(e: Expr, rng: random.Random, mode: Mode, memory: list[Expr]) -> Expr:
    actions = ["replace", "swap", "wrap_exp", "wrap_left", "hoist"]
    if mode.use_lad:
        actions.extend([
            "lad_exp",
            "lad_log",
            "lad_square",
            "lad_add",
            "lad_add_current",
            "lad_mul",
            "lad_mul_current",
            "insert_memory",
            "insert_memory",
        ])
    action = rng.choice(actions)

    if action == "replace":
        path = rng.choice(paths(e))
        remaining_depth = min(RANDOM_TREE_DEPTH, max(0, MAX_DEPTH - len(path)))
        new_tree = random_expr(rng, rng.randint(0, remaining_depth))
        return clamp_depth(replace_at(e, path, new_tree), rng)

    if action == "swap" and not e.is_leaf():
        internal_paths = [path for path in paths(e) if not get_at(e, path).is_leaf()]
        path = rng.choice(internal_paths)
        sub = get_at(e, path)
        return replace_at(e, path, eml(sub.right, sub.left))  # type: ignore[arg-type]

    if action == "wrap_exp":
        return clamp_depth(eml_exp(e), rng)

    if action == "wrap_left":
        return clamp_depth(eml(ONE, e), rng)

    if action == "hoist" and not e.is_leaf():
        internal_paths = [path for path in paths(e) if not get_at(e, path).is_leaf()]
        path = rng.choice(internal_paths)
        sub = get_at(e, path)
        return replace_at(e, path, sub.left if rng.random() < 0.5 else sub.right)  # type: ignore[arg-type]

    if action == "lad_exp":
        path = rng.choice(paths(e))
        return clamp_depth(replace_at(e, path, eml_exp(get_at(e, path))), rng)

    if action == "lad_log":
        path = rng.choice(paths(e))
        return clamp_depth(replace_at(e, path, eml_log(get_at(e, path))), rng)

    if action == "lad_square":
        path = rng.choice(paths(e))
        return clamp_depth(replace_at(e, path, eml_square(get_at(e, path))), rng)

    if action == "lad_add" and memory:
        left = rng.choice(memory)
        right = rng.choice(memory + [e])
        return clamp_depth(eml_add(left, right), rng)

    if action == "lad_add_current" and memory:
        other = rng.choice(memory)
        return clamp_depth(eml_add(e, other) if rng.random() < 0.5 else eml_add(other, e), rng)

    if action == "lad_mul" and memory:
        left = rng.choice(memory)
        right = rng.choice(memory + [e])
        return clamp_depth(eml_mul(left, right), rng)

    if action == "lad_mul_current" and memory:
        other = rng.choice(memory)
        return clamp_depth(eml_mul(e, other) if rng.random() < 0.5 else eml_mul(other, e), rng)

    if action == "insert_memory" and memory:
        path = rng.choice(paths(e))
        return clamp_depth(replace_at(e, path, rng.choice(memory)), rng)

    return random_expr(rng, RANDOM_TREE_DEPTH)


def train_test_xs(domain: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = domain
    train = np.linspace(lo, hi, 64, dtype=np.float64)
    step = (hi - lo) / 127.0
    test = np.linspace(lo + 0.5 * step, hi - 0.5 * step, 64, dtype=np.float64)
    return train, test


def make_benchmarks() -> list[Benchmark]:
    return [
        Benchmark("exp(x)", np.exp, (0.25, 3.0), eml_exp(X)),
        Benchmark("log(x)", np.log, (0.25, 3.0), eml_log(X)),
        Benchmark("log(log(x))", lambda x: np.log(np.log(x)), (1.25, 3.0), eml_log(eml_log(X))),
        Benchmark("x^2", lambda x: x ** 2, (0.25, 3.0), eml_square(X)),
        Benchmark("x*log(x)", lambda x: x * np.log(x), (1.25, 3.0), eml_mul(X, eml_log(X))),
        Benchmark("exp(log(x))", lambda x: np.exp(np.log(x)), (0.25, 3.0), eml_exp(eml_log(X))),
        Benchmark("log(exp(x))", lambda x: np.log(np.exp(x)), (0.25, 3.0), eml_log(eml_exp(X))),
        Benchmark(
            "exp(x)+log(x)",
            lambda x: np.exp(x) + np.log(x),
            (1.25, 3.0),
            eml_add(eml_exp(X), eml_log(X)),
        ),
        Benchmark(
            "x^2+log(x)",
            lambda x: x ** 2 + np.log(x),
            (1.25, 3.0),
            eml_add(eml_square(X), eml_log(X)),
        ),
        # New targets (2026-04-26): EML universality 主張を sin/cos/tanh で追試。
        # witness 未指定 (raw 探索 + bridge mining で発見できるか測る)。
        Benchmark("sin(x)", np.sin, (0.25, 1.5), None),
        Benchmark("cos(x)", np.cos, (0.25, 1.5), None),
        Benchmark("tanh(x)", np.tanh, (0.25, 2.0), None),
    ]


MODES = [
    Mode("raw_eml", "raw EML", use_lad=False, use_kathara=False),
    Mode("lad_seeded", "EML + hand LaD", use_lad=True, use_kathara=False, memory_policy="witness"),
    Mode("kathara_lad", "EML + Kathara + hand LaD", use_lad=True, use_kathara=True, memory_policy="witness"),
    Mode("macro_mined", "EML + mined LaD", use_lad=True, use_kathara=False,
         memory_policy="mined", updates_macro_bank=True),
    Mode("kathara_mined", "EML + Kathara + mined LaD", use_lad=True, use_kathara=True,
         memory_policy="mined", updates_macro_bank=True),
    Mode("mined_no_weight", "Ablation: mined LaD without weighted memory", use_lad=True, use_kathara=False,
         memory_policy="mined", updates_macro_bank=True, memory_weighted=False, family="ablation"),
    Mode("mined_no_seed", "Ablation: mined LaD without memory seeding", use_lad=True, use_kathara=False,
         memory_policy="mined", updates_macro_bank=True, seed_memory=False, family="ablation"),
    Mode("mined_no_cert", "Ablation: mined LaD without compact certificates", use_lad=True, use_kathara=False,
         memory_policy="mined", updates_macro_bank=True, learns_certificates=False, family="ablation"),
    Mode("kathara_mined_no_weight", "Ablation: Kathara mined without weighted memory", use_lad=True, use_kathara=True,
         memory_policy="mined", updates_macro_bank=True, memory_weighted=False, family="ablation"),
    Mode("kathara_mined_no_seed", "Ablation: Kathara mined without memory seeding", use_lad=True, use_kathara=True,
         memory_policy="mined", updates_macro_bank=True, seed_memory=False, family="ablation"),
    Mode("kathara_mined_no_cert", "Ablation: Kathara mined without compact certificates", use_lad=True, use_kathara=True,
         memory_policy="mined", updates_macro_bank=True, learns_certificates=False, family="ablation"),
]


ABLATION_TARGETS = ["log(log(x))", "x*log(x)", "exp(x)+log(x)", "x^2+log(x)"]
ABLATION_MODES = [
    "macro_mined",
    "mined_no_weight",
    "mined_no_seed",
    "mined_no_cert",
    "kathara_mined",
    "kathara_mined_no_weight",
    "kathara_mined_no_seed",
    "kathara_mined_no_cert",
]


def evaluate_best(e: Expr, bench: Benchmark) -> tuple[dict, dict]:
    if bench.fn is None:
        raise ValueError("unsupported benchmark has no target function")
    train_xs, test_xs = train_test_xs(bench.domain)
    train_target = bench.fn(train_xs)
    test_target = bench.fn(test_xs)
    return loss_parts(e, train_xs, train_target), loss_parts(e, test_xs, test_target)


def run_mode(
    bench: Benchmark,
    bench_index: int,
    mode: Mode,
    trial_index: int,
    macro_bank: MacroBank | None = None,
) -> dict:
    if bench.fn is None:
        return {
            "target": bench.name,
            "mode": mode.name,
            "status": "unsupported",
            "unsupported_reason": bench.unsupported_reason,
        }

    seed = RUN_SEED + 100000 * trial_index + 1009 * bench_index + 97 * MODES.index(mode)
    rng = random.Random(seed)
    train_xs, test_xs = train_test_xs(bench.domain)
    train_target = bench.fn(train_xs)
    test_target = bench.fn(test_xs)

    macro_bank_before = len(macro_bank) if macro_bank is not None else None
    memory = memory_for_mode(mode, macro_bank) if mode.use_lad else []
    population = seed_population(rng, mode, memory)
    scores = [score_expr(e, train_xs, train_target) for e in population]
    evals = len(population)
    history = []
    best_generation = 0

    for gen in range(GENERATIONS):
        new_population: list[Expr] = []
        new_scores: list[float] = []

        for i, current in enumerate(population):
            candidates = [current, perturb(current, rng, mode, memory)]

            if mode.use_kathara:
                for j in KATHARA_NEIGHBORS[i]:
                    candidates.append(population[j])
                    candidates.append(perturb(population[j], rng, mode, memory))

            candidate_scores = [score_expr(candidate, train_xs, train_target) for candidate in candidates]
            evals += len(candidates) - 1
            winner = int(np.argmax(candidate_scores))
            new_population.append(candidates[winner])
            new_scores.append(candidate_scores[winner])

        population = new_population
        scores = new_scores
        elite_order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        elite = population[elite_order[0]]
        best_generation = gen + 1

        if mode.use_lad:
            memory = refresh_memory(memory, [population[idx] for idx in elite_order[:4]], mode, macro_bank)

        train_loss, test_loss = evaluate_best(elite, bench)
        if gen % 20 == 0 or (train_loss["total"] < SUCCESS_LOSS and test_loss["total"] < SUCCESS_LOSS):
            history.append(
                {
                    "generation": gen + 1,
                    "best_score": float(scores[elite_order[0]]),
                    "train_total": train_loss["total"],
                    "test_total": test_loss["total"],
                    "best_expr": expr_to_str(elite),
                    "depth": expr_depth(elite),
                    "size": expr_size(elite),
                    "memory_size": len(memory),
                }
            )

        if train_loss["total"] < SUCCESS_LOSS and test_loss["total"] < SUCCESS_LOSS:
            break

    best_idx = int(np.argmax(scores))
    best = population[best_idx]
    train_loss, test_loss = evaluate_best(best, bench)
    success = train_loss["total"] < SUCCESS_LOSS and test_loss["total"] < SUCCESS_LOSS
    status = "success" if success else "miss"
    macro_added = 0
    if success and mode.updates_macro_bank and macro_bank is not None:
        macro_added = macro_bank.learn_from(best, bench.name)
        if mode.learns_certificates and bench.witness is not None:
            macro_added += macro_bank.learn_from(
                bench.witness,
                f"{bench.name}:compact_certificate",
                full_kind="certificate",
                part_kind="certificate_part",
            )
    macro_bank_after = len(macro_bank) if macro_bank is not None else None

    return {
        "target": bench.name,
        "trial": trial_index,
        "mode": mode.name,
        "mode_label": mode.label,
        "mode_family": mode.family,
        "mode_features": {
            "use_lad": mode.use_lad,
            "use_kathara": mode.use_kathara,
            "memory_policy": mode.memory_policy,
            "updates_macro_bank": mode.updates_macro_bank,
            "memory_weighted": mode.memory_weighted,
            "seed_memory": mode.seed_memory,
            "learns_certificates": mode.learns_certificates,
        },
        "status": status,
        "success": success,
        "seed": seed,
        "generations_run": best_generation,
        "evals": evals,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "best_score": float(scores[best_idx]),
        "best_expr": expr_to_str(best),
        "best_depth": expr_depth(best),
        "best_size": expr_size(best),
        "macro_bank_before": macro_bank_before,
        "macro_bank_after": macro_bank_after,
        "macro_added": macro_added,
        "history": history,
    }


def witness_report(benchmarks: list[Benchmark]) -> dict:
    out = {}
    for bench in benchmarks:
        if not bench.supported or bench.witness is None or bench.fn is None:
            out[bench.name] = {
                "supported": False,
                "unsupported_reason": bench.unsupported_reason,
            }
            continue
        train_loss, test_loss = evaluate_best(bench.witness, bench)
        out[bench.name] = {
            "supported": True,
            "expr": expr_to_str(bench.witness),
            "depth": expr_depth(bench.witness),
            "size": expr_size(bench.witness),
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
    return out


def add_raw_speedups(runs: list[dict]) -> None:
    by_target: dict[tuple[str, object], dict[str, dict]] = {}
    for run in runs:
        if run.get("status") == "unsupported":
            continue
        by_target.setdefault((run["target"], run.get("trial")), {})[run["mode"]] = run

    for target_runs in by_target.values():
        raw = target_runs.get("raw_eml")
        if not raw:
            for run in target_runs.values():
                run["speedup_vs_raw_evals"] = None
                run["speedup_vs_raw_generations"] = None
                run["raw_baseline_status"] = "missing"
            continue

        raw_evals = max(1, int(raw.get("evals", 1)))
        raw_gens = max(1, int(raw.get("generations_run", 1)))
        raw_status = raw.get("status", "unknown")

        if raw_status != "success":
            for run in target_runs.values():
                run["raw_baseline_status"] = raw_status
                if run.get("status") == "success":
                    run["speedup_vs_raw_evals"] = raw_evals / max(1, int(run["evals"]))
                    run["speedup_vs_raw_generations"] = raw_gens / max(1, int(run["generations_run"]))
                elif run is raw:
                    run["speedup_vs_raw_evals"] = 1.0
                    run["speedup_vs_raw_generations"] = 1.0
                else:
                    run["speedup_vs_raw_evals"] = None
                    run["speedup_vs_raw_generations"] = None
            continue

        for run in target_runs.values():
            run["raw_baseline_status"] = raw_status
            if run.get("status") == "success":
                run["speedup_vs_raw_evals"] = raw_evals / max(1, int(run["evals"]))
                run["speedup_vs_raw_generations"] = raw_gens / max(1, int(run["generations_run"]))
            else:
                run["speedup_vs_raw_evals"] = None
                run["speedup_vs_raw_generations"] = None


def format_float(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(number):
        return "inf"
    if math.isnan(number):
        return "nan"
    if abs(number) >= 1.0e4 or (0 < abs(number) < 1.0e-3):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def aggregate_runs(runs: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        if run.get("status") == "unsupported":
            continue
        groups.setdefault((run["target"], run["mode"]), []).append(run)

    out = []
    modes_by_name = {mode.name: mode for mode in MODES}
    for (target, mode), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        successes = [row for row in rows if row.get("success")]
        mode_config = modes_by_name.get(mode)
        out.append(
            {
                "target": target,
                "mode": mode,
                "mode_label": mode_config.label if mode_config else mode,
                "mode_family": mode_config.family if mode_config else "unknown",
                "trials": len(rows),
                "successes": len(successes),
                "success_rate": len(successes) / max(1, len(rows)),
                "mean_generations_success": mean_or_none([float(row["generations_run"]) for row in successes]),
                "mean_evals_success": mean_or_none([float(row["evals"]) for row in successes]),
                "mean_train_loss_success": mean_or_none([float(row["train_loss"]["total"]) for row in successes]),
                "mean_test_loss_success": mean_or_none([float(row["test_loss"]["total"]) for row in successes]),
                "mean_speedup_vs_raw_evals_success": mean_or_none(
                    [
                        float(row["speedup_vs_raw_evals"])
                        for row in successes
                        if row.get("speedup_vs_raw_evals") is not None
                    ]
                ),
                "best_expr_examples": sorted({row["best_expr"] for row in successes})[:3],
            }
        )
    return out


def aggregate_map(results: dict) -> dict[tuple[str, str], dict]:
    return {(row["target"], row["mode"]): row for row in results["aggregates"]}


def aggregate_cell(row: dict | None) -> str:
    if row is None:
        return "-"
    if row["successes"] == 0:
        return f"0/{row['trials']}"
    return "{successes}/{trials}, {gens} gen / {evals} evals".format(
        successes=row["successes"],
        trials=row["trials"],
        gens=format_float(row["mean_generations_success"], 1),
        evals=format_float(row["mean_evals_success"], 1),
    )


def make_markdown(results: dict) -> str:
    lines = [
        "# Zenron-EML Smoke Benchmark v8",
        "",
        "Command:",
        "",
        "```powershell",
        "python experiments/zenron_eml_bridge/run_eml_zenron_smoke.py",
        "```",
        "",
        "## Aggregate Result Table",
        "",
        "| target | mode | success | mean gens | mean evals | mean test loss | eval speedup vs raw budget | best expr examples |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in results["aggregates"]:
        examples = "<br>".join(f"`{expr}`" for expr in row["best_expr_examples"]) or "-"
        lines.append(
            "| {target} | {mode} | {successes}/{trials} | {gens} | {evals} | {test} | {speed}x | {examples} |".format(
                target=row["target"],
                mode=row["mode"],
                successes=row["successes"],
                trials=row["trials"],
                gens=format_float(row["mean_generations_success"], 1),
                evals=format_float(row["mean_evals_success"], 1),
                test=format_float(row["mean_test_loss_success"], 2),
                speed=format_float(row["mean_speedup_vs_raw_evals_success"], 2),
                examples=examples,
            )
        )

    by_aggregate = aggregate_map(results)
    lines.extend(
        [
            "",
            "## Ablation Focus Table",
            "",
            "This table isolates which Zenron-EML accelerators matter on hard targets.",
            "",
            "| target | macro_mined | no weight | no seed | no certificate | kathara_mined | kathara no weight | kathara no seed | kathara no certificate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for target in ABLATION_TARGETS:
        cells = [aggregate_cell(by_aggregate.get((target, mode))) for mode in ABLATION_MODES]
        lines.append(f"| `{target}` | " + " | ".join(cells) + " |")

    unsupported = [run for run in results["runs"] if run.get("status") == "unsupported"]
    if unsupported:
        lines.extend(["", "## Unsupported", ""])
        for run in unsupported:
            lines.append(f"- `{run['target']}`: {run['unsupported_reason']}")

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `raw_eml` uses only terminals `{1, x}` and `E(a,b)` mutations.",
            "- `lad_seeded` keeps the same pure EML representation, but starts with hand witness macros for `exp` and `log`.",
            "- `kathara_lad` adds the Kathara 12-node / 30-edge neighbor sharing loop on top of hand witness macros.",
            "- `macro_mined` starts without hand `exp/log` witnesses and grows a cross-target macro bank from successful expressions.",
            "- `kathara_mined` combines the mined macro bank with Kathara sharing.",
            "- v5 gives mined macros priority weights: repeated full successes and reusable subtrees are sampled more often.",
            "- v6 also seeds the starting population from LaD memory, so remembered subtrees begin on the Kathara workbench.",
            "- v7 distills compact EML certificates after a mined mode succeeds, turning deep discoveries into reusable smaller macros.",
            "- v8 adds ablation modes: `*_no_weight`, `*_no_seed`, and `*_no_cert`.",
            "- Aggregate rows are over fixed deterministic seeds.",
            "- If `raw_eml` misses in a trial, speedup is computed against the raw budget spent before the miss.",
            "- Unsupported rows are explicit misses, not hidden failures.",
            "",
            "## Mined Macro Banks",
            "",
        ]
    )

    display_banks = [bank for bank in results.get("macro_banks", []) if bank["mode"] in {"macro_mined", "kathara_mined"}]
    for bank in display_banks:
        lines.append(f"### trial {bank['trial']} / {bank['mode']}")
        rows = bank["macros"][:24]
        if not rows:
            lines.append("")
            lines.append("(empty)")
            lines.append("")
            continue
        lines.append("")
        lines.append("| expr | depth | size | priority | kinds | sources |")
        lines.append("|---|---:|---:|---:|---|---|")
        for row in rows:
            sources = ", ".join(row["sources"])
            kinds = ", ".join(row.get("kinds", []))
            lines.append(f"| `{row['expr']}` | {row['depth']} | {row['size']} | {row.get('priority', '-')} | {kinds} | {sources} |")
        if len(bank["macros"]) > len(rows):
            lines.append(f"| ... | ... | ... | ... | ... | {len(bank['macros']) - len(rows)} more macros in JSON |")
        lines.append("")
    return "\n".join(lines) + "\n"


def print_console_summary(results: dict) -> None:
    print("=" * 78)
    print("Zenron x EML bridge v8 smoke benchmark")
    print("=" * 78)
    print("Primitive: E(a,b)=exp(a)-log(b), terminals: {1, x}")
    print("Modes: main comparisons plus mined/Kathara ablations")
    print(f"Trials: {results['metadata']['n_trials']} deterministic seeds")
    print(f"Kathara node 0 neighbors: {KATHARA_NEIGHBORS[0]}")
    print()

    print("Constructive EML witnesses:")
    for name, row in results["witnesses"].items():
        if not row.get("supported", False):
            print(f"  {name:14s} unsupported: {row['unsupported_reason']}")
            continue
        print(
            f"  {name:14s} depth={row['depth']} size={row['size']} "
            f"train={row['train_loss']['total']:.2e} test={row['test_loss']['total']:.2e}"
        )
        print(f"    {row['expr']}")

    print("\nAggregate search runs:")
    for row in results["aggregates"]:
        print(
            f"  {row['target']:12s} {row['mode']:14s} "
            f"success={row['successes']}/{row['trials']} "
            f"gens={format_float(row['mean_generations_success'], 1):>6s} "
            f"evals={format_float(row['mean_evals_success'], 1):>7s} "
            f"speed={format_float(row['mean_speedup_vs_raw_evals_success'], 2)}x"
        )

    for run in results["runs"]:
        if run.get("status") == "unsupported":
            print(f"  [UNSUPPORTED] {run['target']}: {run['unsupported_reason']}")

    print("\nMined macro banks:")
    for bank in results.get("macro_banks", []):
        rows = bank["macros"]
        preview = ", ".join(row["expr"] for row in rows[:6])
        suffix = "..." if len(rows) > 6 else ""
        print(f"  trial {bank['trial']} / {bank['mode']}: {len(rows)} macros {preview}{suffix}")


def mode_metadata(mode: Mode) -> dict:
    return {
        "name": mode.name,
        "label": mode.label,
        "family": mode.family,
        "use_lad": mode.use_lad,
        "use_kathara": mode.use_kathara,
        "memory_policy": mode.memory_policy,
        "updates_macro_bank": mode.updates_macro_bank,
        "memory_weighted": mode.memory_weighted,
        "seed_memory": mode.seed_memory,
        "learns_certificates": mode.learns_certificates,
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    benchmarks = make_benchmarks()
    runs: list[dict] = []
    macro_bank_summaries = []

    for trial_index in range(N_TRIALS):
        macro_banks = {
            mode.name: MacroBank()
            for mode in MODES
            if mode.updates_macro_bank
        }

        for bench_index, bench in enumerate(benchmarks):
            if not bench.supported:
                if trial_index == 0:
                    runs.append(
                        {
                            "target": bench.name,
                            "mode": "all",
                            "status": "unsupported",
                            "unsupported_reason": bench.unsupported_reason,
                        }
                    )
                continue
            for mode in MODES:
                runs.append(run_mode(bench, bench_index, mode, trial_index, macro_banks.get(mode.name)))

        for name, bank in macro_banks.items():
            macro_bank_summaries.append(
                {
                    "trial": trial_index,
                    "mode": name,
                    "macros": bank.summary(),
                }
            )

    add_raw_speedups(runs)
    aggregates = aggregate_runs(runs)
    results = {
        "metadata": {
            "seed": RUN_SEED,
            "n_trials": N_TRIALS,
            "success_loss": SUCCESS_LOSS,
            "generations": GENERATIONS,
            "population_size": POPULATION_SIZE,
            "max_depth": MAX_DEPTH,
            "modes": [mode.name for mode in MODES],
            "mode_configs": [mode_metadata(mode) for mode in MODES],
        },
        "witnesses": witness_report(benchmarks),
        "runs": runs,
        "aggregates": aggregates,
        "macro_banks": macro_bank_summaries,
    }

    out_json = here / "eml_zenron_smoke_result.json"
    out_md = here / "eml_zenron_smoke_summary.md"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(make_markdown(results), encoding="utf-8")

    print_console_summary(results)
    print()
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")
    print()
    print("Takeaway:")
    print("  EML is a compact expression substrate.")
    print("  Hand witnesses, mined macros, and Kathara sharing are the practical search accelerators.")


if __name__ == "__main__":
    main()
