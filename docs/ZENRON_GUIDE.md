# Zenron — Practical Guide

## 最初にやること: 構造を見つけてproxyを作れ

```
Step 1: truth(結果への影響)とconnectivity(他との連動)を定義
Step 2: MS 20回eval → importance = sqrt(truth × connectivity) → 構造が見える
Step 3: 構造から近似指標(proxy)を作る → 本番evalの数万倍速い
Step 4: proxyで高速最適化 → 1回本番検証 → 完了
実証: 6分で完了（世界: GPU256台×3週間）
```

## The Formula

`x_i ← best(perturb(x_i), share(neighbors_i))` — applies to every domain.

## When to Use What

| x_i type | Use | Entry point |
|---|---|---|
| **Any params + eval_fn (recommended)** | **`MirrorAgent`** | **`MirrorAgent(eval_fn=f, param_ranges=r).run()`** |
| Anything (grid, string, tree) | `zenron_solve()` | `zenron_solve(x_init, eval_fn=f)` — perturb auto |
| Float list with ranges | `optimize()` | `optimize(f, ranges, time_budget=60)` — ≤5 params only |
| "Just optimize this folder" | `MirrorAgent` | `MirrorAgent(folder="x/").run()` |
| Combo | zenron finds structure → MA refines | |

## 3 Steps

**Step 1: Map** — what is x_i? eval_fn? (perturb auto-generated if omitted)

**Step 2: Choose pattern**

```python
# A — eval_fn only (simplest). perturb auto-generated.
zenron_solve(x_init, eval_fn=my_eval, time_budget=60)

# B — 12x parallel
zenron_solve(x_init, eval_fn=my_eval, time_budget=60, parallel=True)

# C — custom perturb (when you know better)
zenron_solve(x_init, perturb_fn=my_perturb, eval_fn=my_eval, parallel=True)

# D — module path (full parallel, no lambda)
kathara_solve(x_init, eval_fn_module="m", eval_fn_name="f", n_workers=12)

# E — MA fully automatic (recommended for 6+ params)
MirrorAgent(eval_fn=my_eval, param_ranges=[(0,1)]*N).run()

# F — MA folder mode
MirrorAgent(folder="my_problem/").run()
```

**Step 3: sigma control** — auto: 0.9× improving, 1.3× stagnant, reset 0.25 if >0.4. Parallel: neighbor sigma blend (0.7×self + 0.3×neighbor).

## AutoPerturb

`perturb_fn` is optional. `auto_perturb(eval_fn, x_init)` detects type + probes sensitivity → generates mutation operator.

| Type | Strategy | Proven |
|---|---|---|
| float_list | Gaussian noise, sensitivity-weighted | -0.55 (5D sphere) |
| int_list | Gaussian round + discrete swap | yes |
| grid (2D) | Random cell change, value range auto-expanded | 100% (3×3 target) |
| str_list | Random char from charset (auto-expanded) | 100% ("ZENRON") |
| string | Random char swap | yes |
| dict | Per-key type-aware mutation | yes |

## Safety Rules (parallel mode)

6 rules in `configs/kathara_safety_rules.json`: regression→rollback | 3-fail→switch | sigma-death→reset | time→finalize | explosion→rollback | collapse→inject. All evolvable.

## Common Mistakes

1. **Binary eval_fn** → use continuous 0-100, not 0/1
2. **Full re-eval per step** → use delta evaluation O(1)
3. **No deterministic rules** → learn certain patterns first, zenron on uncertain
4. **Lambda in parallel** → use importable functions (lambda auto-falls back to 1 worker)

## Key Files

| File | Purpose |
|---|---|
| `twelve/zenron.py` | Solver. perturb optional. `parallel=True` for 12-worker |
| `twelve/auto_perturb.py` | Type detection + sensitivity → perturb_fn |
| `twelve/agent/kathara_at.py` | 12-node coordinator |
| `twelve/agent/worker.py` | Child process: zenron + mailbox + SafetyHooks |
| `twelve/agent/safety_hooks.py` | LaD safety rules engine |

## See Also

- `twelve/agent/AT.md` — Full AT guide: 7 phases, AutoDream, EvalTool, Strategy, distributed
- `docs/ZENRON_APPLICATIONS.md` — 38 domains, engine internals
- `docs/全論.md` — Master theory: formula derivation, evidence, cosmology

---

## Appendix: MirrorScan 詳細 (from CLAUDE.md 移管、2026-04-21)

### Formula

```
x_i ← best( perturb(x_i), share(neighbors_i) )
```
"Perturb self, compare with neighbors, keep the better." DNA / galaxies / neurons all follow this.
Mapping: owl = (measurements, proxy, optimize); Sentinel = (owl, guard_fn, pivot); mimir = (owl, Reigen, scipy).

### MirrorScan — importance (3-layer + pairs)

```
truth[i]         = |corr(param_i, scores)|             # does it affect score?
connectivity[i]  = mean(|corr(param_i, param_j)|) j≠i  # does it co-move?
importance[i]    = (truth × max(connectivity, floor))^exp     # multiplication kills noise
```
Values from `configs/ma_meta_params.json`: exp=0.3064, floor=0.1411 (fallback hardcode: 0.5, 0.01). `_mp()` resolves JSON > hardcode.

| truth | conn | importance | meaning |
|---|---|---|---|
| hi | hi | **hi** | real structure — optimize this |
| hi | lo | lo | accidental correlation (overfit risk) |
| lo | hi | lo | co-moves, no effect |
| lo | lo | ~0 | dead dim — safe to ignore |

**Multiplication kills noise.** Core principle.

### Pair interaction (Layer 4)

```
interaction_imp[i,j] = (|corr(x_i*x_j, scores)| × max(|corr(x_i,x_j)|, floor))^exp
```
High-importance pairs → add `x_i × x_j` terms to proxy. Proven R² 0.38→0.79 on same 53 meas.

### Proxy generation

```
{zenron, zenron_interact, linear} × {raw, log} = up to 6 candidates
best R² wins (or force via force_proxy_type).
  zenron: importance-weighted   zenron_interact: +pair terms   linear: plain
  raw: linear systems           log: multiplicative (PPL, NN); requires same-sign scores
```
Proxy extracts structure not noise → stable under 100K+ opts. Extrapolation not guaranteed → use verify_fn. Dead dims compress search: 206D → 21D ≈ 10^185× reduction.

### Fragility (death-side dual)

```
isolation[i] = 1.0 - connectivity[i]
fragility[i] = (truth × max(isolation, floor))^exp   # active dims only
```
Important AND isolated = fragile. Proven on 306-LLM: `hidden_size` most fragile despite highest connectivity.
