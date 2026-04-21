# mimir / Reigen — Remote Optimization for Friends

## What it does

Your optimization task, powered by our meta-dispatcher **mimir** on my server.

- **You** provide: a Python function `eval_fn(params) -> score`
- **My server** provides: mimir (auto-routing to owl + Reigen + scipy.basinhopping)
- Your `eval_fn` runs on **your** machine. My server only does the optimization math.

Use it for: LLM hyperparameter tuning, model calibration, scientific problems,
game AI tuning, anything where you want "given ranges, find the best params".

**2026-04-21 以降**: `run_mimir_remote()` を推奨 (旧 `run_reigen_remote` も互換で使える)。
mimir は内部で最適なサブアルゴリズム (owl / Reigen / scipy) を自動選択する。

## Setup (one-time)

Ask me for:
1. **Server URL** (my ngrok URL, looks like `https://xxxx.ngrok-free.dev`)
2. **API key** (a shared secret string)

Copy the client helper to your machine:
- [reigen_friend_client.py](reigen_friend_client.py)

That's it. No installation needed beyond Python 3.10+ (uses stdlib only; no `pip install`).

## Quickstart (mimir 推奨、2026-04-21 以降) — 10 lines

```python
from reigen_friend_client import run_mimir_remote

def my_eval(params):
    # params is a list of floats; return a number (higher = better)
    return -sum((x - 1.0) ** 2 for x in params)

result = run_mimir_remote(
    server_url="https://xxxx.ngrok-free.dev",
    api_key="YOUR_SHARED_KEY",
    eval_fn=my_eval,
    param_ranges=[(-3, 3)] * 5,
    time_budget=180,
)

print(result["best_params"], result["best_score"], result["tool_used"])
# tool_used: "owl" | "owl+reigen" | "owl+scipy" ...
```

## Quickstart (Reigen 直使い、legacy で OK)

```python
from reigen_friend_client import run_reigen_remote
result = run_reigen_remote(
    server_url="https://xxxx.ngrok-free.dev",
    api_key="YOUR_SHARED_KEY",
    eval_fn=my_eval,
    user_param_ranges=[(-3, 3)] * 5,
    experience_id="genesis",
    time_budget=180,
)
print(result["user_best_params"], result["user_best_score"])
```

## Demo (no coding needed)

```bash
python reigen_friend_client.py --server https://xxxx.ngrok-free.dev --key YOUR_KEY --time-budget 60
```

Runs a 5D Rosenbrock problem end-to-end. Finishes in ~2 minutes. Good for smoke testing.

## Parameters you pass

| arg | type | default | meaning |
|---|---|---|---|
| `server_url` | str | — | My ngrok URL (no trailing slash) |
| `api_key` | str | — | Bearer token (I send via Discord/DM) |
| `eval_fn` | callable | — | Your function: takes list, returns float |
| `user_param_ranges` | list of (lo, hi) | — | Search space. **Never include 0** in ranges |
| `user_param_names` | list of str | None | Optional labels (for readability only) |
| `experience_id` | str | `"genesis"` | **Keep default** unless you want isolated namespace |
| `time_budget` | int | 300 | Outer Reigen seconds budget |
| `inner_time_budget` | int | 2 | Per-inner-Sentinel seconds |
| `inner_learn` | bool | False | Accumulate per-eval fossils on server |
| `poll_timeout` | int | 15 | HTTP long-poll timeout (1-60) |
| `verbose` | bool | True | Print progress every 5 evals |

## What you get back

```python
{
    "user_best_params":   [...],     # Your answer (list of floats, same shape as your ranges)
    "user_best_score":    0.847,     # Best eval_fn value Reigen found
    "self_best_params":   {...},     # How Reigen tuned itself (dict of 12 knobs)
    "self_diagnostic":    {...},     # {self_informative, recommend_freeze, drift_from_defaults}
    "verdict":            "approved" | "pivoted" | "failed",
    "elapsed_s":          180.5,
    # ... more (see Reigen docs)
}
```

## Why "genesis" for `experience_id`?

Reigen accumulates knowledge across runs via the `experience_id` namespace.
When you and I both use `"genesis"`, our runs **pool self-knowledge**
(how to tune Sentinel internals). More runs = Reigen gets smarter for everyone.

Your actual problem parameters (user_param_ranges) are automatically filtered
per task, so no contamination. Just self-tuning knobs transfer.

If you want an isolated namespace (e.g., for benchmarking), pick your own id:
`experience_id="friend1_isolated"`. But default is always better.

## What your eval_fn can do

ANYTHING that returns a float:

```python
def llm_calib_eval(params):
    # You run Qwen/Llama/etc. on your GPU, measure HellaSwag, return score
    apply_scales(params)
    score = run_hellaswag_eval()
    restore_scales()
    return score

def game_ai_eval(params):
    # Tune game AI weights by running simulated matches
    wins = simulate_matches(params, n=100)
    return wins / 100.0

def chemistry_eval(params):
    # Bind affinity calculation, etc.
    return -dock_score(params)  # negate if lower-is-better
```

GPU, subprocess, remote API calls — all fine. Your side, your resources.

## Rules you must follow

1. **Never include 0 in `user_param_ranges`** — `(0.0, 1.0)` breaks; use `(0.01, 1.0)`.
   Reason: multiplicative proxy collapses at zero scale.
2. **Return higher=better float** — if you're measuring loss (PPL, etc.), negate it: `return -ppl`.
3. **Don't leak test set** — eval on train/dev, never on benchmark answers.
4. **Max ~8 user dims** — if you need more, let me know (I'll switch to larger preset).
5. **Apply → measure → restore** — if your eval modifies global state, restore it before returning.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `HTTP 401` | wrong / missing API key | re-check key from me |
| `HTTP 404 unknown session` | session expired (30 min idle) | start a new session |
| `action: "wait"` loops forever | server busy | OK, just poll again; client does this automatically |
| Client hangs | network glitch | client auto-retries after 3s; give it a minute |
| Score always 0 | your eval_fn raised an exception | check verbose logs; fix the eval_fn |
| Wall time 2-3x budget | Reigen autonomous loop | normal. set lower `time_budget` if in a hurry |

## Limits (while I host this)

- **1-2 concurrent sessions** practical
- **30 min per session** hard wall (after that, cleanup runs)
- **~5-10 min budget** is sweet spot
- Server up during: ask me in Discord

## When server is down

Client raises `urllib.error.URLError`. Wait a bit, retry. I'll announce downtime in Discord.

## Questions?

DM me. But not: "how does the optimization work internally?" That's under NDA-equivalent (trade secret).
