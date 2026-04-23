"""LLM council benchmark — GPU contention test on localhost:8282 (Qwen 3.6).

目的: 4 specialist council が local LLM を叩く時、GPU 取り合いで逐次化するか、
llama-server の concurrent 処理で並列化するかを実測する。

eval_fn: POST /v1/chat/completions with params=[temperature, top_p]
  - 固定 prompt ("Answer with one word: is the sky blue?")
  - score = response 長さ (話が長いほど良い、適当な proxy)
  - 1 call = ~2-5 秒 (GPU 推論時間)

比較:
  S  = mimir(fn, ranges) 単独
  D  = mimir_brunnir(fn, ranges) 4 specialist 並列

測定:
  - wall time (council が 1.5× 以内なら GPU 並列 OK、3-4× なら逐次化)
  - eval_fn 呼出総数 (LaD が内部で 20× 膨らむ件も確認)
  - best_score (council が負けないか)
"""
from __future__ import annotations

import json
import os
import sys
import time
from urllib import request as urlrequest
from urllib.error import URLError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.mimir import mimir
from twelve.agent.mimir_brunnir import mimir_brunnir


LLM_URL = "http://127.0.0.1:8282/v1/chat/completions"
MODEL = "qwen36-heretic-ithink-zenron_core_xl"
PROMPT = "Answer with one word: is the sky blue?"
TIME_BUDGET = 90.0  # per specialist
CALL_COUNTER = {"n": 0}  # shared counter (thread-safe-ish)


def llm_eval(p):
    """eval_fn: params → LLM API call → score (response length).

    p[0] = temperature [0.3, 1.0]
    p[1] = top_p       [0.5, 1.0]
    Returns higher-better scalar.
    """
    CALL_COUNTER["n"] += 1
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": float(p[0]),
        "top_p": float(p[1]),
        "max_tokens": 30,
    }).encode("utf-8")
    req = urlrequest.Request(
        LLM_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return float(len(content))  # longer response = higher score
    except (URLError, TimeoutError, Exception) as e:
        return 0.0  # failed call = worst score


def main():
    ranges = [(0.3, 1.0), (0.5, 1.0)]

    # warmup (1 call)
    print("warmup ...", flush=True)
    warm = llm_eval([0.7, 0.9])
    print(f"  warmup score={warm:.1f} (1 call)", flush=True)
    CALL_COUNTER["n"] = 0

    # -----------------------------------------------------------------
    # Stage 1: single mimir (default)
    # -----------------------------------------------------------------
    print()
    print("=" * 60)
    print("Stage 1: single mimir (default)")
    print("=" * 60)
    CALL_COUNTER["n"] = 0
    t0 = time.time()
    r_single = mimir(llm_eval, ranges,
                     time_budget=TIME_BUDGET,
                     experience_id="llm_bench_single")
    elapsed_single = time.time() - t0
    calls_single = CALL_COUNTER["n"]
    print(f"  best_score={r_single.get('best_score'):.2f}  "
          f"tool={r_single.get('tool_used')}  "
          f"t={elapsed_single:.1f}s  "
          f"calls={calls_single}")

    # -----------------------------------------------------------------
    # Stage 2: council (4 specialists, thread executor — GPU contention test)
    # -----------------------------------------------------------------
    print()
    print("=" * 60)
    print("Stage 2: council (4 specialists, thread)")
    print("=" * 60)
    CALL_COUNTER["n"] = 0
    t0 = time.time()
    r_council = mimir_brunnir(
        llm_eval, ranges,
        time_budget=TIME_BUDGET,
        executor="thread",  # closure eval_fn not picklable; also LLM is I/O
        experience_id="llm_bench_council",
        verbose=True,
    )
    elapsed_council = time.time() - t0
    calls_council = CALL_COUNTER["n"]
    print(f"  winner={r_council.get('specialist')}  "
          f"best_score={r_council.get('best_score'):.2f}  "
          f"t={elapsed_council:.1f}s  "
          f"calls={calls_council}  "
          f"variance={r_council.get('council_variance_std'):.2f}")
    print(f"  council: {r_council.get('council')}")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    wall_ratio = elapsed_council / max(elapsed_single, 0.1)
    calls_ratio = calls_council / max(calls_single, 1)
    print(f"  wall time ratio (council / single): {wall_ratio:.2f}×")
    print(f"  eval calls ratio  (council / single): {calls_ratio:.2f}×")

    if wall_ratio < 1.5:
        verdict = "✅ GPU concurrent OK — council usable for LLM eval"
    elif wall_ratio < 2.5:
        verdict = "⚠ partial parallelism — council ~2× slower than single"
    else:
        verdict = "❌ GPU serialized — council not worth it for LLM eval"
    print(f"  verdict: {verdict}")

    out = {
        "problem": "LLM eval on localhost:8282 Qwen 3.6",
        "model": MODEL,
        "time_budget_per_specialist_s": TIME_BUDGET,
        "single": {
            "best_score": r_single.get("best_score"),
            "tool_used": r_single.get("tool_used"),
            "elapsed_s": elapsed_single,
            "eval_calls": calls_single,
        },
        "council": {
            "winner": r_council.get("specialist"),
            "best_score": r_council.get("best_score"),
            "council": r_council.get("council"),
            "variance": r_council.get("council_variance_std"),
            "elapsed_s": elapsed_council,
            "eval_calls": calls_council,
        },
        "verdict": {
            "wall_ratio": wall_ratio,
            "calls_ratio": calls_ratio,
            "conclusion": verdict,
        },
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_council_llm.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
