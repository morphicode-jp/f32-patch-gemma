"""Reigen Friend Client — client-side Reigen helper.

自分の PC で eval_fn を動かしつつ、Reigen 最適化ロジックは server 側に任せる。

Usage:
    from reigen_friend_client import run_reigen_remote

    def my_eval(params):
        # ここに自分の評価ロジック (LLM 推論、計算、なんでも)
        return score_float   # higher = better

    result = run_reigen_remote(
        server_url="https://rundown-jailer-till.ngrok-free.dev",
        api_key="your_token_here",              # 友達と共有
        eval_fn=my_eval,
        user_param_ranges=[(0.5, 1.5)] * 8,     # 最適化したい範囲
        experience_id="genesis",                 # 同じ ID で cross-task 学習
        time_budget=300,                         # server 側 outer budget
    )

    print(result["user_best_params"])
    print(result["user_best_score"])
    print(result["self_best_params"])           # Reigen 発見ハイパラ
"""
import time
import urllib.request
import urllib.error
import json


def _post(url, body, headers, timeout=30):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url, headers, timeout=30):
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_reigen_remote(
    server_url,
    api_key,
    eval_fn,
    user_param_ranges,
    user_param_names=None,
    experience_id="genesis",
    time_budget=300,
    inner_time_budget=2,
    inner_learn=False,
    poll_timeout=15,
    verbose=True,
):
    """Run Reigen on the remote server, with eval_fn running LOCALLY.

    Args:
        server_url: e.g. "https://xxxx.ngrok-free.dev" (no trailing slash)
        api_key: shared Bearer token
        eval_fn: callable(params_list) -> float (higher = better)
        user_param_ranges: list of (lo, hi) tuples
        user_param_names: optional list of str (same length as ranges)
        experience_id: shared ID for cross-task learning (default "genesis")
        time_budget: server-side outer Sentinel time budget in seconds
        inner_time_budget: per-inner-Sentinel budget in seconds
        inner_learn: whether inner Sentinel writes to experience
        poll_timeout: how long each /reigen/next call waits (1-60 sec)
        verbose: print progress

    Returns:
        dict with keys: user_best_params, user_best_score, self_best_params,
                         self_diagnostic, verdict, elapsed_s, and all Sentinel keys.
    """
    server = server_url.rstrip("/")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # 1. Start session
    start_body = {
        "user_param_ranges": [list(r) for r in user_param_ranges],
        "experience_id": experience_id,
        "time_budget": int(time_budget),
        "inner_time_budget": int(inner_time_budget),
        "inner_learn": bool(inner_learn),
    }
    if user_param_names:
        start_body["user_param_names"] = list(user_param_names)

    start_resp = _post(f"{server}/reigen/start", start_body, headers)
    sid = start_resp.get("session_id")
    if not sid:
        raise RuntimeError(f"Failed to start session: {start_resp}")
    if verbose:
        print(f"[Reigen session started] sid={sid}")

    # 2. Event loop
    t0 = time.time()
    n_evals = 0
    while True:
        # Poll for next action
        try:
            resp = _get(
                f"{server}/reigen/next?sid={sid}&timeout={int(poll_timeout)}",
                headers,
                timeout=poll_timeout + 10,
            )
        except urllib.error.URLError as e:
            # Network hiccup — retry after short wait
            if verbose:
                print(f"  [net error] {e} — retrying in 3s")
            time.sleep(3)
            continue

        action = resp.get("action")
        if action == "done":
            result = resp.get("result", {})
            if verbose:
                elapsed = time.time() - t0
                print(f"[Reigen done] {n_evals} evals, {elapsed:.0f}s total")
            return result

        if action == "wait":
            # server busy, short pause
            time.sleep(0.5)
            continue

        if action == "eval":
            params = resp.get("params")
            if params is None:
                raise RuntimeError(f"Unexpected response: {resp}")
            # Run local eval
            try:
                score = float(eval_fn(list(params)))
            except Exception as e:
                if verbose:
                    print(f"  [eval_fn error] {e} — returning 0.0")
                score = 0.0
            n_evals += 1
            if verbose and n_evals % 5 == 0:
                elapsed = time.time() - t0
                print(f"  [eval #{n_evals}] score={score:.4f} ({elapsed:.0f}s)")
            # Submit score
            _post(
                f"{server}/reigen/score",
                {"sid": sid, "score": score, "params": list(params)},
                headers,
            )
            continue

        raise RuntimeError(f"Unknown action: {resp}")


def run_mimir_remote(
    server_url,
    api_key,
    eval_fn,
    param_ranges,
    param_names=None,
    experience_id="genesis",
    time_budget=300,
    eval_cost_hint=None,
    mode="optimize",
    poll_timeout=15,
    verbose=True,
):
    """Run mimir on the remote server, eval_fn running LOCALLY.

    mimir は内部で owl (Phase 1) と Reigen (Phase 2 cascade) と
    scipy.basinhopping (高次元 gradient) を自動分岐する meta-dispatcher。

    Args:
        server_url: e.g. "https://xxxx.ngrok-free.dev" (no trailing slash)
        api_key: shared Bearer token
        eval_fn: callable(params_list) -> float (higher = better)
        param_ranges: list of (lo, hi) tuples
        param_names: optional list of str
        experience_id: shared ID for cross-task learning (default "genesis")
        time_budget: wall budget (seconds)
        eval_cost_hint: optional float (seconds/call hint, default auto-measure)
        mode: "optimize" (default) or "structure_only"
        poll_timeout: /mimir/next wait per call (1-60 sec)
        verbose: print progress

    Returns:
        dict with keys: best_params, best_score, tool_used, route, proxy_r2,
                        dead_dims, active_dims, fragility, elapsed_s, ...
    """
    server = server_url.rstrip("/")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    start_body = {
        "param_ranges": [list(r) for r in param_ranges],
        "experience_id": experience_id,
        "time_budget": int(time_budget),
        "mode": mode,
    }
    if param_names:
        start_body["param_names"] = list(param_names)
    if eval_cost_hint is not None:
        start_body["eval_cost_hint"] = float(eval_cost_hint)

    start_resp = _post(f"{server}/mimir/start", start_body, headers)
    sid = start_resp.get("session_id")
    if not sid:
        raise RuntimeError(f"Failed to start session: {start_resp}")
    if verbose:
        print(f"[mimir session started] sid={sid}")

    t0 = time.time()
    n_evals = 0
    while True:
        try:
            resp = _get(
                f"{server}/mimir/next?sid={sid}&timeout={int(poll_timeout)}",
                headers,
                timeout=poll_timeout + 10,
            )
        except urllib.error.URLError as e:
            if verbose:
                print(f"  [net error] {e} — retrying in 3s")
            time.sleep(3)
            continue

        action = resp.get("action")
        if action == "done":
            result = resp.get("result", {})
            if verbose:
                elapsed = time.time() - t0
                tu = result.get("tool_used", "?")
                print(f"[mimir done] {n_evals} evals, {elapsed:.0f}s, tool_used={tu}")
            return result

        if action == "wait":
            time.sleep(0.5)
            continue

        if action == "eval":
            params = resp.get("params")
            if params is None:
                raise RuntimeError(f"Unexpected response: {resp}")
            try:
                score = float(eval_fn(list(params)))
            except Exception as e:
                if verbose:
                    print(f"  [eval_fn error] {e} — returning 0.0")
                score = 0.0
            n_evals += 1
            if verbose and n_evals % 5 == 0:
                elapsed = time.time() - t0
                print(f"  [eval #{n_evals}] score={score:.4f} ({elapsed:.0f}s)")
            _post(
                f"{server}/mimir/score",
                {"sid": sid, "score": score, "params": list(params)},
                headers,
            )
            continue

        raise RuntimeError(f"Unknown action: {resp}")


def run_odin_remote(
    server_url,
    api_key,
    eval_fn,
    param_ranges,
    param_names=None,
    experience_id="genesis_odin",
    time_budget=300,
    poll_timeout=15,
    verbose=True,
):
    """Run オーディン (mimir_odin) on the remote server, eval_fn running LOCALLY.

    mimir_odin は 4 specialist (default / lad / expensive / scipy-forced) を
    並列実行し、最良 best_score を採用する。remote context では server 側で
    thread 並列、client の eval_fn は serial で呼ばれる (Queue(1) による逐次化)。

    Args:
        server_url: e.g. "https://xxxx.ngrok-free.dev" (no trailing slash)
        api_key: shared Bearer token
        eval_fn: callable(params_list) -> float (higher = better)
        param_ranges: list of (lo, hi) tuples
        param_names: optional list of str
        experience_id: shared ID (default "genesis_odin")
        time_budget: wall budget per specialist (seconds)
        poll_timeout: /odin/next wait per call (1-60 sec)
        verbose: print progress

    Returns:
        dict with keys:
          best_params, best_score       # 勝者 specialist の結果
          specialist                    # 勝者名 ("default"/"lad"/"expensive"/"scipy-forced")
          council                       # [(name, score), ...] 全員降順
          council_variance_std          # 問題難易度 signal
          dead_dims, active_dims, ...   # 勝者の構造発見結果
    """
    server = server_url.rstrip("/")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    start_body = {
        "param_ranges": [list(r) for r in param_ranges],
        "experience_id": experience_id,
        "time_budget": int(time_budget),
    }
    if param_names:
        start_body["param_names"] = list(param_names)

    start_resp = _post(f"{server}/odin/start", start_body, headers)
    sid = start_resp.get("session_id")
    if not sid:
        raise RuntimeError(f"Failed to start session: {start_resp}")
    if verbose:
        print(f"[オーディン session started] sid={sid}")

    t0 = time.time()
    n_evals = 0
    while True:
        try:
            resp = _get(
                f"{server}/odin/next?sid={sid}&timeout={int(poll_timeout)}",
                headers,
                timeout=poll_timeout + 10,
            )
        except urllib.error.URLError as e:
            if verbose:
                print(f"  [net error] {e} — retrying in 3s")
            time.sleep(3)
            continue

        action = resp.get("action")
        if action == "done":
            result = resp.get("result", {})
            if verbose:
                elapsed = time.time() - t0
                specialist = result.get("specialist", "?")
                variance = result.get("council_variance_std", 0.0)
                print(f"[オーディン done] {n_evals} evals, {elapsed:.0f}s, "
                      f"winner={specialist}, council_std={variance:.3f}")
                council = result.get("council")
                if council:
                    print(f"  council: {council}")
            return result

        if action == "wait":
            time.sleep(0.5)
            continue

        if action == "eval":
            params = resp.get("params")
            if params is None:
                raise RuntimeError(f"Unexpected response: {resp}")
            try:
                score = float(eval_fn(list(params)))
            except Exception as e:
                if verbose:
                    print(f"  [eval_fn error] {e} — returning 0.0")
                score = 0.0
            n_evals += 1
            if verbose and n_evals % 5 == 0:
                elapsed = time.time() - t0
                print(f"  [eval #{n_evals}] score={score:.4f} ({elapsed:.0f}s)")
            _post(
                f"{server}/odin/score",
                {"sid": sid, "score": score, "params": list(params)},
                headers,
            )
            continue

        raise RuntimeError(f"Unknown action: {resp}")


# -------------------------------------------------------------------
# Demo main
# -------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import math

    parser = argparse.ArgumentParser(description="Reigen remote client demo")
    parser.add_argument("--server", required=True,
                        help="Server URL, e.g. https://xxxx.ngrok-free.dev")
    parser.add_argument("--key", required=True, help="Bearer API key")
    parser.add_argument("--time-budget", type=int, default=60)
    args = parser.parse_args()

    # Demo eval: 5D Rosenbrock (optimum = 0 at params = [1,1,1,1,1])
    def demo_eval(params):
        total = 0.0
        for i in range(len(params) - 1):
            total -= 100.0 * (params[i+1] - params[i]**2) ** 2 + (1 - params[i]) ** 2
        return total

    ranges = [(-2.0, 3.0)] * 5

    result = run_reigen_remote(
        server_url=args.server,
        api_key=args.key,
        eval_fn=demo_eval,
        user_param_ranges=ranges,
        experience_id="friend_demo",
        time_budget=args.time_budget,
        inner_time_budget=1,
    )
    print("\n" + "=" * 50)
    print(f"user_best_params: {result.get('user_best_params')}")
    print(f"user_best_score:  {result.get('user_best_score')}")
    print(f"self_best_params: {result.get('self_best_params')}")
    print(f"verdict:          {result.get('verdict')}")
