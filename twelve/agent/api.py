"""AT API — Agentic Twelve をHTTP APIとして公開

使い方:
  python twelve/agent/api.py

エンドポイント:
  POST /run                  — ATを起動して最適化を実行
  POST /owl                   — Owl: 測定データだけで最適化（eval_fn不要）
  POST /reigen                — Reigen (零玄): ローカル Python module で実行
  POST /reigen/start          — Reigen (session) 開始 → session_id
  GET  /reigen/next?sid=...   — 次の params (action=eval) or 完了 (action=done)
  POST /reigen/score          — score 送信 (body: {sid, params, score})
  POST /mimir                 — mimir: ローカル Python module で実行 (2026-04-21)
  POST /mimir/start           — mimir (session) 開始 → session_id
  GET  /mimir/next?sid=...    — mimir session の次 action (reigen と共用)
  POST /mimir/score           — mimir session score 送信 (reigen と共用)
  POST /status               — 実行中のATの状態を取得
  POST /stop                 — 実行中のATを停止
  GET  /health               — ヘルスチェック

認証:
  環境変数 REIGEN_API_KEY 設定時は全 /reigen*, /mimir*, /owl に Bearer token 必須。
  未設定なら認証スキップ (ローカル開発モード)。
"""

import io
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from http.server import HTTPServer

_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_DIR))
sys.path.insert(0, _PROJECT)

from twelve.agent import EvolutionAgent

# グローバル状態
_agent = None
_thread = None
_status = {"state": "idle", "result": None, "error": None, "started_at": None, "logs": []}
_lock = threading.Lock()

# Reigen session state (1 friend でも複数 session 持てるよう dict)
# session_id → _ReigenSession
_sessions = {}
_sessions_lock = threading.Lock()
SESSION_TIMEOUT_S = 1800   # 30 min で放置セッション自動削除


class _ReigenSession:
    """1 つの Reigen セッション。eval_fn を queue 経由で client にラウンドトリップ."""

    def __init__(self):
        self.params_q = queue.Queue(maxsize=1)   # server → client
        self.score_q = queue.Queue(maxsize=1)    # client → server
        self.done = threading.Event()
        self.result = None
        self.thread = None
        self.created_at = time.time()
        self.last_contact = time.time()


def _remote_eval_factory(session, timeout):
    """Client に params を送って score を待つ eval_fn を返す."""

    def eval_fn(params):
        # client 側が長時間返してこない時の保険 — timeout 秒で諦める
        session.params_q.put(list(float(x) for x in params))
        try:
            score = session.score_q.get(timeout=timeout)
        except queue.Empty:
            # client 応答なし → 0.0 返して進める (owl が次のサンプル試す)
            score = 0.0
        return float(score)

    return eval_fn


def _run_reigen_thread(session, config):
    """Background thread: Reigen 本体を回す. eval_fn は queue 経由で client へ."""
    try:
        from twelve.agent.reigen import Reigen

        # client eval 1 回あたりの server-side 待ち時間 (sec). 長めに取る.
        client_timeout = int(config.get("client_eval_timeout", 600))
        eval_fn = _remote_eval_factory(session, client_timeout)
        guard_fn = eval_fn   # single-metric mode (friend の eval_fn は 1 つ想定)

        user_ranges = [tuple(x) for x in config["user_param_ranges"]]

        r = Reigen(
            eval_fn=eval_fn, guard_fn=guard_fn,
            user_param_ranges=user_ranges,
            user_param_names=config.get("user_param_names"),
            experience_id=config.get("experience_id", "genesis"),
            inner_time_budget=int(config.get("inner_time_budget", 2)),
            inner_learn=bool(config.get("inner_learn", False)),
        ).run(
            time_budget=int(config.get("time_budget", 300)),
            wall_time_factor=float(config.get("wall_time_factor", 3.0)),
            verbose=False,
        )
        session.result = r
    except Exception as e:
        session.result = {"error": str(e), "traceback": traceback.format_exc()}
    finally:
        session.done.set()


def _run_mimir_thread(session, config):
    """Background thread: mimir (owl + Reigen + scipy cascade) を回す.

    client の eval_fn が sequential で global state を触ることもあるため、
    mimir の `thread_safe_eval=False` をデフォルト適用 (cascade を逐次化)。
    """
    try:
        from twelve.agent.mimir import mimir as _mimir

        client_timeout = int(config.get("client_eval_timeout", 600))
        eval_fn = _remote_eval_factory(session, client_timeout)

        user_ranges = [tuple(x) for x in config["user_param_ranges"]]

        r = _mimir(
            eval_fn=eval_fn,
            param_ranges=user_ranges,
            param_names=config.get("user_param_names"),
            experience_id=config.get("experience_id", "genesis"),
            time_budget=float(config.get("time_budget", 300)),
            eval_cost_hint=(float(config["eval_cost_hint"])
                            if "eval_cost_hint" in config else None),
            mode=config.get("mode", "optimize"),
            # Remote eval over network is inherently sequential per client →
            # parallel cascade is already serialized by the queue, but we force
            # it explicitly to avoid any race on client-side global state.
            thread_safe_eval=False,
            verbose=False,
        )
        # Trim owl_result raw dict before network send (large)
        for k in ("owl_result", "reigen_result", "scipy_result"):
            r.pop(k, None)
        session.result = r
    except Exception as e:
        session.result = {"error": str(e), "traceback": traceback.format_exc()}
    finally:
        session.done.set()


def _cleanup_expired_sessions():
    """放置セッション定期削除 (daemon thread)."""
    while True:
        time.sleep(60)
        now = time.time()
        with _sessions_lock:
            expired = [
                sid for sid, s in _sessions.items()
                if now - s.last_contact > SESSION_TIMEOUT_S
            ]
            for sid in expired:
                _sessions.pop(sid, None)


_cleanup_thread = threading.Thread(target=_cleanup_expired_sessions, daemon=True)
_cleanup_thread.start()

_PATH_KEYS = ("config_path", "folder", "strategies_dir", "benchmark")


def _validate_paths(config):
    """Return error message if any path contains '..', else None."""
    for key in _PATH_KEYS:
        val = config.get(key)
        if val and ".." in str(val):
            return f"path traversal not allowed in '{key}'"
    return None


class _LogCapture(io.TextIOBase):
    """Tee stdout writes into _status['logs'] list."""

    def __init__(self, original):
        self._original = original

    def write(self, s):
        if s and s.strip():
            with _lock:
                _status.setdefault("logs", []).append(s.rstrip("\n"))
        if self._original:
            return self._original.write(s)
        return len(s)

    def flush(self):
        if self._original:
            self._original.flush()


def _run_agent(config):
    """Run AT in background thread."""
    global _agent, _status
    old_stdout = sys.stdout
    sys.stdout = _LogCapture(old_stdout)
    try:
        with _lock:
            _status = {"state": "running", "result": None, "error": None,
                       "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "config": config, "logs": []}

        if "folder" in config:
            _agent = EvolutionAgent(folder=config["folder"])
        elif "strategies_dir" in config and "benchmark" in config:
            _agent = EvolutionAgent(
                strategies_dir=config["strategies_dir"],
                benchmark=config["benchmark"],
            )
        elif "config_path" in config:
            _agent = EvolutionAgent(config["config_path"])
        else:
            raise ValueError("folder, strategies_dir+benchmark, or config_path required")

        result = _agent.run(
            max_iterations=config.get("max_iterations", 5),
            skip_first_benchmark=config.get("skip_first_benchmark", False),
        )

        with _lock:
            _status["state"] = "completed"
            _status["result"] = result

    except Exception as e:
        with _lock:
            _status["state"] = "error"
            _status["error"] = {"message": str(e), "traceback": traceback.format_exc()}
    finally:
        sys.stdout = old_stdout


class ATHandler(BaseHTTPRequestHandler):
    """ATのHTTP API"""

    # -------- auth --------
    def _auth_required_path(self, path):
        """Auth required for /owl*, /reigen*, /mimir* paths."""
        return (path.startswith("/reigen")
                or path == "/owl"
                or path.startswith("/mimir"))

    def _check_auth(self):
        """Bearer token チェック. 環境変数 REIGEN_API_KEY 未設定なら常に許可."""
        expected = os.environ.get("REIGEN_API_KEY")
        if not expected:
            return True   # dev mode: no auth
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return False
        return auth[len("Bearer "):].strip() == expected

    def _parse_query(self, path):
        """GET /path?a=1&b=2 → (path, {a: "1", b: "2"})"""
        from urllib.parse import urlparse, parse_qs
        u = urlparse(path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        return u.path, q

    def do_GET(self):
        path, query = self._parse_query(self.path)
        if path == "/health":
            self._json_response({"status": "ok", "agent": "AT (Agentic Twelve)"})
            return
        if path == "/client.py":
            self._serve_static("reigen_friend_client.py", "text/x-python; charset=utf-8")
            return
        if path == "/readme":
            self._serve_static("FRIEND_README.md", "text/markdown; charset=utf-8")
            return
        if self._auth_required_path(path) and not self._check_auth():
            self._json_response({"error": "unauthorized"}, 401)
            return
        if path == "/reigen/next":
            self._handle_reigen_next(query)
        elif path == "/mimir/next":
            self._handle_reigen_next(query)  # session loop is algorithm-agnostic
        else:
            self._json_response({"error": "not found"}, 404)

    def _serve_static(self, filename, content_type):
        """Serve a file from the project root (no auth — public helper files only)."""
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        )
        file_path = os.path.join(project_root, filename)
        if not os.path.isfile(file_path):
            self._json_response({"error": f"{filename} not found on server"}, 404)
            return
        with open(file_path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {}

        path, _ = self._parse_query(self.path)

        # Auth gate (dev mode: always pass)
        if self._auth_required_path(path) and not self._check_auth():
            self._json_response({"error": "unauthorized"}, 401)
            return

        if path == "/run":
            self._handle_run(data)
        elif path == "/owl":
            self._handle_owl(data)
        elif path == "/reigen":
            self._handle_reigen(data)
        elif path == "/reigen/start":
            self._handle_reigen_start(data)
        elif path == "/reigen/score":
            self._handle_reigen_score(data)
        elif path == "/mimir":
            self._handle_mimir(data)
        elif path == "/mimir/start":
            self._handle_mimir_start(data)
        elif path == "/mimir/score":
            self._handle_reigen_score(data)  # score path is tool-agnostic
        elif path == "/status":
            self._handle_status()
        elif path == "/stop":
            self._handle_stop()
        else:
            self._json_response({"error": "not found"}, 404)

    def _handle_run(self, config):
        global _thread

        path_err = _validate_paths(config)
        if path_err:
            self._json_response({"error": path_err}, 400)
            return

        with _lock:
            if _status["state"] == "running":
                self._json_response({"error": "already running"}, 409)
                return

        _thread = threading.Thread(target=_run_agent, args=(config,), daemon=True)
        _thread.start()
        self._json_response({"status": "started", "config": config})

    def _handle_owl(self, data):
        """Owl — 測定データだけで最適化。見えない構造を見つける。"""
        measurements = data.get("measurements")
        if not measurements:
            self._json_response({"error": "measurements required"}, 400)
            return

        try:
            from twelve.optimize import owl
            result = owl(
                measurements=measurements,
                time_budget=data.get("time_budget", 60),
                experience_id=data.get("experience_id"),
                verbose=True,
            )
            self._json_response(result)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _handle_reigen(self, data):
        """Reigen (零玄) — ローカル Python モジュール内の eval_fn/guard_fn を指定して実行.

        Request body (JSON):
          {
            "eval_module":     "my_pkg.my_evals",       # Python import path
            "eval_fn":         "my_eval",               # function name in module
            "guard_fn":        "my_guard",              # optional (defaults to eval_fn)
            "user_param_ranges": [[0.5, 1.5], ...],     # list of [lo, hi]
            "experience_id":   "genesis",               # optional, default "genesis"
            "time_budget":     300,                     # seconds
            "inner_time_budget": 2,                     # seconds per inner Sentinel
            "wall_time_factor": 3.0,                    # soft cap on wall time
          }

        Local-only by design: imports by module path, no code eval. Use HTTPS+auth
        if exposing (but primary use case is localhost).
        """
        try:
            import importlib
            mod_name = data.get("eval_module")
            eval_name = data.get("eval_fn")
            guard_name = data.get("guard_fn") or eval_name
            user_ranges = data.get("user_param_ranges")
            if not mod_name or not eval_name or not user_ranges:
                self._json_response({
                    "error": "eval_module, eval_fn, user_param_ranges required"
                }, 400)
                return

            # Sanitize module path (no traversal, no __ trickery)
            if ".." in mod_name or mod_name.startswith("_"):
                self._json_response(
                    {"error": "invalid eval_module name"}, 400)
                return

            mod = importlib.import_module(mod_name)
            eval_fn = getattr(mod, eval_name)
            guard_fn = getattr(mod, guard_name)

            # Normalize ranges (list-of-list → list-of-tuple)
            ranges = [tuple(x) for x in user_ranges]

            from twelve.agent.reigen import Reigen
            result = Reigen(
                eval_fn=eval_fn, guard_fn=guard_fn,
                user_param_ranges=ranges,
                user_param_names=data.get("user_param_names"),
                experience_id=data.get("experience_id", "genesis"),
                inner_time_budget=int(data.get("inner_time_budget", 2)),
                inner_learn=bool(data.get("inner_learn", False)),
            ).run(
                time_budget=int(data.get("time_budget", 300)),
                verbose=True,
                wall_time_factor=float(data.get("wall_time_factor", 3.0)),
            )
            self._json_response(result)
        except Exception as e:
            self._json_response({
                "error": str(e),
                "traceback": traceback.format_exc(),
            }, 500)

    # ==========================================================
    # Reigen session API (client-side eval_fn)
    # ==========================================================
    def _handle_reigen_start(self, data):
        """Start a new Reigen session.

        Request body:
          {
            "user_param_ranges": [[lo, hi], ...],   # required
            "user_param_names": [...],              # optional
            "experience_id": "genesis",             # optional
            "time_budget": 300,                     # optional
            "inner_time_budget": 2,                 # optional
            "inner_learn": false,                   # optional
            "client_eval_timeout": 600              # sec to wait per client eval
          }
        Response: {"session_id": "abc12345"}
        """
        user_ranges = data.get("user_param_ranges")
        if not user_ranges or not isinstance(user_ranges, list):
            self._json_response({"error": "user_param_ranges required"}, 400)
            return

        sid = uuid.uuid4().hex[:8]
        session = _ReigenSession()

        with _sessions_lock:
            _sessions[sid] = session

        session.thread = threading.Thread(
            target=_run_reigen_thread,
            args=(session, data),
            daemon=True,
        )
        session.thread.start()

        self._json_response({"session_id": sid})

    def _handle_reigen_next(self, query):
        """Poll for the next action (eval params, wait, or done).

        Query: ?sid=abc12345&timeout=30
        Response:
          {"action": "eval", "params": [0.5, 0.6, ...]}    ← client should evaluate
          {"action": "wait"}                                ← server busy, retry
          {"action": "done", "result": {...}}               ← session complete
        """
        sid = query.get("sid") or query.get("session_id")
        if not sid:
            self._json_response({"error": "sid required"}, 400)
            return
        try:
            timeout = float(query.get("timeout", "15"))
        except (TypeError, ValueError):
            timeout = 15.0
        # Clamp to safe range
        timeout = max(1.0, min(60.0, timeout))

        with _sessions_lock:
            session = _sessions.get(sid)
        if session is None:
            self._json_response({"error": "unknown session"}, 404)
            return
        session.last_contact = time.time()

        # Done?
        if session.done.is_set():
            result = session.result or {}
            # Cleanup (one-time read)
            with _sessions_lock:
                _sessions.pop(sid, None)
            self._json_response({"action": "done", "result": result})
            return

        # Wait for next params from Reigen thread
        try:
            params = session.params_q.get(timeout=timeout)
            self._json_response({"action": "eval", "params": params})
        except queue.Empty:
            # Maybe it finished in the meantime
            if session.done.is_set():
                result = session.result or {}
                with _sessions_lock:
                    _sessions.pop(sid, None)
                self._json_response({"action": "done", "result": result})
            else:
                self._json_response({"action": "wait"})

    def _handle_reigen_score(self, data):
        """Submit a score for the most recent params.

        Request body:
          {"sid": "abc12345", "score": 0.87, "params": [...] (optional, for logs)}
        Response: {"ok": true}
        """
        sid = data.get("sid") or data.get("session_id")
        if not sid:
            self._json_response({"error": "sid required"}, 400)
            return
        if "score" not in data:
            self._json_response({"error": "score required"}, 400)
            return

        with _sessions_lock:
            session = _sessions.get(sid)
        if session is None:
            self._json_response({"error": "unknown session"}, 404)
            return
        session.last_contact = time.time()

        try:
            score = float(data["score"])
        except (TypeError, ValueError):
            self._json_response({"error": "score must be a number"}, 400)
            return

        try:
            session.score_q.put(score, timeout=5.0)
        except queue.Full:
            self._json_response({"error": "score already submitted, wait for next"}, 409)
            return
        self._json_response({"ok": True})

    # ==========================================================
    # mimir endpoints (2026-04-21)
    # ==========================================================
    def _handle_mimir(self, data):
        """mimir — local Python module 内の eval_fn を指定して mimir 実行.

        Request body (JSON):
          {
            "eval_module":        "my_pkg.my_evals",
            "eval_fn":            "my_eval",
            "param_ranges":       [[lo, hi], ...],       # required
            "param_names":        [...],                 # optional
            "experience_id":      "genesis",             # optional
            "time_budget":        300,                   # seconds
            "eval_cost_hint":     null,                  # optional float
            "mode":               "optimize"              # or "structure_only"
          }

        Local-only by design (importlib, no code eval).
        """
        try:
            import importlib
            mod_name = data.get("eval_module")
            eval_name = data.get("eval_fn")
            ranges = data.get("param_ranges") or data.get("user_param_ranges")
            if not mod_name or not eval_name or not ranges:
                self._json_response({
                    "error": "eval_module, eval_fn, param_ranges required"
                }, 400)
                return
            if ".." in mod_name or mod_name.startswith("_"):
                self._json_response({"error": "invalid eval_module"}, 400)
                return

            mod = importlib.import_module(mod_name)
            eval_fn = getattr(mod, eval_name)
            ranges = [tuple(x) for x in ranges]

            from twelve.agent.mimir import mimir as _mimir
            result = _mimir(
                eval_fn=eval_fn,
                param_ranges=ranges,
                param_names=data.get("param_names"),
                experience_id=data.get("experience_id", "genesis"),
                time_budget=float(data.get("time_budget", 300)),
                eval_cost_hint=(float(data["eval_cost_hint"])
                                if "eval_cost_hint" in data else None),
                mode=data.get("mode", "optimize"),
                verbose=False,
            )
            # Trim raw internal dicts for network
            for k in ("owl_result", "reigen_result", "scipy_result"):
                result.pop(k, None)
            self._json_response(result)
        except Exception as e:
            self._json_response({
                "error": str(e),
                "traceback": traceback.format_exc(),
            }, 500)

    def _handle_mimir_start(self, data):
        """Start a new mimir session (client-side eval_fn).

        Request body: same as /reigen/start with `param_ranges` (alias of
        `user_param_ranges` accepted for backwards-compat). Optional keys:
          `eval_cost_hint`, `mode`.
        Response: {"session_id": "abc12345"}

        Client flow: /mimir/start → poll /mimir/next?sid=X → submit /mimir/score
        → next / done.
        """
        ranges = data.get("param_ranges") or data.get("user_param_ranges")
        if not ranges or not isinstance(ranges, list):
            self._json_response({"error": "param_ranges required"}, 400)
            return
        # Normalize for downstream (thread expects user_param_ranges)
        config = dict(data)
        config["user_param_ranges"] = ranges
        config.setdefault("user_param_names", data.get("param_names"))

        sid = uuid.uuid4().hex[:8]
        session = _ReigenSession()
        with _sessions_lock:
            _sessions[sid] = session
        session.thread = threading.Thread(
            target=_run_mimir_thread,
            args=(session, config),
            daemon=True,
        )
        session.thread.start()
        self._json_response({"session_id": sid})

    def _handle_status(self):
        with _lock:
            self._json_response(dict(_status))

    def _handle_stop(self):
        global _agent
        with _lock:
            if _status["state"] != "running":
                self._json_response({"error": "not running"}, 400)
                return
            _status["state"] = "stopping"
        if _agent is not None:
            _agent.stop()
        self._json_response({"status": "stop requested"})

    def _json_response(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        if not getattr(self.server, 'quiet', False):
            super().log_message(format, *args)


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Concurrent request handling — required because /reigen/next blocks
    while waiting for the Reigen thread to produce params."""
    daemon_threads = True
    allow_reuse_address = True


def serve(host="0.0.0.0", port=8282, quiet=False):
    """AT APIサーバーを起動"""
    server = _ThreadingHTTPServer((host, port), ATHandler)
    server.quiet = quiet
    print(f"AT API running on http://{host}:{port}", flush=True)
    auth_mode = "Bearer required" if os.environ.get("REIGEN_API_KEY") else "no auth (dev)"
    print(f"  auth: {auth_mode}", flush=True)
    print(f"  POST /run                   - start AT", flush=True)
    print(f"  POST /owl                   - Owl: measurements only (no eval_fn)", flush=True)
    print(f"  POST /reigen                - Reigen (local eval_module mode)", flush=True)
    print(f"  POST /reigen/start          - Reigen session: start → session_id", flush=True)
    print(f"  GET  /reigen/next?sid=...   - Reigen session: poll for next action", flush=True)
    print(f"  POST /reigen/score          - Reigen session: submit score", flush=True)
    print(f"  POST /mimir                 - mimir (local eval_module mode)", flush=True)
    print(f"  POST /mimir/start           - mimir session: start → session_id", flush=True)
    print(f"  GET  /mimir/next?sid=...    - mimir session: poll (shares reigen impl)", flush=True)
    print(f"  POST /mimir/score           - mimir session: submit score", flush=True)
    print(f"  POST /status                - status", flush=True)
    print(f"  POST /stop                  - stop", flush=True)
    print(f"  GET  /health                - health", flush=True)
    print(f"  GET  /client.py             - download reigen_friend_client.py (public)", flush=True)
    print(f"  GET  /readme                - download FRIEND_README.md (public)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AT API Server")
    parser.add_argument("--port", type=int, default=8282)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--quiet", action="store_true", help="Suppress HTTP request logs")
    args = parser.parse_args()
    serve(args.host, args.port, quiet=args.quiet)
