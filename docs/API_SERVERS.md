# HTTP API & Servers

**用途**: mimir/Reigen を HTTP / MCP で外部から叩く時、もしくはリモートフレンドが自分の eval_fn を使って共同最適化する時。ローカル作業では使わない。

## Endpoints

```
Python:  reigen(eval, guard, user_ranges)                ← recommended (local)
         Sentinel(eval, guard, ranges).run()
         owl(data, verify_fn=..., autonomous=...)
         optimize(eval, ranges, ...)                      ← primitive
MCP:     owl(measurements_json, ...)                      ← data-to-answer only

HTTP (local/ngrok-shareable, auth via REIGEN_API_KEY env var):
  POST /owl                                 ← measurements-only optimization
  POST /reigen                              ← local eval_module (server-side fn)
  POST /reigen/start                        ← session: start → {session_id}
  GET  /reigen/next?sid=X&timeout=15        ← session: poll → {action: eval|wait|done}
  POST /reigen/score                        ← session: submit {sid, score, params}
  POST /mimir/start                         ← 単独 mimir session
  POST /odin/start                          ← オーディン (4 specialist 並列、2026-04-23 default)
  GET  /odin/next?sid=X                     ← session protocol は /reigen と同じ
  POST /odin/score                          ← 同上
```

## Launch

```bash
# Launch (requires .reigen_api_key file with your bearer token):
./serve_reigen.sh [PORT]
# In another terminal:
ngrok http 8282

# Without launcher (manual):
REIGEN_API_KEY="your_token" python twelve/agent/api.py --port 8282
```

## Session API — remote friend's eval_fn runs CLIENT-side

`/reigen/start + /next + /score` let a remote client keep their `eval_fn` local (on their GPU/LLM) while the server does Reigen bookkeeping only (CPU, ~1 core).

Client helper: `reigen_friend_client.py` — `run_reigen_remote(server, key, eval_fn, ranges, ...)`.
Distribute `reigen_friend_client.py` + `FRIEND_README.md` to friends; they `pip install` nothing (stdlib only).

**experience_id sharing**: everyone (you + all friends) should use `experience_id="genesis"` (the default). Self-params accumulate across tasks and users; user-param fossils auto-filter by dim/range mismatch. More shared runs → smarter Reigen for everyone (Rule 9).

## /reigen body (legacy local-module mode)

```json
{
  "eval_module": "my_pkg.evals",            // importable Python module on server FS
  "eval_fn": "my_eval",
  "guard_fn": "my_guard",                    // optional (defaults to eval_fn)
  "user_param_ranges": [[0.5,1.5], [0.1,1.0]],
  "user_param_names": ["layer_scale","dropout"],
  "experience_id": "genesis",
  "time_budget": 300,
  "inner_time_budget": 2,
  "wall_time_factor": 3.0
}
```
Local-only (`importlib.import_module`, no code eval). HTTPS + auth needed beyond localhost.

## /reigen/start body (session mode — recommended for remote friends)

```json
{
  "user_param_ranges": [[-5, 5], ...],       // required
  "user_param_names": ["x1", "x2", ...],      // optional
  "experience_id": "genesis",                 // default (shared learning)
  "time_budget": 300,
  "inner_time_budget": 2,
  "inner_learn": false,
  "client_eval_timeout": 600                  // per-eval wait on server (sec)
}
```

Response: `{"session_id": "abc12345"}`. Session TTL 30 min idle; all 3 endpoints need `Authorization: Bearer <REIGEN_API_KEY>` when env var is set.

## /odin/start body (オーディン — 4 specialist 並列、新 default)

```json
{
  "param_ranges": [[-5, 5], ...],             // required
  "param_names": ["x1", "x2", ...],            // optional
  "experience_id": "genesis_odin",             // default
  "time_budget": 300,                          // per specialist (並列なので wall time 同じ)
  "client_eval_timeout": 600
}
```

Response: `{"session_id": "abc12345"}`. Final result の追加キー:
- `specialist` — 勝者名 (default / lad / expensive / scipy-forced)
- `council` — [(name, score), ...] 全員降順
- `council_variance_std` — 問題難易度 signal

Client: `reigen_friend_client.run_odin_remote(server, key, eval_fn, ranges, ...)`。

**注意**: server 側は `executor="thread"` で 4 specialist 並列、client 側の eval_fn は Queue(1) で逐次化される。実測 eval 総数は単独 mimir より多い (4-20×)、wall time は同程度。
