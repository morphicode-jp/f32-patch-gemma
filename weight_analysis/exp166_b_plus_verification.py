"""exp166: B+ ((L25 計算 / L26 メタ-verify + emergent timing)) 統計確定実験.

4 patch × 4 prompt × 32 seeds = 512 runs。

設計:
- patch: baseline / L25 単体 ×1.5 / L26 単体 ×1.5 / mix (paper v1 ×1.5)
- prompt:
  P1 (漸化式)      : コイン 6 回 3 連続表 確率 → 5/16
  P2 (combinatorial): 7 人 円卓 並び方 → 720
  P3 (narrative)   : 桃太郎 3 行要約 (no compute, control)
  P4 (multi-step logic): 4 人 身長 ordering → C
- 固定 sampling: temperature=0.7, top_p=0.95, top_k=40, max_tokens=2048
- seed: 0..31

出力:
- exp166_runs.jsonl: 全 512 runs の raw output + metadata
- 後 で exp166_analyze.py で regex 解析 + 統計

実時間: ~3-4 時間 (server start 4 回 + 512 gen × ~20s + cleanup)
"""
import os, sys, time, json, subprocess, signal, re
import urllib.request, urllib.error
from pathlib import Path

BASE_GGUF = "C:/Users/morph/llm/models/google_gemma-4-31B-it-Q2_K.gguf"
PUBLISH_DIR = "publish/release_v1"

MODELS = [
    ("baseline", BASE_GGUF),
    ("L25_x1.5", f"{PUBLISH_DIR}/gemma-4-31B-it-L25x1.5-Q2_K.gguf"),
    ("L26_x1.5", f"{PUBLISH_DIR}/gemma-4-31B-it-L26x1.5-Q2_K.gguf"),
    ("L25L26_x1.5_paperv1", f"{PUBLISH_DIR}/gemma-4-31B-it-L25L26x1.5-Q2_K.gguf"),
]

PROMPTS = {
    "P1_coin_recurrence": {
        "text": (
            "コインを6回投げたとき、表が3回連続して出る確率を求めてください。 "
            "「途中に出てもOK」ではなく「最低1回連続3回」の意味です。 "
            "ステップごとに考えてください。"
        ),
        "ground_truth_patterns": [r"5\s*/\s*16", r"0\.3125", r"31\.25\s*%"],
    },
    "P2_combinatorial": {
        "text": (
            "7 人 が 円卓 に 並ぶ 方法 は 何 通り ありますか。 "
            "ステップ ごと に 考えて 説明 して ください。"
        ),
        "ground_truth_patterns": [r"\b720\b", r"6\s*!"],
    },
    "P3_narrative_control": {
        "text": "桃太郎 の 物語 を 3 行 で 要約 して ください。",
        "ground_truth_patterns": None,  # control, no compute
    },
    "P4_logic_ordering": {
        "text": (
            "A は B より 背 が 高く、 B は C より 背 が 低い。 "
            "C は D より 背 が 高く、 A は D より 背 が 低い。 "
            "誰 が 一番 背 が 高い ですか。 順番 に 推論 して ください。"
        ),
        "ground_truth_patterns": [r"答え.{0,20}\bC\b", r"\bC\s*が\s*一番", r"一番.{0,20}\bC\b"],
    },
}

N_SEEDS = 32
OUT_FILE = "weight_analysis/results/phase2/exp166_runs.jsonl"
LLAMA_SERVER = os.path.expanduser("~/llm/llama-b9016-cuda13.1/llama-server.exe")

# Regex for behavioral coding
RE_MID_WAIT = re.compile(r"(?i)\b(wait|let me (re-?check|verify|reconsider|re-?evaluate|re-?examine)|hmm|actually|hold on)\b")
RE_TAIL_REFLEX = re.compile(r"(?i)(double[- ]?check|let me verify|let me recompute|let me reconsider|let me check again)")
RE_AUX = re.compile(r"(?i)(\(?\s*補\s*足\s*\)?|note that|具体的に|for clarity|in other words|additional|参考|なお)")


def start_server(model_path: str) -> subprocess.Popen:
    """Start llama-server, wait for 'server is listening', return Popen."""
    print(f"  [server] starting {os.path.basename(model_path)}...", flush=True)
    log_path = f"weight_analysis/results/phase2/exp166_server_{os.path.basename(model_path)}.log"
    log_f = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [
            LLAMA_SERVER, "-m", model_path,
            "-ngl", "99", "-c", "4096",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "-fa", "on",
            "--batch-size", "2048", "--ubatch-size", "512",
            "--host", "127.0.0.1", "--port", "8080",
        ],
        stdout=log_f, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    # poll log for "server is listening"
    t0 = time.time()
    while time.time() - t0 < 180:
        if proc.poll() is not None:
            log_f.close()
            raise RuntimeError(f"server died: see {log_path}")
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            if "server is listening" in content:
                time.sleep(2)
                log_f.close()
                print(f"  [server] READY ({time.time()-t0:.0f}s)", flush=True)
                return proc
        except FileNotFoundError:
            pass
        time.sleep(1)
    proc.terminate()
    log_f.close()
    raise RuntimeError(f"server timeout: see {log_path}")


def stop_server(proc: subprocess.Popen) -> None:
    print(f"  [server] stopping...", flush=True)
    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass
    time.sleep(3)


def call_chat(prompt: str, seed: int, timeout: int = 600) -> dict:
    """POST to llama-server OpenAI-compat endpoint.

    Gemma 4 thinking mode emits reasoning tokens in `reasoning_content` field
    (separate from `content`). We retrieve both.
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_tokens": 6144,  # bumped from 2048: Gemma 4 thinking can be verbose
        "seed": seed,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_full_text(result: dict) -> tuple[str, str, str]:
    """Return (full_text, content, reasoning_content) from chat response.

    full_text concatenates reasoning + content for behavioral regex analysis.
    """
    msg = result["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    full = (reasoning + "\n\n" + content).strip() if reasoning else content
    return full, content, reasoning


def analyze_response(content: str, gt_patterns: list | None) -> dict:
    """Apply pre-registered regex to extract behavior + compute_correct."""
    # split into head (mid-stream) and tail (final 30%)
    n = len(content)
    tail_start = max(0, n - max(300, n // 4))
    head = content[:tail_start]
    tail = content[tail_start:]

    mid_wait_count = len(RE_MID_WAIT.findall(head))
    tail_reflex = 1 if RE_TAIL_REFLEX.search(tail) else 0
    aux_spontaneous = 1 if RE_AUX.search(content) else 0

    compute_correct = None
    if gt_patterns is not None:
        compute_correct = 1 if any(re.search(p, content) for p in gt_patterns) else 0

    return {
        "mid_wait_count": mid_wait_count,
        "tail_reflex": tail_reflex,
        "aux_spontaneous": aux_spontaneous,
        "compute_correct": compute_correct,
    }


def main():
    Path("weight_analysis/results/phase2").mkdir(parents=True, exist_ok=True)
    if os.path.exists(OUT_FILE):
        print(f"[resume] existing {OUT_FILE} found, will append (resume mode)", flush=True)

    # Track done (model, prompt, seed) for resume
    done = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("status") == "ok":
                        done.add((d["model"], d["prompt"], d["seed"]))
                except Exception:
                    pass

    print(f"=== exp166 B+ verification ({len(MODELS)} models × {len(PROMPTS)} prompts × {N_SEEDS} seeds = {len(MODELS)*len(PROMPTS)*N_SEEDS} runs) ===", flush=True)
    print(f"[resume] {len(done)} runs already done", flush=True)

    t_total = time.time()
    n_done_session = 0
    n_planned = len(MODELS) * len(PROMPTS) * N_SEEDS - len(done)

    for model_name, model_path in MODELS:
        # Skip if all runs for this model already done
        model_to_run = [
            (prompt_name, seed)
            for prompt_name in PROMPTS
            for seed in range(N_SEEDS)
            if (model_name, prompt_name, seed) not in done
        ]
        if not model_to_run:
            print(f"\n=== {model_name}: SKIP (all {len(PROMPTS)*N_SEEDS} done) ===", flush=True)
            continue

        print(f"\n=== {model_name}: {len(model_to_run)} runs needed ===", flush=True)
        proc = start_server(model_path)
        try:
            for prompt_name, seed in model_to_run:
                prompt_text = PROMPTS[prompt_name]["text"]
                gt_patterns = PROMPTS[prompt_name]["ground_truth_patterns"]

                t0 = time.time()
                rec = {
                    "model": model_name, "prompt": prompt_name, "seed": seed,
                    "ts": time.time(),
                }
                try:
                    result = call_chat(prompt_text, seed)
                    full_text, content, reasoning = extract_full_text(result)
                    tokens = result.get("usage", {}).get("completion_tokens", 0)
                    rec.update({
                        "content": content,
                        "reasoning": reasoning,
                        "full_text_len": len(full_text),
                        "tokens": tokens,
                        "elapsed_s": time.time() - t0,
                        # analyze on full_text (reasoning + content combined)
                        # so "Wait" etc. in Gemma 4 thinking tokens are captured
                        "analysis": analyze_response(full_text, gt_patterns),
                        "status": "ok",
                    })
                except Exception as e:
                    rec.update({
                        "elapsed_s": time.time() - t0,
                        "error": str(e),
                        "status": "error",
                    })

                with open(OUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                n_done_session += 1
                a = rec.get("analysis", {})
                eta_s = (time.time() - t_total) / n_done_session * (n_planned - n_done_session)
                print(
                    f"  [{n_done_session}/{n_planned}] {prompt_name} seed={seed:2d}: "
                    f"tok={rec.get('tokens', 0):4d} mid={a.get('mid_wait_count', 0):2d} "
                    f"tail={a.get('tail_reflex', 0)} aux={a.get('aux_spontaneous', 0)} "
                    f"correct={a.get('compute_correct')} "
                    f"({rec.get('elapsed_s', 0):.0f}s, eta {eta_s/60:.0f}m)",
                    flush=True,
                )
        finally:
            stop_server(proc)

    print(f"\n[done] wall = {(time.time()-t_total)/60:.1f} min, saved to {OUT_FILE}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
