"""Pipeline — AT投機的/パイプライン実行。

Phase重複でイテレーション時間を短縮:
  1. Benchmark pre-warm: Phase 4中にPhase 5のタスクファイルを先読み
  2. Diagnosis pre-compute: Phase 1中に部分結果から診断を先行開始
  3. Parallel validation: 初期タスクで回帰検出→早期終了
  4. Surrogate warm-up: Phase 4直後にサロゲートへデータ投入

全閾値: configs/pipeline_params.json（optimize()で進化可能）
"""

import hashlib
import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


# === パラメータロード ===

_PARAMS_JSON = os.path.join(
    os.path.dirname(__file__), "configs", "pipeline_params.json"
)

_DEFAULT_PARAMS = {
    "prewarm": {
        "enabled": True,
        "prefetch_task_files": True,
        "precompute_signatures": True,
        "result_timeout": 30.0,
    },
    "incremental_diagnosis": {
        "enabled": True,
        "partial_interval": 10,
        "min_tasks_before_start": 5,
        "poll_interval": 2.0,
        "stop_timeout": 5.0,
    },
    "early_exit": {
        "enabled": True,
        "probe_count": 5,
        "regression_threshold": 5.0,
        "min_sample_size": 10,
    },
    "surrogate_warmup": {
        "enabled": True,
        "feed_immediately": True,
        "wait_timeout": 5.0,
    },
    "cleanup": {
        "surrogate_timeout": 2.0,
    },
}


def load_pipeline_params(json_path=None):
    """パイプラインパラメータをロード。無ければデフォルト。"""
    path = json_path or _PARAMS_JSON
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # _commentを除去してデフォルトとマージ
            data = {k: v for k, v in data.items() if not k.startswith("_")}
            merged = {}
            for section, defaults in _DEFAULT_PARAMS.items():
                if isinstance(defaults, dict):
                    merged[section] = dict(defaults)
                    if section in data and isinstance(data[section], dict):
                        merged[section].update(data[section])
                else:
                    merged[section] = data.get(section, defaults)
            return merged
        except Exception:
            pass
    return {k: (dict(v) if isinstance(v, dict) else v)
            for k, v in _DEFAULT_PARAMS.items()}


# === 1. Benchmark Pre-warm ===

class BenchmarkPrewarmer:
    """Phase 4 (optimize) 中にPhase 5のベンチマーク準備をバックグラウンド実行。

    タスクファイルの先読み + シグネチャ事前計算でI/O待ちを排除。
    """

    def __init__(self, params: dict):
        self._params = params.get("prewarm", {})
        self._thread: Optional[threading.Thread] = None
        self._result: Dict[str, Any] = {}
        self._ready = threading.Event()

    @property
    def enabled(self) -> bool:
        return self._params.get("enabled", True)

    def start(self, task_dir: str, task_globs: List[str] = None):
        """バックグラウンドでタスクファイルを先読み開始。"""
        if not self.enabled:
            return
        if task_globs is None:
            task_globs = ["*.json"]

        self._ready.clear()
        self._result = {}
        self._thread = threading.Thread(
            target=self._prewarm_worker,
            args=(task_dir, task_globs),
            daemon=True,
        )
        self._thread.start()

    def _prewarm_worker(self, task_dir: str, task_globs: List[str]):
        """ワーカー: ファイル一覧取得 + 内容先読み + シグネチャ計算。"""
        try:
            import glob as glob_mod

            files = []
            for pattern in task_globs:
                full_pattern = os.path.join(task_dir, pattern)
                files.extend(glob_mod.glob(full_pattern))

            cache = {}
            signatures = {}

            for fpath in files:
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    cache[fpath] = content

                    if self._params.get("precompute_signatures", True):
                        sig = hashlib.md5(content.encode("utf-8")).hexdigest()[:12]
                        signatures[fpath] = sig
                except Exception:
                    continue

            self._result = {
                "files": files,
                "cache": cache,
                "signatures": signatures,
                "count": len(files),
                "timestamp": time.time(),
            }
        except Exception as e:
            self._result = {"error": str(e), "files": [], "cache": {}, "signatures": {}}
        finally:
            self._ready.set()

    def get_result(self, timeout: float = None) -> Dict[str, Any]:
        """先読み結果を取得（完了まで待機）。"""
        if not self.enabled or self._thread is None:
            return {"files": [], "cache": {}, "signatures": {}}
        if timeout is None:
            timeout = self._params.get("result_timeout", 30.0)
        self._ready.wait(timeout=timeout)
        return self._result

    def is_ready(self) -> bool:
        """先読みが完了しているか。"""
        return self._ready.is_set()


# === 2. Incremental Diagnosis ===

class IncrementalDiagnoser:
    """Phase 1 (benchmark) 実行中に部分結果から診断を先行計算。

    ベンチマークが10タスク完了するごとに統計を更新。
    Phase 1完了時には診断の80%が済んでいる。
    """

    def __init__(self, params: dict):
        self._params = params.get("incremental_diagnosis", {})
        self._partial_stats: Dict[str, Any] = {}
        self._task_count = 0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def enabled(self) -> bool:
        return self._params.get("enabled", True)

    @property
    def partial_interval(self) -> int:
        return self._params.get("partial_interval", 10)

    @property
    def min_tasks_before_start(self) -> int:
        return self._params.get("min_tasks_before_start", 5)

    def start_monitoring(self, result_json_path: str, domain_config,
                         compute_stats_fn: Callable):
        """結果JSONファイルを監視し、部分結果から統計を逐次計算。

        Args:
            result_json_path: ベンチマーク結果JSONのパス
            domain_config: DomainConfig
            compute_stats_fn: _compute_stats相当の関数
        """
        if not self.enabled:
            return

        self._stop_event.clear()
        self._partial_stats = {}
        self._task_count = 0

        self._thread = threading.Thread(
            target=self._monitor_worker,
            args=(result_json_path, domain_config, compute_stats_fn),
            daemon=True,
        )
        self._thread.start()

    def _monitor_worker(self, result_json_path: str, domain_config,
                        compute_stats_fn: Callable):
        """ワーカー: 結果JSONを定期監視して部分統計を計算。"""
        interval = self.partial_interval
        min_start = self.min_tasks_before_start
        last_count = 0

        while not self._stop_event.is_set():
            poll = self._params.get("poll_interval", 2.0)
            self._stop_event.wait(timeout=poll)
            if self._stop_event.is_set():
                break

            try:
                if not os.path.exists(result_json_path):
                    continue

                with open(result_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                details = data.get("results", data.get("details", []))
                current_count = len(details) if details else 0

                if current_count < min_start:
                    continue

                # interval件ごとに更新（または最終結果）
                if current_count - last_count >= interval or current_count == data.get("n_tasks", 0):
                    stats = compute_stats_fn(data, domain_config)
                    with self._lock:
                        self._partial_stats = stats
                        self._task_count = current_count
                    last_count = current_count

            except (json.JSONDecodeError, IOError):
                # ファイル書き込み中の場合はスキップ
                continue
            except Exception:
                continue

    def stop(self):
        """監視を停止。"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            t = self._params.get("stop_timeout", 5.0)
            self._thread.join(timeout=t)

    def get_partial_stats(self) -> Tuple[Dict[str, Any], int]:
        """現在の部分統計を取得。

        Returns:
            (stats_dict, task_count)
        """
        with self._lock:
            return dict(self._partial_stats), self._task_count


# === 3. Early-Exit Validation ===

class EarlyExitValidator:
    """Phase 5で初期タスクのみ実行し、明確な回帰を早期検出。

    benchmark_sample_size > 10の場合、最初のprobe_count件だけ先に実行。
    regression_threshold%以上の悪化なら残りをスキップ。
    """

    def __init__(self, params: dict):
        self._params = params.get("early_exit", {})

    @property
    def enabled(self) -> bool:
        return self._params.get("enabled", True)

    @property
    def probe_count(self) -> int:
        return self._params.get("probe_count", 5)

    @property
    def regression_threshold(self) -> float:
        return self._params.get("regression_threshold", 5.0)

    @property
    def min_sample_size(self) -> int:
        return self._params.get("min_sample_size", 10)

    def should_early_exit(self, benchmark_sample_size: int) -> bool:
        """早期終了チェックが有効かどうか。"""
        if not self.enabled:
            return False
        return benchmark_sample_size > self.min_sample_size

    def check_probe_results(self, probe_result: dict,
                            baseline_score: float,
                            eval_metrics: list) -> dict:
        """プローブ結果から回帰判定。

        Args:
            probe_result: 最初のprobe_count件の結果
            baseline_score: 前回のスコア
            eval_metrics: メトリクス定義

        Returns:
            {
                "early_exit": bool,
                "reason": str,
                "probe_score": float,
                "delta_pct": float,
            }
        """
        from .eval_composer import extract_score

        probe_score = extract_score(probe_result, eval_metrics)

        if baseline_score <= 0:
            return {
                "early_exit": False,
                "reason": "baseline_zero",
                "probe_score": probe_score,
                "delta_pct": 0.0,
            }

        delta_pct = ((probe_score - baseline_score) / abs(baseline_score)) * 100.0

        if delta_pct < -self.regression_threshold:
            return {
                "early_exit": True,
                "reason": f"regression_{delta_pct:.1f}%",
                "probe_score": probe_score,
                "delta_pct": delta_pct,
            }

        return {
            "early_exit": False,
            "reason": "within_threshold",
            "probe_score": probe_score,
            "delta_pct": delta_pct,
        }


# === 4. Surrogate Warm-up ===

class SurrogateWarmer:
    """Phase 4 (optimize) 完了直後にサロゲートへデータを非同期投入。

    Phase 5の完了を待たずに、optimize()の結果をサロゲートの
    観測データとして先行登録する。
    """

    def __init__(self, params: dict):
        self._params = params.get("surrogate_warmup", {})
        self._thread: Optional[threading.Thread] = None

    @property
    def enabled(self) -> bool:
        return self._params.get("enabled", True)

    def feed_async(self, surrogate, params: list, predicted_score: float):
        """サロゲートに予測データをバックグラウンドで投入。

        注: これは「予測」値。Phase 5完了後に実測値でrecord()する。
        ここではサロゲートのhistoryに追加してpredict精度を上げる準備。
        """
        if not self.enabled or surrogate is None:
            return
        if not self._params.get("feed_immediately", True):
            return

        self._thread = threading.Thread(
            target=self._feed_worker,
            args=(surrogate, params, predicted_score),
            daemon=True,
        )
        self._thread.start()

    def _feed_worker(self, surrogate, params: list, predicted_score: float):
        """ワーカー: サロゲートにデータポイント追加。"""
        try:
            import numpy as np
            params_arr = np.array(params, dtype=np.float64)
            # historyに追加（predict精度向上のウォームアップ）
            surrogate.history.append((params_arr.copy(), predicted_score))
        except Exception:
            pass

    def wait(self, timeout: float = None):
        """投入完了を待機。"""
        if self._thread and self._thread.is_alive():
            if timeout is None:
                timeout = self._params.get("wait_timeout", 5.0)
            self._thread.join(timeout=timeout)


# === 統合: PipelineContext ===

class PipelineContext:
    """パイプライン全体を管理するコンテキスト。

    agent_loop.pyから使う。pipeline=Trueの時だけ有効。
    pipeline=Falseなら全てno-op。
    """

    def __init__(self, enabled: bool = True, params_json: str = None):
        self.enabled = enabled
        if enabled:
            self.params = load_pipeline_params(params_json)
        else:
            self.params = _DEFAULT_PARAMS

        self.prewarmer = BenchmarkPrewarmer(self.params)
        self.diagnoser = IncrementalDiagnoser(self.params)
        self.early_exit = EarlyExitValidator(self.params)
        self.surrogate_warmer = SurrogateWarmer(self.params)

        # タイミング記録
        self.timings: Dict[str, float] = {}

    # --- Phase 1 helpers ---

    def start_incremental_diagnosis(self, result_json_path: str,
                                     domain_config, compute_stats_fn):
        """Phase 1開始時に呼ぶ。部分診断を開始。"""
        if not self.enabled:
            return
        self.diagnoser.start_monitoring(
            result_json_path, domain_config, compute_stats_fn
        )

    def stop_incremental_diagnosis(self) -> Tuple[Dict[str, Any], int]:
        """Phase 1完了時に呼ぶ。部分診断結果を取得。"""
        if not self.enabled:
            return {}, 0
        self.diagnoser.stop()
        return self.diagnoser.get_partial_stats()

    # --- Phase 4 helpers ---

    def start_prewarm(self, task_dir: str, task_globs: List[str] = None):
        """Phase 4開始時に呼ぶ。Phase 5用タスクファイルを先読み。"""
        if not self.enabled:
            return
        self.prewarmer.start(task_dir, task_globs)

    def get_prewarm_result(self, timeout: float = 30.0) -> Dict[str, Any]:
        """Phase 5開始前に呼ぶ。先読み結果を取得。"""
        if not self.enabled:
            return {"files": [], "cache": {}, "signatures": {}}
        return self.prewarmer.get_result(timeout)

    def feed_surrogate(self, surrogate, params: list, predicted_score: float):
        """Phase 4完了直後に呼ぶ。サロゲートにデータ投入。"""
        if not self.enabled:
            return
        self.surrogate_warmer.feed_async(surrogate, params, predicted_score)

    # --- Phase 5 helpers ---

    def check_early_exit(self, probe_result: dict,
                         baseline_score: float,
                         eval_metrics: list,
                         benchmark_sample_size: int) -> dict:
        """Phase 5で初期タスク結果を受けて回帰判定。"""
        if not self.enabled:
            return {"early_exit": False, "reason": "pipeline_disabled"}
        if not self.early_exit.should_early_exit(benchmark_sample_size):
            return {"early_exit": False, "reason": "sample_too_small"}
        return self.early_exit.check_probe_results(
            probe_result, baseline_score, eval_metrics
        )

    # --- Timing ---

    def record_timing(self, phase: str, elapsed: float):
        """フェーズの実行時間を記録。"""
        self.timings[phase] = elapsed

    def get_timing_report(self) -> str:
        """タイミングレポートを1行で返す。"""
        if not self.timings:
            return "N/A"
        parts = [f"{k}={v:.1f}s" for k, v in self.timings.items()]
        total = sum(self.timings.values())
        parts.append(f"total={total:.1f}s")
        return ", ".join(parts)

    # --- Cleanup ---

    def cleanup(self):
        """全バックグラウンドスレッドを停止。"""
        self.diagnoser.stop()
        t = self.params.get("cleanup", {}).get("surrogate_timeout", 2.0)
        self.surrogate_warmer.wait(timeout=t)
