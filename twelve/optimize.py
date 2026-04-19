"""Twelve Engine 統合最適化API — eval_fn + ranges だけで全機能が動く。

使い方（3行）:
    from twelve.optimize import optimize
    best_params, best_score, info = optimize(
        eval_fn=lambda p: my_score(p),
        param_ranges=[(0, 10), (0.1, 1.0), ...],
        time_budget=60,
    )

カテゴリ付き（カテゴリ間をKathara 30辺で協調最適化）:
    best_params, best_score, info = optimize(
        eval_fn=my_eval,
        param_ranges=ranges,
        categories=["quant", "quant", "lora", "lora", ...],
        time_budget=600,
    )

学習モード（使うほど賢くなる）:
    # 1回目: ゼロから探索
    best, score, info = optimize(eval_fn, ranges, learn=True)
    # 2回目: 前回の経験を活かして探索（ウォームスタート+ボトルネック検出+戦略学習）
    best, score, info = optimize(eval_fn, ranges, learn=True)
    # N回目: 蓄積された経験で同じ時間でもっと良いスコア

メタ進化モード（meta=Trueで3段目を追加）:
    best, score, info = optimize(
        eval_fn, ranges, time_budget=120,
        meta=True,
        meta_config={"components": ["lad_accel", "k7", "k8"]},
    )

内部動作:
    Phase 1: TwelveParallel — カテゴリ並列HC + Kathara 30辺共有（広域探索）
    Phase 2: KatharaParamOptimizer — 12候補×5戦略×3ブリッジ（精密探索）
    Phase 3: メタ進化レイヤー — LadAccel / K7 / K8 / K9 / Knowledge（meta=True時）
    学習層: ExperienceStore — 化石記録、ボトルネック検出、戦略学習（B層）
"""

import hashlib
import json
import math
import os
import random
import time


# ═══════════════════════════════════════════════════════
# Meta Parameters — LaD原則: ドメイン知識はJSONに外出し
# ═══════════════════════════════════════════════════════

_TWELVE_DIR = os.path.dirname(os.path.abspath(__file__))
_META_PARAMS_PATH = os.path.join(_TWELVE_DIR, "configs", "twelve_meta_params.json")

def _load_meta_params():
    """twelve/configs/twelve_meta_params.json をロード。
    ファイルが無い or 壊れている場合はハードコードデフォルトにフォールバック。
    ATのフォルダスキャンで発見・最適化可能。
    """
    try:
        with open(_META_PARAMS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

def _mp(section, key, default):
    """Meta parameterを取得。JSON > default のフォールバックチェーン。"""
    return _META_PARAMS.get(section, {}).get(key, default)

# モジュールロード時に1回読む。reload_meta_params()で再読み込み可能。
_META_PARAMS = _load_meta_params()

def reload_meta_params():
    """twelve/configs/twelve_meta_params.json を再読み込み。
    ATがパラメータを書き換えた後に呼ぶ。
    """
    global _META_PARAMS
    _META_PARAMS = _load_meta_params()
    return _META_PARAMS


# ═══════════════════════════════════════════════════════
# ExperienceStore — 最適化経験の永続化（汎用B層）
# ═══════════════════════════════════════════════════════

_EXP_DIR = os.path.join(_TWELVE_DIR, "experience")


class ExperienceStore:
    """最適化経験の永続化 — 使うほど賢くなる。

    記録するもの:
      - 化石記録: 歴代ベストのparams+score（ロールバック保護）
      - カテゴリ統計: 各カテゴリの収束速度・改善量（ボトルネック検出）
      - 戦略統計: 各変異戦略の成功率（戦略学習）
      - メタパラメータ: 最適σ、Phase配分比率（自己調整）
    """

    def __init__(self, experience_id, exp_dir=None):
        self._dir = exp_dir or _EXP_DIR
        os.makedirs(self._dir, exist_ok=True)
        self._path = os.path.join(self._dir, f"{experience_id}.json")
        self._data = self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 既存データのマイグレーション（Phase 3フィールド追加）
                meta = data.get("meta", {})
                if "phase3_ratio" not in meta:
                    meta["phase3_ratio"] = 0.2
                if "meta_genome" not in meta:
                    meta["meta_genome"] = None
                if "k7_best_genome" not in meta:
                    meta["k7_best_genome"] = None
                if "meta_runs" not in meta:
                    meta["meta_runs"] = 0
                data["meta"] = meta
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "version": 1,
            "runs": 0,
            "total_evals": 0,
            "fossil_record": [],          # [{score, params, run}] 歴代ベスト
            "category_stats": {},         # {cat: {total_improvement, runs, convergence_rate, best_sigma}}
            "strategy_stats": {           # 変異戦略ごとの成功率
                "small": [0, 0],          # [improvements, attempts]
                "medium": [0, 0],
                "large": [0, 0],
                "reset": [0, 0],
            },
            "meta": {
                "phase1_ratio": _mp("phase_allocation", "phase1_default_ratio", 0.6),
                "best_score_ever": None,
                "best_params_ever": None,
                "phase3_ratio": _mp("phase_allocation", "phase3_default_ratio", 0.2),
                "meta_genome": None,      # LadAccelerator最良genome
                "k7_best_genome": None,   # K7最良genome
                "meta_runs": 0,           # Phase3実行回数
            },
        }

    def save(self):
        """Atomic write: unique tmp per writer + os.replace. Retries on
        Windows PermissionError caused by brief contention (antivirus scan,
        indexer, or another writer mid-replace). Readers always see a
        consistent snapshot; same-ID parallel writers have lost-update (last
        writer wins). Mirrors reigen._mk_save() pattern.
        """
        import threading as _th
        import time as _time
        # Per-caller unique tmp: PID + thread id + monotonic ns
        tmp = (f"{self._path}.{os.getpid()}.{_th.get_ident()}."
               f"{_time.monotonic_ns()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            # Windows os.replace can fail with PermissionError under heavy
            # concurrency (another writer holding handle briefly). Retry up
            # to 5 times with exponential backoff; after that, raise so the
            # caller is aware.
            last_err = None
            for attempt in range(5):
                try:
                    os.replace(tmp, self._path)
                    return
                except PermissionError as e:
                    last_err = e
                    _time.sleep(0.01 * (2 ** attempt))  # 10ms, 20, 40, 80, 160
            raise last_err
        finally:
            # If os.replace never succeeded, remove our orphan tmp.
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    @property
    def runs(self):
        return self._data["runs"]

    # ─── 化石記録 ───

    def get_best_params(self):
        """歴代ベストのparamsを返す（ウォームスタート用）"""
        meta = self._data["meta"]
        if meta.get("best_params_ever") is not None:
            return meta["best_params_ever"]
        return None

    def get_best_score(self):
        meta = self._data["meta"]
        return meta.get("best_score_ever")

    def record_fossil(self, params, score):
        """化石記録に追加。ベスト更新なら保存"""
        fossil = self._data["fossil_record"]
        fossil.append({
            "score": round(score, 6),
            "run": self._data["runs"],
        })
        # 最新N件だけ保持（paramsは最良だけ）
        max_records = int(_mp("experience_learning", "fossil_max_records", 10))
        if len(fossil) > max_records:
            self._data["fossil_record"] = fossil[-max_records:]

        meta = self._data["meta"]
        if meta.get("best_score_ever") is None or score > meta["best_score_ever"]:
            meta["best_score_ever"] = round(score, 6)
            meta["best_params_ever"] = [round(p, 8) for p in params]

    # ─── ボトルネック検出 ───

    def get_category_budgets(self, cat_names, total_budget):
        """カテゴリごとの予算配分を返す。ボトルネック（改善余地大）に多く配分。

        Returns: {cat_name: budget_fraction}
        """
        stats = self._data["category_stats"]
        n = len(cat_names)
        if not stats or self._data["runs"] < 1:
            # 経験なし → 均等配分
            return {cat: 1.0 / n for cat in cat_names}

        # 各カテゴリの「改善余地」を推定
        # convergence_rate が低い = まだ改善中 = ボトルネック = 予算多め
        headroom_floor = _mp("category_budget", "headroom_floor", 0.1)
        headroom = {}
        for cat in cat_names:
            cs = stats.get(cat, {})
            conv_rate = cs.get("convergence_rate", 0.5)
            # convergence_rate: 0→全く収束してない（予算多く欲しい）, 1→完全収束（少なくていい）
            headroom[cat] = max(headroom_floor, 1.0 - conv_rate)

        total = sum(headroom.values())
        return {cat: h / total for cat, h in headroom.items()}

    def record_category_result(self, cat_name, improvement, n_evals, budget):
        """カテゴリの結果を記録"""
        stats = self._data["category_stats"]
        if cat_name not in stats:
            stats[cat_name] = {
                "total_improvement": 0.0,
                "runs": 0,
                "convergence_rate": 0.0,
            }
        cs = stats[cat_name]
        cs["runs"] += 1
        cs["total_improvement"] += improvement

        # 収束率の推定: 改善量が小さい→収束に近い
        # 指数移動平均で更新
        alpha = _mp("experience_learning", "convergence_ema_alpha", 0.3)
        ct = _META_PARAMS.get("convergence_thresholds", {})
        t1 = ct.get("tier1_improvement", 0.01)
        r1 = ct.get("tier1_rate", 1.0)
        t2 = ct.get("tier2_improvement", 0.1)
        r2 = ct.get("tier2_rate", 0.8)
        t3 = ct.get("tier3_improvement", 1.0)
        r3 = ct.get("tier3_rate", 0.4)
        r4 = ct.get("tier4_rate", 0.1)
        if improvement < t1:
            new_conv = r1
        elif improvement < t2:
            new_conv = r2
        elif improvement < t3:
            new_conv = r3
        else:
            new_conv = r4
        cs["convergence_rate"] = cs["convergence_rate"] * (1 - alpha) + new_conv * alpha

    # ─── グローバル経験集計 ───
    # share()が強すぎるとperturb()が死ぬ。
    # → グローバル経験を参考にしつつ、一定確率で無視して独自探索。
    # ローカル経験が育つほどグローバル依存度を下げる（自立）。

    def _exploration_rate(self):
        """グローバル経験を無視して独自探索する確率。
        ローカルrunが多いほど自立（探索率低下）。
        runs=0: 0.3（30%独自探索）, runs=5+: 0.05（5%独自探索）
        """
        runs = self._data["runs"]
        initial = _mp("experience_learning", "exploration_rate_initial", 0.3)
        decay = _mp("experience_learning", "exploration_rate_decay", 0.7)
        floor = _mp("experience_learning", "exploration_rate_floor", 0.05)
        return max(floor, initial * (decay ** runs))

    @staticmethod
    def _read_experience_file(fpath):
        """1つのexperienceファイルを読む（ThreadPoolExecutor用）。"""
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            return None

    def _list_experience_files(self):
        """自分以外のexperienceファイルパスを列挙。"""
        try:
            return [os.path.join(self._dir, fn) for fn in os.listdir(self._dir)
                    if fn.endswith(".json") and os.path.join(self._dir, fn) != self._path]
        except OSError:
            return []

    def _collect_global_strategy_stats(self):
        """全experienceファイルのstrategy_statsを並列集計。
        K²改善: ThreadPoolExecutorでファイルI/O並列化。
        """
        from concurrent.futures import ThreadPoolExecutor

        global_stats = {"small": [0, 0], "medium": [0, 0],
                        "large": [0, 0], "reset": [0, 0]}
        files = self._list_experience_files()
        if not files:
            return global_stats

        try:
            with ThreadPoolExecutor(max_workers=min(8, len(files))) as pool:
                results = list(pool.map(self._read_experience_file, files))
        except Exception:
            results = [self._read_experience_file(f) for f in files]

        for data in results:
            if data is None:
                continue
            ss = data.get("strategy_stats", {})
            for strat in global_stats:
                if strat in ss:
                    global_stats[strat][0] += ss[strat][0]
                    global_stats[strat][1] += ss[strat][1]
        return global_stats

    def _collect_global_best_genome(self, component):
        """全experienceから指定コンポーネントの最良genomeを並列探索。
        K²改善: ThreadPoolExecutorでファイルI/O並列化。
        """
        from concurrent.futures import ThreadPoolExecutor

        files = self._list_experience_files()
        if not files:
            return None

        try:
            with ThreadPoolExecutor(max_workers=min(8, len(files))) as pool:
                results = list(pool.map(self._read_experience_file, files))
        except Exception:
            results = [self._read_experience_file(f) for f in files]

        best_genome = None
        best_score = -1e30
        for data in results:
            if data is None:
                continue
            meta_results = data.get("meta_results", {})
            comp_result = meta_results.get(component, {})
            score = comp_result.get("best_score", -1e30)
            genome = comp_result.get("best_genome")
            if genome and score > best_score:
                best_score = score
                best_genome = genome
        return best_genome

    # ─── 戦略学習 ───

    def get_strategy_weights(self):
        """学習済み戦略重みを返す。成功率が高い戦略を優先。
        自分のデータが不足(< 100 attempts)の場合、全experienceのグローバル集計を参考にする。
        探索率に基づきグローバル知識と均等探索をブレンド（share×perturb均衡）。

        Returns: {"small": weight, "medium": weight, ...}
        """
        stats = self._data["strategy_stats"]
        total_attempts = sum(s[1] for s in stats.values())

        if total_attempts >= 100:
            # 自分のデータで十分 → 自分のstatsのみ使用
            use_stats = stats
        else:
            # データ不足 → グローバル集計 + 自分のデータをマージ
            global_stats = self._collect_global_strategy_stats()
            use_stats = {}
            for strat in ["small", "medium", "large", "reset"]:
                local = stats.get(strat, [0, 0])
                glob = global_stats.get(strat, [0, 0])
                use_stats[strat] = [local[0] + glob[0], local[1] + glob[1]]
            merged_total = sum(s[1] for s in use_stats.values())
            min_attempts = _mp("experience_learning", "min_attempts_for_learned", 100)
            if merged_total < min_attempts:
                defaults = _META_PARAMS.get("strategy_weights_default",
                    {"small": 1.0, "medium": 1.0, "large": 1.0, "reset": 1.0})
                return {k: v for k, v in defaults.items() if not k.startswith("_")}

        # 集合知から重みを計算
        success_mult = _mp("experience_learning", "strategy_success_multiplier", 4.0)
        weight_floor = _mp("experience_learning", "strategy_weight_floor", 0.2)
        learned = {}
        for strat, val in use_stats.items():
            imps, atts = val if isinstance(val, list) else (0, 0)
            if atts == 0:
                learned[strat] = 1.0
            else:
                learned[strat] = max(weight_floor, imps / atts * success_mult)

        # 探索率でブレンド: (1-ε)×集合知 + ε×均等
        # share()が強すぎるとperturb()が死ぬ → 常に独自探索の余地を残す
        eps = self._exploration_rate()
        weights = {}
        for strat in learned:
            weights[strat] = (1.0 - eps) * learned[strat] + eps * 1.0
        return weights

    def record_strategy_result(self, strategy, improved):
        """戦略の結果を記録"""
        stats = self._data["strategy_stats"]
        if strategy not in stats:
            stats[strategy] = [0, 0]
        stats[strategy][1] += 1
        if improved:
            stats[strategy][0] += 1

    # ─── メタ ───

    def get_phase1_ratio(self):
        """Phase1のtime_budget配分比率（学習済み）"""
        default = _mp("phase_allocation", "phase1_default_ratio", 0.6)
        return self._data["meta"].get("phase1_ratio", default)

    def record_phase_effectiveness(self, p1_improvement, p2_improvement):
        """Phase配分の学習。改善が大きいPhaseに寄せる"""
        total = abs(p1_improvement) + abs(p2_improvement) + 0.001
        p1_share = abs(p1_improvement) / total
        # 現在値と混合（急激に変えない）
        alpha = _mp("experience_learning", "phase_ema_alpha", 0.3)
        old = self._data["meta"]["phase1_ratio"]
        self._data["meta"]["phase1_ratio"] = round(old * (1 - alpha) + p1_share * alpha, 3)
        # 範囲制限
        p1_min = _mp("phase_allocation", "phase1_min_ratio", 0.3)
        p1_max = _mp("phase_allocation", "phase1_max_ratio", 0.8)
        self._data["meta"]["phase1_ratio"] = max(p1_min, min(p1_max,
            self._data["meta"]["phase1_ratio"]))

    # ─── Phase 3 メタ進化 ───

    def record_meta_result(self, component, result_dict):
        """Phase 3コンポーネントの結果を記録"""
        if "meta_results" not in self._data:
            self._data["meta_results"] = {}
        self._data["meta_results"][component] = result_dict
        # best genomeを全コンポーネント保存
        if result_dict.get("best_genome"):
            self._data["meta"][f"{component}_best_genome"] = result_dict["best_genome"]
        # 後方互換: lad_accelはmeta_genomeにも保存
        if component == "lad_accel" and result_dict.get("best_genome"):
            self._data["meta"]["meta_genome"] = result_dict["best_genome"]
        self._data["meta"]["meta_runs"] = self._data["meta"].get("meta_runs", 0) + 1

    def get_meta_genome(self):
        """前回のLadAccelerator最良genomeを返す。
        自分に無ければグローバル経験から探す。
        探索率に基づき一定確率でNoneを返し、ゼロから探索を強制する。"""
        local = self._data.get("meta", {}).get("meta_genome")
        if local is not None:
            return local
        # グローバルフォールバック（探索率で独自探索の余地を残す）
        if random.random() < self._exploration_rate():
            return None  # 独自探索を強制
        return self._collect_global_best_genome("lad_accel")

    def get_k7_genome(self):
        """前回のK7最良genomeを返す。
        自分に無ければグローバル経験から最良を探す。
        探索率に基づき一定確率でNoneを返す。"""
        local = self._data.get("meta", {}).get("k7_best_genome")
        if local is not None:
            return local
        if random.random() < self._exploration_rate():
            return None
        return self._collect_global_best_genome("k7")

    def get_component_genome(self, component):
        """任意コンポーネントの最良genomeを返す。
        自分に無ければグローバル経験から最良を探す。
        探索率に基づき一定確率でNoneを返す。"""
        local = self._data.get("meta", {}).get(f"{component}_best_genome")
        if local is not None:
            return local
        if random.random() < self._exploration_rate():
            return None
        return self._collect_global_best_genome(component)

    def finish_run(self, total_evals):
        """ラン完了記録"""
        self._data["runs"] += 1
        self._data["total_evals"] += total_evals


def _make_experience_id(param_ranges, categories):
    """param_ranges + categories からユニークIDを生成"""
    sig = f"n={len(param_ranges)}"
    if categories:
        sig += f",cats={sorted(set(categories))}"
    # 短いハッシュ
    h = hashlib.md5(sig.encode()).hexdigest()[:12]
    return f"exp_{len(param_ranges)}d_{h}"


# ═══════════════════════════════════════════════════════
# 統合最適化API
# ═══════════════════════════════════════════════════════

def optimize(eval_fn, param_ranges, time_budget=60,
             categories=None, initial_params=None, verbose=False,
             learn=False, experience_id=None,
             meta=False,
             meta_config=None,
             eval_sec_hint=None,
             multi_start=None):
    """Twelve Engine 統合最適化。

    Args:
        eval_fn: (params: list[float]) -> float  スコア関数（高いほど良い）
        param_ranges: list[(lo, hi)]  各パラメータの範囲
        time_budget: 秒（デフォルト60）
        categories: list[str] | None  各パラメータのカテゴリ名
        initial_params: list[float] | None  初期値
        verbose: ログ出力
        learn: True で経験永続化ON（使うほど賢くなる）
        experience_id: 経験ファイルID（Noneで自動生成）
        meta: True でPhase 3（メタ進化レイヤー）を有効化
        meta_config: Phase 3設定辞書（Noneでデフォルト）
        multi_start: 多地点スカウト数（None=自動, 1=従来動作）

    Returns:
        (best_params, best_score, info) タプル
    """
    from .twelve_parallel import TwelveParallel, LayerSpec
    from .twelve_auto_stack import KatharaParamOptimizer

    # ATが書き換えた最新値を反映
    reload_meta_params()

    # Phase 3 メタ進化デフォルト設定 (JSONから読み込み)
    mp3 = _META_PARAMS.get("meta_phase3", {})
    DEFAULT_META_CONFIG = {
        "time_ratio": mp3.get("time_ratio", 0.2),
        "components": mp3.get("default_components", ["lad_accel", "k7"]),
        "lad_accel_budget": mp3.get("lad_accel_budget", 5),
        "k7_budget": None,
        "k8_budget": None,
        "k9_budget": None,
        "k10_budget": None,
        "k11_budget": None,
        "k12_budget": None,
        "knowledge_budget": None,
    }

    t0 = time.time()
    n_params = len(param_ranges)

    # ─── Phase 3 時間配分 ───
    if meta:
        cfg = {**DEFAULT_META_CONFIG, **(meta_config or {})}
        phase3_time = time_budget * cfg["time_ratio"]
        remaining_for_p1p2 = time_budget - phase3_time
    else:
        cfg = None
        phase3_time = 0
        remaining_for_p1p2 = time_budget

    # ─── 経験ロード ───
    exp = None
    if learn:
        eid = experience_id or _make_experience_id(param_ranges, categories)
        exp = ExperienceStore(eid)
        if verbose and exp.runs > 0:
            print(f"  [学習] 経験ロード: {exp.runs}回目, 歴代ベスト: {exp.get_best_score()}")

    # ─── 初期値（化石記録からウォームスタート） ───
    n_params = len(param_ranges)
    if initial_params is None:
        if exp and exp.get_best_params() is not None:
            fossil_params = exp.get_best_params()
            if len(fossil_params) == n_params:
                initial_params = fossil_params
                if verbose:
                    print(f"  [学習] ウォームスタート: 化石記録から初期値復元")
            else:
                if verbose:
                    print(f"  [学習] 化石パラメータ長不一致({len(fossil_params)}!={n_params}), デフォルト使用")
                initial_params = [(lo + hi) / 2 for lo, hi in param_ranges]
        else:
            initial_params = [(lo + hi) / 2 for lo, hi in param_ranges]
    # 安全弁: initial_paramsの長さがparam_rangesと一致しない場合はデフォルトにフォールバック
    if len(initial_params) != n_params:
        if verbose:
            print(f"  [警告] initial_params長({len(initial_params)})!=param_ranges({n_params}), デフォルト使用")
        initial_params = [(lo + hi) / 2 for lo, hi in param_ranges]

    # eval速度測定
    if eval_sec_hint is not None and eval_sec_hint > 0:
        # 外部から実測値が渡された場合はそれを信頼
        baseline_score = eval_fn(initial_params)
        eval_sec = eval_sec_hint
    else:
        t_m = time.time()
        baseline_score = eval_fn(initial_params)
        eval_sec = max(0.00001, time.time() - t_m)
    evals_per_sec = 1.0 / eval_sec

    # ─── Multi-start スカウティング ───
    _ms_cfg = _META_PARAMS.get("multi_start", {})
    if multi_start is None:
        n_starts = _ms_cfg.get("default_starts", 5) if time_budget >= 60 else 1
    else:
        n_starts = max(1, int(multi_start))

    if n_starts > 1:
        scout_ratio = _ms_cfg.get("budget_ratio", 0.1)
        scout_budget = min(time_budget * scout_ratio, _ms_cfg.get("max_seconds", 10.0))
        t_scout = time.time()
        scouts = [(list(initial_params), baseline_score)]
        rng_scout = random.Random(42)
        while time.time() - t_scout < scout_budget:
            p = [rng_scout.uniform(lo, hi) for lo, hi in param_ranges]
            s = eval_fn(p)
            scouts.append((p, s))
            if len(scouts) >= n_starts * 50:
                break
        scouts.sort(key=lambda x: x[1], reverse=True)
        if scouts[0][1] > baseline_score:
            initial_params = list(scouts[0][0])
            baseline_score = scouts[0][1]
        remaining_for_p1p2 -= (time.time() - t_scout)
        if verbose:
            print(f"  Multi-start: {len(scouts)}地点スカウト, "
                  f"最良={scouts[0][1]:.4f} ({len(scouts) - 1}候補)")

    if verbose:
        run_label = f" (Run #{exp.runs + 1})" if exp else ""
        print(f"Twelve optimize{run_label}: {n_params}パラメータ, {time_budget}秒")
        print(f"  eval速度: {evals_per_sec:.0f}/秒, ベースライン: {baseline_score:.4f}")

    # カテゴリ判定
    unique_cats = set(categories) if categories else set()
    use_parallel = categories is not None and len(unique_cats) >= 2

    p1_result = {}
    p1_score = baseline_score
    p1_evals = 0
    current_params = list(initial_params)

    # ─── Phase配分（経験から学習） ───
    phase1_ratio = exp.get_phase1_ratio() if exp else _mp("phase_allocation", "phase1_default_ratio", 0.6)

    # ─── 戦略重み（経験から学習） ───
    strategy_weights = exp.get_strategy_weights() if exp else None

    # カテゴリ別改善追跡（学習記録用）
    cat_improvements = {}

    if use_parallel:
        # ═══ Phase 1: TwelveParallel — カテゴリ並列 + Kathara 30辺共有 ═══
        phase1_time = remaining_for_p1p2 * phase1_ratio

        # カテゴリ別にパラメータをグルーピング
        cat_groups = {}
        for i, cat in enumerate(categories):
            cat_groups.setdefault(cat, []).append(i)

        if len(cat_groups) > 12:
            sorted_cats = sorted(cat_groups.items(), key=lambda x: len(x[1]))
            while len(sorted_cats) > 12:
                c1_name, c1_idxs = sorted_cats.pop(0)
                c2_name, c2_idxs = sorted_cats.pop(0)
                merged_name = f"{c1_name}+{c2_name}"
                sorted_cats.insert(0, (merged_name, c1_idxs + c2_idxs))
            cat_groups = dict(sorted_cats)

        # ─── ボトルネック検出 → 予算傾斜配分 ───
        cat_budgets = None
        if exp and exp.runs > 0:
            cat_budgets = exp.get_category_budgets(
                list(cat_groups.keys()),
                phase1_time * evals_per_sec,
            )

        cat_best = {}
        layer_specs = []

        # mutation sigma をJSON configから取得（クロージャ外で1回だけ）
        _ms = _META_PARAMS.get("mutation_sigma", {})
        _small_sigma = _ms.get("small_sigma", 0.05)
        _medium_sigma = _ms.get("medium_sigma", 0.15)
        _medium_max_dims = int(_ms.get("medium_max_dims", 3))
        _large_max_dims = int(_ms.get("large_max_dims", 5))

        for cat_name, global_indices in sorted(cat_groups.items()):
            tracker = {"params": None, "score": -float("inf"),
                       "start_score": None, "strategy_log": []}
            cat_best[cat_name] = (global_indices, tracker)

            def make_fns(indices, trk, base_params, ranges, strat_weights):
                def init_fn():
                    return [base_params[i] for i in indices]

                def mutate_fn(state):
                    new = state[:]
                    # 戦略選択（経験がある場合は重み付き）
                    if strat_weights:
                        strats = list(strat_weights.keys())
                        weights = [strat_weights[s] for s in strats]
                        total_w = sum(weights)
                        r = random.random() * total_w
                        cumul = 0
                        strat = strats[-1]
                        for s, w in zip(strats, weights):
                            cumul += w
                            if r <= cumul:
                                strat = s
                                break
                    else:
                        strat = random.choice(["small", "medium", "large", "reset"])

                    if strat == "small":
                        idx = random.randint(0, len(new) - 1)
                        lo, hi = ranges[indices[idx]]
                        new[idx] = max(lo, min(hi, new[idx] + random.gauss(0, (hi - lo) * _small_sigma)))
                    elif strat == "medium":
                        for _ in range(random.randint(1, min(_medium_max_dims, len(new)))):
                            idx = random.randint(0, len(new) - 1)
                            lo, hi = ranges[indices[idx]]
                            new[idx] = max(lo, min(hi, new[idx] + random.gauss(0, (hi - lo) * _medium_sigma)))
                    elif strat == "large":
                        for _ in range(random.randint(1, max(1, min(len(new) // 2, _large_max_dims)))):
                            idx = random.randint(0, len(new) - 1)
                            lo, hi = ranges[indices[idx]]
                            new[idx] = random.uniform(lo, hi)
                    else:
                        idx = random.randint(0, len(new) - 1)
                        lo, hi = ranges[indices[idx]]
                        new[idx] = random.uniform(lo, hi)

                    # 戦略ログ記録
                    trk["strategy_log"].append(strat)
                    return new

                def score_fn(state):
                    full = list(base_params)
                    for local_idx, global_idx in enumerate(indices):
                        full[global_idx] = state[local_idx]
                    score = eval_fn(full)
                    if trk["start_score"] is None:
                        trk["start_score"] = score
                    if score > trk["score"]:
                        # 戦略成功ログ
                        if trk["strategy_log"]:
                            trk.setdefault("strategy_successes", []).append(
                                trk["strategy_log"][-1])
                        trk["score"] = score
                        trk["params"] = state[:]
                    return score

                return init_fn, mutate_fn, score_fn

            i_fn, m_fn, s_fn = make_fns(
                global_indices, tracker, current_params, param_ranges, strategy_weights)

            # カテゴリ別予算傾斜（scaleで調整）
            scale = 1.0
            if cat_budgets:
                n_cats = len(cat_groups)
                scale = cat_budgets.get(cat_name, 1.0 / n_cats) * n_cats
                bs_min = _mp("category_budget", "budget_scale_min", 0.3)
                bs_max = _mp("category_budget", "budget_scale_max", 3.0)
                scale = max(bs_min, min(bs_max, scale))

            layer_specs.append(LayerSpec(cat_name, i_fn, m_fn, s_fn, scale))

        base_budget = int(phase1_time * evals_per_sec / len(layer_specs))
        # 重いeval_fn（数秒〜数十秒/eval）でも最低限探索できるよう下限を調整
        _pb = _META_PARAMS.get("parallel_budget", {})
        max_bpl = int(_pb.get("max_budget_per_layer", 500000))
        min_b_frac = _pb.get("min_budget_fraction", 0.5)
        min_b_abs = int(_pb.get("min_budget_absolute", 3))
        max_min_b = int(_pb.get("max_min_budget", 500))
        min_budget = max(min_b_abs, min(max_min_b, int(phase1_time * evals_per_sec * min_b_frac)))
        budget_per_layer = max(min_budget, min(base_budget, max_bpl))

        if verbose:
            print(f"  Phase 1: TwelveParallel ({len(layer_specs)}カテゴリ並列)"
                  f" [配分比率: {phase1_ratio:.0%}]")
            for cat_name, (indices, _) in sorted(cat_best.items()):
                budget_info = ""
                if cat_budgets:
                    s = cat_budgets.get(cat_name, 0) * len(cat_groups)
                    if abs(s - 1.0) > 0.1:
                        budget_info = f" ×{s:.1f}" if s > 1.0 else f" ×{s:.1f}"
                print(f"    {cat_name:15s}: {len(indices)}パラメータ{budget_info}")
            print(f"    budget/layer: {budget_per_layer}")

        share_div = int(_pb.get("share_interval_divisor", 10))
        share_min = int(_pb.get("share_interval_min", 100))
        parallel = TwelveParallel(layer_specs, verbose=verbose)

        # --- K² Phase 1 チャンク分割: 早期収束でPhase 2に遷移 ---
        n_p1_chunks = int(_pb.get("n_chunks", 4))
        chunk_budget = max(1, budget_per_layer // n_p1_chunks)
        prev_chunk_score = baseline_score
        convergence_thresh = _pb.get("convergence_threshold", 0.001)
        p1_result = {}

        for p1_chunk in range(n_p1_chunks):
            chunk_result = parallel.run(
                budget_per_layer=chunk_budget,
                share_interval=max(share_min, chunk_budget // max(share_div, 1)),
            )
            p1_result = chunk_result

            # チャンク間収束判定: スコア改善が閾値未満なら早期Phase 2遷移
            for cat_name, (indices, tracker) in cat_best.items():
                if tracker["params"] is not None:
                    for local_idx, global_idx in enumerate(indices):
                        current_params[global_idx] = tracker["params"][local_idx]
            chunk_score = eval_fn(current_params)

            improvement = abs(chunk_score - prev_chunk_score)
            if p1_chunk > 0 and improvement < convergence_thresh * abs(prev_chunk_score + 1e-10):
                if verbose:
                    print(f"  Phase 1 chunk {p1_chunk+1}/{n_p1_chunks}: converged "
                          f"(delta={improvement:.6f} < thresh)")
                break
            prev_chunk_score = chunk_score

            if verbose and n_p1_chunks > 1:
                print(f"  Phase 1 chunk {p1_chunk+1}/{n_p1_chunks}: "
                      f"score={chunk_score:.4f} (delta={improvement:.6f})")

        # Phase 1 結果反映 + カテゴリ改善記録
        for cat_name, (indices, tracker) in cat_best.items():
            if tracker["params"] is not None:
                for local_idx, global_idx in enumerate(indices):
                    current_params[global_idx] = tracker["params"][local_idx]

            # カテゴリ改善量を記録
            start = tracker.get("start_score") or baseline_score
            end = tracker["score"] if tracker["score"] > -float("inf") else start
            cat_improvements[cat_name] = end - start

        p1_score = eval_fn(current_params)
        p1_evals = p1_result.get("total_evals", 0)

        if verbose:
            print(f"  Phase 1 完了: {baseline_score:.4f} → {p1_score:.4f} "
                  f"(+{p1_score - baseline_score:.4f})")

    # ═══ Phase 2: KatharaParamOptimizer — 全パラメータ精密探索 ═══
    phase2_time = remaining_for_p1p2 - (time.time() - t0)

    _p2 = _META_PARAMS.get("phase2", {})
    p2_min_time = _p2.get("min_time_threshold", 2)
    p2_improvement = 0.0
    if phase2_time > p2_min_time:
        if verbose:
            print(f"  Phase 2: KatharaParamOptimizer "
                  f"({n_params}パラメータ, {phase2_time:.0f}秒)")

        surr_thresh = _p2.get("surrogate_eval_threshold", 0.01)
        use_surrogate = eval_sec > surr_thresh
        optimizer = KatharaParamOptimizer(
            param_ranges=param_ranges,
            eval_fn=eval_fn,
            initial_params=current_params,
            use_surrogate=use_surrogate,
        )

        # eval速度から世代数を推定: time_budget内に収まる最大世代数
        candidates = int(_p2.get("candidates_per_gen", 12))
        gen_sec = max(0.0001, eval_sec * candidates)
        min_gens = int(_p2.get("min_generations", 10000))
        max_gens = max(min_gens, int(phase2_time / gen_sec))
        best_params, best_score = optimizer.run(
            max_generations=max_gens,
            time_budget=phase2_time,
        )
        p2_info = optimizer.introspect()
        p2_improvement = best_score - p1_score
    else:
        best_params = current_params
        best_score = p1_score
        p2_info = {}

    # ═══ 化石記録ロールバック保護 ═══
    if exp:
        prev_best = exp.get_best_score()
        rollback_thresh = _mp("experience_learning", "fossil_rollback_threshold", 0.5)
        if prev_best is not None and best_score < prev_best - rollback_thresh:
            # 大幅退行 → 化石記録から復元
            fossil_params = exp.get_best_params()
            if fossil_params is not None and len(fossil_params) != n_params:
                if verbose:
                    print(f"  [学習] 化石ロールバック: 次元不一致({len(fossil_params)}!={n_params}), スキップ")
                fossil_params = None
            fossil_score = None
            if fossil_params is not None:
                fossil_score = eval_fn(fossil_params)
            if fossil_score is not None and fossil_score > best_score:
                if verbose:
                    print(f"  [学習] ロールバック保護: {best_score:.4f} < 化石 {fossil_score:.4f} → 復元")
                best_params = fossil_params
                best_score = fossil_score

    # ═══ Phase 3: メタ進化レイヤー（meta=True時のみ） ═══
    phase3_info = None
    if meta:
        # self_evolveの経験も参照（メタgenomeのウォームスタート）
        meta_exp = exp
        if meta_exp is None:
            meta_exp = ExperienceStore("self_evolve")
            if meta_exp.runs == 0:
                meta_exp = None  # self_evolve未実行なら無視
        phase3_info = _run_phase3(cfg, phase3_time, verbose, exp=meta_exp)
        # Phase 3の経験蓄積（learn=True時）
        if learn and exp:
            for comp in phase3_info.get("components_run", []):
                comp_result = phase3_info.get(comp)
                if comp_result:
                    exp.record_meta_result(comp, comp_result)

    elapsed = time.time() - t0
    p2_evals = p2_info.get("eval_count", 0)

    # ═══ 経験記録 ═══
    if exp:
        # 化石記録
        exp.record_fossil(best_params, best_score)

        # カテゴリ統計
        for cat_name, improvement in cat_improvements.items():
            exp.record_category_result(cat_name, improvement, 0, 0)

        # 戦略統計（Phase 1のログから集計）
        if use_parallel:
            for cat_name, (_, tracker) in cat_best.items():
                successes = set(tracker.get("strategy_successes", []))
                for strat in tracker.get("strategy_log", [])[-1000:]:  # 最新1000件
                    exp.record_strategy_result(strat, strat in successes)

        # Phase配分学習
        p1_improvement = p1_score - baseline_score
        exp.record_phase_effectiveness(p1_improvement, p2_improvement)

        # 完了
        exp.finish_run(p1_evals + p2_evals)
        exp.save()

        if verbose:
            sw = exp.get_strategy_weights()
            print(f"  [学習] 経験保存: Run #{exp.runs}, "
                  f"化石ベスト: {exp.get_best_score():.4f}")
            if exp.runs > 1:
                print(f"  [学習] 戦略重み: " + ", ".join(
                    f"{k}={v:.2f}" for k, v in sw.items()))
                print(f"  [学習] 次回Phase1配分: {exp.get_phase1_ratio():.0%}")

    info = {
        "phase1_score": round(p1_score, 4),
        "phase1_evals": p1_evals,
        "phase1_layers": p1_result.get("layers", {}),
        "phase2_evals": p2_evals,
        "total_evals": p1_evals + p2_evals,
        "elapsed": round(elapsed, 2),
        "evals_per_sec": round(evals_per_sec, 0),
        "baseline_score": round(baseline_score, 4),
        "run_number": exp.runs if exp else 0,
        "best_score_ever": exp.get_best_score() if exp else round(best_score, 4),
    }

    # Phase 3結果をinfoに追加
    if phase3_info:
        info["phase3"] = phase3_info

    if verbose:
        print(f"  完了: {baseline_score:.4f} → {best_score:.4f} "
              f"({elapsed:.1f}秒, {info['total_evals']}eval)")

    return best_params, best_score, info


# ═══════════════════════════════════════════════════════
# Phase 3: メタ進化レイヤー
# ═══════════════════════════════════════════════════════

def _run_phase3(cfg, time_budget, verbose, exp=None):
    """Phase 3: メタ進化レイヤー実行。

    LadAccelerator / K7(データ進化) / K8(パラメータ進化) /
    K9(プログラム進化) / Knowledge(知識進化) を順次実行。
    各コンポーネントはtry-exceptで保護し、1つ失敗しても他は続行する。
    expが渡された場合、前回の最良genomeでウォームスタートする。
    """
    t0 = time.time()
    _mp3 = _META_PARAMS.get("meta_phase3", {})
    min_comp_time = _mp3.get("min_component_time", 2)
    min_opt_time = _mp3.get("min_optional_time", 5)

    components = cfg.get("components", _mp3.get("default_components", ["lad_accel", "k7"]))
    # "all"指定で全コンポーネント有効化
    if "all" in components:
        components = ["lad_accel", "k7", "k8", "k9", "k10", "k11", "k12", "knowledge"]

    result = {
        "enabled": True,
        "components_run": [],
        "lad_accel": None,
        "k7": None,
        "k8": None,
        "k9": None,
        "k10": None,
        "k11": None,
        "k12": None,
        "knowledge": None,
    }

    remaining = time_budget

    # --- K² Phase 3 並列化: 独立コンポーネントをThreadPoolExecutorで同時実行 ---
    # Group A (独立): lad_accel, k7, k8, k9 → 並列実行
    # Group B (依存): k10(←k7), k11(←k8), k12(←k9) → Group A後に並列実行
    # Group C (最後): knowledge → 全完了後
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_component(comp_name, comp_budget):
        """Phase 3コンポーネントを1つ実行（ThreadPoolExecutor用）。"""
        try:
            if comp_name == "lad_accel":
                from .twelve_lad_accelerator import TwelveLadAccelerator
                lad_gens = int(_mp3.get("lad_accel_generations", 10))
                prev_genome = exp.get_meta_genome() if exp else None
                accel = TwelveLadAccelerator(verbose=False, initial_genome=prev_genome)
                r = accel.run_accel(meta_generations=lad_gens, meta_time_budget=comp_budget)
                return {"initial_score": r.get("initial_accel_score"),
                        "best_score": r.get("best_accel_score"),
                        "best_genome": r.get("best_genome"),
                        "elapsed": r.get("elapsed", 0)}
            elif comp_name == "k7":
                from .twelve_meta_data_evolution import TwelveMetaDataEvolution, MetaDataGenome
                k7 = TwelveMetaDataEvolution(verbose=False, enable_real_eval=False)
                prev_k7 = exp.get_component_genome("k7") if exp else None
                if prev_k7:
                    try:
                        k7._best_genome = MetaDataGenome.from_dict(prev_k7)
                        k7._best_score = k7._surrogate.evaluate(k7._best_genome)
                    except Exception:
                        pass
                r = k7.run(max_generations=int(_mp3.get("k7_generations", 50)),
                           time_budget=comp_budget,
                           population_size=int(_mp3.get("k7_population", 12)))
                return {"initial_score": r.get("initial_score"),
                        "best_score": r.get("best_score"),
                        "best_genome": r.get("best_genome"),
                        "generations": r.get("generations"),
                        "elapsed": r.get("elapsed", 0)}
            elif comp_name == "k8":
                from .twelve_meta_param_evolution import TwelveMetaParamEvolution, MetaParamGenome
                k8 = TwelveMetaParamEvolution(verbose=False)
                prev_k8 = exp.get_component_genome("k8") if exp else None
                if prev_k8:
                    try:
                        k8._best_genome = MetaParamGenome.from_dict(prev_k8)
                        k8._best_score = k8._surrogate.evaluate(k8._best_genome)
                    except Exception:
                        pass
                r = k8.run(max_generations=int(_mp3.get("k8_generations", 200)),
                           time_budget=comp_budget,
                           population_size=int(_mp3.get("k8_population", 12)))
                return {"initial_score": r.get("initial_score"),
                        "best_score": r.get("best_score"),
                        "best_genome": r.get("best_genome"),
                        "generations": r.get("generations"),
                        "elapsed": r.get("elapsed", 0)}
            elif comp_name == "k9":
                from .twelve_meta_program_evolution import TwelveMetaProgramEvolution, MetaProgramGenome
                k9 = TwelveMetaProgramEvolution(verbose=False)
                prev_k9 = exp.get_component_genome("k9") if exp else None
                if prev_k9:
                    try:
                        k9._best_genome = MetaProgramGenome.from_dict(prev_k9)
                        k9._best_score = k9._surrogate.evaluate(k9._best_genome)
                    except Exception:
                        pass
                r = k9.run(max_generations=int(_mp3.get("k9_generations", 200)),
                           time_budget=comp_budget,
                           population_size=int(_mp3.get("k9_population", 12)))
                return {"initial_score": r.get("initial_score"),
                        "best_score": r.get("best_score"),
                        "best_genome": r.get("best_genome"),
                        "generations": r.get("generations"),
                        "elapsed": r.get("elapsed", 0)}
            elif comp_name == "k10":
                from .twelve_k10_data_evolution import TwelveK10DataEvolution, K10Genome
                k10 = TwelveK10DataEvolution(verbose=False)
                prev_k10 = exp.get_component_genome("k10") if exp else None
                if prev_k10:
                    try:
                        k10._best_genome = K10Genome.from_dict(prev_k10)
                        k10._best_score = k10._surrogate.evaluate(k10._best_genome)
                    except Exception:
                        pass
                r = k10.run(max_generations=int(_mp3.get("k10_generations", 200)),
                            time_budget=comp_budget,
                            population_size=int(_mp3.get("k10_population", 12)))
                return {"initial_score": r.get("initial_score"),
                        "best_score": r.get("best_score"),
                        "best_genome": r.get("best_genome"),
                        "generations": r.get("generations"),
                        "elapsed": r.get("elapsed", 0)}
            elif comp_name == "k11":
                from .twelve_k11_param_evolution import TwelveK11ParamEvolution, K11Genome
                k11 = TwelveK11ParamEvolution(verbose=False)
                prev_k11 = exp.get_component_genome("k11") if exp else None
                if prev_k11:
                    try:
                        k11._best_genome = K11Genome.from_dict(prev_k11)
                        k11._best_score = k11._surrogate.evaluate(k11._best_genome)
                    except Exception:
                        pass
                r = k11.run(max_generations=int(_mp3.get("k11_generations", 200)),
                            time_budget=comp_budget,
                            population_size=int(_mp3.get("k11_population", 12)))
                return {"initial_score": r.get("initial_score"),
                        "best_score": r.get("best_score"),
                        "best_genome": r.get("best_genome"),
                        "generations": r.get("generations"),
                        "elapsed": r.get("elapsed", 0)}
            elif comp_name == "k12":
                from .twelve_k12_program_evolution import TwelveK12ProgramEvolution, K12Genome
                k12 = TwelveK12ProgramEvolution(verbose=False)
                prev_k12 = exp.get_component_genome("k12") if exp else None
                if prev_k12:
                    try:
                        k12._best_genome = K12Genome.from_dict(prev_k12)
                        k12._best_score = k12._surrogate.evaluate(k12._best_genome)
                    except Exception:
                        pass
                r = k12.run(max_generations=int(_mp3.get("k12_generations", 200)),
                            time_budget=comp_budget,
                            population_size=int(_mp3.get("k12_population", 12)))
                return {"initial_score": r.get("initial_score"),
                        "best_score": r.get("best_score"),
                        "best_genome": r.get("best_genome"),
                        "generations": r.get("generations"),
                        "elapsed": r.get("elapsed", 0)}
            elif comp_name == "knowledge":
                from .twelve_knowledge_evolution import TwelveKnowledgeEvolution
                ke = TwelveKnowledgeEvolution(time_budget=comp_budget, verbose=False)
                r = ke.run_knowledge_evolution()
                return {"initial_score": r.get("initial_score"),
                        "final_score": r.get("final_score"),
                        "improvement": r.get("improvement"),
                        "elapsed": r.get("elapsed", 0)}
        except Exception as e:
            return {"error": str(e)}
        return None

    # Group A: 独立コンポーネント並列実行
    group_a = [c for c in ["lad_accel", "k7", "k8", "k9"] if c in components]
    if group_a and remaining > min_comp_time:
        budget_per_comp = remaining / max(len(group_a), 1)
        with ThreadPoolExecutor(max_workers=min(4, len(group_a))) as pool:
            futures = {pool.submit(_run_component, c, budget_per_comp): c
                       for c in group_a}
            for future in as_completed(futures):
                comp = futures[future]
                r = future.result()
                if r and "error" not in r:
                    result[comp] = r
                    result["components_run"].append(comp)
                    if verbose:
                        bs = r.get("best_score", r.get("final_score", 0))
                        print(f"  Phase 3 {comp}: score={bs:.1f}")
                elif r and verbose:
                    print(f"  Phase 3 {comp} error: {r.get('error')}")
        remaining = time_budget - (time.time() - t0)

    # Group B: 依存コンポーネント並列実行 (k10←k7, k11←k8, k12←k9)
    group_b = [c for c in ["k10", "k11", "k12"] if c in components]
    if group_b and remaining > min_opt_time:
        budget_per_comp = remaining / max(len(group_b), 1)
        with ThreadPoolExecutor(max_workers=min(3, len(group_b))) as pool:
            futures = {pool.submit(_run_component, c, budget_per_comp): c
                       for c in group_b}
            for future in as_completed(futures):
                comp = futures[future]
                r = future.result()
                if r and "error" not in r:
                    result[comp] = r
                    result["components_run"].append(comp)
                    if verbose:
                        bs = r.get("best_score", 0)
                        print(f"  Phase 3 {comp}: score={bs:.1f}")
        remaining = time_budget - (time.time() - t0)

    # Group C: Knowledge（最後、単独）
    if "knowledge" in components and "knowledge" not in result["components_run"] and remaining > min_opt_time:
        r = _run_component("knowledge", remaining)
        if r and "error" not in r:
            result["knowledge"] = r
            result["components_run"].append("knowledge")

    # 並列化で未処理のコンポーネントがあれば逐次フォールバック
    remaining = time_budget - (time.time() - t0)
    # 並列で既に処理済みのコンポーネントはスキップ（remaining > 0でもresultに入ってればスキップ）

    # ─── Step 1: LadAccelerator (並列で未処理の場合のみ) ───
    if "lad_accel" in components and "lad_accel" not in result["components_run"] and remaining > min_comp_time:
        try:
            from .twelve_lad_accelerator import TwelveLadAccelerator
            budget = min(cfg.get("lad_accel_budget", _mp3.get("lad_accel_budget", 5)) or 5, remaining)
            lad_gens = int(_mp3.get("lad_accel_generations", 10))
            # ウォームスタート: 前回の最良genomeをロード
            prev_genome = exp.get_meta_genome() if exp else None
            accel = TwelveLadAccelerator(verbose=verbose, initial_genome=prev_genome)
            r = accel.run_accel(meta_generations=lad_gens, meta_time_budget=budget)
            result["lad_accel"] = {
                "initial_score": r.get("initial_accel_score"),
                "best_score": r.get("best_accel_score"),
                "best_genome": r.get("best_genome"),
                "elapsed": r.get("elapsed", 0),
            }
            result["components_run"].append("lad_accel")
            remaining -= r.get("elapsed", 0)
            if verbose:
                print(f"  Phase 3 LadAccel: "
                      f"{r.get('initial_accel_score', 0):.1f} → "
                      f"{r.get('best_accel_score', 0):.1f}")
        except Exception as e:
            if verbose:
                print(f"  Phase 3 LadAccel エラー: {e}")

    # ─── Step 2: K7（メタデータ進化） ───
    if "k7" in components and "k7" not in result["components_run"] and remaining > min_comp_time:
        try:
            from .twelve_meta_data_evolution import TwelveMetaDataEvolution, MetaDataGenome
            budget = cfg.get("k7_budget") or remaining * _mp3.get("k7_time_fraction", 0.6)
            budget = min(budget, remaining)
            k7 = TwelveMetaDataEvolution(verbose=verbose, enable_real_eval=False)
            # ウォームスタート: 前回の最良genomeを初期集団に注入
            prev_k7 = exp.get_component_genome("k7") if exp else None
            if prev_k7:
                try:
                    k7._best_genome = MetaDataGenome.from_dict(prev_k7)
                    k7._best_score = k7._surrogate.evaluate(k7._best_genome)
                except Exception:
                    pass
            r = k7.run(max_generations=int(_mp3.get("k7_generations", 50)),
                       time_budget=budget,
                       population_size=int(_mp3.get("k7_population", 12)))
            result["k7"] = {
                "initial_score": r.get("initial_score"),
                "best_score": r.get("best_score"),
                "best_genome": r.get("best_genome"),
                "generations": r.get("generations"),
                "elapsed": r.get("elapsed", 0),
            }
            result["components_run"].append("k7")
            remaining -= r.get("elapsed", 0)
            if verbose:
                print(f"  Phase 3 K7: "
                      f"{r.get('initial_score', 0):.1f} → "
                      f"{r.get('best_score', 0):.1f}")
        except Exception as e:
            if verbose:
                print(f"  Phase 3 K7 エラー: {e}")

    # ─── Step 3: K8（メタパラメータ進化、オプション） ───
    if "k8" in components and "k8" not in result["components_run"] and remaining > min_opt_time:
        try:
            from .twelve_meta_param_evolution import TwelveMetaParamEvolution, MetaParamGenome
            budget = cfg.get("k8_budget") or remaining * _mp3.get("k8_time_fraction", 0.5)
            budget = min(budget, remaining)
            k8 = TwelveMetaParamEvolution(verbose=verbose)
            prev_k8 = exp.get_component_genome("k8") if exp else None
            if prev_k8:
                try:
                    k8._best_genome = MetaParamGenome.from_dict(prev_k8)
                    k8._best_score = k8._surrogate.evaluate(k8._best_genome)
                except Exception:
                    pass
            r = k8.run(max_generations=int(_mp3.get("k8_generations", 200)),
                       time_budget=budget,
                       population_size=int(_mp3.get("k8_population", 12)))
            result["k8"] = {
                "initial_score": r.get("initial_score"),
                "best_score": r.get("best_score"),
                "best_genome": r.get("best_genome"),
                "generations": r.get("generations"),
                "elapsed": r.get("elapsed", 0),
            }
            result["components_run"].append("k8")
            remaining -= r.get("elapsed", 0)
        except Exception as e:
            if verbose:
                print(f"  Phase 3 K8 エラー: {e}")

    # ─── Step 4: K9（メタプログラム進化、オプション） ───
    if "k9" in components and "k9" not in result["components_run"] and remaining > min_opt_time:
        try:
            from .twelve_meta_program_evolution import TwelveMetaProgramEvolution, MetaProgramGenome
            budget = cfg.get("k9_budget") or remaining * _mp3.get("k9_time_fraction", 0.5)
            budget = min(budget, remaining)
            k9 = TwelveMetaProgramEvolution(verbose=verbose)
            prev_k9 = exp.get_component_genome("k9") if exp else None
            if prev_k9:
                try:
                    k9._best_genome = MetaProgramGenome.from_dict(prev_k9)
                    k9._best_score = k9._surrogate.evaluate(k9._best_genome)
                except Exception:
                    pass
            r = k9.run(max_generations=int(_mp3.get("k9_generations", 200)),
                       time_budget=budget,
                       population_size=int(_mp3.get("k9_population", 12)))
            result["k9"] = {
                "initial_score": r.get("initial_score"),
                "best_score": r.get("best_score"),
                "best_genome": r.get("best_genome"),
                "generations": r.get("generations"),
                "elapsed": r.get("elapsed", 0),
            }
            result["components_run"].append("k9")
            remaining -= r.get("elapsed", 0)
        except Exception as e:
            if verbose:
                print(f"  Phase 3 K9 エラー: {e}")

    # ─── Step 5: K10（メタメタデータ進化 — K7テンプレート集を進化） ───
    if "k10" in components and "k10" not in result["components_run"] and remaining > min_opt_time:
        try:
            from .twelve_k10_data_evolution import TwelveK10DataEvolution, K10Genome
            budget = cfg.get("k10_budget") or remaining * _mp3.get("k10_time_fraction", 0.4)
            budget = min(budget, remaining)
            k10 = TwelveK10DataEvolution(verbose=verbose)
            prev_k10 = exp.get_component_genome("k10") if exp else None
            if prev_k10:
                try:
                    k10._best_genome = K10Genome.from_dict(prev_k10)
                    k10._best_score = k10._surrogate.evaluate(k10._best_genome)
                except Exception:
                    pass
            r = k10.run(max_generations=int(_mp3.get("k10_generations", 200)),
                        time_budget=budget,
                        population_size=int(_mp3.get("k10_population", 12)))
            result["k10"] = {
                "initial_score": r.get("initial_score"),
                "best_score": r.get("best_score"),
                "best_genome": r.get("best_genome"),
                "generations": r.get("generations"),
                "elapsed": r.get("elapsed", 0),
            }
            result["components_run"].append("k10")
            remaining -= r.get("elapsed", 0)
            if verbose:
                print(f"  Phase 3 K10: "
                      f"{r.get('initial_score', 0):.1f} → "
                      f"{r.get('best_score', 0):.1f}")
        except Exception as e:
            if verbose:
                print(f"  Phase 3 K10 エラー: {e}")

    # ─── Step 6: K11（メタメタパラメータ進化 — K8探索範囲を進化） ───
    if "k11" in components and "k11" not in result["components_run"] and remaining > min_opt_time:
        try:
            from .twelve_k11_param_evolution import TwelveK11ParamEvolution, K11Genome
            budget = cfg.get("k11_budget") or remaining * _mp3.get("k11_time_fraction", 0.5)
            budget = min(budget, remaining)
            k11 = TwelveK11ParamEvolution(verbose=verbose)
            prev_k11 = exp.get_component_genome("k11") if exp else None
            if prev_k11:
                try:
                    k11._best_genome = K11Genome.from_dict(prev_k11)
                    k11._best_score = k11._surrogate.evaluate(k11._best_genome)
                except Exception:
                    pass
            r = k11.run(max_generations=int(_mp3.get("k11_generations", 200)),
                        time_budget=budget,
                        population_size=int(_mp3.get("k11_population", 12)))
            result["k11"] = {
                "initial_score": r.get("initial_score"),
                "best_score": r.get("best_score"),
                "best_genome": r.get("best_genome"),
                "generations": r.get("generations"),
                "elapsed": r.get("elapsed", 0),
            }
            result["components_run"].append("k11")
            remaining -= r.get("elapsed", 0)
            if verbose:
                print(f"  Phase 3 K11: "
                      f"{r.get('initial_score', 0):.1f} → "
                      f"{r.get('best_score', 0):.1f}")
        except Exception as e:
            if verbose:
                print(f"  Phase 3 K11 エラー: {e}")

    # ─── Step 7: K12（メタメタプログラム進化 — K9オペレータを進化） ───
    if "k12" in components and "k12" not in result["components_run"] and remaining > min_opt_time:
        try:
            from .twelve_k12_program_evolution import TwelveK12ProgramEvolution, K12Genome
            budget = cfg.get("k12_budget") or remaining * _mp3.get("k12_time_fraction", 0.5)
            budget = min(budget, remaining)
            k12 = TwelveK12ProgramEvolution(verbose=verbose)
            prev_k12 = exp.get_component_genome("k12") if exp else None
            if prev_k12:
                try:
                    k12._best_genome = K12Genome.from_dict(prev_k12)
                    k12._best_score = k12._surrogate.evaluate(k12._best_genome)
                except Exception:
                    pass
            r = k12.run(max_generations=int(_mp3.get("k12_generations", 200)),
                        time_budget=budget,
                        population_size=int(_mp3.get("k12_population", 12)))
            result["k12"] = {
                "initial_score": r.get("initial_score"),
                "best_score": r.get("best_score"),
                "best_genome": r.get("best_genome"),
                "generations": r.get("generations"),
                "elapsed": r.get("elapsed", 0),
            }
            result["components_run"].append("k12")
            remaining -= r.get("elapsed", 0)
            if verbose:
                print(f"  Phase 3 K12: "
                      f"{r.get('initial_score', 0):.1f} → "
                      f"{r.get('best_score', 0):.1f}")
        except Exception as e:
            if verbose:
                print(f"  Phase 3 K12 エラー: {e}")

    # ─── Step 8: Knowledge（知識進化、オプション） ───
    if "knowledge" in components and "knowledge" not in result["components_run"] and remaining > min_opt_time:
        try:
            from .twelve_knowledge_evolution import TwelveKnowledgeEvolution
            budget = cfg.get("knowledge_budget") or remaining
            budget = min(budget, remaining)
            ke = TwelveKnowledgeEvolution(time_budget=budget, verbose=verbose)
            r = ke.run_knowledge_evolution()
            result["knowledge"] = {
                "initial_score": r.get("initial_score"),
                "final_score": r.get("final_score"),
                "improvement": r.get("improvement"),
                "elapsed": r.get("elapsed", 0),
            }
            result["components_run"].append("knowledge")
        except Exception as e:
            if verbose:
                print(f"  Phase 3 Knowledge エラー: {e}")

    result["time_used"] = time.time() - t0
    return result


# ═══════════════════════════════════════════════════════
# self_evolve — Phase 3専用自己進化ループ
# ═══════════════════════════════════════════════════════

def self_evolve(time_budget=300, components=None, verbose=True,
                experience_id="self_evolve"):
    """Twelveエンジン本体の自己進化。eval_fn不要 — エンジン自身を最適化する。

    Phase 3（LadAccel/K7/K8/K9/Knowledge）だけをループ実行し、
    エンジン内部のメタパラメータを蓄積的に進化させる。

    Args:
        time_budget: 総時間（秒）
        components: 実行コンポーネント（Noneで全部）
        verbose: ログ出力
        experience_id: 経験ファイルID

    Returns:
        dict: ラウンドごとの結果
    """
    if components is None:
        components = ["lad_accel", "k7", "k8", "k9", "k10", "k11", "k12", "knowledge"]

    t0 = time.time()
    exp = ExperienceStore(experience_id)
    round_num = 0
    results = []

    if verbose:
        print(f"{'=' * 60}")
        print(f"Twelve self_evolve: エンジン本体の自己進化")
        print(f"  時間予算: {time_budget}秒")
        print(f"  コンポーネント: {components}")
        print(f"  経験蓄積: {exp.runs}回目から継続")
        print(f"{'=' * 60}")

    _se = _META_PARAMS.get("self_evolve", {})
    se_frac = _se.get("round_time_fraction", 0.3)
    se_max = _se.get("round_time_max", 60)
    se_min = _se.get("round_time_min", 5)

    while time.time() - t0 < time_budget:
        round_num += 1
        elapsed = time.time() - t0
        remaining = time_budget - elapsed
        # 1ラウンドの予算: 残り時間のN%かM秒の小さい方
        per_round = min(se_max, remaining * se_frac)
        if per_round < se_min:
            break

        if verbose:
            print(f"\n--- Round {round_num} (経過 {elapsed:.0f}秒 / 残り {remaining:.0f}秒) ---")

        # 各コンポーネントに均等配分（K10-K12にも確実に時間が回るように）
        n_comps = len(components)
        per_comp = max(2, per_round / max(1, n_comps))
        cfg = {
            "components": components,
            "lad_accel_budget": min(per_comp, per_round * _se.get("lad_accel_round_fraction", 0.1)),
            "k7_budget": per_comp,
            "k8_budget": per_comp,
            "k9_budget": per_comp,
            "k10_budget": per_comp,
            "k11_budget": per_comp,
            "k12_budget": per_comp,
            "knowledge_budget": per_comp,
        }

        r = _run_phase3(cfg, per_round, verbose, exp=exp)

        # 経験蓄積
        for comp in r.get("components_run", []):
            comp_result = r.get(comp)
            if comp_result:
                exp.record_meta_result(comp, comp_result)

        exp._data["runs"] += 1
        exp.save()

        results.append({
            "round": round_num,
            "components_run": r["components_run"],
            "time_used": r["time_used"],
            "lad_accel": r.get("lad_accel"),
            "k7": r.get("k7"),
            "k8": r.get("k8"),
            "k9": r.get("k9"),
            "k10": r.get("k10"),
            "k11": r.get("k11"),
            "k12": r.get("k12"),
            "knowledge": r.get("knowledge"),
        })

        if verbose:
            comps = r["components_run"]
            print(f"  Round {round_num}: {comps} ({r['time_used']:.1f}秒)")

    total_time = time.time() - t0

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"self_evolve完了: {round_num}ラウンド, {total_time:.1f}秒")
        print(f"  経験蓄積: {exp.runs}回")
        meta = exp._data.get("meta", {})
        if meta.get("meta_genome"):
            print(f"  最良LadAccel genome: {meta['meta_genome']}")
        print(f"  meta_runs: {meta.get('meta_runs', 0)}")
        print(f"{'=' * 60}")

    return {
        "rounds": round_num,
        "total_time": total_time,
        "results": results,
        "experience_runs": exp.runs,
        "meta_genome": exp.get_meta_genome(),
        "k7_genome": exp.get_k7_genome(),
    }


# ═══════════════════════════════════════════════════════
# Multi-Observer Owl — 複数スコアで構造の安定性を検証
# ═══════════════════════════════════════════════════════

def _owl_multi_observer(measurements, param_names=None, verbose=False):
    """複数観測者でMirrorScanを実行し、構造の安定性を返す。

    入力: [{"params": {...}, "scores": {"obs1": float, "obs2": float, ...}}, ...]
    出力: stable_active, stable_dead, observer_dependent + 各観測者の詳細
    """
    from .agent.mirror_agent import MirrorScan

    first = measurements[0]
    score_keys = sorted(first["scores"].keys())
    n_obs = len(score_keys)

    # パラメータ名推定
    first_params = first["params"]
    if param_names is None:
        if isinstance(first_params, dict):
            param_names = sorted(first_params.keys())
        else:
            param_names = [f"p{i}" for i in range(len(first_params))]
    n_dims = len(param_names)

    if verbose:
        print(f"  [Owl-MultiObs] {n_obs} observers: {score_keys}, {len(measurements)} measurements")

    # 各観測者でMirrorScan実行
    observers = {}
    for sk in score_keys:
        ms = MirrorScan.from_measurements(measurements, param_names=param_names, score_key=sk)
        proxy_fn, r2, ptype = ms.build_proxy()
        observers[sk] = {
            "importance": ms.importance.tolist(),
            "fragility": ms.fragility.tolist(),
            "dead_dims": ms.dead_dims,
            "active_dims": ms.active_dims,
            "proxy_r2": float(r2) if r2 else 0.0,
            "proxy_type": ptype,
        }

    # Dead observer検出: R² < 0.3 → この観測者では構造が見えない
    min_alive_r2 = 0.3
    for sk in score_keys:
        obs = observers[sk]
        obs["alive"] = obs["proxy_r2"] >= min_alive_r2

    alive_observers = {k: v for k, v in observers.items() if v["alive"]}

    # Consensus: alive観測者のみでstable/dependent分類
    if alive_observers:
        all_dead_sets = [set(obs["dead_dims"]) for obs in alive_observers.values()]
        all_active_sets = [set(obs["active_dims"]) for obs in alive_observers.values()]
    else:
        all_dead_sets = [set(obs["dead_dims"]) for obs in observers.values()]
        all_active_sets = [set(obs["active_dims"]) for obs in observers.values()]

    stable_dead = set(range(n_dims))
    for s in all_dead_sets:
        stable_dead &= s

    stable_active = set(range(n_dims))
    for s in all_active_sets:
        stable_active &= s

    observer_dependent = set(range(n_dims)) - stable_dead - stable_active

    # Pairwise observer相関: importanceベクトル間のPearson相関
    import numpy as _np
    obs_names = score_keys
    imp_matrix = _np.array([observers[sk]["importance"] for sk in obs_names])
    n_o = len(obs_names)
    observer_correlations = {}
    for i in range(n_o):
        for j in range(i + 1, n_o):
            std_i = _np.std(imp_matrix[i])
            std_j = _np.std(imp_matrix[j])
            if std_i > 1e-10 and std_j > 1e-10:
                c = _np.corrcoef(imp_matrix[i], imp_matrix[j])[0, 1]
                if not _np.isnan(c):
                    observer_correlations[f"{obs_names[i]}×{obs_names[j]}"] = round(float(c), 3)

    result = {
        "stable_active": sorted(stable_active),
        "stable_dead": sorted(stable_dead),
        "observer_dependent": sorted(observer_dependent),
        "observers": observers,
        "observer_correlations": observer_correlations,
        "dead_observers": [sk for sk in score_keys if not observers[sk]["alive"]],
        "param_names": param_names,
        "n_measurements": len(measurements),
        "n_observers": n_obs,
    }

    if verbose:
        sa = [param_names[i] for i in result["stable_active"]]
        sd = [param_names[i] for i in result["stable_dead"]]
        od = [param_names[i] for i in result["observer_dependent"]]
        dead_obs = result["dead_observers"]
        print(f"  [Owl-MultiObs] stable_active={sa}")
        print(f"  [Owl-MultiObs] stable_dead={sd}")
        print(f"  [Owl-MultiObs] observer_dependent={od}")
        if dead_obs:
            print(f"  [Owl-MultiObs] dead_observers={dead_obs}")
        # 高相関ペア（冗長候補）
        for pair, corr in observer_correlations.items():
            if corr > 0.95:
                print(f"  [Owl-MultiObs] redundant pair: {pair} (corr={corr})")

    return result


# ═══════════════════════════════════════════════════════
# Owl — 見えない構造を見つけて最適化する
# ═══════════════════════════════════════════════════════

def owl(
    measurements=None,
    param_ranges=None,
    param_names=None,
    time_budget=60,
    experience_id=None,
    verify_fn=None,
    n_rounds=1,
    min_r_squared=0.7,
    autonomous=False,
    max_iterations=10,
    verbose=False,
    kathara="auto",
    stagnation_threshold=None,   # NEW: overrides autonomous stagnation threshold (default 3)
    force_proxy_type=None,       # NEW: forces a specific proxy (zenron/zenron_interact/linear); None = R²-best
    curated_measurements=None,   # Explicit alias of `measurements`. Domain-curated prior data; use this name to signal intent.
    n_seed_samples=None,         # Empty-data path: seed count (default max(5, n_dims+2))
    seed_rng_state=42,           # Empty-data path: RNG seed for uniform sampling
    guard_fn=None,               # Optional safety: guard_fn(params) → float. Baseline set via guard_threshold or midpoint eval.
    guard_threshold=None,        # Minimum acceptable guard_score. If None and guard_fn given, computed at midpoint params.
    safe_dim_analysis=False,     # If True + guard_fn + guard fails, run multi-observer analysis to surface safe_dims.
):
    """Owl — 見えない構造を見つけて最適化する。

    測定データだけでTwelve最適化。eval_fnを書く必要なし。

    内部動作:
      1. measurementsからMirrorScan構造分析（全論公式: importance = sqrt(truth × connectivity)）
      2. build_proxy()で3-6候補（zenron/zenron_interact/linear × raw/log）からR²が高い方を自動選択
      3. R² ≥ min_r_squared → proxyで高速最適化（0.001ms/eval × 100,000回）
      4. verify_fnがあれば本番検証 → n_rounds>1なら結果をmeasurementsに追加して再ループ

    自動成長ループ（verify_fn + n_rounds > 1）:
      proxy最適解を本番検証 → 実測値をmeasurementsに追加 → proxy再構築 → 繰り返し
      毎ラウンド「最も知りたい場所」のデータが増える → proxyが賢くなる

    autonomousモード（verify_fn + autonomous=True）:
      AT的自律ループ。proxy最適化→verify→探索サンプリング→proxy再構築→停滞検知。
      停滞3回でMS再分析+探索範囲摂動。max_iterations回繰り返す。

    Args:
        measurements: 測定データ
            [{"params": {"temp": 0.7, "top_p": 0.9}, "score": 85.0}, ...]
            or [{"params": [0.7, 0.9], "score": 85.0}, ...]
        param_ranges: [(lo, hi), ...] 省略時はmeasurementsから自動推定
        param_names: パラメータ名リスト（省略時は自動検出）
        time_budget: 1ラウンドあたりの最適化時間（秒）
        experience_id: 経験ID（省略時は自動生成）
        verify_fn: (params_list) -> float 本番検証関数（省略可）
        n_rounds: ラウンド数（verify_fn必須。毎回proxyを再構築して育てる）
        min_r_squared: proxy品質閾値（デフォルト0.7）
        autonomous: AT的自律ループを有効化（verify_fn必須）
        max_iterations: autonomousモードの最大イテレーション数
        verbose: ログ出力
        kathara: K²モード（6ノード並列）。
            "auto"(デフォルト): verify_fnがimportable + budget≥120sで自動有効
            True: 強制K²（verify_fn必須、lambda時はフォールバック）
            False: 強制シングルノード

    Returns:
        dict: {
            "best_params": dict or list,  最適パラメータ
            "best_score": float,          予測スコア（proxy）
            "verified_score": float|None, verify_fn実行時のみ
            "proxy_r2": float,            proxy品質
            "proxy_type": str,            "zenron"/"linear"/"zenron_log"/"linear_log"
            "active_dims": list,          有効次元
            "dead_dims": list,            死に次元
            "param_names": list,          パラメータ名
            "confidence": str,            "high"/"low"/"insufficient"
            "n_measurements": int,        最終データ数
            "rounds_completed": int,      完了ラウンド数
        }
    """
    from .agent.mirror_agent import MirrorScan

    t0 = time.time()   # owl's own wall clock (for autonomous budget-aware loop)

    # Resolve measurements input: explicit `measurements` wins over `curated_measurements`
    # alias. Alias exists to make domain-curated data intent explicit in user code.
    if measurements is None:
        measurements = curated_measurements if curated_measurements is not None else []
    elif curated_measurements is not None and verbose:
        print("  [owl] both measurements and curated_measurements given; using measurements")

    # Empty-data autonomous start: if no measurements but verify_fn + ranges available,
    # seed initial samples by calling verify_fn on uniform-random points in ranges.
    # This removes the hard 5-point minimum for callers with just a callable.
    if len(measurements) == 0:
        if verify_fn is None or param_ranges is None:
            raise ValueError(
                "owl(): empty measurements requires both verify_fn and param_ranges "
                "for autonomous seed sampling. Provide measurements, or both verify_fn "
                "and param_ranges."
            )
        import random as _random
        n_dims = len(param_ranges)
        n_seed = n_seed_samples if n_seed_samples is not None else max(5, n_dims + 2)
        _rng = _random.Random(seed_rng_state)
        if verbose:
            print(f"  [owl empty-start] seeding {n_seed} points in {n_dims}d via verify_fn")
        measurements = []
        for _ in range(n_seed):
            sample = [_rng.uniform(lo, hi) for lo, hi in param_ranges]
            try:
                s = float(verify_fn(sample))
            except Exception as e:
                if verbose:
                    print(f"    seed eval error ({type(e).__name__}), skipping")
                continue
            use_params = sample
            if param_names:
                use_params = {n: v for n, v in zip(param_names, sample)}
            measurements.append({"params": use_params, "score": s})
        if len(measurements) < 2:
            raise RuntimeError(
                f"owl() empty-start: only {len(measurements)}/{n_seed} seed evals succeeded; "
                f"verify_fn failing too often"
            )

    # Multi-observer検出: "scores" dictがあればmulti-observer分析にディスパッチ
    if measurements and "scores" in measurements[0] and isinstance(measurements[0]["scores"], dict):
        return _owl_multi_observer(measurements, param_names=param_names, verbose=verbose)

    # 経験読み込み (UnifiedExperience)
    _ue = None
    _ue_dead_hint = None
    if experience_id:
        try:
            from .agent.unified_experience import UnifiedExperience
            _ue = UnifiedExperience(experience_id)
            _ue_dead_hint = _ue.get_dead_dims()
            ws = _ue.get_warm_start()
            if ws and verbose:
                print(f"  [Experience] warm_start={len(ws)}d, "
                      f"dead_hint={len(_ue_dead_hint) if _ue_dead_hint else 0}")
        except Exception:
            _ue = None

    # param_namesを確定
    first_params = measurements[0]["params"]
    is_dict = isinstance(first_params, dict)
    if is_dict:
        names = param_names or sorted(first_params.keys())
    else:
        names = param_names or [f"p{i}" for i in range(len(first_params))]

    # --- Kathara K² モード: 6ノード並列最適化 ---
    # auto判定: verify_fnがimportable + budget≥120s → 自動K²
    _use_k2 = False
    if kathara == "auto" and verify_fn is not None and time_budget >= 120:
        _k2_mod = _k2_name = None
        if hasattr(verify_fn, "__module__") and hasattr(verify_fn, "__qualname__"):
            _m, _n = verify_fn.__module__, verify_fn.__qualname__
            if "<" not in _n and _m != "__main__":
                _k2_mod, _k2_name = _m, _n
                _use_k2 = True
    elif kathara == "auto":
        pass  # verify_fnなし or budget不足 → シングルノード
    elif kathara is True:
        if verify_fn is None:
            raise ValueError("kathara=True requires verify_fn")
        _k2_mod = _k2_name = None
        if hasattr(verify_fn, "__module__") and hasattr(verify_fn, "__qualname__"):
            _m, _n = verify_fn.__module__, verify_fn.__qualname__
            if "<" not in _n and _m != "__main__":
                _k2_mod, _k2_name = _m, _n
                _use_k2 = True
        if not _use_k2 and verbose:
            print("  [Owl-K2] verify_fn is lambda/closure -> single-node fallback")
    # kathara=False → _use_k2=False のまま

    if _use_k2:
        _k2_ranges = param_ranges
        if _k2_ranges is None:
            _n_p = len(names)
            _mins = [float('inf')] * _n_p
            _maxs = [float('-inf')] * _n_p
            for _me in measurements:
                _p = _me["params"]
                for _i, _nm in enumerate(names):
                    _v = _p[_nm] if is_dict else _p[_i]
                    _mins[_i] = min(_mins[_i], float(_v))
                    _maxs[_i] = max(_maxs[_i], float(_v))
            _k2_ranges = []
            for _lo, _hi in zip(_mins, _maxs):
                _sp = _hi - _lo
                if _sp < 1e-10:
                    _sp = abs(_lo) * 0.2 or 1.0
                _k2_ranges.append((_lo - _sp * 0.1, _hi + _sp * 0.1))

        # S2(Bind): 初期測定データをワーカーに引き渡し
        import tempfile as _tempfile
        _meas_dir = _tempfile.mkdtemp(prefix="owl_k2_")
        _meas_path = os.path.join(_meas_dir, "init_measurements.json")
        try:
            import json as _json
            with open(_meas_path, "w", encoding="utf-8") as _f:
                _json.dump(measurements, _f, ensure_ascii=False)
        except Exception:
            _meas_path = None

        from .agent.kathara_ma import KatharaMACoordinator
        if verbose:
            print(f"  [Owl-K2] 6ノード並列起動: {len(names)}次元, {time_budget}秒"
                  f", 初期データ{len(measurements)}件")

        _coord = KatharaMACoordinator(
            eval_fn_module=_k2_mod,
            eval_fn_name=_k2_name,
            param_ranges=_k2_ranges,
            time_budget=time_budget,
            experience_id=experience_id or "owl_k2",
            max_iterations=max_iterations,
            verbose=verbose,
            init_measurements_path=_meas_path,
        )
        _k2r = _coord.run()

        _k2_bp = _k2r.get("best_params")
        if is_dict and isinstance(_k2_bp, (list, tuple)):
            _k2_bp = {n: v for n, v in zip(names, _k2_bp)}

        return {
            "best_params": _k2_bp,
            "best_score": _k2r.get("best_score"),
            "verified_score": _k2r.get("best_score"),
            "proxy_r2": 0.0,
            "proxy_type": "kathara_k2",
            "active_dims": _k2r.get("active_dims", []),
            "dead_dims": _k2r.get("dead_dims", []),
            "recovered_dims": [],
            "param_names": names,
            "confidence": "high",
            "n_measurements": len(measurements),
            "rounds_completed": _k2r.get("polls", 0),
            "n_ma_nodes": _k2r.get("n_ma_nodes"),
        }

    # autonomousモード: verify_fnなしでもproxy自己検証で反復可能
    if autonomous:
        n_rounds = max_iterations
    elif verify_fn is not None and n_rounds <= 1:
        n_rounds = 5
    elif verify_fn is None and not autonomous:
        n_rounds = 1

    # 作業用コピー（元データを変更しない）
    growing_data = list(measurements)

    best_result = None
    _stagnant_count = 0
    _prev_best_verified = float('-inf')

    # Budget-aware autonomous extension (added 2026-04-19):
    # When autonomous=True, continue past n_rounds while time_budget remains and
    # verify_fn cost is tracked. This ensures cheap eval_fn (e.g., synthetic
    # benchmarks) fully utilize the given budget instead of exiting after
    # max_iterations rounds (default 10) regardless of wall time.
    # Hard cap at 200 rounds to prevent runaway.
    _AUTONOMOUS_HARD_CAP = 200
    _verify_eval_times = []           # rolling cost estimate of verify_fn
    _global_t0 = t0                   # for elapsed check

    round_i = -1
    while True:
        round_i += 1

        # Budget check at top of every autonomous round (total-budget semantics).
        # time_budget is treated as the TOTAL owl wall budget for autonomous mode.
        # This prevents overruns when per-round optimize() calls accumulate.
        if autonomous:
            _elapsed = time.time() - _global_t0
            if _elapsed > time_budget:
                if verbose:
                    print(f"  [Auto] total time_budget {time_budget}s exhausted "
                          f"at round {round_i} ({_elapsed:.1f}s elapsed)")
                break

        # Stopping condition A: non-autonomous, fixed n_rounds
        if not autonomous and round_i >= n_rounds:
            break

        # Stopping condition B: autonomous, extend past n_rounds if budget allows
        if autonomous and round_i >= n_rounds:
            if round_i >= _AUTONOMOUS_HARD_CAP:
                if verbose:
                    print(f"  [Auto] hard cap {_AUTONOMOUS_HARD_CAP} reached")
                break
            if _verify_eval_times:
                _avg_cost = sum(_verify_eval_times) / len(_verify_eval_times)
                _elapsed = time.time() - _global_t0
                _remaining = time_budget - _elapsed
                if _remaining < _avg_cost * 2:
                    if verbose:
                        print(f"  [Auto] budget exhausted extending at round {round_i + 1}")
                    break
            else:
                # No cost data → don't extend
                break
        if verbose and n_rounds > 1:
            print(f"\n{'='*50}")
            print(f"  [Round {round_i+1}/{n_rounds}] {len(growing_data)}件のデータ")
            print(f"{'='*50}")

        # --- Step 1: MirrorScanで構造分析 ---
        ms = MirrorScan.from_measurements(growing_data, param_names=names)

        if verbose:
            print(f"  [Measurements] {len(growing_data)}件, {len(names)}次元")
            print(f"  [MS] active={len(ms.active_dims)}, dead={len(ms.dead_dims)}")

        # --- Step 2: proxy自動生成（4候補自動選択） ---
        proxy_fn, proxy_r2, proxy_name = ms.build_proxy(force_proxy_type=force_proxy_type)

        proxy_type = proxy_name or "auto"

        if verbose:
            print(f"  [Proxy] R²={proxy_r2:.3f}")

        # --- Step 3: R²に応じた処理 ---
        if proxy_fn is None or proxy_r2 < 0.3:
            confidence = "insufficient"
            if verbose:
                print(f"  [Proxy] R²={proxy_r2:.3f} < 0.3: 構造発見できず。")
            best_result = {
                "best_params": None,
                "best_score": None,
                "verified_score": None,
                "proxy_r2": proxy_r2,
                "proxy_type": proxy_type,
                "active_dims": ms.active_dims,
                "dead_dims": ms.dead_dims,
                "recovered_dims": getattr(ms, 'recovered_dims', []),
                "param_names": names,
                "confidence": confidence,
                "n_measurements": len(growing_data),
                "rounds_completed": round_i + 1,
            }
            # Autonomous + low R² + verify_fn: try direct-optimize fallback on
            # real eval_fn first (Sentinel's trick ported to owl). Warm-start
            # from best measurement. If proxy is fundamentally inadequate
            # (multimodal/deceptive landscape), this path wins big.
            if autonomous and verify_fn is not None and param_ranges is not None \
                    and len(growing_data) > 0:
                _best_m = max(growing_data, key=lambda m: m["score"])
                _warm = (_best_m["params"] if not isinstance(_best_m["params"], dict)
                         else [_best_m["params"].get(n, 0.0) for n in names])
                _elapsed = time.time() - _global_t0
                _fb_budget = min(_inner_budget if '_inner_budget' in dir() else time_budget,
                                 max(1.0, (time_budget - _elapsed) / 3.0))
                if verbose:
                    print(f"  [Fallback-low-R²] direct optimize on verify_fn "
                          f"from warm-start, {_fb_budget:.1f}s")
                try:
                    _fb_bp, _fb_bs, _ = optimize(
                        eval_fn=verify_fn,
                        param_ranges=param_ranges,
                        initial_params=_warm,
                        time_budget=_fb_budget,
                        learn=True,
                        experience_id=experience_id or "owl_fallback_lowR",
                        verbose=False,
                    )
                    if _fb_bs is not None:
                        if is_dict:
                            growing_data.append({
                                "params": {n: float(v) for n, v in zip(names, _fb_bp)},
                                "score": float(_fb_bs)})
                        else:
                            growing_data.append({"params": [float(v) for v in _fb_bp],
                                                 "score": float(_fb_bs)})
                        # Promote into best_result so fallback improves aren't lost
                        best_result = {
                            "best_params": ({n: float(v) for n, v in zip(names, _fb_bp)}
                                            if is_dict else [float(v) for v in _fb_bp]),
                            "best_score": float(_fb_bs),
                            "verified_score": float(_fb_bs),
                            "proxy_r2": proxy_r2,
                            "proxy_type": proxy_type,
                            "active_dims": ms.active_dims,
                            "dead_dims": ms.dead_dims,
                            "recovered_dims": getattr(ms, 'recovered_dims', []),
                            "param_names": names,
                            "confidence": "direct",
                            "n_measurements": len(growing_data),
                            "rounds_completed": round_i + 1,
                        }
                        if verbose:
                            print(f"  [Fallback-low-R²] direct best={_fb_bs:.4f}")
                except Exception as _fb_e:
                    if verbose:
                        print(f"  [Fallback-low-R²] error: {type(_fb_e).__name__}")

                # Also explore random points to grow data for next proxy attempt
                _expl_rng = random.Random(round_i * 17 + 3)
                for _ in range(3):
                    _sample = [_expl_rng.uniform(lo, hi) for lo, hi in param_ranges]
                    try:
                        _vt0 = time.time()
                        _s = float(verify_fn(_sample))
                        _verify_eval_times.append(time.time() - _vt0)
                    except Exception:
                        continue
                    if is_dict:
                        growing_data.append({"params": {n: v for n, v in zip(names, _sample)}, "score": _s})
                    else:
                        growing_data.append({"params": _sample, "score": _s})
                continue   # retry proxy build next round
            break

        confidence = "high" if proxy_r2 >= min_r_squared else "low"

        # --- Step 4: proxyで高速最適化 ---
        ranges = param_ranges if param_ranges is not None else ms.param_ranges

        # In autonomous mode, scale inner optimize budget to remaining total
        # time so we don't over-commit a whole time_budget per round.
        _inner_budget = time_budget
        if autonomous:
            _remaining = time_budget - (time.time() - _global_t0)
            if _remaining > 0:
                # Divide remaining among expected remaining rounds; keep at least 1s
                _rounds_left = max(1, n_rounds - round_i)
                _inner_budget = max(1.0, _remaining / _rounds_left)
            else:
                _inner_budget = 1.0

        best_params, best_score, info = optimize(
            eval_fn=proxy_fn,
            param_ranges=ranges,
            time_budget=_inner_budget,
            learn=True,
            experience_id=experience_id or "measurements_auto",
            verbose=verbose,
        )

        # 次元検証: ExperienceStoreの化石ロールバックで異なる次元が返る場合がある
        n_expected = len(names)
        if best_params is not None and len(best_params) != n_expected:
            if verbose:
                print(f"  [Owl] dim mismatch: got {len(best_params)}, expected {n_expected} -> truncate/pad")
            bp = list(best_params)
            if len(bp) > n_expected:
                best_params = bp[:n_expected]
            else:
                best_params = bp + [(lo + hi) / 2 for lo, hi in ranges[len(bp):]]

        # --- Step 5: verify_fnで本番検証 ---
        verified_score = None
        if verify_fn is not None:
            try:
                _verify_t0 = time.time()
                verified_score = float(verify_fn(best_params))
                _verify_eval_times.append(time.time() - _verify_t0)
                if verbose:
                    print(f"  [Verify] proxy={best_score:.4f} → real={verified_score:.4f}")

                # 実測結果をmeasurementsに追加（次ラウンドでproxyが育つ）
                # best_paramsはnumpy arrayの場合があるのでfloat()に変換
                if is_dict:
                    new_entry = {"params": {n: float(v) for n, v in zip(names, best_params)},
                                 "score": verified_score}
                else:
                    new_entry = {"params": [float(v) for v in best_params], "score": verified_score}
                growing_data.append(new_entry)

            except Exception as e:
                if verbose:
                    print(f"  [Verify] error: {e}")

        # --- Step 5b: proxy untrustworthy → direct-optimize fallback ---
        # Ported from Sentinel._optimize (2026-04-19). When the proxy predicts
        # a best that underperforms the best actually measured (meaning the
        # proxy is misleading / landscape is non-smooth), skip ahead and run
        # optimize() directly on verify_fn with a warm-start from the best
        # measurement. This is exactly what makes Sentinel win on multimodal
        # landscapes — moving the capability into owl itself.
        # Triggers only when autonomous + verify_fn + we have real measurements.
        if (autonomous and verify_fn is not None and len(growing_data) > 0
                and ranges is not None):
            _max_measured = max(m["score"] for m in growing_data)
            _measurement_winning = (verified_score is None
                                    or verified_score < _max_measured)
            _proxy_weak = proxy_r2 < min_r_squared
            if _measurement_winning and _proxy_weak:
                # Find the best measurement's params to warm-start
                _best_m = max(growing_data, key=lambda m: m["score"])
                _warm = (_best_m["params"] if not isinstance(_best_m["params"], dict)
                         else [_best_m["params"].get(n, 0.0) for n in names])
                # Budget: half of remaining autonomous time
                _elapsed = time.time() - _global_t0
                _remaining = max(1.0, time_budget - _elapsed)
                _fb_budget = min(_inner_budget, _remaining / 2.0)
                if verbose:
                    print(f"  [Fallback] proxy weak (R²={proxy_r2:.2f}) + "
                          f"measurement-winning ({_max_measured:.3f} > verified "
                          f"{verified_score}); direct optimize() on verify_fn "
                          f"for {_fb_budget:.1f}s")
                try:
                    _fb_bp, _fb_bs, _ = optimize(
                        eval_fn=verify_fn,
                        param_ranges=ranges,
                        initial_params=_warm,
                        time_budget=_fb_budget,
                        learn=True,
                        experience_id=experience_id or "owl_fallback",
                        verbose=False,
                    )
                    if _fb_bs is not None and _fb_bs > _max_measured:
                        best_params = list(_fb_bp)
                        best_score = float(_fb_bs)
                        verified_score = float(_fb_bs)
                        # Add to growing_data
                        if is_dict:
                            growing_data.append({
                                "params": {n: float(v) for n, v in zip(names, best_params)},
                                "score": verified_score})
                        else:
                            growing_data.append({"params": [float(v) for v in best_params],
                                                 "score": verified_score})
                        if verbose:
                            print(f"  [Fallback] improved: verified={verified_score:.4f}")
                except Exception as _fb_e:
                    if verbose:
                        print(f"  [Fallback] error: {type(_fb_e).__name__}: {_fb_e}")

        # --- autonomous: 停滞検知 + 探索サンプリング ---
        if autonomous:
            _tracking = verified_score if verified_score is not None else best_score
            if _tracking is not None and _tracking > _prev_best_verified:
                _prev_best_verified = _tracking
                _stagnant_count = 0
            elif _tracking is not None:
                _stagnant_count += 1

            _stag_thr = int(stagnation_threshold) if stagnation_threshold is not None else 3
            if verbose:
                print(f"  [Auto] stagnant={_stagnant_count}/{_stag_thr}, "
                      f"best={_prev_best_verified:.4f}")

            # 停滞 N 回: 探索範囲摂動 (N defaults to 3, Reigen can override)
            if _stagnant_count >= _stag_thr:
                _stagnant_count = 0
                if verbose:
                    print(f"  [Auto] Stagnation -> range perturbation")
                _rng = random.Random(round_i)
                ranges = param_ranges if param_ranges is not None else ms.param_ranges
                ranges = [
                    (lo + _rng.uniform(-0.1, 0.1) * (hi - lo),
                     hi + _rng.uniform(-0.1, 0.1) * (hi - lo))
                    for lo, hi in ranges
                ]

            # 近傍3点サンプリング → データ追加（並列評価）
            if best_params is not None:
                _rng2 = random.Random(round_i * 1000 + 7)
                _ranges = param_ranges if param_ranges is not None else ms.param_ranges

                # 3サンプルを事前生成
                _neighbor_samples = []
                for _si in range(3):
                    sample = [
                        v + _rng2.gauss(0, 0.05 * (hi - lo))
                        for v, (lo, hi) in zip(best_params, _ranges)
                    ]
                    sample = [max(lo, min(hi, v)) for v, (lo, hi) in zip(sample, _ranges)]
                    _neighbor_samples.append(sample)

                # 評価関数を決定して実行
                _eval_target = verify_fn if verify_fn is not None else proxy_fn
                if _eval_target is not None:
                    def _eval_sample(s):
                        try:
                            return float(_eval_target(s))
                        except Exception:
                            return None

                    if verify_fn is not None:
                        # verify_fnはユーザー実装→スレッド安全性不明→逐次実行
                        _scores = [_eval_sample(s) for s in _neighbor_samples]
                    else:
                        # proxy_fnは内部生成の純粋関数→並列安全
                        from concurrent.futures import ThreadPoolExecutor
                        with ThreadPoolExecutor(max_workers=3) as _pool:
                            _scores = list(_pool.map(_eval_sample, _neighbor_samples))

                    for sample, s_score in zip(_neighbor_samples, _scores):
                        if s_score is None:
                            continue
                        if is_dict:
                            growing_data.append({"params": {n: v for n, v in zip(names, sample)},
                                                 "score": s_score})
                        else:
                            growing_data.append({"params": sample, "score": s_score})
                        if s_score > _prev_best_verified:
                            _prev_best_verified = s_score
                            best_params = sample
                            if verify_fn is not None:
                                verified_score = s_score

        # listをdictに戻す（元がdictの場合）
        if is_dict and best_params is not None:
            best_dict = {name: val for name, val in zip(names, best_params)}
        else:
            best_dict = best_params

        best_result = {
            "best_params": best_dict,
            "best_score": float(best_score) if best_score is not None else None,
            "verified_score": verified_score,
            "proxy_r2": float(proxy_r2),
            "proxy_type": proxy_type,
            "active_dims": ms.active_dims,
            "dead_dims": ms.dead_dims,
            "fragility": ms.fragility.tolist() if hasattr(ms, 'fragility') else [],
            "recovered_dims": getattr(ms, 'recovered_dims', []),
            "param_names": names,
            "confidence": confidence,
            "n_measurements": len(growing_data),
            "rounds_completed": round_i + 1,
        }


    # 経験保存
    if _ue and best_result:
        try:
            _ue.save_ms_result(
                dead_dims=best_result.get("dead_dims", []),
                active_dims=best_result.get("active_dims", []),
                importance=None,
                best_params=best_result.get("best_params"),
                best_score=best_result.get("verified_score") or best_result.get("best_score", 0),
            )
        except Exception:
            pass

    # --- guard_fn analysis (post-hoc, Sentinel's pivot logic ported here) ---
    # Non-invasive: existing return schema preserved, new keys only added when
    # guard_fn is provided. No change in behavior for legacy callers.
    if guard_fn is not None and best_result and best_result.get("best_params") is not None:
        try:
            bp = best_result["best_params"]
            bp_list = ([bp.get(n, 0.0) for n in best_result.get("param_names", [])]
                       if isinstance(bp, dict) else list(bp))

            # Baseline: midpoint or caller-supplied threshold
            if guard_threshold is None:
                ranges_for_gb = param_ranges or (ms.param_ranges if 'ms' in dir() else None)
                if ranges_for_gb is not None:
                    mid = [(lo + hi) / 2 for lo, hi in ranges_for_gb]
                    baseline_guard = float(guard_fn(mid))
                else:
                    baseline_guard = float("-inf")
            else:
                baseline_guard = float(guard_threshold)

            best_guard = float(guard_fn(bp_list))
            guard_ok = best_guard >= baseline_guard
            best_result["guard_score"] = best_guard
            best_result["baseline_guard"] = baseline_guard
            best_result["guard_verdict"] = "approved" if guard_ok else "failed"

            # Optional multi-observer safe-dim analysis when guard fails.
            # Uses growing_data (eval scores) + fresh guard evals to identify
            # dims that are safe under both objectives.
            if safe_dim_analysis and not guard_ok and len(growing_data) >= 5:
                try:
                    multi_data = []
                    for m in growing_data[:50]:  # cap for cost
                        p = m["params"] if isinstance(m["params"], list) \
                            else [m["params"].get(n, 0.0) for n in names]
                        try:
                            g = float(guard_fn(p))
                        except Exception:
                            continue
                        multi_data.append({
                            "params": p,
                            "scores": {"eval": float(m["score"]), "guard": g},
                        })
                    if len(multi_data) >= 5:
                        mo = _owl_multi_observer(multi_data, param_names=names, verbose=False)
                        best_result["safe_dims"] = mo.get("stable_active", [])
                        best_result["conflict_dims"] = mo.get("observer_dependent", [])
                        best_result["multi_observer"] = mo
                except Exception as _e:
                    if verbose:
                        print(f"  [guard/safe_dim_analysis] skipped: {type(_e).__name__}")
        except Exception as _e:
            if verbose:
                print(f"  [guard_fn] evaluation failed: {type(_e).__name__}")

    return best_result


# 後方互換エイリアス
optimize_from_measurements = owl
