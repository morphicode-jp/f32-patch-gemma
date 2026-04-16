"""MirrorAgent — MA最外殻。円環の全自動実行。

Usage:
    # API mode (inline eval_fn — recommended)
    from twelve.agent.mirror_agent import MirrorAgent
    MirrorAgent(eval_fn=my_func, param_ranges=ranges).run()

    # Folder mode (Zenron)
    MirrorAgent(folder="my_problem/").run()

MA ⊃ EA ⊃ TL:
    ① MS初期スキャン → 死に次元除去
    ② ED(EvalDesigner) → eval_fn品質管理
    ③ EA(EvolutionAgent) → TL最適化 + 停滞時MS自動削減
    ④ 円環: EA¹→MS¹→EA²→MS²→EA³→EA⁵→EA¹
"""
import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from twelve.optimize import optimize

# --- MA Meta Parameters (JSON-driven, tunable by EA⁵) ---
_MA_META_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "configs", "ma_meta_params.json")
def _load_ma_meta():
    try:
        with open(_MA_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

_MA_META = _load_ma_meta()

def _mp(section, key, default):
    """MA meta parameter. JSON > default fallback."""
    return _MA_META.get(section, {}).get(key, default)


def _compute_truth_connectivity(X, y):
    """全論公式の共通計算: truth, connectivity を返す。

    ベクトル化版: np.corrcoef()を1回だけ呼び全相関を一括計算。
    旧実装のO(n_dims²)ループ→O(1)行列演算で100-1000倍高速化。

    Args:
        X: (n_samples, n_dims) パラメータ行列
        y: (n_samples,) スコアベクトル
    Returns:
        truth: (n_dims,) |corr(param_i, score)|
        connectivity: (n_dims,) mean(|corr(param_i, param_j)|)
    """
    n_dims = X.shape[1]
    truth = np.zeros(n_dims)
    connectivity = np.zeros(n_dims)

    # 定数列を除外（std ≈ 0）
    stds = np.std(X, axis=0)
    active_mask = stds > 1e-10
    active_indices = np.where(active_mask)[0]

    if len(active_indices) == 0:
        return truth, connectivity

    # yが定数の場合、全truthは0（相関計算不能）
    if np.std(y) < 1e-10:
        # connectivityだけ計算可能
        X_active = X[:, active_mask]
        n_active = len(active_indices)
        if n_active > 1:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                C_x = np.corrcoef(X_active.T)
                C_x = np.nan_to_num(C_x)
            C_params = np.abs(C_x)
            np.fill_diagonal(C_params, 0.0)
            connectivity[active_mask] = C_params.sum(axis=1) / (n_active - 1)
        return truth, connectivity

    # active列 + y を結合して1回のcorrcoefで全相関を計算
    X_active = X[:, active_mask]
    data = np.vstack([X_active.T, y.reshape(1, -1)])  # (n_active+1, n_samples)
    C = np.corrcoef(data)  # (n_active+1, n_active+1)
    C = np.nan_to_num(C)

    # truth: 各パラメータとスコアの相関（最後の行/列）
    truth[active_mask] = np.abs(C[:-1, -1])

    # connectivity: パラメータ間の平均|相関|（自己相関を除く）
    n_active = len(active_indices)
    if n_active > 1:
        C_params = np.abs(C[:-1, :-1])
        np.fill_diagonal(C_params, 0.0)
        connectivity[active_mask] = C_params.sum(axis=1) / (n_active - 1)

    return truth, connectivity


def _importance_formula(truth, connectivity):
    """全論重要度公式: (truth × max(connectivity, floor)) ^ exp
    v5: exp=0.306, cf=0.141. Hybrid gap+threshold dead detection."""
    _exp = _mp("mirror_scan", "importance_exponent", 0.3064)
    _cf = _mp("mirror_scan", "connectivity_floor", 0.1411)
    return (truth * np.maximum(connectivity, _cf)) ** _exp


def _fragility_formula(truth, connectivity, active_dims):
    """死の側公式: fragility = (truth × max(isolation, floor)) ^ exp.

    importanceの双対。同じexp/floorでスケール統一。
    importance = truth × connectivity (接続 → 構造の一部 → 最適化せよ)
    fragility  = truth × isolation   (孤立 → 代替なし → 守れ)

    Returns:
        fragility: (n_dims,) 脆弱度。active_dimsのみ非ゼロ。
    """
    _exp = _mp("mirror_scan", "importance_exponent", 0.3064)
    _cf = _mp("mirror_scan", "connectivity_floor", 0.1411)
    isolation = np.clip(1.0 - connectivity, 0.0, 1.0)
    raw = (truth * np.maximum(isolation, _cf)) ** _exp
    # dead dimsはfragility=0（依存がないので壊れても影響ない）
    mask = np.zeros_like(raw)
    for i in active_dims:
        mask[i] = 1.0
    return raw * mask


def _dead_dims_hybrid(importance):
    """Hybrid dead detection: gap-based (S3:Branch) + threshold fallback.

    importanceを降順に並べ、最大ギャップが中央値の gap_multiplier 倍を超えたら
    そこで切る（構造的分岐点）。超えなければ従来の ratio+floor 閾値で切る。
    """
    nd = len(importance)
    if nd < 2:
        return []
    mx = np.max(importance) if np.max(importance) > 0 else 1e-10
    ratio = _mp("mirror_scan", "dead_threshold_ratio", 0.5)
    floor = _mp("mirror_scan", "dead_threshold_floor", 0.0226)
    gap_mult = _mp("mirror_scan", "gap_multiplier", 2.0)

    # threshold baseline
    thr = max(mx * ratio, floor)
    dead_thr = [i for i in range(nd) if importance[i] < thr]

    # gap detection (need >= 3 dims)
    if nd >= 3:
        s = np.sort(importance)[::-1]
        rng = s[0] - s[-1]
        if rng > 1e-10:
            gaps = np.array([s[i] - s[i + 1] for i in range(len(s) - 1)])
            median_gap = np.median(gaps)
            best_idx = int(np.argmax(gaps))
            if gaps[best_idx] > median_gap * gap_mult and best_idx > 0:
                gap_threshold = s[best_idx + 1]
                dead_gap = [i for i in range(nd) if importance[i] <= gap_threshold]
                if len(dead_gap) < nd:
                    return dead_gap

    # fallback: threshold
    if len(dead_thr) < nd:
        return dead_thr
    return []


class MirrorScan:
    """MSの診断エンジン。eval_fnを数回呼んで次元の重要度を分析する。"""

    def __init__(self, eval_fn, param_ranges, n_samples=None, seed=42):
        self.eval_fn = eval_fn
        self.param_ranges = param_ranges
        self.n_dims = len(param_ranges)
        # 動的サンプル数: パラメータ数に応じて調整
        min_samples = _mp("mirror_scan", "initial_samples", 20)
        if n_samples is not None:
            self.n_samples = n_samples
        else:
            self.n_samples = max(min_samples, self.n_dims * 2)
        self.rng = np.random.RandomState(seed)
        self.history = []
        self.importance = np.ones(self.n_dims)
        self.fragility = np.zeros(self.n_dims)
        self.dead_dims = []
        self.active_dims = list(range(self.n_dims))

    def initial_scan(self, n_samples=None):
        """初期スキャン: 安定するまで自動的にサンプル数を調整。

        eval_fnの実行時間を自動検出してスキャン量を調整。
        """
        # 最初の2回でeval_fn速度を測定（1回目はキャッシュ/warmup効果を除く）
        import time as _time
        times = []
        for _ in range(2):
            test_params = [self.rng.uniform(lo, hi) for lo, hi in self.param_ranges]
            t0 = _time.time()
            self.eval_fn(test_params)
            times.append(_time.time() - t0)
            self.history.append({'params': test_params, 'score': 0})
        eval_time = max(times)  # 最も遅い方を採用

        # eval_fn速度に応じてスキャン量を調整
        # 重いevalほどサンプル数を絞る（eval×samples = MS総時間）
        if eval_time > 30:    # PPL等の超重いeval (120s × 15 = 30min)
            min_samples = n_samples or 15
        elif eval_time > 5:   # 重めのeval (10s × 20 = 200s)
            min_samples = n_samples or 20
        elif eval_time > 0.5: # 中程度 (1s × 20 = 20s)
            min_samples = n_samples or max(20, min(self.n_dims, 30))
        else:                 # ISS等の軽いeval (0.01s × 400 = 4s)
            min_samples = n_samples or max(20, self.n_dims * 2)

        batch_size = max(5, min_samples // 3)
        max_total = min_samples * 3 if eval_time < 0.5 else min_samples  # 重いevalは追加スキャンしない
        print(f"  [MS] eval_time={eval_time:.1f}s → min_samples={min_samples}", flush=True)

        print(f"  [MS] Adaptive scan: min={min_samples}, batch={batch_size}, "
              f"max={max_total} on {self.n_dims} dims...", flush=True)

        samples = []
        prev_dead_count = -1
        stable_rounds = 0
        total = 0

        while total < max_total:
            # バッチでサンプリング
            for _ in range(batch_size):
                params = [self.rng.uniform(lo, hi) for lo, hi in self.param_ranges]
                score = self.eval_fn(params)
                samples.append((params, score))
                self.history.append({'params': params, 'score': score})
            total += batch_size

            # 現時点での死に次元を計算
            params_arr = np.array([s[0] for s in samples])
            scores = np.array([s[1] for s in samples])
            self._compute_importance(params_arr, scores)
            current_dead = len(self.dead_dims)

            if total >= min_samples:
                if current_dead == prev_dead_count:
                    stable_rounds += 1
                else:
                    stable_rounds = 0
                prev_dead_count = current_dead

                # 2バッチ連続で同じ結果 → 安定した → 終了
                if stable_rounds >= 2:
                    print(f"  [MS] Stabilized at {total} samples "
                          f"({current_dead} dead, {self.n_dims - current_dead} active)",
                          flush=True)
                    break

        # --- 3シードコンセンサス: サンプルを3分割し独立にimportance計算 ---
        # 各シードは独立 → ThreadPoolExecutorで並列実行
        n_seeds = _mp("mirror_scan", "consensus_seeds", 3)
        if len(samples) >= n_seeds * 3:
            chunk = len(samples) // n_seeds
            chunks = []
            for si in range(n_seeds):
                s_start = si * chunk
                s_end = s_start + chunk if si < n_seeds - 1 else len(samples)
                chunks.append(samples[s_start:s_end])

            def _compute_seed(s_chunk):
                p_arr = np.array([s[0] for s in s_chunk])
                sc_arr = np.array([s[1] for s in s_chunk])
                truth, conn = _compute_truth_connectivity(p_arr, sc_arr)
                imp = _importance_formula(truth, conn)
                dead = _dead_dims_hybrid(imp)
                return set(dead), imp

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n_seeds) as pool:
                results = list(pool.map(_compute_seed, chunks))
            seed_dead_sets = [r[0] for r in results]
            seed_importances = [r[1] for r in results]

            # majority vote: n_seeds中過半数がdeadと判定 → 確定
            from collections import Counter
            all_dead_votes = Counter()
            for ds in seed_dead_sets:
                for d in ds:
                    all_dead_votes[d] += 1
            majority = (n_seeds // 2) + 1
            consensus_dead = [d for d, cnt in all_dead_votes.items() if cnt >= majority]

            # importance = 全シードの平均
            avg_importance = np.mean(seed_importances, axis=0)
            self.importance = avg_importance
            self.dead_dims = sorted(consensus_dead)
            self.active_dims = [i for i in range(self.n_dims) if i not in set(self.dead_dims)]

            # fragility: consensus後の全データで再計算（旧コードの不整合も修正）
            all_params = np.array([s[0] for s in samples])
            all_scores = np.array([s[1] for s in samples])
            truth_all, conn_all = _compute_truth_connectivity(all_params, all_scores)
            self.fragility = _fragility_formula(truth_all, conn_all, self.active_dims)

            print(f"  [MS] Consensus ({n_seeds} seeds, majority={majority}): "
                  f"votes={dict(all_dead_votes)}", flush=True)
        else:
            # サンプル少なすぎる場合は従来方式
            params_arr = np.array([s[0] for s in samples])
            scores = np.array([s[1] for s in samples])
            self._compute_importance(params_arr, scores)

        report = {
            'dead_dims': self.dead_dims,
            'active_dims': self.active_dims,
            'importance': self.importance.tolist(),
            'n_dead': len(self.dead_dims),
            'n_active': len(self.active_dims),
            'reduction': f"{self.n_dims} -> {len(self.active_dims)}",
        }

        print(f"  [MS] Scan complete: {len(self.dead_dims)} dead, "
              f"{len(self.active_dims)} active ({self.n_dims}->{len(self.active_dims)})",
              flush=True)

        if self.dead_dims:
            top_dead = sorted(self.dead_dims,
                              key=lambda i: self.importance[i])[:5]
            print(f"  [MS] Dead dims (top 5): {top_dead}", flush=True)

        top_active = sorted(self.active_dims,
                            key=lambda i: self.importance[i], reverse=True)[:5]
        print(f"  [MS] Most important: {top_active} "
              f"(corr={[round(self.importance[i],3) for i in top_active]})", flush=True)

        return report

    def stagnation_reduce(self):
        """停滞時の次元削減。historyからさらに死に次元を見つける。"""
        if len(self.history) < 10:
            return None

        params_arr = np.array([h['params'] for h in self.history])
        scores = np.array([h['score'] for h in self.history])

        pct = _mp("mirror_scan", "stagnation_top_bottom_pct", 0.1)
        n = len(scores)
        top_idx = np.argsort(scores)[-max(1, int(n * pct)):]
        bot_idx = np.argsort(scores)[:max(1, int(n * pct)):]

        top_mean = np.mean(params_arr[top_idx], axis=0)
        bot_mean = np.mean(params_arr[bot_idx], axis=0)
        diff = np.abs(top_mean - bot_mean)

        threshold = np.max(diff) * _mp("mirror_scan", "stagnation_diff_threshold_ratio", 0.05)
        newly_dead = [i for i in self.active_dims if diff[i] < threshold]

        if not newly_dead:
            return None

        self.dead_dims = list(set(self.dead_dims + newly_dead))
        self.active_dims = [i for i in range(self.n_dims) if i not in self.dead_dims]

        print(f"  [MS] Stagnation reduce: {len(newly_dead)} more dead, "
              f"now {len(self.active_dims)} active", flush=True)

        return {
            'newly_dead': newly_dead,
            'active_dims': self.active_dims,
            'n_active': len(self.active_dims),
        }

    def _compute_importance(self, params_arr, scores):
        """全論公式でimportanceを計算し、dead/active_dimsを更新。"""
        truth, connectivity = _compute_truth_connectivity(params_arr, scores)
        self.importance = _importance_formula(truth, connectivity)

        self.dead_dims = _dead_dims_hybrid(self.importance)
        self.active_dims = [i for i in range(self.n_dims) if i not in self.dead_dims]

        # 死の側: fragility = √(truth × isolation)
        self.fragility = _fragility_formula(truth, connectivity, self.active_dims)

    @classmethod
    def from_measurements(cls, measurements, param_names=None, score_key=None):
        """外部測定データからMirrorScanを構築。eval_fn不要。

        ユーザーが持っている測定データ（params + score のペア）から
        全論公式で構造分析し、build_proxy()が呼べる状態のインスタンスを返す。

        Args:
            measurements: [{"params": dict or list, "score": float}, ...]
                dictの場合: {"temp": 0.7, "top_p": 0.9}
                listの場合: [0.7, 0.9]
                multi-observer: {"params": ..., "scores": {"obs1": float, ...}}
            param_names: パラメータ名リスト（dict keyの順序指定用、省略時は自動検出）
            score_key: "scores" dict内の特定キー。None時は "score" フィールド使用。

        Returns:
            MirrorScan instance (history注入済み、構造分析済み)
            .build_proxy() で proxy eval_fn を自動生成可能
        """
        if not measurements or len(measurements) < 5:
            raise ValueError(f"測定データが不足（{len(measurements) if measurements else 0}件）。最低5件必要。")

        # スコア抽出: score_key指定時は scores[score_key]、なければ score
        def _get_score(m):
            if score_key and "scores" in m:
                return m["scores"][score_key]
            return m["score"]

        first_params = measurements[0]["params"]
        is_dict = isinstance(first_params, dict)

        if is_dict:
            names = param_names or sorted(first_params.keys())
            history = [
                {"params": [m["params"][k] for k in names], "score": _get_score(m)}
                for m in measurements
            ]
        else:
            names = param_names or [f"p{i}" for i in range(len(first_params))]
            history = [
                {"params": list(m["params"]), "score": _get_score(m)}
                for m in measurements
            ]

        n_dims = len(history[0]["params"])

        # param_rangesをデータのmin/max + 10%マージンで自動推定
        param_ranges = []
        for d in range(n_dims):
            vals = [h["params"][d] for h in history]
            lo, hi = min(vals), max(vals)
            margin = max((hi - lo) * 0.1, 1e-6)
            param_ranges.append((lo - margin, hi + margin))

        # ダミーeval_fnでインスタンス化
        instance = cls(eval_fn=lambda p: 0.0, param_ranges=param_ranges)
        instance.history = history

        # 全論公式で構造分析
        params_arr = np.array([h["params"] for h in history])
        scores = np.array([h["score"] for h in history])
        instance._compute_importance(params_arr, scores)

        return instance

    def add_history(self, params, score):
        """TLの評価結果をMSに報告"""
        self.history.append({'params': list(params), 'score': score})

    def build_proxy(self, active_dims=None):
        """MSのhistoryからproxy eval_fnを自動構築。

        6種類のproxyを試して最良を返す:
          1. 全論公式proxy (raw): sqrt(truth × connectivity)ベース
          2. 全論交互作用proxy (raw): connectivity-guided interaction terms
          3. 線形回帰proxy (raw): Σ(weight × param)
          4-6. 上記3種のlog空間版

        全論交互作用proxy: connectivity(i,j)が高いパラメータペアの
        交互作用項 x_i×x_j を自動追加。全論公式のペア拡張。
        interaction_importance = sqrt(pair_truth × pair_connectivity)
        → 実在する交互作用だけ選択的に追加 → 過学習を防ぎつつ非線形性をキャプチャ。

        log候補はスコアのダイナミックレンジが大きい系
        （PPL: 49〜10^18等）で威力を発揮する。
        かけ算の構造（ニューラルネット、化学反応等）をlog→足し算に変換。

        Returns:
            (proxy_fn, proxy_quality) or (None, 0) if data insufficient
            proxy_fn: params(full_dims) → float (0.001ms)
            proxy_quality: R²値 (1.0=完全一致, <0.3=使えない)
        """
        if len(self.history) < 10:
            return None, 0.0, None

        dims = active_dims if active_dims is not None else self.active_dims
        if not dims:
            return None, 0.0, None

        X = np.array([[h['params'][d] for d in dims] for h in self.history])
        y = np.array([h['score'] for h in self.history])

        # log変換可能か判定（全スコアが同符号で非ゼロ）
        can_log = False
        y_log = None
        log_sign = 1.0
        if np.all(y > 0):
            can_log = True
            y_log = np.log(y)
            log_sign = 1.0
        elif np.all(y < 0):
            can_log = True
            y_log = np.log(-y)
            log_sign = -1.0

        # --- 全候補を生成 ---
        candidates = []  # (name, fn, r2)

        for use_log in ([False, True] if can_log else [False]):
            y_fit = y_log if use_log else y
            label_suffix = "_log" if use_log else ""

            # 共通: importance を事前計算（zenron + interact で共有）
            try:
                truth, connectivity = _compute_truth_connectivity(X, y_fit)
                imp = _importance_formula(truth, connectivity)
            except Exception:
                imp = None

            # --- 全論公式proxy ---
            try:
                if imp is None:
                    raise ValueError("imp not computed")
                X_weighted = X * imp[np.newaxis, :]
                X_wb = np.column_stack([X_weighted, np.ones(len(X))])
                result = np.linalg.lstsq(X_wb, y_fit, rcond=None)
                w = result[0]

                # R²は元のy空間で計算（log proxyならexp変換後）
                if use_log:
                    y_pred_log = X_wb @ w
                    y_pred = log_sign * np.exp(y_pred_log)
                else:
                    y_pred = X_wb @ w
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

                f_imp = imp.copy()
                f_w = w[:-1].copy()
                f_b = float(w[-1])
                f_dims = list(dims)
                f_use_log = use_log
                f_sign = log_sign

                def _make_zenron_fn(_imp, _w, _b, _dims, _log, _sign):
                    def fn(params):
                        total = _b
                        for j, dim in enumerate(_dims):
                            total += _w[j] * _imp[j] * params[dim]
                        if _log:
                            return float(_sign * np.exp(total))
                        return float(total)
                    return fn

                fn = _make_zenron_fn(f_imp, f_w, f_b, f_dims, f_use_log, f_sign)
                candidates.append((f"zenron{label_suffix}", fn, float(r2)))
            except Exception:
                pass

            # --- 全論交互作用proxy (connectivity-guided interactions) ---
            try:
                if imp is None:
                    raise ValueError("imp not computed")
                _exp = _mp("mirror_scan", "importance_exponent", 0.5)
                _cf = _mp("mirror_scan", "connectivity_floor", 0.01)
                n_d = len(dims)
                pair_conn = np.zeros((n_d, n_d))
                pair_truth = np.zeros((n_d, n_d))

                # ベクトル化: pair_conn を1回のcorrcoefで一括計算
                import warnings as _w
                stds_x = np.std(X, axis=0)
                active_cols = stds_x > 1e-10
                if np.any(active_cols):
                    with _w.catch_warnings():
                        _w.simplefilter("ignore", RuntimeWarning)
                        C_x = np.corrcoef(X.T)  # (n_d, n_d)
                    C_x = np.nan_to_num(C_x)
                    pair_conn_full = np.abs(C_x)
                    np.fill_diagonal(pair_conn_full, 0.0)
                    # 上三角のみ保持（旧コードと同じ）
                    pair_conn = np.triu(pair_conn_full, k=1)

                # ベクトル化: pair_truth — 交互作用列を一括生成→一括相関
                valid_pairs = [(j, k) for j in range(n_d) for k in range(j + 1, n_d)
                               if stds_x[j] > 1e-10 and stds_x[k] > 1e-10]
                if valid_pairs:
                    int_cols = np.column_stack([X[:, j] * X[:, k] for j, k in valid_pairs])
                    int_stds = np.std(int_cols, axis=0)
                    valid_int_mask = int_stds > 1e-10
                    if np.any(valid_int_mask):
                        data_int = np.vstack([int_cols[:, valid_int_mask].T,
                                              y_fit.reshape(1, -1)])
                        with _w.catch_warnings():
                            _w.simplefilter("ignore", RuntimeWarning)
                            C_int = np.corrcoef(data_int)
                        C_int = np.nan_to_num(C_int)
                        corrs = np.abs(C_int[:-1, -1])
                        idx = 0
                        for pi, (j, k) in enumerate(valid_pairs):
                            if valid_int_mask[pi]:
                                pair_truth[j, k] = corrs[idx]
                                idx += 1

                pair_imp = (pair_truth * np.maximum(pair_conn, _cf)) ** _exp
                _imf = _mp("proxy", "interaction_median_floor", 0.1)
                median_imp = np.median(imp[imp > 0.01]) if np.any(imp > 0.01) else _imf

                int_pairs = [(j, k) for j in range(n_d) for k in range(j + 1, n_d)
                             if pair_imp[j, k] > median_imp]

                # Rule 5: n_measurements > n_terms
                max_terms = len(X) // 2
                if n_d + len(int_pairs) + 1 > max_terms:
                    int_pairs = sorted(int_pairs,
                                       key=lambda jk: pair_imp[jk[0], jk[1]],
                                       reverse=True)[:max(max_terms - n_d - 1, 0)]

                if int_pairs:
                    X_int = X * imp[np.newaxis, :]
                    for j, k in int_pairs:
                        X_int = np.column_stack([X_int, X[:, j] * X[:, k]])
                    X_int = np.column_stack([X_int, np.ones(len(X))])
                    result_int = np.linalg.lstsq(X_int, y_fit, rcond=None)
                    w_int = result_int[0]

                    if use_log:
                        y_pred_int = log_sign * np.exp(X_int @ w_int)
                    else:
                        y_pred_int = X_int @ w_int
                    ss_res_int = np.sum((y - y_pred_int) ** 2)
                    ss_tot_int = np.sum((y - np.mean(y)) ** 2)
                    r2_int = 1.0 - ss_res_int / max(ss_tot_int, 1e-10)

                    f_imp_i = imp.copy()
                    f_w_i = w_int.copy()
                    f_dims_i = list(dims)
                    f_pairs_i = list(int_pairs)
                    f_nb = n_d
                    f_log_i = use_log
                    f_sgn_i = log_sign

                    def _make_interact_fn(_imp, _w, _dims, _pairs, _nb, _log, _sgn):
                        def fn(params):
                            n = len(params)
                            total = float(_w[-1])
                            for j, dim in enumerate(_dims):
                                if dim < n:
                                    total += float(_w[j]) * _imp[j] * params[dim]
                            for idx, (j, k) in enumerate(_pairs):
                                if _dims[j] < n and _dims[k] < n:
                                    total += float(_w[_nb + idx]) * params[_dims[j]] * params[_dims[k]]
                            if _log:
                                return float(_sgn * np.exp(total))
                            return float(total)
                        return fn

                    fn_i = _make_interact_fn(f_imp_i, f_w_i, f_dims_i, f_pairs_i,
                                             f_nb, f_log_i, f_sgn_i)
                    candidates.append((f"zenron_interact{label_suffix}", fn_i, float(r2_int)))
            except Exception:
                pass

            # --- 線形回帰proxy ---
            try:
                X_bias = np.column_stack([X, np.ones(len(X))])
                result = np.linalg.lstsq(X_bias, y_fit, rcond=None)
                weights = result[0]

                if use_log:
                    y_pred_log = X_bias @ weights
                    y_pred = log_sign * np.exp(y_pred_log)
                else:
                    y_pred = X_bias @ weights
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

                f_w2 = weights[:-1].copy()
                f_b2 = float(weights[-1])
                f_dims2 = list(dims)
                f_use_log2 = use_log
                f_sign2 = log_sign

                def _make_linear_fn(_w, _b, _dims, _log, _sign):
                    def fn(params):
                        total = _b
                        for j, dim in enumerate(_dims):
                            total += _w[j] * params[dim]
                        if _log:
                            return float(_sign * np.exp(total))
                        return float(total)
                    return fn

                fn = _make_linear_fn(f_w2, f_b2, f_dims2, f_use_log2, f_sign2)
                candidates.append((f"linear{label_suffix}", fn, float(r2)))
            except Exception:
                pass

        # --- 最良候補を選択 ---
        if not candidates:
            return None, 0.0, None

        candidates.sort(key=lambda x: x[2], reverse=True)
        best_name, best_fn, best_r2 = candidates[0]

        summary = ", ".join(f"{n} R2={r:.3f}" for n, _, r in candidates)
        print(f"  [MS Proxy] {summary} -> {best_name} selected")

        # --- 残差分析: dead_dims の中に本物が埋まっていないか ---
        self.recovered_dims = []
        if self.dead_dims and best_fn and len(self.history) >= 10:
            try:
                all_params = np.array([h['params'] for h in self.history])
                scores = np.array([h['score'] for h in self.history])
                preds = np.array([best_fn(list(p)) for p in all_params])
                resid = scores - preds
                if np.std(resid) > 1e-10:
                    for d in self.dead_dims:
                        col = all_params[:, d]
                        if np.std(col) < 1e-10:
                            continue
                        r = np.corrcoef(resid, col)[0, 1]
                        if not np.isnan(r) and abs(r) > 0.3:
                            self.recovered_dims.append((d, float(abs(r))))
                    self.recovered_dims.sort(key=lambda x: x[1], reverse=True)
                    if self.recovered_dims:
                        labels = [f"dim{d}(r={r:.2f})" for d, r in self.recovered_dims]
                        print(f"  [MS] Recovered from dead: {', '.join(labels)}")
            except Exception:
                pass

        return best_fn, best_r2, best_name


class MirrorAgent:
    """MA最外殻エージェント。MS→ED→EA→TL全層を円環で自動実行。

    Two modes:
        API mode:    MirrorAgent(eval_fn=fn, param_ranges=ranges).run()
        Folder mode: MirrorAgent(folder="my_problem/").run()
    """

    _EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "experience")

    def __init__(self, eval_fn=None, param_ranges=None, initial_params=None,
                 folder=None, config=None, kathara="auto",
                 n_ma_nodes=None, n_inner_workers=None, **kwargs):
        """
        Args:
            kathara: K²モード制御。
                "auto" (default): eval_fnがmodule-importableならK²、lambda/closureなら旧MA
                True:  強制K²（lambdaなら旧MAにフォールバック）
                False: 強制旧MA（K²を使わない）
        """
        self.eval_fn = eval_fn
        self.param_ranges = param_ranges
        self.initial_params = initial_params
        self.folder = folder
        self.config_path = config
        self.kathara = kathara
        self.n_ma_nodes = n_ma_nodes
        self.n_inner_workers = n_inner_workers
        self.kwargs = kwargs
        self.mr_scan = None
        self.ma_exp = None

    def _load_ma_experience(self, experience_id):
        """MA経験をJSONから読み込み"""
        if not experience_id:
            return {}
        path = os.path.join(self._EXP_DIR, f"ma_{experience_id}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_ma_experience(self, experience_id, data):
        """MA経験をJSONに保存（ロールバック保護付き）"""
        if not experience_id:
            return
        os.makedirs(self._EXP_DIR, exist_ok=True)
        path = os.path.join(self._EXP_DIR, f"ma_{experience_id}.json")
        existing = self._load_ma_experience(experience_id)

        # ロールバック判定: 前回より悪かったら前回のbest_paramsを保持
        prev_best = existing.get("best_score")
        new_best = data.get("best_score", 0)
        if prev_best is not None and new_best < prev_best:
            print(f"  [MA Rollback] Score dropped: {prev_best} → {new_best}. "
                  f"Keeping prior best_params.", flush=True)
            # best_params/best_scoreは前回を維持
            data["best_params"] = existing.get("best_params", data.get("best_params"))
            data["best_score"] = prev_best
            data["rollback_count"] = existing.get("rollback_count", 0) + 1
        else:
            data["rollback_count"] = existing.get("rollback_count", 0)

        # 前回設定のバックアップ（MA設定JSONのロールバック用）
        data["prev_config"] = self._backup_ma_config()

        existing.update(data)
        existing["runs"] = existing.get("runs", 0) + 1
        existing["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def _backup_ma_config(self):
        """現在のMA設定JSONのスナップショットを返す"""
        try:
            with open(_MA_META_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _rollback_ma_config(self, experience_id):
        """前回のMA設定に戻す"""
        exp = self._load_ma_experience(experience_id)
        prev_config = exp.get("prev_config")
        if prev_config:
            with open(_MA_META_PATH, "w", encoding="utf-8") as f:
                json.dump(prev_config, f, indent=2, ensure_ascii=False)
            print(f"  [MA Rollback] Config restored from previous run.", flush=True)
            return True
        return False

    def _should_use_kathara(self):
        """eval_fnがmodule-importableかどうかでK²使用を自動判定。

        K²に必要: eval_fnがimportableモジュール関数であること。
        lambda/closure/__main__定義 → Falseを返し旧MAにフォールバック。
        """
        if self.eval_fn is None:
            return False
        fn = self.eval_fn
        if hasattr(fn, "__module__") and hasattr(fn, "__qualname__"):
            mod = fn.__module__
            name = fn.__qualname__
            if "<" in name:
                # lambda or closure
                return False
            if mod == "__main__":
                print("  [MA] eval_fn is in __main__ -> single-node MA. "
                      "For K2, define eval_fn in an importable module.",
                      flush=True)
                return False
            return True
        return False

    def run(self, max_iterations=10, mr_samples=20, time_budget=3600,
            experience_id=None, kathara=None):
        """円環実行。

        API mode:    MS → TL → 停滞検出 → MS削減 → TL再開
        Folder mode: MS → ED → EA(AT^n) → TL → 円環
        kathara:
            "auto"/None: eval_fnがimportableならK²、lambdaなら旧MA
            True:  K²（lambdaなら旧MAにフォールバック）
            False: 旧MA
        """
        # kathara引数 > __init__のkathara > auto判定
        k = kathara if kathara is not None else self.kathara

        if k == "auto":
            use_kathara = self._should_use_kathara()
        elif k is True:
            use_kathara = True
        else:
            use_kathara = False

        if self.eval_fn is not None:
            if use_kathara:
                return self._run_api_kathara(max_iterations, mr_samples,
                                             time_budget, experience_id)
            return self._run_api(max_iterations, mr_samples, time_budget,
                                 experience_id)
        else:
            return self._run_folder(max_iterations, mr_samples)

    def _run_api(self, max_iterations, mr_samples, time_budget, experience_id):
        """API mode: eval_fn → 初期測定 → owl(autonomous) → 結果。

        owl() が MS→proxy→optimize→verify→自動成長を全て処理する。
        旧実装の AutoEscalation(EA¹-EA⁵) は owl autonomous の
        停滞検知+範囲摂動+近傍サンプリングに統合済み。
        """
        from twelve.optimize import owl
        import random as _random

        eval_fn = self.eval_fn
        param_ranges = self.param_ranges
        initial = self.initial_params or [
            (lo + hi) / 2 for lo, hi in param_ranges
        ]
        exp_id = experience_id or 'ma_api'

        print("=" * 60)
        print(f"  MA API -> owl(autonomous) | {len(param_ranges)}D, {time_budget}s")
        print("=" * 60)

        # Step 1: 初期測定データ収集
        measurements = []
        rng = _random.Random(42)
        s = eval_fn(initial)
        measurements.append({"params": list(initial), "score": float(s)})
        for _ in range(max(1, mr_samples) - 1):
            sample = [rng.uniform(lo, hi) for lo, hi in param_ranges]
            s = eval_fn(sample)
            measurements.append({"params": sample, "score": float(s)})

        # Step 2: owl autonomous（MS→proxy→optimize→verify→自動成長）
        result = owl(
            measurements=measurements,
            param_ranges=param_ranges,
            verify_fn=eval_fn,
            time_budget=time_budget,
            experience_id=exp_id,
            autonomous=True,
            max_iterations=max_iterations,
            verbose=True,
        )

        # Step 3: 結果整形（list形式に統一）
        bp = result.get('best_params')
        if isinstance(bp, dict):
            bp = [bp.get(f'p{i}', (lo + hi) / 2)
                  for i, (lo, hi) in enumerate(param_ranges)]

        self.mr_scan = None

        best_score = result.get('verified_score') or result.get('best_score', float('-inf'))
        active = result.get('active_dims', [])
        dead = result.get('dead_dims', [])

        print(f"\n  MA Complete: score={best_score:.4f}, "
              f"active={len(active)}/{len(param_ranges)}, "
              f"R2={result.get('proxy_r2', 0):.3f}")

        return {
            'best_params': bp or list(initial),
            'best_score': best_score,
            'active_dims': active,
            'dead_dims': dead,
            'n_active': len(active),
            'n_total': len(param_ranges),
            'iterations': result.get('rounds_completed', 0),
            'proxy_r2': result.get('proxy_r2', 0),
            'confidence': result.get('confidence', 'unknown'),
        }

    def _run_api_kathara(self, max_iterations, mr_samples, time_budget,
                          experience_id):
        """K-squared: MA-level Kathara (6ノード並列) × TL-level Kathara。

        owl(kathara=True) 経由でも呼べる。
        eval_fnがモジュールインポート可能な場合のみ動作。
        lambda/closureの場合は自動でシングルノードにフォールバック。
        """
        # eval_fnのモジュールパスを解決
        eval_fn = self.eval_fn
        eval_fn_module = None
        eval_fn_name = None

        if hasattr(eval_fn, "__module__") and hasattr(eval_fn, "__qualname__"):
            mod = eval_fn.__module__
            name = eval_fn.__qualname__
            if "<" not in name and mod != "__main__":
                eval_fn_module = mod
                eval_fn_name = name

        if eval_fn_module is None:
            print("  [MA-K2] eval_fn is lambda/closure -> single-node fallback")
            return self._run_api(max_iterations, mr_samples, time_budget,
                                 experience_id)

        from .kathara_ma import KatharaMACoordinator

        kw = {}
        if self.n_ma_nodes is not None:
            kw["n_ma_nodes"] = self.n_ma_nodes
        if self.n_inner_workers is not None:
            kw["n_inner_workers"] = self.n_inner_workers

        coordinator = KatharaMACoordinator(
            eval_fn_module=eval_fn_module,
            eval_fn_name=eval_fn_name,
            param_ranges=self.param_ranges,
            initial_params=self.initial_params,
            time_budget=time_budget,
            experience_id=experience_id or "ma_k2",
            max_iterations=max_iterations,
            mr_samples=mr_samples,
            **kw,
        )

        try:
            result = coordinator.run()
        except Exception as e:
            print(f"  [MA-K2] Coordinator failed: {e} -> single-node fallback")
            return self._run_api(max_iterations, mr_samples, time_budget,
                                 experience_id)

        # MA Experience保存（旧MAと同じフォーマットで互換性維持）
        self._save_ma_experience(experience_id, {
            "best_score": result.get("best_score"),
            "best_params": result.get("best_params"),
            "ms_dead_dims": result.get("dead_dims", []),
            "ms_active_dims": result.get("active_dims", []),
            "ms_importance": [],  # K²ではノード集約なし（次回MSで再計算される）
            "n_active": result.get("n_active"),
            "n_total": result.get("n_total"),
            "mode": "kathara_k2",
            "n_ma_nodes": result.get("n_ma_nodes"),
        })

        # 戻り値を旧MAと互換にする（共通キー + K²固有キー）
        return {
            'best_params': result.get('best_params'),
            'best_score': result.get('best_score'),
            'active_dims': result.get('active_dims', []),
            'n_active': result.get('n_active', 0),
            'n_total': result.get('n_total', 0),
            'iterations': result.get('polls', 0),  # K²ではポーリング回数
            # K²固有
            'dead_dims': result.get('dead_dims', []),
            'n_ma_nodes': result.get('n_ma_nodes'),
            'n_inner_workers': result.get('n_inner_workers'),
            'elapsed': result.get('elapsed'),
        }

    def _run_folder(self, max_iterations, mr_samples):
        """Folder mode: Zenron経由でEA+MS+ED全自動。"""
        from twelve.agent import EvolutionAgent

        print("=" * 60)
        print("  MA (MirrorAgent) - Folder Mode")
        print("  MS(scan) -> ED(quality) -> EA(escalation) -> TL(optimize)")
        print("=" * 60)

        if self.folder:
            agent = EvolutionAgent(folder=self.folder, **self.kwargs)
        elif self.config_path:
            agent = EvolutionAgent(self.config_path)
        else:
            raise ValueError("folder or config required")

        agent.config.mirror = True
        agent.config.auto_escalation = True
        agent.config.escalation_threshold = 3

        def mirror_callback(action, info):
            if self.mr_scan is None:
                self.mr_scan = getattr(agent.config, '_mirror_scan', None)
            if self.mr_scan is None:
                return None
            if action == "mirror_reduce":
                return self.mr_scan.stagnation_reduce()
            elif action == "mirror_redesign":
                print("  [MS] eval_fn redesign not yet implemented", flush=True)
                return None
            return None

        agent.config._mirror_callback = mirror_callback

        print(f"  MS samples: {mr_samples}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Auto-escalation: ON (threshold=3)")
        print()

        result = agent.run(max_iterations=max_iterations)

        print("\n" + "=" * 60)
        print("  MA Complete.")
        if self.mr_scan:
            print(f"  MS history: {len(self.mr_scan.history)} evaluations")
            print(f"  Active dims: {len(self.mr_scan.active_dims)}/{self.mr_scan.n_dims}")
        print("=" * 60)

        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MA (MirrorAgent)")
    parser.add_argument("--folder", help="eval_fn.pyがあるフォルダ")
    parser.add_argument("--config", help="EAのJSON設定ファイル")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--mr-samples", type=int, default=20)
    parser.add_argument("--kathara", choices=["auto", "true", "false"],
                        default="auto", help="K2 mode (auto/true/false)")
    args = parser.parse_args()

    k = {"auto": "auto", "true": True, "false": False}[args.kathara]
    ma = MirrorAgent(folder=args.folder, config=args.config, kathara=k)
    ma.run(max_iterations=args.iterations, mr_samples=args.mr_samples)
