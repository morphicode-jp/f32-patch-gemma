"""Nous — 自律的構造理解エージェント（RL + LaD + Owl 統合）。

Owlは「データを渡す→構造を返す」（受動的な脳）。
Nousは「自分で行動→自分でデータ化→構造を理解→理解に基づいて行動を改善」（自律的知性）。

3層アーキテクチャ:
    RL  = 環境内で行動し経験を得る（手足）
    LaD = 経験を数値測定に変換する（翻訳）
    Owl = 測定から構造を発見する（脳）

Usage (RL mode):
    from twelve.agent.nous import Nous, BalanceEnv

    env = BalanceEnv()
    result = Nous(env=env).run(time_budget=60)
    result["best_params"]   # 最適ポリシー (W行列 + bias)
    result["importance"]    # どのobs→act接続が重要か
    result["dead_dims"]     # 不要な接続 (pruning候補)

Usage (eval_fn mode — 後方互換):
    result = Nous(eval_fn, param_ranges=[(lo, hi)] * n).run(time_budget=300)

核心の洞察: RL環境はeval_fnの自動生成機。
    env + policy_params → run_episode → total_reward = eval_fn(params)
    この変換により、既存の4フェーズループ・MirrorScan・owl()が全て無変更で動く。
"""

import math
import os
import sys
import time
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from twelve.agent.mirror_agent import MirrorScan


# ================================================================
# Environment プロトコル
# ================================================================

class Environment:
    """環境インターフェース。gym不要、自己完結。

    継承して reset() / step() / obs_size / n_actions を実装する。
    """

    def reset(self):
        """環境をリセット → 初期観測 (list[float]) を返す。"""
        raise NotImplementedError

    def step(self, action):
        """行動を実行 → (observation, reward, done) を返す。

        Args:
            action: int (離散行動のインデックス)
        Returns:
            observation: list[float]
            reward: float
            done: bool
        """
        raise NotImplementedError

    @property
    def obs_size(self):
        """観測ベクトルの次元数。"""
        raise NotImplementedError

    @property
    def n_actions(self):
        """離散行動の数。"""
        raise NotImplementedError


# ================================================================
# BalanceEnv — 倒立振子デモ環境（CartPole相当、外部依存なし）
# ================================================================

class BalanceEnv(Environment):
    """倒立振子。OpenAI Gym CartPole-v1 と同じ物理。

    obs = [位置, 速度, 角度, 角速度]  (4次元)
    actions = [左(-10N), 右(+10N)]    (2択)
    報酬: 倒れなかった1ステップにつき+1
    終了: |位置| > 2.4 or |角度| > 12度 or 200ステップ
    """

    GRAVITY = 9.8
    CART_MASS = 1.0
    POLE_MASS = 0.1
    TOTAL_MASS = CART_MASS + POLE_MASS
    POLE_HALF_LEN = 0.5
    FORCE_MAG = 10.0
    DT = 0.02
    MAX_STEPS = 200
    X_THRESHOLD = 2.4
    THETA_THRESHOLD = 12.0 * math.pi / 180.0

    def __init__(self):
        self._state = None
        self._step_count = 0
        self._rng = np.random.RandomState(None)

    @property
    def obs_size(self):
        return 4

    @property
    def n_actions(self):
        return 2

    def reset(self):
        self._state = self._rng.uniform(-0.05, 0.05, size=4).tolist()
        self._step_count = 0
        return list(self._state)

    def step(self, action):
        x, x_dot, theta, theta_dot = self._state
        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # CartPole dynamics (Euler)
        temp = (force + self.POLE_MASS * self.POLE_HALF_LEN * theta_dot ** 2 * sin_t) / self.TOTAL_MASS
        theta_acc = (self.GRAVITY * sin_t - cos_t * temp) / (
            self.POLE_HALF_LEN * (4.0 / 3.0 - self.POLE_MASS * cos_t ** 2 / self.TOTAL_MASS)
        )
        x_acc = temp - self.POLE_MASS * self.POLE_HALF_LEN * theta_acc * cos_t / self.TOTAL_MASS

        x += self.DT * x_dot
        x_dot += self.DT * x_acc
        theta += self.DT * theta_dot
        theta_dot += self.DT * theta_acc

        self._state = [x, x_dot, theta, theta_dot]
        self._step_count += 1

        done = (abs(x) > self.X_THRESHOLD or
                abs(theta) > self.THETA_THRESHOLD or
                self._step_count >= self.MAX_STEPS)

        reward = 1.0 if not done or self._step_count >= self.MAX_STEPS else 0.0

        return list(self._state), reward, done


# ================================================================
# FlyWorldEnv — FlyWorld adapter for Environment interface
# ================================================================

class FlyWorldEnv(Environment):
    """FlyWorldV3 を Environment インターフェースに適合。

    FlyWorld: 2D空間で食物を探すハエ。12Dセンサー。
    離散化: 3×3 = 9アクション (turn × speed)
    """

    def __init__(self, version=3, max_steps=60, **kwargs):
        self._version = version
        self._max_steps = max_steps
        self._kwargs = kwargs
        self._env = None
        self._step_count = 0
        self._total_reward = 0.0

    def _make_env(self):
        import importlib.util
        import os
        # kathara_brain_sim_v8.py からFlyWorldクラスを動的ロード
        base = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        path = os.path.join(base, "kathara_brain_sim_v8.py")
        spec = importlib.util.spec_from_file_location("kbsv8", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls_map = {1: "FlyWorldV1", 2: "FlyWorldV2", 3: "FlyWorldV3"}
        cls = getattr(mod, cls_map.get(self._version, "FlyWorldV3"))
        return cls(**self._kwargs)

    @property
    def obs_size(self):
        return 12

    @property
    def n_actions(self):
        return 9  # 3 turn × 3 speed

    def reset(self):
        if self._env is None:
            self._env = self._make_env()
        obs = self._env.reset()
        self._step_count = 0
        self._total_reward = 0.0
        return list(obs.astype(float))

    def step(self, action):
        # 9 actions: 3 turn × 3 speed
        turn_idx = action // 3  # 0=left, 1=straight, 2=right
        speed_idx = action % 3  # 0=slow, 1=medium, 2=fast
        nav = turn_idx / 2.0    # 0.0, 0.5, 1.0
        cen = speed_idx / 2.0   # 0.0, 0.5, 1.0

        result = self._env.step(nav, cen)
        obs = result[0]
        dist = result[1]
        reached = result[2]

        self._step_count += 1

        # reward: approach bonus + reach bonus
        reward = -dist * 0.01  # 近づくほど良い
        if reached:
            reward += 10.0

        done = reached or self._step_count >= self._max_steps
        return list(obs.astype(float)), float(reward), done


# ================================================================
# KatharaPolicy — Kathara(12) hidden layer policy
# ================================================================

class KatharaPolicy:
    """Kathara(12,{1,4,6}) 隠れ層ポリシー。

    Architecture: obs → W_in → tanh → Kathara_propagate → tanh → W_out → action

    Parameter layout:
        W_in:       obs_size × n_hidden     (input projection)
        edge_w:     n_edges                  (Kathara edge weights)
        W_out:      n_hidden × n_actions     (output projection)
        bias:       n_actions                (output bias)
    """

    def __init__(self, obs_size, n_actions, n_hidden=12, offsets=None):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.offsets = offsets or [1, 4, 6]
        self.adj = self._build_circulant(n_hidden, self.offsets)
        # edge indices for efficient packing
        self.edge_list = []
        for i in range(n_hidden):
            for j in range(i + 1, n_hidden):
                if self.adj[i, j]:
                    self.edge_list.append((i, j))
        self.n_edges = len(self.edge_list)
        # parameter counts
        self.n_in = obs_size * n_hidden
        self.n_out = n_hidden * n_actions
        self.n_bias = n_actions
        self.n_params = self.n_in + self.n_edges + self.n_out + self.n_bias

    @staticmethod
    def _build_circulant(n, offsets):
        adj = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for o in offsets:
                j = (i + o) % n
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        return adj

    def forward(self, obs, params):
        """obs (list/array) + params (list/array) → action (int)."""
        obs = np.asarray(obs, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)

        # Unpack
        off = 0
        W_in = params[off:off + self.n_in].reshape(self.obs_size, self.n_hidden)
        off += self.n_in
        edge_w = params[off:off + self.n_edges]
        off += self.n_edges
        W_out = params[off:off + self.n_out].reshape(self.n_hidden, self.n_actions)
        off += self.n_out
        bias = params[off:off + self.n_bias]

        # Forward: input → hidden
        hidden = np.tanh(obs @ W_in)

        # Kathara propagation: weighted adjacency × hidden
        weighted_adj = np.zeros_like(self.adj)
        for idx, (i, j) in enumerate(self.edge_list):
            weighted_adj[i, j] = edge_w[idx]
            weighted_adj[j, i] = edge_w[idx]
        hidden = np.tanh(weighted_adj @ hidden)

        # Output
        logits = hidden @ W_out + bias
        return int(np.argmax(logits))

    def param_names(self):
        """人間が読める名前を生成。"""
        names = []
        for o in range(self.obs_size):
            for h in range(self.n_hidden):
                names.append(f"in[{o}][{h}]")
        for idx, (i, j) in enumerate(self.edge_list):
            names.append(f"e[{i}-{j}]")
        for h in range(self.n_hidden):
            for a in range(self.n_actions):
                names.append(f"out[{h}][{a}]")
        for a in range(self.n_actions):
            names.append(f"b[{a}]")
        return names


# ================================================================
# Nous — 自律的構造理解エージェント
# ================================================================

class Nous:
    """自律的構造理解エージェント。

    4フェーズで自律的に探索→理解→最適化:
        EXPLORE: ランダム探索で全体把握
        FOCUS:   importance比例サンプリングで重要次元を深堀り
        DEEPEN:  interaction pairs重点探索
        EXPLOIT: proxy事前スクリーニングで効率的最適化

    RL mode (env引数): 環境から自動的にeval_fnを構築。
    eval_fn mode: 従来通り直接eval_fnを渡す。
    """

    def __init__(self, eval_fn=None, param_ranges=None, param_names=None,
                 experience_id=None, env=None, policy_type="linear"):
        self.env = env
        self.policy_type = policy_type
        self._kathara_policy = None

        # RL mode: 環境からeval_fn/param_ranges/param_namesを自動構築
        if env is not None and eval_fn is None:
            if policy_type == "kathara":
                self._kathara_policy = KatharaPolicy(
                    env.obs_size, env.n_actions)
            n_policy = self._policy_dims()
            eval_fn = lambda params: self._run_episode(params)
            param_ranges = [(-2.0, 2.0)] * n_policy
            param_names = self._make_policy_names()

        self.eval_fn = eval_fn
        self.param_ranges = param_ranges
        self.n_dims = len(param_ranges)
        self.param_names = param_names or [f"p{i}" for i in range(self.n_dims)]
        self.experience_id = experience_id

        # 内部状態
        self.measurements = []
        self.importance = np.ones(self.n_dims)
        self.dead_dims = []
        self.active_dims = list(range(self.n_dims))
        self.proxy_fn = None
        self.proxy_r2 = 0.0
        self.proxy_type = None
        self.phase = "EXPLORE"
        self.best_params = None
        self.best_score = float('-inf')
        self.rng = np.random.RandomState(42)

        # フェーズ遷移追跡
        self._dead_history = []
        self._phase_log = []
        self._understand_count = 0

    # ================================================================
    # ポリシー層 (RL mode)
    # ================================================================

    def _policy_dims(self):
        """ポリシーのパラメータ数。"""
        if self._kathara_policy is not None:
            return self._kathara_policy.n_params
        return self.env.n_actions * self.env.obs_size + self.env.n_actions

    def _make_policy_names(self):
        """ポリシーパラメータの名前を生成。"""
        if self._kathara_policy is not None:
            return self._kathara_policy.param_names()
        names = []
        for a in range(self.env.n_actions):
            for o in range(self.env.obs_size):
                names.append(f"W[{a}][{o}]")
        for a in range(self.env.n_actions):
            names.append(f"b[{a}]")
        return names

    def _params_to_action(self, params, obs):
        """ポリシーパラメータ + 観測 → 行動選択。"""
        if self._kathara_policy is not None:
            return self._kathara_policy.forward(obs, params)
        na = self.env.n_actions
        ns = self.env.obs_size
        W = np.array(params[:na * ns]).reshape(na, ns)
        b = np.array(params[na * ns:na * ns + na])
        scores = W @ np.array(obs) + b
        return int(np.argmax(scores))

    def _run_episode(self, params, n_episodes=3):
        """ポリシーパラメータでn_episodes回実行 → 平均報酬。"""
        total = 0.0
        for _ in range(n_episodes):
            obs = self.env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action = self._params_to_action(params, obs)
                obs, reward, done = self.env.step(action)
                ep_reward += reward
            total += ep_reward
        return total / n_episodes

    # ================================================================
    # メインループ
    # ================================================================

    def run(self, time_budget=300, verbose=True):
        """メインループ。フェーズ自動遷移。"""
        t0 = time.time()
        self._budget = time_budget

        mode = "RL" if self.env is not None else "eval_fn"
        if verbose:
            print("=" * 60)
            print(f"  Nous | {self.n_dims}D, {time_budget}s, mode={mode}")
            print("=" * 60)

        # 経験読み込み
        warm_params = self._load_experience()

        # EXPLORE: 初期ランダム探索
        self.phase = "EXPLORE"
        self._log_phase(t0, len(self.measurements), verbose)

        n_initial = max(20, self.n_dims)

        # warm startがあれば最初に評価
        if warm_params is not None:
            score = float(self.eval_fn(warm_params))
            self.measurements.append({"params": list(warm_params), "score": score})
            if score > self.best_score:
                self.best_score = score
                self.best_params = list(warm_params)
            n_initial -= 1

        for _ in range(n_initial):
            sample = self._random_sample()
            score = float(self.eval_fn(sample))
            self.measurements.append({"params": sample, "score": score})
            if score > self.best_score:
                self.best_score = score
                self.best_params = sample

        # 初回構造理解
        self._understand(verbose)

        # メインループ
        batch_size = 5
        while time.time() - t0 < time_budget:
            prev_phase = self.phase
            self._check_phase_transition(t0)
            if self.phase != prev_phase:
                self._log_phase(t0, len(self.measurements), verbose)

            if self.phase == "EXPLORE":
                samples = [self._random_sample() for _ in range(batch_size)]
            elif self.phase == "FOCUS":
                samples = [self._importance_guided_sample() for _ in range(batch_size)]
            elif self.phase == "DEEPEN":
                samples = [self._interaction_sample() for _ in range(batch_size)]
            elif self.phase == "EXPLOIT":
                if self.proxy_fn is not None:
                    batch = self._proxy_screened_sample(n_candidates=100,
                                                        n_eval=batch_size)
                    for m in batch:
                        self.measurements.append(m)
                        if m["score"] > self.best_score:
                            self.best_score = m["score"]
                            self.best_params = m["params"]
                    self._understand(verbose)
                    continue
                else:
                    samples = [self._importance_guided_sample()
                               for _ in range(batch_size)]

            # eval実行 + 蓄積
            for s in samples:
                score = float(self.eval_fn(s))
                self.measurements.append({"params": s, "score": score})
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = s

            # 定期的に構造理解を更新
            if len(self.measurements) % 10 == 0:
                self._understand(verbose)

        # 最終owl()で最適化仕上げ
        remaining = max(10, time_budget - (time.time() - t0) + 30)
        self._final_optimize(remaining, verbose)

        # 経験保存
        self._save_experience()

        elapsed = time.time() - t0

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Nous Complete: score={self.best_score:.4f}, "
                  f"{len(self.measurements)} evals, {elapsed:.1f}s")
            print(f"  active={len(self.active_dims)}/{self.n_dims}, "
                  f"dead={len(self.dead_dims)}, R2={self.proxy_r2:.3f}")
            print(f"  Phases: {' -> '.join(p['phase'] for p in self._phase_log)}")
            print(f"{'=' * 60}")

        result = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "importance": self.importance.tolist(),
            "dead_dims": self.dead_dims,
            "active_dims": self.active_dims,
            "proxy_r2": self.proxy_r2,
            "proxy_type": self.proxy_type,
            "phase": self.phase,
            "phase_log": self._phase_log,
            "n_measurements": len(self.measurements),
            "elapsed_s": elapsed,
        }

        # RL mode: ポリシーの構造解釈を追加
        if self.env is not None:
            result["policy_interpretation"] = self._interpret_policy()

        return result

    # ================================================================
    # サンプリング戦略
    # ================================================================

    def _random_sample(self):
        """均一ランダムサンプリング。"""
        return [float(self.rng.uniform(lo, hi))
                for lo, hi in self.param_ranges]

    def _importance_guided_sample(self):
        """importance比例サンプリング。dead_dimsは中央固定。

        核心: 重要な次元を広く探索、不重要な次元を狭く探索。
        """
        imp_max = np.max(self.importance) if np.max(self.importance) > 0 else 1.0
        sample = []
        for i, (lo, hi) in enumerate(self.param_ranges):
            if i in self.dead_dims:
                # dead dim: best_paramsの値 or 中央
                if self.best_params is not None:
                    sample.append(self.best_params[i])
                else:
                    sample.append((lo + hi) / 2)
            else:
                # active dim: importanceに比例した幅で探索
                imp_ratio = self.importance[i] / imp_max
                center = self.best_params[i] if self.best_params else (lo + hi) / 2
                # imp高 → 幅広 (全レンジ)、imp低 → 幅狭 (10%レンジ)
                width = (hi - lo) * (0.1 + 0.9 * imp_ratio)
                val = float(self.rng.uniform(
                    max(lo, center - width / 2),
                    min(hi, center + width / 2)
                ))
                sample.append(val)
        return sample

    def _interaction_sample(self):
        """interaction pairsを重点的に変動させるサンプリング。"""
        sample = self._importance_guided_sample()

        # interaction pairsがなければ通常のimportance-guidedにフォールバック
        if not hasattr(self, '_interaction_pairs') or not self._interaction_pairs:
            return sample

        # 上位interaction pairの両次元を同時に大きく変動
        for dim_i, dim_j in self._interaction_pairs[:3]:
            lo_i, hi_i = self.param_ranges[dim_i]
            lo_j, hi_j = self.param_ranges[dim_j]
            sample[dim_i] = float(self.rng.uniform(lo_i, hi_i))
            sample[dim_j] = float(self.rng.uniform(lo_j, hi_j))

        return sample

    def _proxy_screened_sample(self, n_candidates=100, n_eval=5):
        """proxy_fnでn_candidates点を事前評価、上位n_eval点だけ本番eval。"""
        candidates = [self._importance_guided_sample()
                      for _ in range(n_candidates)]
        proxy_scores = np.array([self.proxy_fn(c) for c in candidates])
        top_indices = np.argsort(proxy_scores)[-n_eval:]

        results = []
        for idx in top_indices:
            score = float(self.eval_fn(candidates[idx]))
            results.append({"params": candidates[idx], "score": score})
        return results

    # ================================================================
    # 構造理解 (MirrorScan利用)
    # ================================================================

    def _understand(self, verbose=False):
        """MirrorScanで構造理解を更新。"""
        if len(self.measurements) < 10:
            return

        self._understand_count += 1
        ms = MirrorScan.from_measurements(self.measurements)

        # 構造理解を更新
        self.importance = ms.importance.copy()
        prev_dead = set(self.dead_dims)
        self.dead_dims = list(ms.dead_dims)
        self.active_dims = list(ms.active_dims)

        # dead_dims安定性追跡
        self._dead_history.append(set(self.dead_dims))

        # proxy構築
        proxy_fn, proxy_r2, proxy_name = ms.build_proxy()
        if proxy_fn is not None:
            self.proxy_fn = proxy_fn
            self.proxy_r2 = float(proxy_r2)
            self.proxy_type = proxy_name

            # proxyで最適値を探索
            best_proxy_score = float('-inf')
            best_proxy_params = None
            for _ in range(1000):
                s = self._importance_guided_sample()
                ps = proxy_fn(s)
                if ps > best_proxy_score:
                    best_proxy_score = ps
                    best_proxy_params = s

            # proxy最適値を実eval
            if best_proxy_params is not None:
                real_score = float(self.eval_fn(best_proxy_params))
                self.measurements.append({
                    "params": best_proxy_params, "score": real_score
                })
                if real_score > self.best_score:
                    self.best_score = real_score
                    self.best_params = best_proxy_params

        # interaction pairs検出
        self._detect_interactions(ms)

        if verbose and self._understand_count % 3 == 1:
            top_dims = np.argsort(self.importance)[-3:][::-1]
            top_str = ", ".join(
                f"{self.param_names[d]}={self.importance[d]:.3f}"
                for d in top_dims
            )
            print(f"  [{self.phase}] n={len(self.measurements)}, "
                  f"dead={len(self.dead_dims)}, R2={self.proxy_r2:.3f}, "
                  f"best={self.best_score:.2f}, top=[{top_str}]")

    def _detect_interactions(self, ms):
        """MirrorScanのhistoryからinteraction pairsを検出。"""
        if len(ms.history) < 20 or len(self.active_dims) < 2:
            self._interaction_pairs = []
            return

        X = np.array([h['params'] for h in ms.history])
        y = np.array([h['score'] for h in ms.history])

        pairs = []
        for idx_i, dim_i in enumerate(self.active_dims):
            for dim_j in self.active_dims[idx_i + 1:]:
                # pair_truth: |corr(x_i * x_j, y)|
                product = X[:, dim_i] * X[:, dim_j]
                if np.std(product) < 1e-10 or np.std(y) < 1e-10:
                    continue
                corr = abs(np.corrcoef(product, y)[0, 1])
                if np.isnan(corr):
                    continue
                pairs.append((dim_i, dim_j, corr))

        # 上位5ペア
        pairs.sort(key=lambda x: x[2], reverse=True)
        self._interaction_pairs = [(i, j) for i, j, _ in pairs[:5]]

    # ================================================================
    # ポリシー解釈 (RL mode)
    # ================================================================

    def _interpret_policy(self):
        """RL modeのポリシーを人間が読める形に解釈。"""
        if self.env is None or self.best_params is None:
            return {}

        # importance上位の接続
        top_dims = np.argsort(self.importance)[-5:][::-1]
        top_connections = []
        for d in top_dims:
            top_connections.append({
                "param": self.param_names[d],
                "importance": round(float(self.importance[d]), 4),
                "weight": round(float(self.best_params[d]), 4),
            })

        result = {
            "top_connections": top_connections,
            "n_dead_connections": len(self.dead_dims),
            "n_active_connections": len(self.active_dims),
            "effective_policy_dims": len(self.active_dims),
            "policy_type": self.policy_type,
        }

        if self._kathara_policy is not None:
            result["n_params"] = self._kathara_policy.n_params
            result["n_hidden"] = self._kathara_policy.n_hidden
            result["n_edges"] = self._kathara_policy.n_edges

        return result

    # ================================================================
    # フェーズ遷移
    # ================================================================

    def _check_phase_transition(self, t0):
        """フェーズ遷移を判定。"""
        elapsed = time.time() - t0

        if self.phase == "EXPLORE":
            # dead_dimsが2回連続安定 → FOCUS
            if len(self._dead_history) >= 3:
                last3 = self._dead_history[-3:]
                if last3[-1] == last3[-2]:
                    self.phase = "FOCUS"
                    return

            # 時間の20%消費 → FOCUS (強制)
            if elapsed > self._budget * 0.2:
                self.phase = "FOCUS"

        elif self.phase == "FOCUS":
            # proxy R2 >= 0.7 → DEEPEN
            if self.proxy_r2 >= 0.7:
                self.phase = "DEEPEN"
                return

            # 時間の40%消費 → DEEPEN (強制)
            if elapsed > self._budget * 0.4:
                self.phase = "DEEPEN"

        elif self.phase == "DEEPEN":
            # 時間の60%消費 → EXPLOIT
            if elapsed > self._budget * 0.6:
                self.phase = "EXPLOIT"

    def _log_phase(self, t0, n_meas, verbose):
        """フェーズ遷移をログ。"""
        entry = {
            "phase": self.phase,
            "elapsed_s": round(time.time() - t0, 1),
            "n_measurements": n_meas,
            "n_dead": len(self.dead_dims),
            "proxy_r2": round(self.proxy_r2, 3),
            "best_score": round(self.best_score, 4) if self.best_score > float('-inf') else None,
        }
        self._phase_log.append(entry)
        if verbose:
            print(f"\n  >>> Phase: {self.phase} | "
                  f"t={entry['elapsed_s']}s, n={n_meas}, "
                  f"dead={len(self.dead_dims)}, R2={self.proxy_r2:.3f}")

    # ================================================================
    # 最終最適化 (owl() 利用)
    # ================================================================

    def _final_optimize(self, time_budget, verbose):
        """蓄積したmeasurementsでowl()最終最適化。"""
        if len(self.measurements) < 10:
            return

        from twelve.optimize import owl

        if verbose:
            print(f"\n  [Final] owl() with {len(self.measurements)} measurements...")

        result = owl(
            measurements=self.measurements,
            param_ranges=self.param_ranges,
            param_names=self.param_names,
            experience_id=self.experience_id,
            verify_fn=self.eval_fn,
            time_budget=min(60, max(10, time_budget)),
            autonomous=False,
            verbose=verbose,
        )

        bp = result.get('best_params')
        if bp is not None:
            if isinstance(bp, dict):
                bp = [bp.get(n, (lo + hi) / 2)
                      for n, (lo, hi) in zip(self.param_names, self.param_ranges)]
            vs = result.get('verified_score') or result.get('best_score')
            if vs is not None and vs > self.best_score:
                self.best_score = float(vs)
                self.best_params = list(bp)

        # owl結果でproxy_r2更新
        r2 = result.get('proxy_r2', self.proxy_r2)
        if r2 > self.proxy_r2:
            self.proxy_r2 = r2

    # ================================================================
    # 経験 (UnifiedExperience)
    # ================================================================

    def _load_experience(self):
        """前回の経験を読み込む。warm_startを返す。"""
        if not self.experience_id:
            return None
        try:
            from twelve.agent.unified_experience import UnifiedExperience
            ue = UnifiedExperience(self.experience_id)
            ws = ue.get_warm_start()
            dead_hint = ue.get_dead_dims()
            if dead_hint:
                self.dead_dims = dead_hint
                self.active_dims = [i for i in range(self.n_dims)
                                    if i not in dead_hint]
            return ws
        except Exception:
            return None

    def _save_experience(self):
        """経験を保存。"""
        if not self.experience_id:
            return
        try:
            from twelve.agent.unified_experience import UnifiedExperience
            ue = UnifiedExperience(self.experience_id)
            ue.save_ms_result(
                dead_dims=self.dead_dims,
                active_dims=self.active_dims,
                importance=self.importance,
                best_params=self.best_params,
                best_score=self.best_score,
            )
        except Exception:
            pass


# ================================================================
# デモ
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nous demo")
    parser.add_argument("--mode", choices=["rl", "rastrigin", "both",
                                            "compare", "flyworld"],
                        default="both")
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--policy", choices=["linear", "kathara", "compare"],
                        default="linear")
    args = parser.parse_args()

    if args.mode == "compare":
        # Linear vs Kathara 比較テスト
        print("\n" + "=" * 60)
        print("  COMPARE: Linear vs Kathara Policy")
        print("=" * 60)
        results = {}
        per_budget = args.budget // 2

        for pt in ["linear", "kathara"]:
            print(f"\n--- {pt.upper()} policy ---")
            env = BalanceEnv()
            nous = Nous(env=env, policy_type=pt)
            r = nous.run(time_budget=per_budget, verbose=True)
            results[pt] = r
            print(f"  {pt}: score={r['best_score']:.1f}, "
                  f"dims={r.get('policy_interpretation', {}).get('n_params', '?')}, "
                  f"dead={len(r['dead_dims'])}/{len(r['importance'])}")

        print(f"\n{'=' * 60}")
        print(f"  RESULT: linear={results['linear']['best_score']:.1f}, "
              f"kathara={results['kathara']['best_score']:.1f}")
        winner = "kathara" if results['kathara']['best_score'] >= results['linear']['best_score'] else "linear"
        print(f"  Winner: {winner}")
        print(f"{'=' * 60}")

    elif args.mode == "flyworld":
        # FlyWorld テスト
        print("\n" + "=" * 60)
        print(f"  FlyWorld test (policy={args.policy})")
        print("=" * 60)
        try:
            env = FlyWorldEnv(version=3, max_steps=60)
            nous = Nous(env=env, policy_type=args.policy)
            result = nous.run(time_budget=args.budget, verbose=True)
            print(f"\n  Best score: {result['best_score']:.4f}")
            print(f"  Dead: {len(result['dead_dims'])}/{len(result['importance'])}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    else:
        if args.mode in ("rl", "both"):
            print("\n" + "=" * 60)
            print(f"  DEMO 1: RL mode (BalanceEnv, policy={args.policy})")
            print("=" * 60)
            env = BalanceEnv()
            nous = Nous(env=env, experience_id="nous_balance",
                        policy_type=args.policy)
            result = nous.run(time_budget=args.budget, verbose=True)

            print(f"\n  Best reward: {result['best_score']:.1f} / 200")
            print(f"  Dead dims: {result['dead_dims']} "
                  f"({len(result['dead_dims'])} dead)")
            print(f"  Active dims: {len(result['active_dims'])} / "
                  f"{len(result['importance'])}")

            interp = result.get("policy_interpretation", {})
            if interp.get("top_connections"):
                print(f"\n  Policy Structure (top connections):")
                for c in interp["top_connections"][:5]:
                    print(f"    {c['param']:12s} imp={c['importance']:.4f}  "
                          f"w={c['weight']:+.3f}")

        if args.mode in ("rastrigin", "both"):
            print("\n" + "=" * 60)
            print("  DEMO 2: eval_fn mode (Rastrigin 10D)")
            print("=" * 60)

            def rastrigin(params):
                A = 10
                n = len(params)
                return -(A * n + sum(x**2 - A * np.cos(2 * np.pi * x)
                                     for x in params))

            ranges = [(-5.12, 5.12)] * 10
            nous = Nous(rastrigin, ranges, experience_id="test_rastrigin")
            result = nous.run(time_budget=min(30, args.budget), verbose=True)
            print(f"\n  Best score: {result['best_score']:.4f}")
            print(f"  Dead dims: {result['dead_dims']}")
