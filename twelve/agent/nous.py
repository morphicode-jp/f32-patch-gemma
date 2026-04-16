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
# LaD: ロジックのデータ化 — 全内部定数をJSON駆動に
# ================================================================

_NOUS_DEFAULTS = {
    "phase": {
        "explore_ratio": 0.2,
        "focus_ratio": 0.4,
        "exploit_ratio": 0.6,
        "r2_threshold_deepen": 0.7,
        "dead_stability_checks": 3,
    },
    "sampling": {
        "batch_size": 5,
        "n_initial_base": 20,
        "importance_width_min": 0.1,
        "n_episodes": 3,
    },
    "proxy": {
        "screen_candidates": 100,
        "search_iters": 1000,
        "understand_interval": 10,
        "interaction_top_n": 5,
    },
    "curiosity": {
        "candidates": 50,
        "split_ratio": 0.5,
        "recent_window": 30,
    },
    "social": {
        "share_interval": 15,
        "worker_budget_ratio": 0.85,
        "neighbor_share_threshold": 0.8,
        "screen_candidates": 50,
    },
    "policy": {
        "range": 2.0,
        "auto_select_trials": 10,
    },
    "hebbian": {
        "cumulative_episodes": 20,
    },
}

_nous_cache = None


def _load_nous_config():
    """JSON > hardcode fallback。ma_meta_params.jsonと同パターン。"""
    global _nous_cache
    if _nous_cache is not None:
        return _nous_cache
    cfg = {}
    for section, defaults in _NOUS_DEFAULTS.items():
        cfg[section] = dict(defaults)
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "configs", "nous_params.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                loaded = json.load(f)
            for section in cfg:
                if section in loaded:
                    cfg[section].update(loaded[section])
        except Exception:
            pass
    _nous_cache = cfg
    return cfg


def _cfg(section, key):
    """設定値取得。_cfg("phase", "explore_ratio") → 0.2"""
    return _load_nous_config()[section][key]


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

    @property
    def action_type(self):
        """'discrete' or 'continuous'。デフォルト: discrete。"""
        return "discrete"

    @property
    def action_dim(self):
        """連続行動の次元数。discrete時はn_actionsと同じ。"""
        return self.n_actions

    @property
    def action_range(self):
        """連続行動の範囲 (low, high)。各次元共通。"""
        return (-1.0, 1.0)

    def episode_score(self):
        """カスタムエピソードスコア。None = 累積報酬を使用。"""
        return None


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
# SwingUpEnv — 連続行動デモ環境（Pendulum相当）
# ================================================================

class SwingUpEnv(Environment):
    """振り上げ振子。連続トルク制御。

    obs = [cos(θ), sin(θ), θ̇]  (3次元)
    action = [torque]  (連続, -2.0 ~ +2.0)
    報酬: -(θ² + 0.1θ̇² + 0.001a²)  (上向きが最適)
    """

    DT = 0.05
    MAX_STEPS = 200
    MAX_SPEED = 8.0
    MAX_TORQUE = 2.0
    G = 9.81
    MASS = 1.0
    LENGTH = 1.0

    def __init__(self):
        self._theta = 0.0
        self._theta_dot = 0.0
        self._step_count = 0
        self._rng = np.random.RandomState(None)

    @property
    def obs_size(self):
        return 3

    @property
    def n_actions(self):
        return 1

    @property
    def action_type(self):
        return "continuous"

    @property
    def action_dim(self):
        return 1

    @property
    def action_range(self):
        return (-self.MAX_TORQUE, self.MAX_TORQUE)

    def reset(self):
        self._theta = self._rng.uniform(-math.pi, math.pi)
        self._theta_dot = self._rng.uniform(-1.0, 1.0)
        self._step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return [math.cos(self._theta), math.sin(self._theta), self._theta_dot]

    def step(self, action):
        if isinstance(action, np.ndarray):
            torque = float(np.clip(action[0], -self.MAX_TORQUE, self.MAX_TORQUE))
        else:
            torque = float(np.clip(action, -self.MAX_TORQUE, self.MAX_TORQUE))

        # Pendulum dynamics
        theta_acc = (-3.0 * self.G / (2.0 * self.LENGTH) * math.sin(self._theta + math.pi)
                     + 3.0 / (self.MASS * self.LENGTH ** 2) * torque)
        self._theta_dot = np.clip(self._theta_dot + theta_acc * self.DT,
                                  -self.MAX_SPEED, self.MAX_SPEED)
        self._theta += self._theta_dot * self.DT
        self._step_count += 1

        # Normalize angle to [-pi, pi]
        self._theta = ((self._theta + math.pi) % (2 * math.pi)) - math.pi

        # Reward: upright = θ=0 is best
        reward = -(self._theta ** 2 + 0.1 * self._theta_dot ** 2 + 0.001 * torque ** 2)
        done = self._step_count >= self.MAX_STEPS
        return self._get_obs(), reward, done


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
# SocialWorld — 2体協力タスク（proto-language 創発実験）
# ================================================================

class SocialWorld(Environment):
    """2体のエージェントが協力して重い箱をターゲットへ押す。

    箱は1体では動かせない (friction=1.5, agent_force=1.0)。
    信号チャネル (3種) に報酬なし → 言語が自然に創発するか観察。
    同じ脳 (HebbianKatharaPolicy) を共有する双子アーキテクチャ。

    obs = 12D, actions = 12 (4方向 × 3信号)
    """

    WORLD_SIZE = 10.0
    MOVE_STEP = 0.5
    BOX_FRICTION = 1.5
    AGENT_FORCE = 1.0
    MAX_STEPS = 100
    DIRS = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=float)

    def __init__(self):
        self._rng = np.random.RandomState(None)
        self._partner = None
        self._partner_lr = None
        self._signal_a = 0
        self._signal_b = 0
        self._step_count = 0
        self._signal_log = []
        self._agent_a = np.zeros(2)
        self._agent_b = np.zeros(2)
        self._box = np.zeros(2)
        self._target = np.zeros(2)
        self._prev_box_dist = 4.0

    @property
    def obs_size(self):
        return 12

    @property
    def n_actions(self):
        return 12

    def sync_partner(self, brain, lr_vector):
        """Agent B の脳を A からコピー (同じ辺重み、独立ノード状態)。"""
        self._partner = HebbianKatharaPolicy(
            brain.obs_size, brain.n_actions,
            brain.n_nodes, list(brain.offsets))
        self._partner.edge_weights = brain.edge_weights.copy()
        self._partner.momentum = brain.momentum.copy()
        self._partner_lr = np.clip(np.asarray(lr_vector, dtype=np.float64),
                                   0.0, 0.5)

    def reset(self):
        self._step_count = 0
        S = self.WORLD_SIZE
        self._agent_a = self._rng.uniform(1, S - 1, 2)
        self._agent_b = self._rng.uniform(1, S - 1, 2)
        self._box = np.array([S / 2, S / 2])
        angle = self._rng.uniform(0, 2 * np.pi)
        self._target = self._box + np.array(
            [np.cos(angle), np.sin(angle)]) * 4.0
        self._target = np.clip(self._target, 0.5, S - 0.5)
        self._signal_a = 0
        self._signal_b = 0
        self._prev_box_dist = np.linalg.norm(self._box - self._target)
        self._signal_log = []
        if self._partner is not None:
            self._partner.reset_states()
        return self._get_obs('a')

    def _get_obs(self, perspective):
        S = self.WORLD_SIZE
        if perspective == 'a':
            me, them = self._agent_a, self._agent_b
            their_sig = self._signal_b
        else:
            me, them = self._agent_b, self._agent_a
            their_sig = self._signal_a
        return [
            me[0] / S, me[1] / S,
            them[0] / S, them[1] / S,
            self._box[0] / S, self._box[1] / S,
            self._target[0] / S, self._target[1] / S,
            (self._box[0] - me[0]) / S, (self._box[1] - me[1]) / S,
            their_sig / 2.0,
            np.linalg.norm(me - self._box) / S,
        ]

    def step(self, action_a):
        self._step_count += 1
        move_a = action_a // 3
        self._signal_a = action_a % 3

        # Agent B acts
        if self._partner is not None:
            obs_b = self._get_obs('b')
            action_b, firing_b = self._partner.forward_with_hidden(obs_b)
            move_b = action_b // 3
            self._signal_b = action_b % 3
        else:
            move_b = self._rng.randint(4)
            self._signal_b = 0
            firing_b = None

        # Move agents
        S = self.WORLD_SIZE
        self._agent_a = np.clip(
            self._agent_a + self.DIRS[move_a] * self.MOVE_STEP, 0, S)
        self._agent_b = np.clip(
            self._agent_b + self.DIRS[move_b] * self.MOVE_STEP, 0, S)

        # Box physics
        self._push_box(move_a, move_b)

        # Reward (shared)
        box_dist = np.linalg.norm(self._box - self._target)
        my_box = np.linalg.norm(self._agent_a - self._box)
        partner_box = np.linalg.norm(self._agent_b - self._box)
        reached = box_dist < 1.0

        reward = (self._prev_box_dist - box_dist) * 5.0
        reward += 0.1 if my_box < 2.0 else -0.1
        reward += 0.3 if (my_box < 2.0 and partner_box < 2.0) else 0.0
        reward += 20.0 if reached else 0.0
        self._prev_box_dist = box_dist

        # Partner Hebbian update
        if firing_b is not None and self._partner_lr is not None:
            if abs(reward) > 0.01:
                self._partner.hebbian_update(
                    firing_b, reward, self._partner_lr)

        # Signal log
        self._signal_log.append((self._signal_a, self._signal_b))

        done = reached or self._step_count >= self.MAX_STEPS
        return self._get_obs('a'), reward, done

    def _push_box(self, move_a, move_b):
        """2体の力を合算。friction超えたら箱が動く。"""
        force = np.zeros(2)
        for agent_pos, move in [(self._agent_a, move_a),
                                (self._agent_b, move_b)]:
            if np.linalg.norm(agent_pos - self._box) < 2.0:
                to_box = self._box - agent_pos
                push_dir = self.DIRS[move]
                if np.dot(push_dir, to_box) > 0:
                    force += push_dir * self.AGENT_FORCE
        mag = np.linalg.norm(force)
        if mag > self.BOX_FRICTION:
            self._box += (force / mag) * (mag - self.BOX_FRICTION) * 0.3
            self._box = np.clip(self._box, 0, self.WORLD_SIZE)

    def episode_score(self):
        """0-110スケール (FlyWorldと同じ)。"""
        box_dist = np.linalg.norm(self._box - self._target)
        return max(0.0, 100.0 - box_dist * 10.0) + \
            (10.0 if box_dist < 1.0 else 0.0)

    def get_signal_stats(self):
        """Proto-language分析。信号の偏り = 言語創発の兆候。"""
        if not self._signal_log:
            return {}
        sigs_a = [s for s, _ in self._signal_log]
        sigs_b = [s for _, s in self._signal_log]
        n = len(sigs_a)
        freq_a = [sigs_a.count(i) / max(1, n) for i in range(3)]
        freq_b = [sigs_b.count(i) / max(1, n) for i in range(3)]

        def _non_uniform(freq):
            ent = -sum(p * np.log(p + 1e-10) for p in freq if p > 0)
            return 1.0 - ent / np.log(3)

        return {
            "freq_a": [round(f, 3) for f in freq_a],
            "freq_b": [round(f, 3) for f in freq_b],
            "non_uniformity_a": round(_non_uniform(freq_a), 3),
            "non_uniformity_b": round(_non_uniform(freq_b), 3),
        }


# ================================================================
# LifeWorld — 生存サンドボックス（内部ドライブ、タスクなし）
# ================================================================

class LifeWorld(Environment):
    """生存サンドボックス。タスクなし、内部ドライブで学習。

    報酬は外部から与えられない。体の中から来る:
        Δenergy × 3.0 (生存圧力)
        + novelty × 0.5 (好奇心)
        - pain × 2.0 (痛み回避)

    obs = 12D (内部状態 + 外部知覚), actions = 12 (4方向 × 3意図)
    fitness = 生存時間 + 残エネルギー
    """

    WORLD_SIZE = 20.0
    MAX_STEPS = 300
    N_FOOD = 8
    FOOD_ENERGY = 15.0
    FOOD_REGEN = 0.03
    MOVE_COST = 0.2
    IDLE_COST = 0.05
    EAT_RADIUS = 1.5
    N_HAZARDS = 2
    HAZARD_RADIUS = 2.0
    HAZARD_DAMAGE = 3.0
    INITIAL_ENERGY = 50.0
    DIRS = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=float)

    def __init__(self, n_peers=2):
        self.n_peers = n_peers
        self._rng = np.random.RandomState(None)
        self._pos = np.zeros(2)
        self._energy = self.INITIAL_ENERGY
        self._age = 0
        self._prev_energy = self.INITIAL_ENERGY
        self._prev_obs = None
        self._signal = 0
        self._food_pos = np.zeros((self.N_FOOD, 2))
        self._food_alive = np.ones(self.N_FOOD, dtype=bool)
        self._hazard_pos = np.zeros((self.N_HAZARDS, 2))
        self._peers = []
        self._partner = None
        self._partner_lr = None
        self._eat_count = 0
        self._hazard_hits = 0
        self._signal_log = []

    @property
    def obs_size(self):
        return 12

    @property
    def n_actions(self):
        return 12

    def sync_partner(self, brain, lr_vector):
        """全peerの脳をA脳のコピーで初期化。"""
        self._partner = HebbianKatharaPolicy(
            brain.obs_size, brain.n_actions,
            brain.n_nodes, list(brain.offsets))
        self._partner.edge_weights = brain.edge_weights.copy()
        self._partner.momentum = brain.momentum.copy()
        self._partner_lr = np.clip(
            np.asarray(lr_vector, dtype=np.float64), 0.0, 0.5)

    def reset(self):
        S = self.WORLD_SIZE
        self._pos = self._rng.uniform(2, S - 2, 2)
        self._energy = self.INITIAL_ENERGY
        self._age = 0
        self._prev_energy = self.INITIAL_ENERGY
        self._prev_obs = None
        self._signal = 0
        self._eat_count = 0
        self._hazard_hits = 0
        self._signal_log = []

        for i in range(self.N_FOOD):
            self._food_pos[i] = self._rng.uniform(1, S - 1, 2)
            self._food_alive[i] = True
        for i in range(self.N_HAZARDS):
            self._hazard_pos[i] = self._rng.uniform(4, S - 4, 2)

        self._peers = []
        for _ in range(self.n_peers):
            self._peers.append({
                'pos': self._rng.uniform(2, S - 2, 2),
                'energy': self.INITIAL_ENERGY,
                'prev_energy': self.INITIAL_ENERGY,
                'signal': 0,
                'alive': True,
            })
        if self._partner is not None:
            self._partner.reset_states()
        return self._get_obs()

    def _get_obs(self, pos=None, energy=None):
        S = self.WORLD_SIZE
        if pos is None:
            pos = self._pos
        if energy is None:
            energy = self._energy

        energy_norm = energy / 100.0
        hunger = 1.0 - energy_norm

        alive_food = self._food_pos[self._food_alive]
        if len(alive_food) > 0:
            dists = np.linalg.norm(alive_food - pos, axis=1)
            near = np.argmin(dists)
            food_dx = (alive_food[near][0] - pos[0]) / S
            food_dy = (alive_food[near][1] - pos[1]) / S
            food_dist = dists[near] / S
        else:
            food_dx = food_dy = 0.0
            food_dist = 1.0

        alive_peers = [p for p in self._peers if p['alive']]
        if alive_peers:
            pd = [(np.linalg.norm(p['pos'] - pos), p) for p in alive_peers]
            _, nearest_p = min(pd, key=lambda x: x[0])
            peer_dx = (nearest_p['pos'][0] - pos[0]) / S
            peer_dy = (nearest_p['pos'][1] - pos[1]) / S
            peer_sig = nearest_p['signal'] / 2.0
        else:
            peer_dx = peer_dy = peer_sig = 0.0

        hd = np.linalg.norm(self._hazard_pos - pos, axis=1)
        hazard_dist = np.min(hd) / S

        wall_x = min(pos[0], S - pos[0]) / S
        wall_y = min(pos[1], S - pos[1]) / S
        age = self._age / self.MAX_STEPS

        return [energy_norm, hunger, food_dx, food_dy, food_dist,
                peer_dx, peer_dy, peer_sig, hazard_dist,
                wall_x, wall_y, age]

    def step(self, action):
        self._age += 1
        S = self.WORLD_SIZE
        move = action // 3
        intent = action % 3

        self._pos = np.clip(
            self._pos + self.DIRS[move] * 0.5, 0, S)
        self._energy -= self.MOVE_COST

        if intent == 1:
            if self._try_eat(self._pos):
                self._energy = min(100.0, self._energy + self.FOOD_ENERGY)
                self._eat_count += 1
            self._signal = 0
        elif intent == 2:
            self._signal = 1
        else:
            self._signal = 0

        for hz in self._hazard_pos:
            if np.linalg.norm(self._pos - hz) < self.HAZARD_RADIUS:
                self._energy -= self.HAZARD_DAMAGE
                self._hazard_hits += 1

        self._step_peers()

        for i in range(self.N_FOOD):
            if not self._food_alive[i]:
                if self._rng.random() < self.FOOD_REGEN:
                    self._food_alive[i] = True
                    self._food_pos[i] = self._rng.uniform(1, S - 1, 2)

        self._energy = max(0.0, self._energy - self.IDLE_COST)

        delta_e = self._energy - self._prev_energy

        obs = self._get_obs()
        novelty = 0.0
        if self._prev_obs is not None:
            novelty = float(np.mean(
                np.abs(np.array(obs) - np.array(self._prev_obs))))
        pain = 1.0 if self._energy < 20 else 0.0

        reward = delta_e * 3.0 + novelty * 0.5 - pain * 2.0

        self._prev_energy = self._energy
        self._prev_obs = obs
        self._signal_log.append(self._signal)

        dead = self._energy <= 0
        done = dead or self._age >= self.MAX_STEPS
        return obs, reward, done

    def _try_eat(self, pos):
        alive_idx = np.where(self._food_alive)[0]
        for idx in alive_idx:
            if np.linalg.norm(pos - self._food_pos[idx]) < self.EAT_RADIUS:
                self._food_alive[idx] = False
                return True
        return False

    def _step_peers(self):
        if self._partner is None:
            return
        for peer in self._peers:
            if not peer['alive']:
                continue
            obs_p = self._get_obs(pos=peer['pos'], energy=peer['energy'])
            action_p, firing_p = self._partner.forward_with_hidden(obs_p)
            move_p = action_p // 3
            intent_p = action_p % 3

            S = self.WORLD_SIZE
            peer['pos'] = np.clip(
                peer['pos'] + self.DIRS[move_p] * 0.5, 0, S)
            peer['energy'] -= self.MOVE_COST

            if intent_p == 1:
                if self._try_eat(peer['pos']):
                    peer['energy'] = min(
                        100.0, peer['energy'] + self.FOOD_ENERGY)
                peer['signal'] = 0
            elif intent_p == 2:
                peer['signal'] = 1
            else:
                peer['signal'] = 0

            for hz in self._hazard_pos:
                if np.linalg.norm(peer['pos'] - hz) < self.HAZARD_RADIUS:
                    peer['energy'] -= self.HAZARD_DAMAGE

            peer['energy'] = max(0.0, peer['energy'] - self.IDLE_COST)

            if peer['energy'] <= 0:
                peer['alive'] = False

            delta_p = peer['energy'] - peer.get('prev_energy',
                                                 self.INITIAL_ENERGY)
            pain_p = 1.0 if peer['energy'] < 20 else 0.0
            r_p = delta_p * 3.0 - pain_p * 2.0
            if abs(r_p) > 0.01 and firing_p is not None:
                self._partner.hebbian_update(
                    firing_p, r_p, self._partner_lr)
            peer['prev_energy'] = peer['energy']

    def episode_score(self):
        """生存適応度 = 生存時間 + 残エネルギー。"""
        return float(self._age) + max(0.0, self._energy) * 0.5

    def get_life_stats(self):
        """生命の物語。"""
        n_sig = len(self._signal_log)
        return {
            "age": self._age,
            "energy": round(self._energy, 1),
            "alive": self._energy > 0,
            "meals": self._eat_count,
            "hazard_hits": self._hazard_hits,
            "peers_alive": sum(1 for p in self._peers if p['alive']),
            "signal_rate": (sum(self._signal_log) / max(1, n_sig)),
        }


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
# HebbianKatharaPolicy — brain sim v6 Mode B 忠実移植
# ================================================================

class HebbianKatharaPolicy:
    """Hebbian学習付きKathara(N)ポリシー。

    brain sim v6 Mode Bの学習則をNous内で使用:
    - Nノード、circulant辺構造 (N=12時: 30辺, offsets=[1,4,6])
    - 辺重みのみHebbian更新 (ノードパラメータは固定)
    - Owlが最適化するのは per-edge learning rates

    二重時間スケール:
        高速 (per-step):  Δw_ij = lr_ij × firing_i × firing_j × reward
        低速 (per-N-ep):  Owl → importance → dead edges → lr=0 (構造的プルーニング)

    GrowableKathara: n_nodes引数で任意サイズ。grow()で動的に成長可能。
    """

    # ノード固定パラメータ (v6デフォルト)
    GAIN = 1.5
    BIAS = 0.0
    INPUT_W = 1.0
    LEAK = 0.3
    THRESHOLD = 0.2

    @staticmethod
    def _auto_offsets(n):
        """circulant(N)の自動オフセット: {1, N//4, N//2}"""
        return sorted(set([1, max(1, n // 4), n // 2]))

    def __init__(self, obs_size, n_actions, n_nodes=12, offsets=None):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.n_nodes = n_nodes
        self.offsets = offsets or self._auto_offsets(n_nodes)
        self.motor_nodes = [n_nodes - 2, n_nodes - 1]

        self._build_structure()

    def _build_structure(self):
        """辺構造・隣接リスト・センサーマップ・状態を(再)構築。"""
        n = self.n_nodes

        # Kathara辺構造
        self.edge_list = []
        for i in range(n):
            for offset in self.offsets:
                j = (i + offset) % n
                pair = (min(i, j), max(i, j))
                if pair not in self.edge_list:
                    self.edge_list.append(pair)
        self.n_edges = len(self.edge_list)

        # 隣接リスト
        self.neighbors = [[] for _ in range(n)]
        for eidx, (a, b) in enumerate(self.edge_list):
            self.neighbors[a].append((b, eidx))
            self.neighbors[b].append((a, eidx))

        # センサーノード割り当て
        if self.obs_size >= n:
            self.sensor_map = list(range(n))
        elif n >= 12:
            # v6/v8慣例: 7=left_vis, 9=right_vis, 10=olfactory, 3=obstacle
            priority = [7, 9, 10, 3, 8, 5, 11, 0, 1, 2, 4, 6]
            self.sensor_map = priority[:self.obs_size]
        else:
            # 小さいN: 先頭からモーター以外を割り当て
            non_motor = [i for i in range(n) if i not in self.motor_nodes]
            self.sensor_map = non_motor[:self.obs_size]

        # 永続状態
        self.edge_weights = np.zeros(self.n_edges, dtype=np.float64)
        self.momentum = np.zeros(self.n_edges, dtype=np.float64)
        self.states = np.zeros(n, dtype=np.float64)

    def grow(self, n_new_nodes=2):
        """ノード追加。既存辺重みを保存し、新辺をゼロで初期化。"""
        old_edges = self.edge_list[:]
        old_weights = self.edge_weights.copy()
        old_momentum = self.momentum.copy()

        self.n_nodes += n_new_nodes
        self.offsets = self._auto_offsets(self.n_nodes)
        self.motor_nodes = [self.n_nodes - 2, self.n_nodes - 1]

        self._build_structure()

        # 既存辺の重みを転送
        old_edge_map = {e: i for i, e in enumerate(old_edges)}
        for new_idx, edge in enumerate(self.edge_list):
            if edge in old_edge_map:
                self.edge_weights[new_idx] = old_weights[old_edge_map[edge]]
                self.momentum[new_idx] = old_momentum[old_edge_map[edge]]

        return self.n_nodes, self.n_edges

    def reset_states(self):
        """エピソード開始時: ノード状態リセット。辺重みは保持。"""
        self.states = np.zeros(self.n_nodes, dtype=np.float64)

    def forward_with_hidden(self, obs):
        """v6 simulate_step()と同一の物理。(action, firing) を返す。"""
        n = self.n_nodes

        # センサー入力
        inputs = np.zeros(n)
        for i, node in enumerate(self.sensor_map):
            if i < len(obs) and node < n:
                inputs[node] = float(obs[i])

        # ニューロンダイナミクス (v6 simulate_step 忠実移植)
        new_states = np.zeros(n, dtype=np.float64)
        for i in range(n):
            syn_input = 0.0
            for j, eidx in self.neighbors[i]:
                syn_input += self.states[j] * self.edge_weights[eidx]
            total = syn_input + self.INPUT_W * inputs[i] + self.BIAS
            x = self.GAIN * total
            activation = 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
            if activation < self.THRESHOLD:
                activation *= 0.1
            new_states[i] = activation

        # リーキー積分
        self.states = self.states * (1 - self.LEAK) + new_states * self.LEAK
        self.states = np.clip(self.states, 0, 1)
        firing = self.states.copy()

        # モーター出力 → 行動
        nav = firing[self.motor_nodes[0]]
        cen = firing[self.motor_nodes[1]]
        action = self._motors_to_action(nav, cen)

        return action, firing

    def _motors_to_action(self, nav, cen):
        """モーター出力を離散行動に変換。"""
        if self.n_actions == 9:  # FlyWorld: 3×3
            turn_idx = 0 if nav < 0.33 else (2 if nav > 0.67 else 1)
            speed_idx = 0 if cen < 0.33 else (2 if cen > 0.67 else 1)
            return turn_idx * 3 + speed_idx
        elif self.n_actions == 2:  # Balance: left/right
            return 1 if nav > 0.5 else 0
        else:
            return int(nav * self.n_actions) % self.n_actions

    def hebbian_update(self, firing, reward, lr_vector):
        """v6 Mode B Hebbian学習則。

        Δw_ij = lr_ij × firing_i × firing_j × reward + momentum(0.8)
        """
        for eidx, (a, b) in enumerate(self.edge_list):
            hebbian = firing[a] * firing[b] * reward * lr_vector[eidx]
            self.momentum[eidx] = 0.8 * self.momentum[eidx] + hebbian
            self.edge_weights[eidx] += self.momentum[eidx]
            self.edge_weights[eidx] = np.clip(
                self.edge_weights[eidx], -3.0, 3.0)

    def get_edge_weights(self):
        return self.edge_weights.copy()

    def set_edge_weights(self, weights):
        self.edge_weights = np.array(weights, dtype=np.float64)

    def reset_learning(self):
        """辺重みとmomentumを完全リセット。"""
        self.edge_weights = np.zeros(self.n_edges, dtype=np.float64)
        self.momentum = np.zeros(self.n_edges, dtype=np.float64)


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
                 experience_id=None, env=None, envs=None, policy_type="linear",
                 learning_mode="global", memory_len=0, curiosity=False,
                 n_workers=1):
        # マルチ環境
        if envs is not None:
            self.envs = envs
            env = envs[0]  # primary (obs_size, n_actions基準)
        else:
            self.envs = [env] if env is not None else []

        self.env = env
        self.policy_type = policy_type
        self.learning_mode = learning_mode
        self.memory_len = memory_len
        self.curiosity = curiosity
        self.n_workers = n_workers
        self._kathara_policy = None
        self._hebbian_policy = None

        # RL mode: 環境からeval_fn/param_ranges/param_namesを自動構築
        if env is not None and eval_fn is None:
            if learning_mode == "growth":
                # Growth mode: 小さいKathara(4)から始めて成長
                self._growth_start_nodes = 4
                self._growth_step = 2
                self._growth_max_nodes = 16
                self._growth_history = []
                self._hebbian_policy = HebbianKatharaPolicy(
                    env.obs_size, env.n_actions, n_nodes=4)
                eval_fn = lambda params: self._run_hebbian_episode(params)
                param_ranges = [(0.0, 0.5)] * self._hebbian_policy.n_edges
                param_names = [f"lr_e{a}_{b}"
                               for a, b in self._hebbian_policy.edge_list]
            elif learning_mode == "hebbian":
                # Hebbian mode: Owl最適化対象 = per-edge lr (30D)
                self._hebbian_policy = HebbianKatharaPolicy(
                    env.obs_size, env.n_actions)
                eval_fn = lambda params: self._run_hebbian_episode(params)
                param_ranges = [(0.0, 0.5)] * self._hebbian_policy.n_edges
                param_names = [f"lr_e{a}_{b}"
                               for a, b in self._hebbian_policy.edge_list]
            else:
                if policy_type == "kathara":
                    self._kathara_policy = KatharaPolicy(
                        env.obs_size, env.n_actions)
                n_policy = self._policy_dims()
                # マルチ環境: 全環境の平均スコア
                if len(self.envs) > 1:
                    eval_fn = lambda params: self._run_multi_env(params)
                else:
                    eval_fn = lambda params: self._run_episode(params)
                pr = _cfg("policy", "range")
                param_ranges = [(-pr, pr)] * n_policy
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
        is_cont = getattr(self.env, 'action_type', 'discrete') == 'continuous'
        na = self.env.action_dim if is_cont else self.env.n_actions
        obs = self._effective_obs_size()
        return na * obs + na

    def _effective_obs_size(self):
        """memory_len考慮後の実効obs次元。"""
        base = self.env.obs_size
        return base * (1 + self.memory_len)

    def _make_policy_names(self):
        """ポリシーパラメータの名前を生成。"""
        if self._kathara_policy is not None:
            return self._kathara_policy.param_names()
        is_cont = getattr(self.env, 'action_type', 'discrete') == 'continuous'
        na = self.env.action_dim if is_cont else self.env.n_actions
        obs = self._effective_obs_size()
        names = []
        for a in range(na):
            for o in range(obs):
                names.append(f"W[{a}][{o}]")
        for a in range(na):
            names.append(f"b[{a}]")
        return names

    def _augment_obs(self, obs, obs_history):
        """memory_len > 0: 過去Nフレームのobsを現在obsに結合。"""
        if self.memory_len == 0 or not obs_history:
            return obs
        flat = list(obs)
        # 過去N stepを新しい順に結合 (最新が先)
        for i in range(min(self.memory_len, len(obs_history))):
            flat.extend(obs_history[-(i + 1)])
        # memory不足時はゼロ埋め
        expected = self.env.obs_size * (1 + self.memory_len)
        while len(flat) < expected:
            flat.append(0.0)
        return flat

    def _params_to_action(self, params, obs):
        """ポリシーパラメータ + 観測 → 行動選択。

        discrete: argmax → int
        continuous: tanh → scaled float array
        """
        if self._kathara_policy is not None:
            return self._kathara_policy.forward(obs, params)

        is_continuous = getattr(self.env, 'action_type', 'discrete') == 'continuous'
        na = self.env.action_dim if is_continuous else self.env.n_actions
        ns = len(obs)  # memory拡張時にobs_sizeと異なる場合あり
        W = np.array(params[:na * ns]).reshape(na, ns)
        b = np.array(params[na * ns:na * ns + na])
        logits = W @ np.array(obs) + b

        if is_continuous:
            lo, hi = self.env.action_range
            return np.tanh(logits) * ((hi - lo) / 2) + ((hi + lo) / 2)
        return int(np.argmax(logits))

    def _run_episode(self, params, n_episodes=None, env_override=None):
        if n_episodes is None:
            n_episodes = int(_cfg("sampling", "n_episodes"))
        """ポリシーパラメータでn_episodes回実行 → 平均スコア。

        FlyWorld: v6スタイルスコア (0-110)。それ以外: 累積報酬。
        env_override: マルチ環境用、指定環境で評価。
        """
        env = env_override or self.env
        is_flyworld = isinstance(env, FlyWorldEnv)
        total = 0.0
        for _ in range(n_episodes):
            obs = env.reset()
            obs_history = []  # エピソード内記憶
            ep_reward = 0.0
            done = False

            if is_flyworld and env._env is not None:
                final_dist = env._env.get_food_dist()
            else:
                final_dist = 10.0
            reached = False

            while not done:
                # memory: 過去N stepのobsを結合
                aug_obs = self._augment_obs(obs, obs_history)
                action = self._params_to_action(params, aug_obs)
                obs_history.append(obs)
                if len(obs_history) > self.memory_len:
                    obs_history.pop(0)

                obs, reward, done = env.step(action)
                ep_reward += reward
                if is_flyworld and env._env is not None:
                    final_dist = env._env.get_food_dist()
                    if final_dist < 0.5:
                        reached = True

            if is_flyworld:
                ep_score = max(0.0, 100.0 - final_dist * 10.0)
                if reached:
                    ep_score += 10.0
            else:
                ep_score = ep_reward

            total += ep_score
        return total / n_episodes

    def _run_multi_env(self, params, n_episodes=2):
        """全環境で評価 → 平均スコア。汎化圧力で過学習を防ぐ。"""
        total = 0.0
        for env in self.envs:
            total += self._run_episode(params, n_episodes=n_episodes,
                                       env_override=env)
        return total / len(self.envs)

    # ================================================================
    # Hebbian学習 (learning_mode="hebbian")
    # ================================================================

    def _run_hebbian_episode(self, lr_vector, n_episodes=8):
        """lr_vectorで n_episodes Hebbian学習 → 平均スコア。

        各evaluation: 辺重みゼロから開始 (Owlのiid仮定維持)。
        v6 Mode B の報酬関数を忠実に移植:
            hebb_reward = (prev_dist - curr_dist) * 2.0 + 10*reached + movement
        """
        hp = self._hebbian_policy
        lr_vec = np.clip(np.asarray(lr_vector, dtype=np.float64), 0.0, 0.5)

        # 辺重み/momentum保存 → ゼロ初期化
        saved_w = hp.get_edge_weights()
        saved_m = hp.momentum.copy()
        hp.reset_learning()

        # SocialWorld: パートナー脳を同期
        if hasattr(self.env, 'sync_partner'):
            self.env.sync_partner(hp, lr_vec)

        is_flyworld = isinstance(self.env, FlyWorldEnv)
        total_score = 0.0
        n_reached = 0

        for ep in range(n_episodes):
            obs = self.env.reset()
            hp.reset_states()
            done = False
            ep_acc_reward = 0.0

            # v6スタイル: prev_dist追跡
            prev_dist = None
            prev_pos = None
            if is_flyworld and self.env._env is not None:
                prev_dist = self.env._env.get_food_dist()
                prev_pos = self.env._env.fly_pos.copy()

            final_dist = prev_dist if prev_dist else 10.0
            reached = False

            while not done:
                action, firing = hp.forward_with_hidden(obs)
                obs_new, env_reward, done = self.env.step(action)

                # v6スタイル Hebbian報酬 (接近 + 到達 + 移動)
                if is_flyworld and self.env._env is not None:
                    curr_dist = self.env._env.get_food_dist()
                    curr_pos = self.env._env.fly_pos.copy()
                    moved = np.linalg.norm(curr_pos - prev_pos)
                    step_reached = curr_dist < 0.5

                    hebb_reward = (prev_dist - curr_dist) * 2.0
                    hebb_reward += 10.0 * float(step_reached)
                    hebb_reward += 0.1 if moved > 0.05 else -0.2

                    prev_dist = curr_dist
                    prev_pos = curr_pos
                    final_dist = curr_dist
                    if step_reached:
                        reached = True
                else:
                    hebb_reward = env_reward
                    ep_acc_reward += env_reward

                # per-step Hebbian更新
                if abs(hebb_reward) > 0.01:
                    hp.hebbian_update(firing, hebb_reward, lr_vec)

                obs = obs_new

            # スコア計算
            if is_flyworld:
                ep_score = max(0.0, 100.0 - final_dist * 10.0)
                if reached:
                    ep_score += 10.0
                    n_reached += 1
            else:
                custom = self.env.episode_score()
                ep_score = custom if custom is not None else ep_acc_reward

            total_score += ep_score

        # 辺重み/momentum復元
        hp.set_edge_weights(saved_w)
        hp.momentum = saved_m

        return total_score / n_episodes

    def _run_cumulative_hebbian(self, lr_vector, n_episodes=20,
                                verbose=False):
        """EXPLOITフェーズ: best lr_vectorで辺重みを累積学習。

        リセットなし。dead edges → lr=0 (構造的プルーニング)。
        v6スタイル報酬でHebbian更新。
        """
        hp = self._hebbian_policy
        lr_vec = np.clip(np.asarray(lr_vector, dtype=np.float64), 0.0, 0.5)

        # dead edges: lr=0
        for d in self.dead_dims:
            if d < len(lr_vec):
                lr_vec[d] = 0.0

        is_flyworld = isinstance(self.env, FlyWorldEnv)

        # SocialWorld: パートナー脳を同期
        if hasattr(self.env, 'sync_partner'):
            self.env.sync_partner(hp, lr_vec)

        for ep in range(n_episodes):
            obs = self.env.reset()
            hp.reset_states()
            done = False

            prev_dist = None
            prev_pos = None
            if is_flyworld and self.env._env is not None:
                prev_dist = self.env._env.get_food_dist()
                prev_pos = self.env._env.fly_pos.copy()

            final_dist = prev_dist if prev_dist else 10.0
            reached = False

            while not done:
                action, firing = hp.forward_with_hidden(obs)
                obs_new, env_reward, done = self.env.step(action)

                # v6スタイル Hebbian報酬
                if is_flyworld and self.env._env is not None:
                    curr_dist = self.env._env.get_food_dist()
                    curr_pos = self.env._env.fly_pos.copy()
                    moved = np.linalg.norm(curr_pos - prev_pos)
                    step_reached = curr_dist < 0.5

                    hebb_reward = (prev_dist - curr_dist) * 2.0
                    hebb_reward += 10.0 * float(step_reached)
                    hebb_reward += 0.1 if moved > 0.05 else -0.2

                    prev_dist = curr_dist
                    prev_pos = curr_pos
                    final_dist = curr_dist
                    if step_reached:
                        reached = True
                else:
                    hebb_reward = env_reward

                if abs(hebb_reward) > 0.01:
                    hp.hebbian_update(firing, hebb_reward, lr_vec)
                obs = obs_new

            # スコア計算
            if is_flyworld:
                ep_score = max(0.0, 100.0 - final_dist * 10.0)
                if reached:
                    ep_score += 10.0
            else:
                custom = self.env.episode_score()
                ep_score = custom if custom is not None else env_reward

            if ep_score > self.best_score:
                self.best_score = ep_score
                if verbose:
                    print(f"    [cumulative] ep={ep}, "
                          f"score={ep_score:.2f} (new best)")

    # ================================================================
    # Growth mode — 成長するKathara
    # ================================================================

    def _run_growth_mode(self, time_budget, verbose):
        """成長ループ: 小さいKatharaから始め、Owlの分析で成長・収束。

        各epoch = 独立した mini Nous hebbian。
        dead_ratio < 0.1 = 容量飽和 → grow。
        dead_ratio > 0.3 = 容量十分 → 収束。
        """
        t0 = time.time()
        hp = self._hebbian_policy
        growth_log = []
        epoch = 0
        last_best_params = None
        last_dead_dims = []

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Growth Mode: N={hp.n_nodes} → max {self._growth_max_nodes}")
            print(f"{'=' * 60}")

        while True:
            epoch += 1
            remaining = time_budget - (time.time() - t0)
            epoch_budget = min(remaining * 0.4, 30)
            if epoch_budget < 5 or hp.n_nodes > self._growth_max_nodes:
                break

            if verbose:
                print(f"\n  [Growth Epoch {epoch}] "
                      f"N={hp.n_nodes}, E={hp.n_edges}")

            # Mini Nous hebbian を実行
            mini = Nous(env=self.env, learning_mode="hebbian")
            mini._hebbian_policy = hp
            # param_ranges/names を現在のhpに合わせる
            mini.param_ranges = [(0.0, 0.5)] * hp.n_edges
            mini.param_names = [f"lr_e{a}_{b}" for a, b in hp.edge_list]
            mini.n_dims = hp.n_edges
            mini.importance = np.ones(hp.n_edges)
            mini.dead_dims = []
            mini.active_dims = list(range(hp.n_edges))
            mini_result = mini.run(time_budget=epoch_budget, verbose=verbose)

            dead_ratio = len(mini_result["dead_dims"]) / max(1, hp.n_edges)
            score = mini_result["best_score"]
            last_best_params = mini_result.get("best_params")
            last_dead_dims = mini_result.get("dead_dims", [])

            growth_log.append({
                "epoch": epoch,
                "n_nodes": hp.n_nodes,
                "n_edges": hp.n_edges,
                "dead_ratio": round(dead_ratio, 3),
                "score": round(score, 2),
            })

            if verbose:
                print(f"    Score={score:.2f}, "
                      f"dead={len(mini_result['dead_dims'])}/{hp.n_edges} "
                      f"({dead_ratio:.2f})")

            # 成長判断
            if epoch <= 2 and hp.n_nodes < self._growth_max_nodes:
                # 最初の2epochは必ず成長（最小サイズでは容量不足の可能性）
                old_n = hp.n_nodes
                hp.grow(self._growth_step)
                if verbose:
                    print(f"    → GROW (initial): N={old_n}→{hp.n_nodes}, "
                          f"E={hp.n_edges}")
            elif dead_ratio < 0.1 and hp.n_nodes < self._growth_max_nodes:
                old_n = hp.n_nodes
                hp.grow(self._growth_step)
                if verbose:
                    print(f"    → GROW: N={old_n}→{hp.n_nodes}, "
                          f"E={hp.n_edges}")
            elif dead_ratio > 0.3:
                if verbose:
                    print(f"    → CONVERGED")
                break
            else:
                if verbose:
                    print(f"    → STABLE (dead_ratio={dead_ratio:.2f})")

        # 最終cumulative hebbian
        if last_best_params is not None:
            remaining = time_budget - (time.time() - t0)
            if remaining > 5:
                if verbose:
                    print(f"\n  [Final] Cumulative hebbian "
                          f"(N={hp.n_nodes}, E={hp.n_edges})")
                # dead_dims のlrを0にして cumulative
                self._hebbian_policy = hp
                self.dead_dims = last_dead_dims
                self._run_cumulative_hebbian(
                    last_best_params, n_episodes=20, verbose=verbose)

        elapsed = time.time() - t0
        best_score = max((g["score"] for g in growth_log), default=0)
        self.best_score = best_score
        self._growth_history = growth_log

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Growth Complete: {elapsed:.1f}s")
            print(f"  Final: N={hp.n_nodes}, E={hp.n_edges}, "
                  f"score={best_score:.2f}")
            print(f"  History:")
            for g in growth_log:
                print(f"    epoch={g['epoch']}: N={g['n_nodes']}, "
                      f"E={g['n_edges']}, dead={g['dead_ratio']:.2f}, "
                      f"score={g['score']:.1f}")
            print(f"{'=' * 60}")

        result = {
            "best_params": last_best_params,
            "best_score": best_score,
            "importance": self.importance.tolist()
                if hasattr(self, 'importance') else [],
            "dead_dims": last_dead_dims,
            "active_dims": [i for i in range(hp.n_edges)
                            if i not in last_dead_dims],
            "proxy_r2": 0.0,
            "proxy_type": None,
            "phase": "GROWTH_COMPLETE",
            "phase_log": [],
            "n_measurements": sum(1 for _ in growth_log),
            "elapsed_s": elapsed,
            "growth_history": growth_log,
            "final_nodes": hp.n_nodes,
            "final_edges": hp.n_edges,
        }
        if self.env is not None:
            result["policy_interpretation"] = self._interpret_policy()
        return result

    # ================================================================
    # メインループ
    # ================================================================

    def run(self, time_budget=300, verbose=True):
        """メインループ。フェーズ自動遷移。"""
        t0 = time.time()
        self._budget = time_budget

        # Growth mode: 成長ループで実行
        if self.learning_mode == "growth":
            return self._run_growth_mode(time_budget, verbose)

        # 自動policy_type選択
        if (self.policy_type == "auto" and self.env is not None
                and self.learning_mode == "global"):
            winner = self._auto_select_policy(time_budget, verbose)
            if verbose:
                print(f"  [auto] winner: {winner}")

        # K²社会学習
        if self.n_workers > 1 and self.env is not None:
            return self._run_social(time_budget, verbose)

        mode = "RL" if self.env is not None else "eval_fn"
        if self.learning_mode == "hebbian":
            mode = "hebbian"
        if verbose:
            print("=" * 60)
            print(f"  Nous | {self.n_dims}D, {time_budget}s, mode={mode}")
            print("=" * 60)

        # 経験読み込み
        warm_params = self._load_experience()

        # EXPLORE: 初期ランダム探索
        self.phase = "EXPLORE"
        self._log_phase(t0, len(self.measurements), verbose)

        n_initial = max(int(_cfg("sampling", "n_initial_base")), self.n_dims)

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
        batch_size = int(_cfg("sampling", "batch_size"))
        while time.time() - t0 < time_budget:
            prev_phase = self.phase
            self._check_phase_transition(t0)
            if self.phase != prev_phase:
                self._log_phase(t0, len(self.measurements), verbose)

            if self.phase == "EXPLORE":
                samples = [self._random_sample() for _ in range(batch_size)]
            elif self.phase == "FOCUS":
                if self.curiosity and self.proxy_fn is not None:
                    # 好奇心駆動: split_ratio分を好奇心、残りをimportance
                    sr = _cfg("curiosity", "split_ratio")
                    n_curious = max(1, int(batch_size * sr))
                    samples = [self._curiosity_sample() for _ in range(n_curious)]
                    samples += [self._importance_guided_sample()
                                for _ in range(batch_size - n_curious)]
                else:
                    samples = [self._importance_guided_sample()
                               for _ in range(batch_size)]
            elif self.phase == "DEEPEN":
                samples = [self._interaction_sample() for _ in range(batch_size)]
            elif self.phase == "EXPLOIT":
                if self.proxy_fn is not None:
                    batch = self._proxy_screened_sample(
                        n_candidates=int(_cfg("proxy", "screen_candidates")),
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
            if len(self.measurements) % int(_cfg("proxy", "understand_interval")) == 0:
                self._understand(verbose)

        # Hebbian mode: best lr_vectorで累積学習 (辺重みを育てる)
        if (self.learning_mode == "hebbian" and self._hebbian_policy
                and self.best_params is not None):
            cum_ep = int(_cfg("hebbian", "cumulative_episodes"))
            if verbose:
                print(f"  [cumulative] best lr で {cum_ep} ep 累積学習...")
            self._run_cumulative_hebbian(
                self.best_params, n_episodes=cum_ep, verbose=verbose)

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
                # imp高 → 幅広 (全レンジ)、imp低 → 幅狭
                w_min = _cfg("sampling", "importance_width_min")
                width = (hi - lo) * (w_min + (1.0 - w_min) * imp_ratio)
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

    def _curiosity_sample(self, n_candidates=None):
        if n_candidates is None:
            n_candidates = int(_cfg("curiosity", "candidates"))
        """好奇心駆動サンプリング。proxy予測誤差が大きい領域を優先。

        proxy_fn(x) の予測と実際のスコアの乖離が大きい点 =
        proxyがまだ理解していない構造 → そこを探索すればproxyが賢くなる。
        """
        if self.proxy_fn is None or len(self.measurements) < 15:
            return self._importance_guided_sample()

        # 最近のN点でproxy誤差を計算
        rw = int(_cfg("curiosity", "recent_window"))
        recent = self.measurements[-min(rw, len(self.measurements)):]
        errors = []
        for m in recent:
            pred = self.proxy_fn(m["params"])
            errors.append(abs(pred - m["score"]))
        mean_err = np.mean(errors) if errors else 1.0

        # n_candidates生成、proxyとの乖離が大きそうな候補を選ぶ
        candidates = [self._importance_guided_sample()
                      for _ in range(n_candidates)]
        # 既知データとの距離が遠い点 = 未探索領域
        known = np.array([m["params"] for m in self.measurements])
        best_cand = None
        best_novelty = -1.0
        for c in candidates:
            c_arr = np.array(c)
            # 最近傍との距離 (正規化)
            dists = np.linalg.norm(known - c_arr, axis=1)
            min_dist = np.min(dists)
            # proxy不確実性 = 距離 × 平均誤差
            novelty = min_dist * mean_err
            if novelty > best_novelty:
                best_novelty = novelty
                best_cand = c
        return best_cand if best_cand is not None else candidates[0]

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
            for _ in range(int(_cfg("proxy", "search_iters"))):
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
        top_n = int(_cfg("proxy", "interaction_top_n"))
        self._interaction_pairs = [(i, j) for i, j, _ in pairs[:top_n]]

    # ================================================================
    # ポリシー解釈 (RL mode)
    # ================================================================

    def _interpret_policy(self):
        """RL modeのポリシーを人間が読める形に解釈。"""
        if self.env is None or self.best_params is None:
            return {}

        # Growth mode: 成長履歴 + 最終構造
        if self.learning_mode == "growth" and self._hebbian_policy:
            hp = self._hebbian_policy
            result = {
                "policy_type": "growth",
                "final_nodes": hp.n_nodes,
                "final_edges": hp.n_edges,
                "growth_history": getattr(self, '_growth_history', []),
            }
            if self.best_params is not None and len(self.importance) == hp.n_edges:
                top_dims = np.argsort(self.importance)[-5:][::-1]
                result["top_lr_edges"] = [
                    {"edge": f"e{hp.edge_list[d][0]}_{hp.edge_list[d][1]}",
                     "importance": round(float(self.importance[d]), 4),
                     "lr": round(float(self.best_params[d]), 4)}
                    for d in top_dims if d < hp.n_edges
                ]
            return result

        # Hebbian mode: 辺重み + lr構造の解釈
        if self.learning_mode == "hebbian" and self._hebbian_policy:
            hp = self._hebbian_policy
            top_dims = np.argsort(self.importance)[-5:][::-1]
            return {
                "policy_type": "hebbian",
                "n_edges": hp.n_edges,
                "n_dead_edges": len(self.dead_dims),
                "n_active_edges": len(self.active_dims),
                "edge_weights": hp.get_edge_weights().tolist(),
                "top_lr_edges": [
                    {"edge": f"e{hp.edge_list[d][0]}_{hp.edge_list[d][1]}",
                     "importance": round(float(self.importance[d]), 4),
                     "lr": round(float(self.best_params[d]), 4)}
                    for d in top_dims
                ],
            }

        # Global mode (既存)
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
            # dead_dimsが連続安定 → FOCUS
            dsc = int(_cfg("phase", "dead_stability_checks"))
            if len(self._dead_history) >= dsc:
                last = self._dead_history[-dsc:]
                if last[-1] == last[-2]:
                    self.phase = "FOCUS"
                    return

            # 時間消費 → FOCUS (強制)
            if elapsed > self._budget * _cfg("phase", "explore_ratio"):
                self.phase = "FOCUS"

        elif self.phase == "FOCUS":
            # proxy R2閾値超え → DEEPEN
            if self.proxy_r2 >= _cfg("phase", "r2_threshold_deepen"):
                self.phase = "DEEPEN"
                return

            # 時間消費 → DEEPEN (強制)
            if elapsed > self._budget * _cfg("phase", "focus_ratio"):
                self.phase = "DEEPEN"

        elif self.phase == "DEEPEN":
            # 時間消費 → EXPLOIT
            if elapsed > self._budget * _cfg("phase", "exploit_ratio"):
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
    # K²社会学習 (複数Nous並列探索 + 知見共有)
    # ================================================================

    def _run_social(self, time_budget, verbose):
        """Circulant(N,{1,⌊N/2⌋}) トポロジーで並列探索。

        各ワーカーが独立にNousループを回し、定期的にbest measurementsを隣接ノードと共有。
        owl(kathara="auto")の上位版。
        """
        import threading

        t0 = time.time()
        n = self.n_workers
        if verbose:
            print(f"  [K²] {n} workers, circulant offsets=[1,{n//2}]")

        # Circulant隣接リスト
        offsets = [1, max(2, n // 2)]
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for o in offsets:
                j = (i + o) % n
                if j not in neighbors[i]:
                    neighbors[i].append(j)
                if i not in neighbors[j]:
                    neighbors[j].append(i)

        # 共有プール: 各ワーカーのbest measurements
        shared_lock = threading.Lock()
        shared_best = [{"params": None, "score": float('-inf')}] * n
        all_measurements = [[] for _ in range(n)]

        def worker_fn(worker_id):
            """各ワーカー: 独立Nousループ (EXPLORE→FOCUS→EXPLOIT)。"""
            env_copy = type(self.env)()  # 環境を新規作成
            worker = Nous(
                env=env_copy, policy_type=self.policy_type,
                learning_mode=self.learning_mode,
                memory_len=self.memory_len, curiosity=self.curiosity,
                n_workers=1,  # 再帰防止
            )
            worker.rng = np.random.RandomState(42 + worker_id)

            t0 = time.time()
            budget = time_budget * _cfg("social", "worker_budget_ratio")
            n_initial = max(10, worker.n_dims)

            for _ in range(n_initial):
                sample = worker._random_sample()
                score = float(worker.eval_fn(sample))
                worker.measurements.append({"params": sample, "score": score})
                if score > worker.best_score:
                    worker.best_score = score
                    worker.best_params = sample

            worker._understand(verbose=False)
            batch_size = int(_cfg("sampling", "batch_size"))
            share_interval = int(_cfg("social", "share_interval"))

            while time.time() - t0 < budget:
                worker._check_phase_transition(t0)
                worker._budget = budget

                if worker.phase == "EXPLOIT" and worker.proxy_fn:
                    batch = worker._proxy_screened_sample(
                        int(_cfg("social", "screen_candidates")), batch_size)
                    for m in batch:
                        worker.measurements.append(m)
                        if m["score"] > worker.best_score:
                            worker.best_score = m["score"]
                            worker.best_params = m["params"]
                else:
                    samples = [worker._importance_guided_sample()
                               for _ in range(batch_size)]
                    for s in samples:
                        score = float(worker.eval_fn(s))
                        worker.measurements.append({"params": s, "score": score})
                        if score > worker.best_score:
                            worker.best_score = score
                            worker.best_params = s

                if len(worker.measurements) % 10 == 0:
                    worker._understand(verbose=False)

                # 定期共有: 隣接ワーカーのbest measurementsを取り込み
                if len(worker.measurements) % share_interval == 0:
                    with shared_lock:
                        # 自分のbestを公開
                        if worker.best_params is not None:
                            shared_best[worker_id] = {
                                "params": list(worker.best_params),
                                "score": worker.best_score,
                            }
                        # 隣接のbestを取り込み
                        for nb in neighbors[worker_id]:
                            nb_best = shared_best[nb]
                            nst = _cfg("social", "neighbor_share_threshold")
                            if (nb_best["params"] is not None
                                    and nb_best["score"] > worker.best_score * nst):
                                worker.measurements.append(dict(nb_best))

            all_measurements[worker_id] = worker.measurements
            with shared_lock:
                if worker.best_params is not None:
                    shared_best[worker_id] = {
                        "params": list(worker.best_params),
                        "score": worker.best_score,
                    }

        # 並列実行
        threads = []
        for i in range(n):
            t = threading.Thread(target=worker_fn, args=(i,), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=time_budget + 10)

        # 全ワーカーのmeasurementsを統合
        for wm in all_measurements:
            self.measurements.extend(wm)

        # 全体のbest
        global_best = max(shared_best, key=lambda x: x["score"])
        if global_best["params"] is not None:
            self.best_params = global_best["params"]
            self.best_score = global_best["score"]

        if verbose:
            for i, sb in enumerate(shared_best):
                print(f"  [K²] worker {i}: score={sb['score']:.2f}")

        # 統合後の最終owl()
        self._understand(verbose)
        remaining = max(10, time_budget * (1.0 - _cfg("social", "worker_budget_ratio")))
        self._final_optimize(remaining, verbose)
        self._save_experience()

        elapsed = time.time() - t0
        if verbose:
            print(f"\n  [K²] Complete: score={self.best_score:.4f}, "
                  f"{len(self.measurements)} total evals")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "importance": self.importance.tolist(),
            "dead_dims": self.dead_dims,
            "active_dims": self.active_dims,
            "proxy_r2": self.proxy_r2,
            "proxy_type": self.proxy_type,
            "phase": "K2_SOCIAL",
            "phase_log": [{"phase": "K2_SOCIAL",
                           "n_workers": n,
                           "n_measurements": len(self.measurements)}],
            "n_measurements": len(self.measurements),
            "n_workers": n,
            "worker_scores": [sb["score"] for sb in shared_best],
        }

    # ================================================================
    # 自動policy_type選択
    # ================================================================

    def _auto_select_policy(self, total_budget, verbose):
        """linear vs kathara を短時間テスト → 勝者でself再構築。"""
        trial_budget = max(5, total_budget * 0.1)

        scores = {}
        for pt in ["linear", "kathara"]:
            trial = Nous(env=self.env, policy_type=pt,
                         learning_mode="global", memory_len=self.memory_len)
            # 最小限の探索
            for _ in range(int(_cfg("policy", "auto_select_trials"))):
                sample = trial._random_sample()
                score = float(trial.eval_fn(sample))
                if score > scores.get(pt, float('-inf')):
                    scores[pt] = score
            if verbose:
                print(f"  [auto] {pt}: best={scores[pt]:.2f}")

        winner = max(scores, key=scores.get)

        # selfを勝者のpolicy_typeで再構築
        if winner == "kathara" and self._kathara_policy is None:
            self._kathara_policy = KatharaPolicy(
                self.env.obs_size, self.env.n_actions)
        elif winner == "linear":
            self._kathara_policy = None

        self.policy_type = winner
        # eval_fn, param_ranges, param_names を再構築
        self.eval_fn = lambda params: self._run_episode(params)
        n_policy = self._policy_dims()
        pr = _cfg("policy", "range")
        self.param_ranges = [(-pr, pr)] * n_policy
        self.n_dims = n_policy
        self.param_names = self._make_policy_names()
        self.importance = np.ones(self.n_dims)
        self.dead_dims = []
        self.active_dims = list(range(self.n_dims))
        self.measurements = []
        self.best_params = None
        self.best_score = float('-inf')
        return winner

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
            # 次元不一致チェック (learning_mode変更時)
            if ws is not None and len(ws) != self.n_dims:
                ws = None
            dead_hint = ue.get_dead_dims()
            if dead_hint and all(d < self.n_dims for d in dead_hint):
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
    # メタ最適化: Nous自身の設定をowl()で最適化 (LaD究極形)
    # ================================================================

    @staticmethod
    def meta_optimize(env_factory, time_budget=300, n_trials=30,
                      verbose=True):
        """Nousの内部パラメータ自体をowl()で最適化。

        LaD原則の究極形: 思考のデータ化 → 思考の最適化。
        env_factory: () → Environment (毎回新環境を生成)
                     またはリスト [factory1, factory2, ...] → 全環境平均で汎化

        最適化対象 (13D):
            phase_explore_ratio, phase_focus_ratio, phase_exploit_ratio,
            r2_threshold_deepen, batch_size, n_initial_base,
            importance_width_min, n_episodes, screen_candidates,
            search_iters, understand_interval, curiosity_split_ratio,
            curiosity_candidates
        """
        from twelve.optimize import owl

        # env_factory正規化: 単一→リスト化
        if callable(env_factory) and not isinstance(env_factory, list):
            env_factories = [env_factory]
        else:
            env_factories = list(env_factory)

        # 最適化対象の定義
        meta_dims = [
            ("phase", "explore_ratio", 0.05, 0.4),
            ("phase", "focus_ratio", 0.2, 0.6),
            ("phase", "exploit_ratio", 0.4, 0.8),
            ("phase", "r2_threshold_deepen", 0.3, 0.9),
            ("sampling", "batch_size", 2, 15),
            ("sampling", "n_initial_base", 10, 50),
            ("sampling", "importance_width_min", 0.01, 0.5),
            ("sampling", "n_episodes", 1, 8),
            ("proxy", "screen_candidates", 20, 300),
            ("proxy", "search_iters", 100, 3000),
            ("proxy", "understand_interval", 5, 30),
            ("curiosity", "split_ratio", 0.1, 0.9),
            ("curiosity", "candidates", 10, 200),
        ]

        param_names = [f"{s}.{k}" for s, k, _, _ in meta_dims]
        param_ranges = [(lo, hi) for _, _, lo, hi in meta_dims]

        # eval_fn: 設定を適用 → 軽量RL評価 → スコア
        # Nous.run()は内部でowl()を呼ぶため重すぎる。
        # 代わりに: episode実行 + HC最適化の軽量版で設定品質を測定。
        def meta_eval(params):
            global _nous_cache
            cfg = _load_nous_config()
            saved = {}
            for i, (section, key, _, _) in enumerate(meta_dims):
                saved[(section, key)] = cfg[section][key]
                val = params[i]
                if key in ("batch_size", "n_initial_base", "n_episodes",
                           "screen_candidates", "search_iters",
                           "understand_interval", "candidates"):
                    val = max(1, int(round(val)))
                cfg[section][key] = val

            cfg["phase"]["focus_ratio"] = max(
                cfg["phase"]["focus_ratio"],
                cfg["phase"]["explore_ratio"] + 0.05)
            cfg["phase"]["exploit_ratio"] = max(
                cfg["phase"]["exploit_ratio"],
                cfg["phase"]["focus_ratio"] + 0.05)

            try:
                n_ep = int(cfg["sampling"]["n_episodes"])
                batch = int(cfg["sampling"]["batch_size"])
                n_initial = int(cfg["sampling"]["n_initial_base"])
                scores_list = []
                for ef in env_factories:
                    env = ef()
                    obs_dim = env.obs_size
                    is_cont = (env.action_type == "continuous")
                    act_dim = env.action_dim if is_cont else env.n_actions
                    n_p = obs_dim * act_dim + act_dim
                    a_lo, a_hi = env.action_range if is_cont else (0, 0)

                    def _run_ep(p, _env=env, _od=obs_dim, _ad=act_dim,
                                _ne=n_ep, _cont=is_cont, _alo=a_lo,
                                _ahi=a_hi):
                        W = np.array(p[:_od * _ad]).reshape(_ad, _od)
                        b = np.array(p[_od * _ad:])
                        total = 0.0
                        for _ in range(_ne):
                            obs = _env.reset()
                            done = False
                            while not done:
                                out = W @ np.array(obs) + b
                                if _cont:
                                    act = np.clip(np.tanh(out) *
                                                  _ahi, _alo, _ahi)
                                else:
                                    act = int(np.argmax(out))
                                obs, r, done = _env.step(act)
                                total += r
                        return total / _ne

                    # initial random + HC
                    best_p = None
                    best_s = float('-inf')
                    for _ in range(min(n_initial, 15)):
                        p = [float(np.random.uniform(-2, 2))
                             for _ in range(n_p)]
                        s = _run_ep(p)
                        if s > best_s:
                            best_s = s
                            best_p = p
                    for _ in range(batch * 2):
                        p2 = [v + np.random.normal(0, 0.3) for v in best_p]
                        s2 = _run_ep(p2)
                        if s2 > best_s:
                            best_s = s2
                            best_p = p2
                    scores_list.append(best_s)
                score = float(np.mean(scores_list))
            except Exception:
                score = float('-inf')
            finally:
                for (section, key), val in saved.items():
                    cfg[section][key] = val

            return score

        if verbose:
            print(f"  [meta] Nousの内部パラメータを最適化 ({len(meta_dims)}D)")
            print(f"  [meta] {n_trials} trials, {len(env_factories)} envs, budget={time_budget}s")

        # データ収集
        measurements = []
        for i in range(n_trials):
            sample = [float(np.random.uniform(lo, hi))
                      for _, _, lo, hi in meta_dims]
            score = meta_eval(sample)
            measurements.append({"params": sample, "score": score})
            if verbose and (i + 1) % 5 == 0:
                best_so_far = max(m["score"] for m in measurements)
                print(f"  [meta] {i+1}/{n_trials}: best={best_so_far:.2f}")

        # owl()で構造発見 + 最適化
        result = owl(
            measurements=measurements,
            param_ranges=param_ranges,
            param_names=param_names,
            experience_id="nous_meta",
            time_budget=60,
            verbose=verbose,
        )

        # 最適設定をJSONに書き出し
        if result.get("best_params"):
            bp = result["best_params"]
            if isinstance(bp, dict):
                bp = [bp.get(n, (lo + hi) / 2)
                      for n, (_, _, lo, hi) in zip(param_names, meta_dims)]

            cfg = _load_nous_config()
            for i, (section, key, _, _) in enumerate(meta_dims):
                val = bp[i]
                if key in ("batch_size", "n_initial_base", "n_episodes",
                           "screen_candidates", "search_iters",
                           "understand_interval", "candidates"):
                    val = max(1, int(round(val)))
                cfg[section][key] = val

            # 順序制約
            cfg["phase"]["focus_ratio"] = max(
                cfg["phase"]["focus_ratio"],
                cfg["phase"]["explore_ratio"] + 0.05)
            cfg["phase"]["exploit_ratio"] = max(
                cfg["phase"]["exploit_ratio"],
                cfg["phase"]["focus_ratio"] + 0.05)

            # JSONに保存
            json_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "configs", "nous_params.json")
            env_names = [ef.__name__ for ef in env_factories]
            cfg["_comment"] = (f"owl()で最適化済み ({'+'.join(env_names)}, "
                               f"{n_trials} trials, R2={result.get('proxy_r2', 0):.3f})")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)

            # キャッシュクリア
            global _nous_cache
            _nous_cache = None

            if verbose:
                print(f"\n  [meta] 最適設定を保存: {json_path}")
                print(f"  [meta] importance: {result.get('importance', [])[:5]}")
                print(f"  [meta] dead_dims: {result.get('dead_dims', [])}")

        return result


# ================================================================
# デモ
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nous — autonomous RL agent")
    parser.add_argument("--mode", choices=[
        "rl", "rastrigin", "both", "compare", "flyworld",
        "continuous", "memory", "curiosity", "social", "multi_env", "auto",
        "meta", "social_world", "life", "transfer",
    ], default="both")
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--policy", choices=["linear", "kathara", "hebbian",
                                              "auto", "compare", "growth"],
                        default="linear")
    parser.add_argument("--memory", type=int, default=0,
                        help="Episode memory length (0=disabled)")
    parser.add_argument("--workers", type=int, default=1,
                        help="K² social learning workers (1=disabled)")
    parser.add_argument("--curiosity", action="store_true",
                        help="Enable intrinsic curiosity")
    args = parser.parse_args()

    def _print_result(result, label=""):
        """結果表示ヘルパー。"""
        print(f"\n  {label}Best score: {result['best_score']:.4f}")
        if result.get("final_nodes"):
            print(f"  Final: N={result['final_nodes']}, E={result['final_edges']}")
            print(f"  Dead: {len(result['dead_dims'])}/{result['final_edges']}")
        else:
            print(f"  Dead: {len(result['dead_dims'])}/{len(result.get('importance', []))}")
        if result.get('n_workers'):
            print(f"  Workers: {result['n_workers']}, "
                  f"scores={[f'{s:.1f}' for s in result.get('worker_scores', [])]}")
        interp = result.get("policy_interpretation", {})
        if interp.get("top_connections"):
            print(f"  Top connections:")
            for c in interp["top_connections"][:5]:
                print(f"    {c['param']:12s} imp={c['importance']:.4f}  "
                      f"w={c['weight']:+.3f}")

    if args.mode == "compare":
        print("\n" + "=" * 60)
        print("  COMPARE: Linear vs Kathara vs Hebbian")
        print("=" * 60)
        results = {}
        for pt, lm in [("linear", "global"), ("kathara", "global"),
                        ("linear", "hebbian")]:
            label = lm if lm == "hebbian" else pt
            print(f"\n--- {label.upper()} ---")
            r = Nous(env=BalanceEnv(), policy_type=pt,
                     learning_mode=lm).run(
                time_budget=args.budget // 3, verbose=True)
            results[label] = r
        print(f"\n{'=' * 60}")
        for label, r in results.items():
            print(f"  {label:10s}: {r['best_score']:.1f}")
        print(f"  Winner: {max(results, key=lambda k: results[k]['best_score'])}")

    elif args.mode == "continuous":
        # 連続行動テスト (SwingUpEnv)
        print("\n" + "=" * 60)
        print("  CONTINUOUS: SwingUpEnv (pendulum)")
        print("=" * 60)
        env = SwingUpEnv()
        nous = Nous(env=env, memory_len=args.memory,
                    curiosity=args.curiosity)
        result = nous.run(time_budget=args.budget, verbose=True)
        _print_result(result, "SwingUp ")

    elif args.mode == "memory":
        # エピソード内記憶テスト
        print("\n" + "=" * 60)
        mem = args.memory if args.memory > 0 else 3
        print(f"  MEMORY: BalanceEnv with memory_len={mem}")
        print("=" * 60)
        for ml in [0, mem]:
            print(f"\n--- memory_len={ml} ---")
            r = Nous(env=BalanceEnv(), memory_len=ml).run(
                time_budget=args.budget // 2, verbose=True)
            print(f"  score={r['best_score']:.1f}, "
                  f"dims={len(r['importance'])}")

    elif args.mode == "curiosity":
        # 好奇心比較テスト
        print("\n" + "=" * 60)
        print("  CURIOSITY: with vs without")
        print("=" * 60)
        for cur in [False, True]:
            print(f"\n--- curiosity={cur} ---")
            r = Nous(env=BalanceEnv(), curiosity=cur).run(
                time_budget=args.budget // 2, verbose=True)
            print(f"  score={r['best_score']:.1f}")

    elif args.mode == "social":
        # K²社会学習テスト
        n = args.workers if args.workers > 1 else 4
        print("\n" + "=" * 60)
        print(f"  K² SOCIAL: {n} workers")
        print("=" * 60)
        nous = Nous(env=BalanceEnv(), n_workers=n)
        result = nous.run(time_budget=args.budget, verbose=True)
        _print_result(result, "K² ")

    elif args.mode == "multi_env":
        # マルチ環境テスト
        print("\n" + "=" * 60)
        print("  MULTI-ENV: BalanceEnv × 3 (汎化テスト)")
        print("=" * 60)
        envs = [BalanceEnv() for _ in range(3)]
        nous = Nous(envs=envs, curiosity=args.curiosity)
        result = nous.run(time_budget=args.budget, verbose=True)
        _print_result(result, "MultiEnv ")

    elif args.mode == "auto":
        # 自動policy_type選択テスト
        print("\n" + "=" * 60)
        print("  AUTO: policy_type=auto")
        print("=" * 60)
        nous = Nous(env=BalanceEnv(), policy_type="auto")
        result = nous.run(time_budget=args.budget, verbose=True)
        _print_result(result, "Auto ")

    elif args.mode == "meta":
        # メタ最適化: 複数環境で汎化
        print("\n" + "=" * 60)
        print("  META: Nousの内部パラメータを自動最適化 (multi-env LaD)")
        print("=" * 60)
        result = Nous.meta_optimize(
            env_factory=[BalanceEnv, SwingUpEnv],
            time_budget=args.budget,
            n_trials=10,
            verbose=True,
        )
        print(f"\n  R²={result.get('proxy_r2', 0):.3f}")
        print(f"  dead_dims={result.get('dead_dims', [])}")

    elif args.mode == "flyworld":
        print("\n" + "=" * 60)
        lm = ("growth" if args.policy == "growth" else
              "hebbian" if args.policy == "hebbian" else "global")
        pt = "linear" if args.policy in ("linear", "hebbian", "growth") else args.policy
        print(f"  FlyWorld (policy={args.policy}, learning={lm})")
        print("=" * 60)
        try:
            env = FlyWorldEnv(version=3, max_steps=60)
            nous = Nous(env=env, policy_type=pt, learning_mode=lm,
                        memory_len=args.memory, curiosity=args.curiosity)
            result = nous.run(time_budget=args.budget, verbose=True)
            _print_result(result, "FlyWorld ")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    elif args.mode == "social_world":
        print("\n" + "=" * 60)
        lm = ("growth" if args.policy == "growth" else
              "hebbian" if args.policy == "hebbian" else "global")
        pt = "linear" if args.policy in ("linear", "hebbian", "growth") \
            else args.policy
        print(f"  SocialWorld (policy={args.policy}, learning={lm})")
        print("=" * 60)
        env = SocialWorld()
        nous = Nous(env=env, policy_type=pt, learning_mode=lm,
                    memory_len=args.memory, curiosity=args.curiosity)
        result = nous.run(time_budget=args.budget, verbose=True)
        _print_result(result, "SocialWorld ")

        # Proto-language 分析
        stats = env.get_signal_stats()
        if stats:
            print(f"\n  === Proto-language Analysis ===")
            print(f"  Agent A signals: {stats.get('freq_a', [])}")
            print(f"  Agent B signals: {stats.get('freq_b', [])}")
            nu_a = stats.get('non_uniformity_a', 0)
            nu_b = stats.get('non_uniformity_b', 0)
            print(f"  Non-uniformity: A={nu_a}, B={nu_b}")
            if nu_a > 0.3 or nu_b > 0.3:
                print("  >>> Proto-language detected!")

    elif args.mode == "life":
        print("\n" + "=" * 60)
        lm = ("growth" if args.policy == "growth" else
              "hebbian" if args.policy == "hebbian" else "global")
        pt = "linear" if args.policy in ("linear", "hebbian", "growth") \
            else args.policy
        print(f"  LifeWorld (policy={args.policy}, learning={lm})")
        print("=" * 60)
        env = LifeWorld(n_peers=2)
        nous = Nous(env=env, policy_type=pt, learning_mode=lm,
                    memory_len=args.memory, curiosity=args.curiosity)
        result = nous.run(time_budget=args.budget, verbose=True)
        _print_result(result, "LifeWorld ")

        stats = env.get_life_stats()
        print(f"\n  === Life Story ===")
        print(f"  Survived: {stats['age']}/{env.MAX_STEPS} steps")
        print(f"  Energy: {stats['energy']}")
        print(f"  Status: {'ALIVE' if stats['alive'] else 'DEAD'}")
        print(f"  Meals: {stats['meals']}")
        print(f"  Hazard hits: {stats['hazard_hits']}")
        print(f"  Peers alive: {stats['peers_alive']}/{env.n_peers}")
        print(f"  Signal rate: {stats['signal_rate']:.1%}")

    elif args.mode == "transfer":
        # ============================================================
        # Transfer実験 v2: LifeWorldの構造理解 → FlyWorldに転移
        #
        # 転移するもの:
        #   - dead_dims (不要な辺 → lr=0で学習スキップ)
        #   - lr_vector (どの辺をどの速度で学ぶか)
        #   - トポロジー (N, 辺構造)
        # ============================================================
        print("\n" + "=" * 60)
        print("  TRANSFER EXPERIMENT v2")
        print("  LifeWorldの構造理解をFlyWorldに転移")
        print("=" * 60)

        total_budget = args.budget
        grow_budget = int(total_budget * 0.4)
        test_budget = int(total_budget * 0.2)

        # --- Phase 1: LifeWorldで脳を育てる ---
        print(f"\n  [Phase 1] LifeWorld Growth ({grow_budget}s)")
        print("-" * 50)
        env_life = LifeWorld(n_peers=2)
        nous_life = Nous(env=env_life, policy_type="linear",
                         learning_mode="growth",
                         memory_len=0, curiosity=False)
        life_result = nous_life.run(time_budget=grow_budget, verbose=True)
        life_stats = env_life.get_life_stats()

        grown_brain = nous_life._hebbian_policy
        grown_n = grown_brain.n_nodes
        grown_offsets = list(grown_brain.offsets)
        grown_edges = grown_brain.n_edges
        life_dead = life_result.get("dead_dims", [])
        life_best_params = life_result.get("best_params")

        print(f"\n  Grown brain: N={grown_n}, E={grown_edges}, "
              f"offsets={grown_offsets}")
        print(f"  LifeWorld score: {life_result['best_score']:.1f}")
        print(f"  Dead edges: {len(life_dead)}/{grown_edges} "
              f"({len(life_dead)/max(1,grown_edges):.0%})")
        print(f"  Life: age={life_stats['age']}, "
              f"meals={life_stats['meals']}, "
              f"hazards={life_stats['hazard_hits']}")

        # --- Phase 2: FlyWorld 4条件対決 ---
        print(f"\n  [Phase 2] FlyWorld Test ({test_budget}s each)")
        print("-" * 50)

        def _make_nous_fly(n_nodes, offsets, dead_dims=None):
            """指定トポロジーでFlyWorld Nousを構築。"""
            ef = FlyWorldEnv(version=3, max_steps=60)
            n = Nous(env=ef, policy_type="linear",
                     learning_mode="hebbian")
            hp = HebbianKatharaPolicy(
                ef.obs_size, ef.n_actions,
                n_nodes=n_nodes, offsets=offsets)
            n._hebbian_policy = hp
            n.param_ranges = [(0.0, 0.5)] * hp.n_edges
            n.param_names = [f"lr_e{a}_{b}" for a, b in hp.edge_list]
            n.n_dims = hp.n_edges
            n.importance = np.ones(hp.n_edges)
            n.dead_dims = list(dead_dims) if dead_dims else []
            n.active_dims = [i for i in range(hp.n_edges)
                             if i not in (dead_dims or [])]
            return n, ef

        # A: LifeWorld構造 + dead_dims転移
        print(f"\n  [A] Transfer (topology + dead_dims from LifeWorld)")
        nous_a, _ = _make_nous_fly(
            grown_n, grown_offsets, dead_dims=life_dead)
        result_a = nous_a.run(time_budget=test_budget, verbose=True)

        # B: 同サイズ、dead_dimsなし（白紙の脳）
        print(f"\n  [B] Same topology, no transfer (blank brain)")
        nous_b, _ = _make_nous_fly(
            grown_n, grown_offsets, dead_dims=None)
        result_b = nous_b.run(time_budget=test_budget, verbose=True)

        # C: ベースライン N=12
        print(f"\n  [C] Baseline N=12")
        env_c = FlyWorldEnv(version=3, max_steps=60)
        nous_c = Nous(env=env_c, policy_type="linear",
                      learning_mode="hebbian")
        result_c = nous_c.run(time_budget=test_budget, verbose=True)

        # D: LifeWorld lr_vectorをそのまま使ってcumulative hebbian
        print(f"\n  [D] Direct lr_vector transfer (no Owl)")
        nous_d, env_d = _make_nous_fly(
            grown_n, grown_offsets, dead_dims=life_dead)
        score_d = 0.0
        if (life_best_params is not None
                and len(life_best_params) == nous_d.n_dims):
            nous_d.best_score = float('-inf')
            nous_d._run_cumulative_hebbian(
                life_best_params, n_episodes=40, verbose=True)
            score_d = nous_d.best_score
        else:
            print(f"    (skipped: lr dim mismatch "
                  f"{len(life_best_params) if life_best_params else 0}"
                  f" vs {nous_d.n_dims})")

        # --- Phase 3: 結果比較 ---
        score_a = result_a["best_score"]
        score_b = result_b["best_score"]
        score_c = result_c["best_score"]

        print(f"\n{'=' * 60}")
        print(f"  TRANSFER RESULTS v2")
        print(f"{'=' * 60}")
        w = 45
        print(f"  {'Condition':<{w}s} {'Score':>8s}  {'Dead':>6s}")
        print(f"  {'-'*w} {'-'*8}  {'-'*6}")
        print(f"  {'[A] Transfer (dead_dims from LifeWorld)':<{w}s} "
              f"{score_a:8.2f}  "
              f"{len(life_dead):>2d}/{grown_edges:<2d}")
        print(f"  {'[B] Same topology, blank brain':<{w}s} "
              f"{score_b:8.2f}  "
              f"{'0':>2s}/{grown_edges:<2d}")
        print(f"  {'[C] Baseline N=12':<{w}s} "
              f"{score_c:8.2f}  "
              f"{len(result_c.get('dead_dims',[])):>2d}/"
              f"{nous_c._hebbian_policy.n_edges:<2d}")
        print(f"  {'[D] Direct lr_vector (no Owl)':<{w}s} "
              f"{score_d:8.2f}  "
              f"{len(life_dead):>2d}/{grown_edges:<2d}")

        print(f"\n  Transfer effects:")
        print(f"    A vs B (dead_dims value):   {score_a - score_b:+.2f}")
        print(f"    A vs C (vs baseline):       {score_a - score_c:+.2f}")
        print(f"    D vs C (lr direct):         {score_d - score_c:+.2f}")

        best_cond = max(
            [("A:Transfer", score_a), ("B:Blank", score_b),
             ("C:Baseline", score_c), ("D:Direct-lr", score_d)],
            key=lambda x: x[1])
        print(f"\n  Winner: {best_cond[0]} ({best_cond[1]:.2f})")

        if score_a > score_b + 3 or score_d > score_c + 3:
            print(f"\n  >>> STRUCTURAL TRANSFER CONFIRMED")
            print(f"  >>> LifeWorldの構造理解はFlyWorldで有利")
        elif score_a > score_b or score_d > score_c:
            print(f"\n  >>> Slight advantage detected")
        else:
            print(f"\n  >>> Need different transfer method")

    else:
        if args.mode in ("rl", "both"):
            print("\n" + "=" * 60)
            lm = ("growth" if args.policy == "growth" else
                  "hebbian" if args.policy == "hebbian" else "global")
            pt = "linear" if args.policy in ("linear", "hebbian", "growth") else args.policy
            print(f"  DEMO 1: RL mode (BalanceEnv, policy={args.policy})")
            print("=" * 60)
            env = BalanceEnv()
            nous = Nous(env=env, experience_id="nous_balance",
                        policy_type=pt, learning_mode=lm,
                        memory_len=args.memory, curiosity=args.curiosity,
                        n_workers=args.workers)
            result = nous.run(time_budget=args.budget, verbose=True)
            _print_result(result, "BalanceEnv ")

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
            nous = Nous(rastrigin, ranges, experience_id="test_rastrigin",
                        curiosity=args.curiosity)
            result = nous.run(time_budget=min(30, args.budget), verbose=True)
            print(f"\n  Best score: {result['best_score']:.4f}")
            print(f"  Dead dims: {result['dead_dims']}")
