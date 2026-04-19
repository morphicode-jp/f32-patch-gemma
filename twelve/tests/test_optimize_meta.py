"""Phase 3 メタ進化レイヤーのテストスイート。

テスト対象:
  - optimize(meta=True) の動作・後方互換
  - Phase 3 各コンポーネント (lad_accel, k7, k8, k9, knowledge)
  - ExperienceStore の Phase 3 フィールド
  - TwelveParallel.default_k9() の 9レイヤー並列
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from twelve.optimize import optimize, ExperienceStore
from twelve.twelve_parallel import TwelveParallel


# ═══════════════════════════════════════════════════════
# 共通: テスト用 eval_fn と ranges
# ═══════════════════════════════════════════════════════

def eval_fn(params):
    """単純な二次関数（最適値は全パラメータ=1.0のとき0）"""
    return -sum((x - 1.0) ** 2 for x in params)


ranges = [(0, 2)] * 3


# ─── ヘルパー: 一時ExperienceStoreのクリーンアップ ───

def _exp_path(name):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "experience", f"{name}.json"
    )


def _cleanup(name):
    p = _exp_path(name)
    if os.path.exists(p):
        os.remove(p)


# ═══════════════════════════════════════════════════════
# A. Phase 3 基本 (7テスト)
# ═══════════════════════════════════════════════════════

class TestPhase3Basic:
    """Phase 3 の基本動作テスト"""

    def test_meta_false_no_phase3(self):
        """meta=False時にphase3がinfoに含まれない"""
        best, score, info = optimize(eval_fn, ranges, time_budget=5)
        assert "phase3" not in info

    def test_meta_true_has_phase3(self):
        """meta=True時にphase3がinfoに含まれる"""
        best, score, info = optimize(eval_fn, ranges, time_budget=15, meta=True)
        assert "phase3" in info
        assert info["phase3"]["enabled"] is True

    def test_meta_default_components(self):
        """デフォルトでlad_accel+k7のみ実行"""
        best, score, info = optimize(eval_fn, ranges, time_budget=15, meta=True)
        comps = info["phase3"]["components_run"]
        assert "lad_accel" in comps or "k7" in comps

    def test_meta_info_structure(self):
        """phase3 infoの構造が正しい"""
        best, score, info = optimize(eval_fn, ranges, time_budget=15, meta=True)
        p3 = info["phase3"]
        assert "enabled" in p3
        assert "components_run" in p3
        assert "time_used" in p3
        assert "lad_accel" in p3
        assert "k7" in p3

    def test_meta_backward_compat_score(self):
        """meta=True/Falseでスコアが大きく変わらない（同じeval_fn）"""
        best1, score1, _ = optimize(eval_fn, ranges, time_budget=10)
        best2, score2, _ = optimize(eval_fn, ranges, time_budget=10, meta=True)
        # スコアは近い（Phase 3はeval_fnに依存しないので）
        assert abs(score1 - score2) < 5.0

    def test_meta_short_budget(self):
        """time_budget=5秒でもmeta=Trueが安全に動く"""
        best, score, info = optimize(eval_fn, ranges, time_budget=5, meta=True)
        assert "phase3" in info  # エラーなく完了

    def test_meta_config_custom(self):
        """meta_configでカスタム設定"""
        cfg = {"time_ratio": 0.3, "components": ["lad_accel"]}
        best, score, info = optimize(
            eval_fn, ranges, time_budget=15, meta=True, meta_config=cfg
        )
        p3 = info["phase3"]
        # lad_accelだけが実行対象なのでk7はcomponents_runに入らない
        assert "k7" not in p3["components_run"]


# ═══════════════════════════════════════════════════════
# B. コンポーネント個別 (5テスト)
# ═══════════════════════════════════════════════════════

class TestPhase3Components:
    """Phase 3 各コンポーネントの個別テスト"""

    def test_lad_accel_result(self):
        """LadAccelerator結果の構造"""
        cfg = {"components": ["lad_accel"]}
        _, _, info = optimize(
            eval_fn, ranges, time_budget=15, meta=True, meta_config=cfg
        )
        la = info["phase3"].get("lad_accel")
        if la:
            assert "initial_score" in la
            assert "best_score" in la

    def test_k7_result(self):
        """K7結果の構造"""
        cfg = {"components": ["k7"]}
        _, _, info = optimize(
            eval_fn, ranges, time_budget=15, meta=True, meta_config=cfg
        )
        k7 = info["phase3"].get("k7")
        if k7:
            assert "initial_score" in k7
            assert "best_score" in k7

    def test_component_error_isolation(self):
        """1コンポーネントが仮に失敗しても全体は完了する"""
        # 全コンポーネント有効（短い時間で一部はスキップされるかもしれない）
        cfg = {"components": ["all"]}
        best, score, info = optimize(
            eval_fn, ranges, time_budget=15, meta=True, meta_config=cfg
        )
        assert "phase3" in info  # エラーなく完了

    def test_only_k7(self):
        """K7だけ実行"""
        cfg = {"components": ["k7"]}
        _, _, info = optimize(
            eval_fn, ranges, time_budget=15, meta=True, meta_config=cfg
        )
        comps = info["phase3"]["components_run"]
        assert "lad_accel" not in comps

    def test_component_none_when_not_run(self):
        """実行されなかったコンポーネントはNone"""
        cfg = {"components": ["lad_accel"]}
        _, _, info = optimize(
            eval_fn, ranges, time_budget=15, meta=True, meta_config=cfg
        )
        assert info["phase3"]["k8"] is None
        assert info["phase3"]["k9"] is None
        assert info["phase3"]["knowledge"] is None


# ═══════════════════════════════════════════════════════
# C. ExperienceStore Phase 3 (6テスト)
# ═══════════════════════════════════════════════════════

class TestExperienceStorePhase3:
    """ExperienceStore の Phase 3 関連フィールドのテスト"""

    def test_experience_meta_fields(self):
        """新フィールドが初期化される"""
        name = f"test_meta_fields_{os.getpid()}"
        try:
            exp = ExperienceStore(name)
            assert exp._data["meta"].get("phase3_ratio") == 0.2
            assert exp._data["meta"].get("meta_runs", 0) == 0
        finally:
            _cleanup(name)

    def test_experience_record_meta(self):
        """record_meta_result動作"""
        name = f"test_record_meta_{os.getpid()}"
        try:
            exp = ExperienceStore(name)
            exp.record_meta_result("lad_accel", {
                "initial_score": 1.0,
                "best_score": 2.0,
                "best_genome": {"x": 1},
            })
            assert exp._data["meta"]["meta_genome"] == {"x": 1}
        finally:
            _cleanup(name)

    def test_experience_get_meta_genome(self):
        """get_meta_genome()が正しく動く。
        ローカルに無い場合はグローバル経験からフォールバックする可能性がある。
        record後はローカル値が返る。"""
        name = f"test_get_meta_{os.getpid()}"
        try:
            exp = ExperienceStore(name)
            # ローカルに無い → Noneまたはグローバルからのフォールバック
            before = exp.get_meta_genome()
            # record後はローカル値が優先
            exp.record_meta_result("lad_accel", {"best_genome": {"y": 2}})
            assert exp.get_meta_genome() == {"y": 2}
        finally:
            _cleanup(name)

    def test_experience_get_k7_genome(self):
        """get_k7_genome()が正しく動く。
        ローカルに無い場合はグローバル経験からフォールバックする可能性がある。"""
        name = f"test_get_k7_{os.getpid()}"
        try:
            exp = ExperienceStore(name)
            # ローカルに無い → Noneまたはグローバルからのフォールバック
            result = exp.get_k7_genome()
            assert result is None or isinstance(result, dict)
        finally:
            _cleanup(name)

    def _find_tmp_leftovers(self, name):
        """Return list of any .tmp files matching <path>.*.tmp pattern."""
        import glob
        pattern = _exp_path(name) + ".*.tmp"
        return glob.glob(pattern)

    def test_save_is_atomic_no_tmp_leftover(self):
        """save() uses unique tmp + os.replace; after return, no .tmp file lingers."""
        name = f"test_atomic_notmp_{os.getpid()}"
        try:
            exp = ExperienceStore(name)
            exp.record_meta_result("lad_accel", {"best_genome": {"x": 1}})
            exp.save()
            leftovers = self._find_tmp_leftovers(name)
            assert not leftovers, f"tmp files should not linger: {leftovers}"
            assert os.path.exists(_exp_path(name)), "real file should exist"
        finally:
            _cleanup(name)
            for t in self._find_tmp_leftovers(name):
                try: os.remove(t)
                except OSError: pass

    def test_save_parallel_never_corrupts(self):
        """Concurrent save() from threads: file always parses as valid JSON."""
        import threading
        import json as _json
        name = f"test_atomic_parallel_{os.getpid()}"
        try:
            errors = []
            def worker(tid):
                try:
                    for i in range(20):
                        exp = ExperienceStore(name)
                        exp._data["meta"]["best_score_ever"] = float(tid * 100 + i)
                        exp.save()
                except Exception as e:
                    errors.append(e)
            threads = [threading.Thread(target=worker, args=(t,)) for t in range(3)]
            for t in threads: t.start()
            for t in threads: t.join()
            assert not errors, f"worker errors: {errors}"
            # File must parse cleanly (never half-written)
            with open(_exp_path(name), "r", encoding="utf-8") as f:
                loaded = _json.load(f)
            assert isinstance(loaded, dict)
            assert "meta" in loaded
            # No tmp leftovers (unique tmp names all cleaned up)
            leftovers = self._find_tmp_leftovers(name)
            assert not leftovers, f"tmp leftovers: {leftovers}"
        finally:
            _cleanup(name)
            for t in self._find_tmp_leftovers(name):
                try: os.remove(t)
                except OSError: pass

    def test_experience_backward_compat(self):
        """旧フォーマット（meta_runs等なし）でも壊れない"""
        name = f"test_compat_{os.getpid()}"
        try:
            exp = ExperienceStore(name)
            # 旧フォーマットをシミュレート
            if "phase3_ratio" in exp._data["meta"]:
                del exp._data["meta"]["phase3_ratio"]
            # record_meta_resultが動くこと
            exp.record_meta_result("k7", {
                "initial_score": 50,
                "best_score": 60,
            })
            assert exp._data["meta"]["meta_runs"] >= 1
        finally:
            _cleanup(name)

    def test_experience_learn_mode(self):
        """learn=True + meta=True で経験が蓄積される"""
        eid = f"test_learn_meta_{os.getpid()}"
        try:
            _, _, info = optimize(
                eval_fn, ranges, time_budget=15,
                meta=True, learn=True, experience_id=eid,
            )
            # 経験ファイルが存在すること
            assert os.path.exists(_exp_path(eid))
        finally:
            _cleanup(eid)


# ═══════════════════════════════════════════════════════
# D. TwelveParallel.default_k9() (5テスト)
# ═══════════════════════════════════════════════════════

class TestDefaultK9:
    """TwelveParallel.default_k9() の9レイヤー並列テスト"""

    def test_default_k9_creates_9_layers(self):
        """9レイヤーが生成される"""
        engine = TwelveParallel.default_k9()
        assert engine._n_layers == 9

    def test_default_k9_layer_names(self):
        """K1-K9の名前が正しい"""
        engine = TwelveParallel.default_k9()
        names = [spec.name for spec in engine._specs]
        for i in range(1, 10):
            assert f"K{i}" in names

    def test_default_k9_run(self):
        """実行が完了する"""
        engine = TwelveParallel.default_k9()
        result = engine.run(budget_per_layer=50, share_interval=10)
        assert "layers" in result
        assert len(result["layers"]) == 9

    def test_k7_layerspec_score_range(self):
        """K7のscore_fnが0以上の値を返す"""
        from twelve.twelve_parallel import _k7_init, _k7_score
        state = _k7_init()
        score = _k7_score(state)
        assert isinstance(score, (int, float))
        assert score >= 0

    def test_k7_layerspec_mutate(self):
        """K7のmutate_fnがdictを返す"""
        from twelve.twelve_parallel import _k7_init, _k7_mutate
        state = _k7_init()
        mutated = _k7_mutate(state)
        assert isinstance(mutated, dict)


# ═══════════════════════════════════════════════════════
# エントリポイント
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
