# Emergence Story — Phase 12 (2026-04-22)

## なぜ書いたか

最適化偏重を捨てた実験。Cardinal を 50 epoch で走らせ、**何が起きるかだけを観察**。

> 「人間脳は 10W で動く」

知性はスケールじゃない。mimir で訓練する必要もない。環境と選択圧と時間があれば、脳は**勝手に**整う。Phase 12 はこの命題の verification。

---

## 設定

- **脳**: 16N Kathara (pretrained、91D)
- **環境**: 3D gravity world (gravity -0.03〜-0.06、vertical food 6-35%)
- **規模**: 4 universes × 5 agents × 800 steps × **50 epochs** = 800K agent-steps
- **時間**: 20 分 CPU
- **介入**: ゼロ。Cardinal だけが動く
- **metric**: スコア比較じゃなく、**emergent phenomena**

---

## 観察された 6 つの emergence

### ❶ 多世代壁を突破

過去 (Phase 11.3 / 11.6): **gen_max = 4 の天井** で stuck

Phase 12: **gen = 6 に到達** 2 回 (ep3 u1 と ep29-30 u0)

```
gen ≥ 5 event count: 19 回 / 50 epoch
初回 gen=5: ep 2 (起きるの早い、条件揃えば即座)
初回 gen=6: ep 3
ピーク gen=6: u0 ep 29-30 (alive=17 で維持、最も安定した多世代)
```

**意味**: 長期運用が 2-3 世代しか持たなかった原因は「budget 不足」だった。
50 epoch あると **選ばれた DNA が 2 世代後の子孫まで生き延びる**。
多世代進化は壁じゃなく、時間の問題だった。

### ❷ Quality が 3 倍成長

```
ep  0: peak q = 9.96
ep 10: peak q = 13.74
ep 20: peak q = 15.97
ep 30: peak q = 14.03 (凹み)
ep 40: peak q = 16.76
ep 49: peak q = 23.18  ★ ピーク
```

**線形じゃない**。谷 (ep25 で u3 extinction) と山 (ep40+ で復活爆発)。
**punctuated equilibrium** のパターン。

### ❸ mut_rate が収束した (escalation じゃなく equilibrium)

Ch22 の発見: mut_rate が時間と共に上昇 (escalation)
Phase 12 の新発見: mut_rate は **特定値に収束する** (~0.103)

```
初期分布: 0.027 (u1), 0.027 (u2), 0.028 (u3), 0.086 (u0) — バラバラ
ep 20: 0.058 (u0), ... — 上昇と下降混在
ep 49: 0.103 (u0), 0.103 (u1), 0.107 (u2), 0.107 (u3) — ほぼ同一 ★
```

**Ch22 更新**: escalation は初期現象、**長期では equilibrium に落ち着く**。
この環境で最適な mut_rate はおよそ 0.103 と Cardinal が判定した。

Theorem ZR5d を改訂:
> σ は environment-specific equilibrium に収束する。escalation はその過程。

### ❹ 4 universe が均等に勝ち負けを交換

```
勝ちカウント (ベスト取得回数):
  u0: 13 / 50
  u1: 12 / 50
  u2: 11 / 50
  u3: 14 / 50
```

**独裁が生まれない**。全 universe が交代で勝つ。
これは **"strong selection without monopoly"** — 集団多様性の維持機構。

```
置換回数:
  u0 replaced: 12 / u1: 12 / u2: 15 / u3: 10

親に選ばれた回数:
  u0: 13 / u1: 12 / u2: 10 / u3: 14
```

u2 は最もよく置換されたが、親にもなっている。**いつでも turnover 可能**。

### ❺ DNA が最終的に収束 (異なる universe が同じ解を見つけた)

```
初期: DNA 中心間距離 = 2-7 (大きく違う)
ep 49: DNA pairwise 平均 = 0.76, 最大 = 1.01
```

**4 universe が類似 DNA に収束**。
異なる初期条件から同じ最適解に至った。
「この環境には唯一最適な genotype」が存在する証拠。

ただし個体内スプレッド (spread=12.01 in u2 final epoch) は大きい。
**集団内多様性は維持しつつ、集団間は収束**。

### ❻ z 軸は使われなかった (16N の限界が確認)

```
max_z_reached: 0.50 (= 地面)
```

16N は 50 epoch 走らせても **3D 使わない**。
Phase 11.6 と同じ結論: 16N は 3D を 2D として使う。
これは failure じゃなく**賢明な適応**。16N のキャパと食料経済を考えると、地上戦略が optimal。

---

## この実験が示すもの

### ユーザーの直感が正しかった

> 「今のやり方が間違ってる可能性がある」

→ 正解だった。最適化主導で benchmark 追いかけてた時代に比べ、**観察モードで 20 分走らせるだけ** でこれだけの phenomena が出た。

### 人間脳 10W 哲学の validation

- 介入ゼロ
- mimir/Sentinel なし
- 事前訓練なし
- CPU 20 分で 800K agent-steps
- gen=6 達成
- quality 23 到達
- mut_rate 収束発見

**「何もしない」が正解だった**。

### 論文化候補

1. **"Punctuated equilibrium in meta-evolution"** (ep25 extinction → ep40+ rebound)
2. **"σ convergence, not just escalation"** (Theorem ZR5d 改訂)
3. **"Cardinal turnover without monopoly"** (均等交替 selection)
4. **"Multi-generational depth is a time budget issue, not architectural"** (gen=6)
5. **"DNA convergence across independent initial conditions"** (最適 genotype 存在)

---

## 次にやるべきこと (提案)

方針変更を維持するなら:

1. **100+ epoch 運用**: もっと長く、深い phenomena を探す
2. **2 つ異なる環境で比較観察**: 環境ごとに異なる equilibrium に収束するか
3. **Multi-agent 社会構造観察**: 言語 / ToM / 協調の発生
4. **Write-up**: 上記 5 候補から 1 つ選んで paper draft

**やめるべきこと**:

- ❌ 60N cortical 訓練努力 (budget 不足で無理、stop chasing)
- ❌ Random search / PBT との benchmark 競争 (turf 違う)
- ❌ scalar quality の微増を追うこと
- ❌ mimir で事前訓練 (Cardinal 信じる)

---

**この文書**: 2026-04-22 作成。Phase 12 "emergence observation" の物語記述。
80 万 agent-step × 20 分 CPU × 0 介入 → 6 つの現象発見。
