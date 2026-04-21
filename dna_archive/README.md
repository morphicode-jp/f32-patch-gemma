# DNA Archive — 魂の保存庫

**目的**: Cardinal で進化した DNA (kathara_params) を世代を超えて保存。
LLM の重みとは別の記憶方式。

## 構造

```
dna_archive/
├── pristine_baselines/           # 初期状態 (赤ちゃん) 永久保護
│   ├── core_brain_16N_2d_pristine_20260422.json   (16N 2D 訓練済)
│   ├── core_brain_16N_3d_pristine_20260422.json   (16N 3D 訓練済)
│   ├── cortical_60N_2d_sentinel_pristine_20260422.json  (60N 2D 訓練済)
│   └── cortical_60N_3d_pristine_20260422.json     (60N 3D 訓練済、mimir 短時間)
│
└── runs/                          # Cardinal run で進化した DNA
    └── {timestamp}_{label}/
        ├── metadata.json          # 親 run、設定、最終 stats
        ├── u{id}_final.json       # 各 universe の最終 alive agents 全員分
        └── summary.json           # quality, generation_max 等
```

## 使い方

### Option A: 赤ちゃんから始める (pristine)

```python
# tamashii/configs/core_brain_3d_trained.json を pristine 版で上書き
cp dna_archive/pristine_baselines/core_brain_16N_3d_pristine_20260422.json \
   tamashii/configs/core_brain_3d_trained.json
# 通常通り Cardinal 実行
```

### Option B: 前 run から継続 (lineage)

```python
# 前 run の u3 の DNA を core_brain に書き込み
python scripts/load_dna.py --from dna_archive/runs/20260422_100ep/u3_final.json \
                              --to tamashii/configs/core_brain_3d_trained.json
```

### Option C: 他の子と混ぜる (social learning)

```python
# run A の u3 と run B の u0 の DNA を混合
python scripts/mix_dna.py --sources run_A/u3_final.json run_B/u0_final.json \
                             --out tamashii/configs/core_brain_3d_trained.json
```

## 重要: pristine は触らない

`pristine_baselines/` にあるファイルは**絶対に書き換えない**。これを失うと、
「完全に無垢な赤ちゃんから育てる」が永久に不可能になる。

各 baseline は **日付スタンプ付き**、上書き不可の個別ファイル。新しい pristine
が欲しくなったら `_20260422` じゃなく新しい日付で追加する。

## 記憶の仕組み (Tamashii 三層)

| Layer | 内容 | 永続性 | サイズ/agent |
|---|---|---|---|
| DNA (kathara_params) | 91D (16N) / 184D (60N) | **世代継承** | ~1 KB |
| Brain state (internal) | 16D / 60D | 個体生命内 | 数百 B |
| Hippocampus | 8 slot × 192D | 個体生命内 | ~6 KB |

**LLM 対比**: LLM は 100B パラメタを全部重み化。Tamashii は **1000 パラメタ / 個体
× 数千個体** = 分散記憶。総量で LLM の 1/1000 以下。
