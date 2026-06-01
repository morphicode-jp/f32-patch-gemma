# publish/basin_b_v1/ — Release v1 ファイル一覧

各ファイルの用途とアップロード先。

## アップロード先別

### → HuggingFace `morphicode_jp/gemma-4-31B-it-basin-b-Q2_K`
- `gemma-4-31B-it-basin-b-Q2_K.gguf` (12.6 GB、basin B 適用済)
- `gemma-4-31B-it-basin-b-Q2_K.gguf.md5` (検証用)
- `HF_README.md` → repo の `README.md` にリネーム

### → GitHub `morphicode_jp/f32-patch-gemma`
- `apply_basin_b.py` (CLI tool 本体、1 ファイル完結)
- `GITHUB_README.md` → repo の `README.md` にリネーム
- `LICENSE` (Apache 2.0)
- `CITATION.cff`
- `REPRODUCE.md`
- `requirements.txt`

### → X (Twitter) `@morphicode_jp`
- `X_THREAD.md` (4 投稿 draft + 補助戦略)

### → Zenn (公開後 1-2 日)
- (準備中) Zenn 日本語記事 draft

### → Reddit r/LocalLLaMA (公開当日)
- (準備中) Reddit post draft

## 公開手順 (推奨順序)

1. **GitHub repo 作成 + push** (5 分)
   ```bash
   cd github_repo
   cp -r publish/basin_b_v1/{apply_basin_b.py,GITHUB_README.md,LICENSE,CITATION.cff,REPRODUCE.md,requirements.txt} .
   mv GITHUB_README.md README.md
   git init && git add -A && git commit -m "Initial release: basin B v1"
   gh repo create morphicode_jp/f32-patch-gemma --public --source=. --push
   ```

2. **HF repo 作成 + upload** (1-2 時間、回線次第)
   ```bash
   huggingface-cli repo create gemma-4-31B-it-basin-b-Q2_K
   huggingface-cli upload morphicode_jp/gemma-4-31B-it-basin-b-Q2_K \
       publish/basin_b_v1/gemma-4-31B-it-basin-b-Q2_K.gguf
   huggingface-cli upload morphicode_jp/gemma-4-31B-it-basin-b-Q2_K \
       publish/basin_b_v1/gemma-4-31B-it-basin-b-Q2_K.gguf.md5
   # README.md upload
   ```

3. **X thread 投稿** (5 分)
   - `X_THREAD.md` 内の Tweet 1-4 を順次投稿
   - Tweet 1 を固定ツイートに設定
   - thread 内に GitHub URL + HF URL 含める

4. **Reddit r/LocalLLaMA 投稿** (5 分)
   - title: 上記 draft 参照
   - body: HF + GitHub link + 簡潔な要約

5. **Zenn 記事公開** (1-2 日後)

## 検証 checklist (公開前最終)

- [ ] `apply_basin_b.py` が dry-run でエラーなく動く
- [ ] patched GGUF の MD5 が `.md5` ファイルと一致
- [ ] HF README に X / GitHub link 含む
- [ ] GitHub README に HF link 含む
- [ ] X thread に HF + GitHub link 両方含む
- [ ] `apply_basin_b.py` の Author 行が `@morphicode_jp`
- [ ] HF README の base_model frontmatter が正しい
- [ ] LICENSE に Gemma Terms 言及あり
- [ ] CITATION.cff の author/email/repo URL が正しい

## ファイル統計

```
publish/basin_b_v1/
├── apply_basin_b.py            ~5 KB  (CLI tool)
├── HF_README.md                ~7 KB  (model card)
├── GITHUB_README.md            ~5 KB  (repo readme)
├── X_THREAD.md                 ~5 KB  (4 tweets draft)
├── REPRODUCE.md                ~3 KB  (reproduction guide)
├── LICENSE                     ~1 KB  (Apache 2.0)
├── CITATION.cff                ~1 KB  (citation metadata)
├── requirements.txt            ~30 B  (gguf, numpy)
├── INDEX.md                    ~3 KB  (this file)
├── gemma-...-basin-b-Q2_K.gguf 12.6 GB (patched model)
└── gemma-...-basin-b-Q2_K.gguf.md5  64 B (checksum)
```

Total disk: ~12.6 GB (mostly GGUF)
Total upload: ~12.6 GB (HF) + ~25 KB (GitHub)
