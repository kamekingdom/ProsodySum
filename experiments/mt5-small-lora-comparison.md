# mT5-small 要約実験: Full Fine-Tuning と LoRA の比較

実行日: 2026-06-07

## 目的

日本語の会見・国会質疑などの文字起こしから要約を生成するタスクで、`google/mt5-small` の full fine-tuning と LoRA fine-tuning を比較した。

入力条件は次の 2 種類で固定した。

- Baseline: 文字起こしのみを入力する
- Proposed: 発話ごとに `act` / `stance` ラベルを付与した文字起こしを入力する

出力は両条件とも同じ要約ターゲットを使用した。

## データ

使用データ:

- train: 14 件
- valid: 3 件
- test: 3 件
- 入力データ: `data/summarization`
- 参照要約: `data/summary`

データ数が非常に少ないため、本結果は最終的な性能評価ではなく、実験パイプラインの動作確認と初期比較として扱う。

## 実行条件

### Full Fine-Tuning

```bash
python data/prog/finetune_mt5_summarizer.py \
  --condition baseline \
  --output-root runs/mt5-small \
  --local-files-only \
  --learning-rate 5e-5
```

```bash
python data/prog/finetune_mt5_summarizer.py \
  --condition proposed \
  --output-root runs/mt5-small \
  --local-files-only \
  --learning-rate 5e-5
```

### LoRA Fine-Tuning

```bash
python data/prog/finetune_mt5_summarizer.py \
  --condition baseline \
  --tuning-mode lora \
  --output-root runs/mt5-small \
  --local-files-only \
  --learning-rate 1e-3
```

```bash
python data/prog/finetune_mt5_summarizer.py \
  --condition proposed \
  --tuning-mode lora \
  --output-root runs/mt5-small \
  --local-files-only \
  --learning-rate 1e-3
```

LoRA 設定:

- `r=8`
- `alpha=16`
- `dropout=0.05`
- target modules: `q`, `v`
- trainable parameters: 344,064
- trainable ratio: 0.1145%

`--fp16` は full fine-tuning で `NaN` loss が出たため、今回の比較では使っていない。

## 結果

| 学習方式 | 入力条件 | test loss | char ROUGE-L |
| --- | --- | ---: | ---: |
| Full FT | Baseline | 8.9049 | 0.0648 |
| Full FT | Proposed | 14.4272 | 0.0303 |
| LoRA FT | Baseline | 10.7541 | 0.1043 |
| LoRA FT | Proposed | 8.6775 | 0.0577 |

## 観察

今回の test set では、最も高い char ROUGE-L は `LoRA FT + Baseline` の 0.1043 だった。`LoRA FT + Proposed` は `Full FT + Proposed` より改善したが、Baseline 系には届かなかった。

LoRA は保存サイズが小さい。今回のローカル出力では full fine-tuning の各条件が約 7.9GB だったのに対し、LoRA の各条件は約 69MB だった。

一方で、生成文には反復が多く、要約品質としてはまだ不十分である。たとえば「国民の国民の...」「そして、そのことについて述べました...」のような繰り返しが残っている。これはデータ数が 20 件しかないこと、入力が長く noisy な文字起こしであること、mT5-small に対して教師データが少なすぎることが主な要因と考えられる。

サンプル別に、生成要約が具体的にどう変化したかは `experiments/mt5-small-output-change-analysis.md` に整理した。

## 暫定結論

現時点では、`act` / `stance` ラベル付き入力が要約性能を改善したとは言えない。少なくとも今回の極小データでは、Baseline 入力の方が安定している。

ただし、LoRA は full fine-tuning より保存サイズが小さく、今回の Baseline 条件では char ROUGE-L も上回ったため、今後の追加実験では LoRA を主軸にしてよい。

次に改善すべき点:

- train / valid / test の件数を増やす
- 文字起こし内の反復や「ご視聴ありがとうございました」などの混入を除去する
- 入力長が 1024 token で切られるため、長い質疑をチャンク分割して要約する
- generation の repetition penalty や no-repeat ngram 制約を導入する
- `act` / `stance` ラベルの精度を確認し、誤ラベルが多い場合は Proposed 条件を再作成する
