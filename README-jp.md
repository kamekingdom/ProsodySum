# ProsodySum

ProsodySum は、日本語の会見・国会質疑・会議音声を対象に、文字起こし、発話ラベル付け、要約モデルの fine-tuning を行うための実験用リポジトリです。

`.wav` 音声を Whisper で文字起こしし、発話ごとに話者・発話行為・スタンス・重要度を付けたテキストを作成します。さらに、そのデータを使って `google/mt5-small` を fine-tuning し、通常の文字起こし入力と、`act` / `stance` ラベル付き入力の 2 条件を比較できます。

## ディレクトリ構成

```text
data/
  audio/          入力音声ファイル置き場
  prog/           文字起こし・データ作成・学習用スクリプト
  text/           文字起こし結果の .txt / .json
  summary/        要約ターゲット
  summarization/  Baseline / Proposed の学習用 JSONL
experiments/      実験メモ
```

文字起こしの中心スクリプトは `data/prog/transcribe.py` です。詳細な文字起こし手順は `data/prog/README.md` にあります。

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r data/prog/requirements.txt
```

通常の `.wav` 入力であれば、Python 側で直接読み込むため `ffmpeg` は基本的に不要です。

## 文字起こし

`data/audio` に `.wav` ファイルを置いて実行します。

```bash
python data/prog/transcribe.py --language ja
```

例として `data/audio/example.wav` を処理すると、`data/text/example.txt` が作成されます。既存の `.txt` は上書きしません。上書きする場合は `--force` を付けます。

```bash
python data/prog/transcribe.py --language ja --force
```

標準出力形式は、下流の fine-tuning や分析に使えるように、発話ごとのラベル付きテキストです。

```text
[Speaker1] act: question | stance: negative | importance: high | text: 今回の対応は遅かったのではないですか。
[Speaker1] act: answer | stance: neutral | importance: high | text: 関係機関との調整に時間を要しました。
```

素の文字起こしだけを保存したい場合は、次のように実行します。

```bash
python data/prog/transcribe.py --language ja --output-format plain
```

## 話者推定と話者分離

標準の話者推定モードは `hybrid` です。文字起こし内に現れる話者名を優先し、見つからない箇所では汎用の話者ラベルを使います。

追加の gated model を使わない簡易的なローカル話者分離は次のコマンドで実行できます。

```bash
python data/prog/transcribe.py \
  --language ja \
  --diarize \
  --diarization-backend local \
  --speaker-attribution hybrid \
  --num-speakers 2 \
  --timestamps \
  --force
```

pyannote を使う場合は Hugging Face のトークンを設定します。既定の pyannote モデルは、Hugging Face 側で利用許諾が必要な場合があります。

```bash
export HF_TOKEN=...
python data/prog/transcribe.py \
  --language ja \
  --diarize \
  --diarization-backend pyannote \
  --num-speakers 2 \
  --force
```

## GPU 実行

CUDA が使える環境では、補助スクリプトを使えます。

```bash
chmod +x data/prog/run_gpu_local.sh
data/prog/run_gpu_local.sh
```

このスクリプトは、ローカル話者分離、hybrid 話者推定、Qwen による文分割、タイムスタンプ出力、`--device cuda` を指定して実行します。

## 既存テキストへのラベル付け

既存の `data/text/*.txt` に対して、文字起こしをやり直さずにラベルだけ付ける場合:

```bash
python data/prog/transcribe.py --label-existing-text --force
```

Hugging Face の instruct モデルで `act`、`stance`、`importance` を推定する場合:

```bash
python data/prog/transcribe.py \
  --label-existing-text \
  --labeler-backend hf-llm \
  --labeler-model-id Qwen/Qwen2.5-3B-Instruct \
  --speaker-attribution text \
  --device cuda \
  --force
```

## JSON からテキストを再生成

既存の `data/text/*.json` から、Whisper を再実行せずに `.txt` を再生成できます。

```bash
python data/prog/transcribe.py \
  --format-existing-json \
  --speaker-attribution hybrid \
  --force
```

## mT5 要約モデルの Fine-Tuning

このリポジトリでは、`google/mt5-small` を使って、日本語の会見・国会質疑などの文字起こしから要約を生成するモデルを fine-tuning できます。

比較する条件は次の 2 つです。

- Baseline FT: 入力は文字起こしのみ
- Proposed FT: 入力は発話ごとに `act` / `stance` ラベルを付与した文字起こし
- 出力は両条件とも同じ要約ターゲット

要約ターゲットは `data/summary` に置きます。現在のデータは train 14 件、valid 3 件、test 3 件です。

## 学習用データの作成

次のコマンドで Baseline / Proposed それぞれの JSONL を作成します。

```bash
python data/prog/prepare_summarization_data.py
```

出力先:

```text
data/summarization/baseline/train.jsonl
data/summarization/baseline/valid.jsonl
data/summarization/baseline/test.jsonl
data/summarization/proposed/train.jsonl
data/summarization/proposed/valid.jsonl
data/summarization/proposed/test.jsonl
```

## Baseline の学習

```bash
python data/prog/finetune_mt5_summarizer.py \
  --condition baseline \
  --output-root runs/mt5-small \
  --local-files-only \
  --learning-rate 5e-5
```

## Proposed の学習

```bash
python data/prog/finetune_mt5_summarizer.py \
  --condition proposed \
  --output-root runs/mt5-small \
  --local-files-only \
  --learning-rate 5e-5
```

各実行では、fine-tuning 済みモデル、`test_predictions.jsonl`、`test_metrics.json` が `runs/mt5-small/<condition>/` に保存されます。`runs/` は容量が大きいため Git 管理外です。

評価指標は文字単位 ROUGE-L です。日本語に対して追加の分かち書きを使わずに比較できる、簡易で再現しやすい指標として使っています。

## LoRA Fine-Tuning

LoRA で学習する場合は `--tuning-mode lora` を付けます。full fine-tuning と同じ Baseline / Proposed 条件を、軽量な adapter 学習として実行できます。

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

既定の LoRA 設定は `r=8`、`alpha=16`、`dropout=0.05`、target modules は `q` と `v` です。出力先は `runs/mt5-small/lora/<condition>/` です。

学習済みモデルから test 予測だけを再生成したい場合:

```bash
python data/prog/finetune_mt5_summarizer.py \
  --condition baseline \
  --output-root runs/mt5-small \
  --local-files-only \
  --predict-only
```

## 初回実験結果

初回の Baseline / Proposed 比較は `experiments/mt5-small-baseline-vs-proposed.md` に記録しています。

LoRA を含めた比較は `experiments/mt5-small-lora-comparison.md` に記録しています。

| 条件 | test loss | char ROUGE-L |
| --- | ---: | ---: |
| Baseline FT | 8.9049 | 0.0648 |
| Proposed FT | 14.4272 | 0.0303 |

`--fp16` も試しましたが `NaN` loss が発生したため、記録している実験では fp32 と `learning-rate 5e-5` を使用しています。

現在のデータ数は非常に少ないため、この結果は最終的な性能評価ではなく、比較パイプラインの smoke test と初期 baseline として扱うのが妥当です。

## 補足

- 既定の Whisper モデル: `openai/whisper-large-v3-turbo`
- 対応音声形式: `.wav`
- 文字起こし出力先: `data/text`
- 要約ターゲット: `data/summary`
- 要約学習用 JSONL: `data/summarization`
- 詳細な文字起こし例: `data/prog/README.md`
