# Whisper transcription

`data/audio` に入れた `.wav` ファイルを、`openai/whisper-large-v3-turbo` で文字起こしして `data/text` に `.txt` として保存します。

## セットアップ

```bash
cd /home/kame-research/Documents/ProsodySum
python3 -m venv .venv
source .venv/bin/activate
pip install -r data/prog/requirements.txt
```

`.wav` は Python 側で直接読み込むため、通常は `ffmpeg` なしで実行できます。

話者分離は2通りあります。

- `--diarization-backend local`: 追加の gated model なしで、音響特徴のクラスタリングにより簡易推定します。
- `--diarization-backend pyannote`: `pyannote.audio` を使います。既定の `pyannote/speaker-diarization-3.1` は Hugging Face 側で利用許諾が必要な場合があります。

## 実行

```bash
python data/prog/transcribe.py --language ja
```

`data/audio/example.wav` を処理すると、`data/text/example.txt` が作られます。既存の `.txt` は上書きしません。上書きしたい場合は `--force` を付けてください。

標準では、ファインチューニング用に次の形式で出力します。

```text
[Speaker1] act: question | stance: negative | importance: high | text: 今回の対応は遅かったのではないですか。
[Speaker1] act: answer | stance: neutral | importance: high | text: 関係機関との調整に時間を要しました。
```

Whisper 単体では話者分離、発話行為、スタンス、重要度は確定できません。そのため、話者は標準で `Speaker1` として出力します。ラベルはルールベースで次の値を推定します。

- `act`: `question`, `answer`, `explanation`, `concern`, `decision`, `action_item`
- `stance`: `positive`, `neutral`, `negative`
- `importance`: `high`, `medium`, `low`

国会や会議音声のように文字起こし内に話者名が出る場合は、標準の `--speaker-attribution hybrid` で「奥田文夫」「高市さなえ」のような話者名を優先して付けます。音声クラスタリングは話者名が見つからない箇所の補助として使います。

```bash
python data/prog/transcribe.py --language ja --force
```

元の文字起こしだけを保存したい場合:

```bash
python data/prog/transcribe.py --language ja --output-format plain
```

2人の話者を交互に仮置きしたい場合:

```bash
python data/prog/transcribe.py --language ja --speaker-mode alternate --num-speakers 2
```

追加ダウンロードなしの簡易話者分離を使う場合:

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

`local` は pyannote より精度は落ちますが、Hugging Face の gated model 許諾なしで動きます。話者数が分かっている場合は `--num-speakers` を指定してください。

GPU で回す場合:

```bash
chmod +x data/prog/run_gpu_local.sh
data/prog/run_gpu_local.sh
```

このスクリプトは `torch.cuda.is_available()` を確認してから、`--device cuda --diarization-backend local` で実行します。Codex の通常サンドボックス内では `/dev/nvidia*` が見えないことがあるため、その場合はサンドボックス外の端末か、許可付き実行で回してください。

`act`, `stance`, `importance` を OSS LLM で推定したい場合:

```bash
python data/prog/transcribe.py \
  --label-existing-text \
  --labeler-backend hf-llm \
  --labeler-model-id Qwen/Qwen2.5-3B-Instruct \
  --speaker-attribution text \
  --device cuda \
  --force
```

`gpt-oss-20b` を試す場合は `--labeler-backend gpt-oss --labeler-model-id openai/gpt-oss-20b` を指定できます。ただし、この環境の `transformers 4.57.x` では `gpt-oss-20b` の MXFP4 実行に追加 kernel 互換が必要です。Whisper と同じ環境を壊さず運用するなら、まず `hf-llm` で軽量 instruct モデルを後段ラベル付けに使う方が安定します。

pyannote で音声から話者分離したい場合:

```bash
export HF_TOKEN=...
python data/prog/transcribe.py --language ja --diarize --diarization-backend pyannote --num-speakers 2 --force
```

pyannote で話者数が不明な場合は `--num-speakers 0` を指定すると、`pyannote.audio` 側の推定に任せます。`local` では `--num-speakers 0` は2人として扱います。

ラベル推定を止めて固定値で出したい場合:

```bash
python data/prog/transcribe.py \
  --language ja \
  --no-infer-act \
  --no-infer-stance \
  --no-infer-importance \
  --default-act explanation \
  --default-stance neutral \
  --default-importance medium
```

既存の `data/text/*.txt` にラベルだけ付けたい場合:

```bash
python data/prog/transcribe.py --label-existing-text --force
```

この場合、元の `.txt` は残し、`train_013.labeled.txt` のような別ファイルを書き出します。

既存の `data/text/*.json` から、文字起こしをやり直さずに `.txt` だけ再生成したい場合:

```bash
python data/prog/transcribe.py \
  --format-existing-json \
  --speaker-attribution hybrid \
  --force
```

`--labeler-backend hf-llm` と組み合わせると、Whisper を再実行せずに OSS LLM でラベルだけ再推定できます。

文の途中で行が分かれる場合は、Qwen で意味上の1文ごとに分割できます。

```bash
python data/prog/transcribe.py \
  --format-existing-json \
  --sentence-splitter qwen \
  --speaker-attribution hybrid \
  --device cuda \
  --force
```

Qwen 分割は `--sentence-splitter-model-id` でモデルを変更できます。標準は `Qwen/Qwen2.5-3B-Instruct` です。

1行が長すぎる場合は、最大文字数を変更できます。

```bash
python data/prog/transcribe.py --language ja --max-line-chars 80
```

タイムスタンプ付きの JSON も保存したい場合:

```bash
python data/prog/transcribe.py --language ja --timestamps
```

`--timestamps --diarize` を併用すると、JSON に `diarization` として話者区間も保存します。
