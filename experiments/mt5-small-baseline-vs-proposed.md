# google/mt5-small Baseline vs Proposed

Date: 2026-06-07

## Data

- Train: 14 samples
- Valid: 3 samples
- Test: 3 samples
- Target summaries: `data/summary/*.txt`
- Baseline input: transcript text from `data/text/*.json`
- Proposed input: transcript lines with `act` and `stance` labels from `data/text/*.txt`

Datasets were generated with:

```bash
python data/prog/prepare_summarization_data.py
```

## Training

Both conditions used:

```bash
python data/prog/finetune_mt5_summarizer.py \
  --condition <baseline|proposed> \
  --output-root runs/mt5-small \
  --local-files-only \
  --learning-rate 5e-5
```

`--fp16` was tested but produced `NaN` loss, so the reported runs use fp32.

## Test Results

| Condition | test loss | char ROUGE-L |
| --- | ---: | ---: |
| Baseline FT | 8.9049 | 0.0648 |
| Proposed FT | 14.4272 | 0.0303 |

## Notes

The current dataset is too small for a reliable quality claim. Generated summaries are often repetitive, so these numbers should be treated as a pipeline smoke test and initial baseline rather than a final experimental result.
