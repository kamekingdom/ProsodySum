#!/usr/bin/env python3
"""Fine-tune a seq2seq model for transcript summarization."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for key in ("id", "input", "summary"):
                if key not in record or not isinstance(record[key], str) or not record[key].strip():
                    raise ValueError(f"{path}:{line_number} missing non-empty string field {key!r}")
            records.append(record)
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def char_rouge_l(prediction: str, reference: str) -> float:
    pred_chars = list(prediction.strip())
    ref_chars = list(reference.strip())
    if not pred_chars or not ref_chars:
        return 0.0

    previous = [0] * (len(ref_chars) + 1)
    for pred_char in pred_chars:
        current = [0]
        for index, ref_char in enumerate(ref_chars, start=1):
            if pred_char == ref_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current

    lcs = previous[-1]
    precision = lcs / len(pred_chars)
    recall = lcs / len(ref_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sanitize_token_ids(token_ids: np.ndarray, pad_token_id: int) -> np.ndarray:
    return np.where(token_ids < 0, pad_token_id, token_ids)


def clean_generated_text(text: str) -> str:
    return re.sub(r"<extra_id_\d+>", "", text).strip()


@dataclass
class TokenizedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


class SummarizationDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, str]],
        tokenizer,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.examples: list[TokenizedExample] = []
        for record in records:
            model_inputs = tokenizer(
                record["input"],
                max_length=max_source_length,
                truncation=True,
            )
            labels = tokenizer(
                text_target=record["summary"],
                max_length=max_target_length,
                truncation=True,
            )
            self.examples.append(
                TokenizedExample(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    labels=labels["input_ids"],
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        return {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "labels": example.labels,
        }


def build_training_args(args: argparse.Namespace) -> Seq2SeqTrainingArguments:
    signature = inspect.signature(Seq2SeqTrainingArguments)
    eval_arg_name = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": True,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "predict_with_generate": True,
        "generation_max_length": args.generation_max_length,
        "generation_num_beams": args.num_beams,
        "report_to": [],
        "fp16": args.fp16,
    }
    kwargs[eval_arg_name] = "epoch"
    return Seq2SeqTrainingArguments(**kwargs)


def write_predictions(path: Path, records: list[dict[str, str]], predictions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record, prediction in zip(records, predictions, strict=True):
            output = {
                "id": record["id"],
                "prediction": prediction,
                "reference": record["summary"],
                "char_rouge_l": char_rouge_l(prediction, record["summary"]),
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("baseline", "proposed"), required=True)
    parser.add_argument("--tuning-mode", choices=("full", "lora"), default="full")
    parser.add_argument("--dataset-dir", type=Path, default=default_data_dir() / "summarization")
    parser.add_argument("--output-root", type=Path, default=Path("runs/mt5-small"))
    parser.add_argument("--model-id", default="google/mt5-small")
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--generation-max-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q", "v"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    args.output_dir = args.output_root / args.condition
    if args.tuning_mode == "lora":
        args.output_dir = args.output_root / "lora" / args.condition
    condition_dir = args.dataset_dir / args.condition
    train_records = load_jsonl(condition_dir / "train.jsonl")
    valid_records = load_jsonl(condition_dir / "valid.jsonl")
    test_records = load_jsonl(condition_dir / "test.jsonl")

    tokenizer_path = args.output_dir if args.predict_only else args.model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=args.local_files_only)

    if args.predict_only and args.tuning_mode == "lora":
        base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, local_files_only=args.local_files_only)
        model = PeftModel.from_pretrained(base_model, args.output_dir, local_files_only=args.local_files_only)
    else:
        model_path = args.output_dir if args.predict_only else args.model_id
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=args.local_files_only)

    if not args.predict_only and args.tuning_mode == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.generation_config.no_repeat_ngram_size = args.no_repeat_ngram_size
    model.generation_config.repetition_penalty = args.repetition_penalty
    model.generation_config.length_penalty = args.length_penalty
    train_dataset = SummarizationDataset(
        train_records,
        tokenizer,
        args.max_source_length,
        args.max_target_length,
    )
    valid_dataset = SummarizationDataset(
        valid_records,
        tokenizer,
        args.max_source_length,
        args.max_target_length,
    )
    test_dataset = SummarizationDataset(
        test_records,
        tokenizer,
        args.max_source_length,
        args.max_target_length,
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics(eval_predictions) -> dict[str, float]:
        generated_ids, label_ids = eval_predictions
        if isinstance(generated_ids, tuple):
            generated_ids = generated_ids[0]
        generated_ids = sanitize_token_ids(generated_ids, tokenizer.pad_token_id)
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
        predictions = [clean_generated_text(text) for text in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]
        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        scores = [char_rouge_l(prediction, reference) for prediction, reference in zip(predictions, references)]
        return {"char_rouge_l": float(np.mean(scores))}

    trainer = Seq2SeqTrainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if not args.predict_only:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)

    predictions_output = trainer.predict(test_dataset, metric_key_prefix="test")
    prediction_ids = predictions_output.predictions
    if isinstance(prediction_ids, tuple):
        prediction_ids = prediction_ids[0]
    predictions = tokenizer.batch_decode(
        sanitize_token_ids(prediction_ids, tokenizer.pad_token_id),
        skip_special_tokens=True,
    )
    predictions = [clean_generated_text(text) for text in predictions]
    write_predictions(args.output_dir / "test_predictions.jsonl", test_records, predictions)

    metrics = {key: float(value) for key, value in predictions_output.metrics.items()}
    metrics_path = args.output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
