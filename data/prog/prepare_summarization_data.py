#!/usr/bin/env python3
"""Build Baseline and Proposed summarization JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LABEL_LINE_RE = re.compile(
    r"^\[(?P<speaker>[^\]]+)\]\s+act:\s*(?P<act>[^|]+)\|\s*stance:\s*(?P<stance>[^|]+)"
    r"(?:\|\s*importance:\s*[^|]+)?\|\s*text:\s*(?P<text>.*)$"
)


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def read_json_text(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{path} does not contain a non-empty string field: text")
    return normalize_text(text)


def read_plain_from_labeled_text(path: Path) -> str:
    parts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = LABEL_LINE_RE.match(stripped)
        parts.append(match.group("text") if match else stripped)
    return normalize_text(" ".join(parts))


def read_proposed_text(path: Path) -> str:
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = LABEL_LINE_RE.match(stripped)
        if match:
            speaker = normalize_text(match.group("speaker"))
            act = normalize_text(match.group("act"))
            stance = normalize_text(match.group("stance"))
            text = normalize_text(match.group("text"))
            lines.append(f"[{speaker}] act: {act} | stance: {stance} | text: {text}")
        else:
            lines.append(stripped)
    return normalize_text("\n".join(lines))


def iter_ids(summary_dir: Path, split: str) -> list[str]:
    return sorted(path.stem for path in summary_dir.glob(f"{split}_*.txt"))


def build_records(data_dir: Path, split: str, condition: str, baseline_source: str) -> list[dict[str, str]]:
    text_dir = data_dir / "text"
    summary_dir = data_dir / "summary"
    records: list[dict[str, str]] = []

    for sample_id in iter_ids(summary_dir, split):
        summary_path = summary_dir / f"{sample_id}.txt"
        transcript_txt_path = text_dir / f"{sample_id}.txt"
        transcript_json_path = text_dir / f"{sample_id}.json"

        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        if not transcript_txt_path.exists():
            raise FileNotFoundError(transcript_txt_path)

        if condition == "baseline":
            if baseline_source == "json":
                if not transcript_json_path.exists():
                    raise FileNotFoundError(transcript_json_path)
                source = read_json_text(transcript_json_path)
            else:
                source = read_plain_from_labeled_text(transcript_txt_path)
        elif condition == "proposed":
            source = read_proposed_text(transcript_txt_path)
        else:
            raise ValueError(f"unknown condition: {condition}")

        summary = normalize_text(summary_path.read_text(encoding="utf-8"))
        if not source:
            raise ValueError(f"empty source for {sample_id}")
        if not summary:
            raise ValueError(f"empty summary for {sample_id}")

        records.append(
            {
                "id": sample_id,
                "split": split,
                "condition": condition,
                "input": f"要約: {source}",
                "summary": summary,
            }
        )

    return records


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=default_data_dir() / "summarization")
    parser.add_argument("--baseline-source", choices=("json", "txt"), default="json")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for condition in ("baseline", "proposed"):
        for split in args.splits:
            records = build_records(args.data_dir, split, condition, args.baseline_source)
            output_path = args.output_dir / condition / f"{split}.jsonl"
            write_jsonl(output_path, records)
            print(f"wrote {len(records):>3} records: {output_path}")


if __name__ == "__main__":
    main()
