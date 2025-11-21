#!/usr/bin/env python
"""Utility to convert simple JSONL clone data into CONCORD pre-training triplets.

The raw data is expected to be JSON lines with at least two fields:
  - "code": the source snippet as a string
  - "label": the class/clone ID (samples sharing a label are treated as positives)
Optional fields such as "index" are preserved but not required.

The generated file is JSON Lines (one JSON object per row) with keys:
    orig_code, positive_code, negative_code, tree_token_ids
which matches the expectation of scripts/run_concord_clone_aware_pretrain.py
once loaded via the datasets library.

The --tokenizer argument accepts either a Hugging Face tokenizer identifier/path
or a raw SentencePiece .model file (optionally living inside a directory).

Tree labels are emitted as zero-only sequences whose length matches the
(tokenized) original snippet. They act as placeholders so that CONCORD's
AST-recovery loss is effectively ignored (zeros are later masked out), but the
format remains compatible with the trainer.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

try:
    import sentencepiece as spm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    spm = None
from transformers import AutoTokenizer

def read_jsonl(path: Path) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            code = record.get("code")
            label = record.get("label")
            if code is None or label is None:
                # Skip malformed rows but continue processing.
                continue
            samples.append({"code": code, "label": str(label)})
    return samples


def group_by_label(samples: List[Dict[str, str]]) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    for sample in samples:
        buckets[sample["label"]].append(sample["code"])
    return buckets


def pick_negative_label(current: str, labels: List[str], rng: random.Random) -> str:
    choices = [label for label in labels if label != current]
    if not choices:
        raise ValueError("Need at least two distinct labels to sample negatives.")
    return rng.choice(choices)


def build_length_fn(tokenizer_path: str, max_length: int):
    """Return a callable that measures tokenized length with truncation."""

    def make_hf_length_fn(hf_tokenizer):
        def length_fn(text: str) -> int:
            tokenized = hf_tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=False,
                add_special_tokens=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            return len(tokenized["input_ids"])

        return length_fn

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return make_hf_length_fn(tokenizer)
    except OSError:
        sp_model_path = Path(tokenizer_path)
        if sp_model_path.is_dir():
            sp_model_path = sp_model_path / "tokenizer.model"
        if not sp_model_path.is_file():
            raise
        if spm is None:
            raise ImportError(
                "sentencepiece is required to load raw .model files; install it or pass a Hugging Face tokenizer"
            )
        sp_processor = spm.SentencePieceProcessor(model_file=str(sp_model_path))
        bos_id = sp_processor.bos_id()
        eos_id = sp_processor.eos_id()

        def length_fn(text: str) -> int:
            ids = list(sp_processor.encode(text, out_type=int))
            if bos_id != -1:
                ids.insert(0, bos_id)
            if eos_id != -1:
                ids.append(eos_id)
            if len(ids) > max_length:
                ids = ids[:max_length]
            return len(ids)

        return length_fn


def tree_stub(code: str, length_fn) -> str:
    seq_len = length_fn(code)
    interior_len = max(seq_len - 2, 0)
    if interior_len == 0:
        return ""  # Will still receive cls/sep from trainer logic
    return " ".join(["0"] * interior_len)


def build_triplets(
    buckets: Dict[str, List[str]],
    length_fn,
    rng: random.Random,
    negatives_per_sample: int,
):
    labels = [label for label, codes in buckets.items() if len(codes) >= 2]
    if len(labels) == 0:
        raise ValueError("No label has at least two samples to form positives.")

    for label in labels:
        codes = buckets[label]
        for idx, orig in enumerate(codes):
            pos_candidates = codes[:idx] + codes[idx + 1 :]
            if not pos_candidates:
                continue
            positive = rng.choice(pos_candidates)
            for _ in range(negatives_per_sample):
                neg_label = pick_negative_label(label, list(buckets.keys()), rng)
                negative = rng.choice(buckets[neg_label])
                yield (
                    orig,
                    positive,
                    negative,
                    tree_stub(orig, length_fn),
                )


def convert(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    length_fn = build_length_fn(args.tokenizer, args.max_length)
    samples = read_jsonl(Path(args.input))
    buckets = group_by_label(samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for orig, positive, negative, tree_ids in build_triplets(
            buckets=buckets,
            length_fn=length_fn,
            rng=rng,
            negatives_per_sample=args.negatives_per_sample,
        ):
            record = {
                "orig_code": orig,
                "positive_code": positive,
                "negative_code": negative,
                "tree_token_ids": tree_ids,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSONL clone data to CONCORD triplets.")
    parser.add_argument("--input", required=True, help="Path to the JSONL source file.")
    parser.add_argument("--output", required=True, help="Destination JSONL file.")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name/path or SentencePiece .model file used for computing lengths.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length; must match DataTrainingArguments.max_seq_length.",
    )
    parser.add_argument(
        "--negatives-per-sample",
        type=int,
        default=1,
        help="How many negative snippets to pair with each original sample.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()
