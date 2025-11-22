#!/usr/bin/env python3
"""Create a smaller, label-balanced subset of the POJ-104 CC dataset.

By default, this script reads the full dataset from the finetune_data.zip
payload downloaded by setup.sh (under downloads/zenodo_8393793 by default,
respecting CONCORD_DOWNLOAD_DIR). If that download is missing, it falls back to
data_processing/finetune_data/poj104_cc. The reduced sample is written to
data_processing/finetune_data/poj104_cc_small with up to 100 lines per split
and at least two examples per label so the negative-sampling dataloader can
operate without errors.
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


def collect_label_counts(path: Path) -> Counter:
    counts = Counter()
    with path.open() as fh:
        for line in fh:
            obj = json.loads(line)
            counts[obj["label"]] += 1
    return counts


def sample_split(split: str, input_dir: Path, output_dir: Path, limit: int, min_per_label: int):
    src = input_dir / f"{split}.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist")

    label_counts = collect_label_counts(src)
    if len(label_counts) < 2:
        raise ValueError(f"{split}: need at least two labels, found {len(label_counts)}")

    per_label_cap = max(min_per_label, math.ceil(limit / len(label_counts)))

    counts = defaultdict(int)
    selected = []
    with src.open() as fh:
        for line in fh:
            obj = json.loads(line)
            label = obj["label"]
            if counts[label] >= per_label_cap:
                continue
            counts[label] += 1
            selected.append(line.rstrip("\n"))
            if len(selected) >= limit:
                break

    if len(selected) < limit:
        raise ValueError(f"{split}: only collected {len(selected)} rows (target {limit}, cap {per_label_cap})")

    min_seen = min(counts.values())
    if min_seen < min_per_label:
        raise ValueError(
            f"{split}: a label has only {min_seen} examples after sampling; "
            f"increase --limit or lower --min-per-label"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / f"{split}.jsonl"
    with dst.open("w") as out:
        for line in selected:
            out.write(line + "\n")

    return {
        "split": split,
        "lines": len(selected),
        "labels": len(counts),
        "min_per_label": min_seen,
        "max_per_label": max(counts.values()),
        "per_label_cap": per_label_cap,
        "dst": dst,
    }


def parse_args():
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    download_root = Path(os.environ.get("CONCORD_DOWNLOAD_DIR", project_root / "downloads" / "zenodo_8393793"))
    # setup.sh unzips finetune_data.zip into <download_root>/finetune_data/data/poj104_cc
    downloaded_input_dir = download_root / "finetune_data" / "data" / "poj104_cc"
    repo_input_dir = base_dir / "finetune_data" / "poj104_cc"
    default_input_dir = downloaded_input_dir if downloaded_input_dir.exists() else repo_input_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help=(
            "Directory containing train/valid/test.jsonl "
            "(default: <CONCORD_DOWNLOAD_DIR>/finetune_data/poj104_cc from setup.sh; "
            "falls back to data_processing/finetune_data/poj104_cc)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "finetune_data" / "poj104_cc_small",
        help="Where to write the sampled splits (default: data_processing/finetune_data/poj104_cc_small)",
    )
    parser.add_argument("--limit", type=int, default=100, help="Number of rows to keep per split (default: 100)")
    parser.add_argument(
        "--min-per-label",
        type=int,
        default=2,
        help="Minimum examples per label to allow (guards negative sampling)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Which splits to sample (default: train valid test)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for split in args.splits:
        stats = sample_split(split, input_dir, output_dir, args.limit, args.min_per_label)
        print(
            f"{split}: wrote {stats['lines']} rows across {stats['labels']} labels "
            f"(per-label between {stats['min_per_label']}-{stats['max_per_label']}, cap {stats['per_label_cap']}) "
            f"-> {stats['dst']}"
        )


if __name__ == "__main__":
    main()
