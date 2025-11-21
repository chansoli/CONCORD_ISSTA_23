#!/usr/bin/env python3
"""Create a smaller, label-balanced subset of the CXG variant-detection dataset.

By default, reads from data_processing/finetune_data/cxg_vd and writes to
data_processing/finetune_data/cxg_vd_small with up to 200 rows per split and at
least a minimum number of examples per label to keep both classes represented.
"""

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def collect_label_counts(path: Path) -> Counter:
    counts = Counter()
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            counts[row["label"]] += 1
    return counts


def sample_split(split: str, input_dir: Path, output_dir: Path, limit: int, min_per_label: int):
    src = input_dir / f"{split}_func.csv"
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist")

    label_counts = collect_label_counts(src)
    if len(label_counts) < 2:
        raise ValueError(f"{split}: need at least two labels, found {len(label_counts)}")

    per_label_cap = max(min_per_label, math.ceil(limit / len(label_counts)))

    counts = defaultdict(int)
    selected = []
    with src.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = reader.fieldnames
        for row in reader:
            label = row["label"]
            if counts[label] >= per_label_cap:
                continue
            counts[label] += 1
            selected.append(row)
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
    dst = output_dir / f"{split}_func.csv"
    with dst.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(selected)

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=base_dir / "finetune_data" / "cxg_vd",
        help="Directory containing *_func.csv splits (default: data_processing/finetune_data/cxg_vd)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "finetune_data" / "cxg_vd_small",
        help="Where to write the sampled splits (default: data_processing/finetune_data/cxg_vd_small)",
    )
    parser.add_argument("--limit", type=int, default=200, help="Number of rows to keep per split (default: 200)")
    parser.add_argument(
        "--min-per-label",
        type=int,
        default=20,
        help="Minimum examples per label to allow (guards against class dropout)",
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
