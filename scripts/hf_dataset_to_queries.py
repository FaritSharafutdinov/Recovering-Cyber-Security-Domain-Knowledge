"""
Sample a larger instruction dataset from Hugging Face and export queries.json-compatible rows.

Use this to move beyond the fixed 50-query set toward a broader benchmark-style evaluation set.

Example (downloads data on first run):
  python scripts/hf_dataset_to_queries.py \\
    --dataset yahma/alpaca-cleaned --split train --n 120 \\
    --field instruction --keyword security --out data/hf_security_queries.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="yahma/alpaca-cleaned")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--field", default="instruction", help="Field to treat as the user instruction.")
    p.add_argument(
        "--keyword",
        default=None,
        help="If set, only keep rows whose instruction matches this regex (case-insensitive).",
    )
    p.add_argument("--out", default="data/hf_sample_queries.json")
    args = p.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    kw = re.compile(args.keyword, re.IGNORECASE) if args.keyword else None

    out = []
    next_id = 1
    for row in ds:
        text = (row.get(args.field) or "").strip()
        if not text:
            continue
        if kw and not kw.search(text):
            continue
        out.append({"id": next_id, "instruction": text, "system": None})
        next_id += 1
        if len(out) >= args.n:
            break

    if len(out) < args.n and kw:
        print(
            f"Warning: only {len(out)} rows matched keyword after scanning the split; "
            "relax --keyword or increase scan range in code if needed."
        )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(out)} queries to {path}")


if __name__ == "__main__":
    main()
