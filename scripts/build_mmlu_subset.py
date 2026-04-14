"""
Build a stratified MMLU subset (default 100 questions) compatible with mcq_to_queries / eval_mcq_probe.

Uses Hugging Face `cais/mmlu` per-subject configs (excludes `all` and `auxiliary_train`).
Allocation: as uniform as possible across subjects (larger remainder subjects come first alphabetically).

Output schema (list):
  id, topic, question, choices {A..D}, answer (letter), mmlu_subject

Example:
  python scripts/build_mmlu_subset.py --out data/mmlu_eval_100.json --total 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

_LETTERS = ["A", "B", "C", "D"]


def _row_to_item(
    row: Dict[str, Any], qid: int, subject: str
) -> Dict[str, Any]:
    choices_list = row["choices"]
    if len(choices_list) != 4:
        raise ValueError("Expected 4 choices")
    choices = {L: str(choices_list[i]) for i, L in enumerate(_LETTERS)}
    ans_idx = int(row["answer"])
    return {
        "id": qid,
        "topic": subject,
        "question": str(row["question"]).strip(),
        "choices": choices,
        "answer": _LETTERS[ans_idx],
        "mmlu_subject": subject,
    }


def _allocate(total: int, n_subjects: int) -> List[int]:
    base = total // n_subjects
    rem = total % n_subjects
    return [base + (1 if i < rem else 0) for i in range(n_subjects)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/mmlu_eval_100.json")
    p.add_argument("--total", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from datasets import get_dataset_config_names, load_dataset  # noqa: PLC0415

    subjects = sorted(
        c
        for c in get_dataset_config_names("cais/mmlu")
        if c not in ("all", "auxiliary_train")
    )
    n = len(subjects)
    if args.total < n:
        raise SystemExit(f"--total must be >= number of subjects ({n}) for stratification")

    rng = random.Random(args.seed)
    counts = _allocate(args.total, n)

    out: List[Dict[str, Any]] = []
    next_id = 100_001

    for subject, k in zip(subjects, counts):
        ds = load_dataset("cais/mmlu", subject, split="test")
        n_available = len(ds)
        if n_available < k:
            raise RuntimeError(
                f"Subject {subject!r} needs {k} rows but test split has only {n_available}. "
                "Lower --total or adjust allocation logic."
            )
        indices = list(range(n_available))
        rng.shuffle(indices)
        pick = sorted(indices[:k])  # stable order after random subset for reproducible file diff
        for idx in pick:
            row = ds[idx]
            out.append(_row_to_item(row, next_id, subject))
            next_id += 1

    if len(out) != args.total:
        raise RuntimeError(f"internal: expected {args.total} items, got {len(out)}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(out)} MMLU items to {path}")


if __name__ == "__main__":
    main()
