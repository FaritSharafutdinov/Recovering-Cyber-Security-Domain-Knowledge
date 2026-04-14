"""
Convert a multiple-choice JSON file into the queries format used by scripts/inference.py.

Example:
  python scripts/mcq_to_queries.py --mcq data/general_capability_mcq.json --out outputs/mcq_queries.json
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mcq", required=True)
    p.add_argument("--out", default="outputs/mcq_queries.json")
    args = p.parse_args()

    with open(args.mcq, encoding="utf-8") as f:
        items = json.load(f)

    queries = []
    for item in items:
        qid = item["id"]
        lines = [
            "Answer with a single letter only: A, B, C, or D.",
            "Put the letter on the last line of your answer, for example: Answer: B",
            "",
            item["question"],
            "",
        ]
        for letter in sorted(item["choices"].keys()):
            lines.append(f"{letter}) {item['choices'][letter]}")
        queries.append({"id": qid, "instruction": "\n".join(lines), "system": None})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(queries)} MCQ prompts to {out}")


if __name__ == "__main__":
    main()
