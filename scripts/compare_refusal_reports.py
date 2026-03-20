"""
Compare refusal rates across multiple response files.
Example:
python scripts/compare_refusal_reports.py --inputs outputs/prompt_ablation/*.json --out outputs/prompt_ablation/summary.md
"""
import argparse
import json
from pathlib import Path

from eval_refusal_rate import classify_refusal


def compute_rate(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    labels = [classify_refusal(item.get("response", "")) for item in data]
    n = len(labels)
    hard = sum(1 for x in labels if x == "hard")
    soft = sum(1 for x in labels if x == "soft")
    refusals = hard + soft
    rate = 100.0 * refusals / n if n else 0.0
    return n, hard, soft, refusals, rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = []
    for input_path in args.inputs:
        n, hard, soft, refusals, rate = compute_rate(input_path)
        rows.append((Path(input_path).name, n, hard, soft, refusals, rate))

    rows.sort(key=lambda x: x[-1])
    print("Profile comparison (lower refusal rate is better):")
    for row in rows:
        print(
            f"- {row[0]}: rate={row[5]:.1f}% (n={row[1]}, hard={row[2]}, soft={row[3]}, refusals={row[4]})"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Prompt Comparison Report",
            "",
            "| File | N | Hard | Soft | Refusals | Refusal Rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for file_name, n, hard, soft, refusals, rate in rows:
            lines.append(f"| {file_name} | {n} | {hard} | {soft} | {refusals} | {rate:.1f}% |")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nMarkdown summary written to {out}")


if __name__ == "__main__":
    main()
