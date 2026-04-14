"""
Analyze trigger-token sensitivity for refusal behavior.
Example:
python scripts/analyze_trigger_tokens.py --queries data/queries.json --responses data/baseline_outputs.json --manual_labels data/baseline_refusal_labels.json --out outputs/trigger_token_report.md
"""
import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lib.refusal import classify_refusal


def compile_patterns(tokens):
    compiled = []
    for token in tokens:
        escaped = re.escape(token)
        # Word-boundary matching for single words; plain phrase matching otherwise.
        if " " in token:
            pattern = re.compile(escaped, re.IGNORECASE)
        else:
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        compiled.append((token, pattern))
    return compiled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default="data/queries.json")
    p.add_argument("--responses", default="data/baseline_outputs.json")
    p.add_argument("--triggers", default="data/trigger_tokens.json")
    p.add_argument("--manual_labels", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.queries, encoding="utf-8") as f:
        queries = json.load(f)
    with open(args.responses, encoding="utf-8") as f:
        responses = json.load(f)
    with open(args.triggers, encoding="utf-8") as f:
        triggers = json.load(f)

    query_by_id = {int(x["id"]): x for x in queries}
    manual = {}
    if args.manual_labels:
        with open(args.manual_labels, encoding="utf-8") as f:
            manual = {int(k): v for k, v in json.load(f).items()}

    patterns = compile_patterns(triggers)
    stats = {token: {"n": 0, "refusals": 0, "hard": 0, "soft": 0, "examples": []} for token in triggers}

    for item in responses:
        qid = int(item.get("id", -1))
        query = query_by_id.get(qid, {})
        instruction = query.get("instruction", "")
        label = manual.get(qid, classify_refusal(item.get("response", "")))
        is_refusal = label in {"hard", "soft"}

        for token, pattern in patterns:
            if pattern.search(instruction):
                row = stats[token]
                row["n"] += 1
                if is_refusal:
                    row["refusals"] += 1
                if label == "hard":
                    row["hard"] += 1
                if label == "soft":
                    row["soft"] += 1
                if len(row["examples"]) < 3:
                    row["examples"].append({"id": qid, "label": label, "instruction": instruction})

    ranked = []
    for token, row in stats.items():
        if row["n"] == 0:
            continue
        rate = 100.0 * row["refusals"] / row["n"]
        ranked.append((token, row["n"], row["refusals"], row["hard"], row["soft"], rate, row["examples"]))
    ranked.sort(key=lambda x: (-x[5], -x[1], x[0]))

    print("Trigger token sensitivity (sorted by refusal rate):")
    for token, n, refusals, hard, soft, rate, _ in ranked:
        print(f"- {token}: rate={rate:.1f}% (n={n}, refusals={refusals}, hard={hard}, soft={soft})")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Trigger Token Sensitivity Report",
            "",
            f"- Queries: `{args.queries}`",
            f"- Responses: `{args.responses}`",
            f"- Trigger list: `{args.triggers}`",
            f"- Manual labels: `{args.manual_labels}`" if args.manual_labels else "- Manual labels: not used",
            "",
            "| Trigger token | N | Refusals | Hard | Soft | Refusal rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for token, n, refusals, hard, soft, rate, _ in ranked:
            lines.append(f"| {token} | {n} | {refusals} | {hard} | {soft} | {rate:.1f}% |")

        lines.extend(["", "## Sample matched queries", ""])
        for token, _, _, _, _, _, examples in ranked:
            lines.append(f"### `{token}`")
            if not examples:
                lines.append("- No examples")
                continue
            for ex in examples:
                lines.append(f"- id={ex['id']} ({ex['label']}): {ex['instruction']}")
            lines.append("")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nMarkdown report written to {out}")


if __name__ == "__main__":
    main()
