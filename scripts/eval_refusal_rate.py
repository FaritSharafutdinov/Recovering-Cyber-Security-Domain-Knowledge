"""
Compute refusal rate from a responses JSON (id, instruction, response).
Run from repo root: python scripts/eval_refusal_rate.py [--input data/baseline_outputs.json]
"""
import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lib.bootstrap import bootstrap_ci
from lib.refusal import classify_refusal
from lib.tiers import load_query_tiers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/baseline_outputs.json")
    p.add_argument("--tiers", default="data/query_tiers.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_markdown", default=None)
    p.add_argument("--save_json", default=None, help="Optional machine-readable metrics (for automation).")
    p.add_argument(
        "--manual_labels",
        default=None,
        help="Optional JSON mapping query id -> {hard|soft|none} to override heuristic labels.",
    )
    p.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Evaluate only the first N rows of the responses JSON (order as in file).",
    )
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    if args.max_items is not None:
        data = data[: args.max_items]
    tiers = load_query_tiers(args.tiers)

    n = len(data)
    manual = {}
    if args.manual_labels:
        with open(args.manual_labels, encoding="utf-8") as f:
            manual = {int(k): v for k, v in json.load(f).items()}

    labels = []
    for item in data:
        qid = int(item.get("id", -1))
        label = manual.get(qid, classify_refusal(item.get("response", "")))
        labels.append(label)
    hard = sum(1 for label in labels if label == "hard")
    soft = sum(1 for label in labels if label == "soft")
    refusals = hard + soft
    binary = [1 if label != "none" else 0 for label in labels]
    rate = 100.0 * refusals / n if n else 0.0
    lo, hi = bootstrap_ci(binary, seed=args.seed)

    tier_stats = {}
    for item, label in zip(data, labels):
        qid = int(item.get("id", -1))
        tier = tiers.get(qid, "unknown")
        if tier not in tier_stats:
            tier_stats[tier] = {"n": 0, "refusals": 0, "hard": 0, "soft": 0}
        tier_stats[tier]["n"] += 1
        if label != "none":
            tier_stats[tier]["refusals"] += 1
        if label == "hard":
            tier_stats[tier]["hard"] += 1
        if label == "soft":
            tier_stats[tier]["soft"] += 1

    print(f"Total responses: {n}")
    print(f"Hard refusals: {hard}")
    print(f"Soft refusals: {soft}")
    print(f"Refusals (heuristic): {refusals}")
    print(f"Refusal rate: {rate:.1f}%")
    print(f"95% bootstrap CI: [{100*lo:.1f}%, {100*hi:.1f}%]")
    print("\nBy difficulty tier:")
    for tier in sorted(tier_stats):
        row = tier_stats[tier]
        tier_rate = 100.0 * row["refusals"] / row["n"] if row["n"] else 0.0
        print(
            f"- {tier}: n={row['n']}, refusals={row['refusals']} "
            f"(hard={row['hard']}, soft={row['soft']}), rate={tier_rate:.1f}%"
        )

    if args.save_markdown:
        out_path = Path(args.save_markdown)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Refusal Evaluation Report",
            "",
            f"- Input: `{args.input}`",
            f"- Manual labels: `{args.manual_labels}`" if args.manual_labels else "- Manual labels: not used",
            f"- Total responses: **{n}**",
            f"- Hard refusals: **{hard}**",
            f"- Soft refusals: **{soft}**",
            f"- Refusal rate: **{rate:.1f}%**",
            f"- 95% bootstrap CI: **[{100*lo:.1f}%, {100*hi:.1f}%]**",
            "",
            "## Difficulty Tier Breakdown",
            "",
            "| Tier | N | Refusals | Hard | Soft | Rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for tier in sorted(tier_stats):
            row = tier_stats[tier]
            tier_rate = 100.0 * row["refusals"] / row["n"] if row["n"] else 0.0
            lines.append(
                f"| {tier} | {row['n']} | {row['refusals']} | {row['hard']} | {row['soft']} | {tier_rate:.1f}% |"
            )
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nMarkdown report written to {out_path}")

    if args.save_json:
        jp = Path(args.save_json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n": n,
            "hard": hard,
            "soft": soft,
            "refusals": refusals,
            "refusal_rate_pct": rate,
            "bootstrap_ci_low_pct": 100.0 * lo,
            "bootstrap_ci_high_pct": 100.0 * hi,
            "tier_stats": tier_stats,
        }
        jp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON metrics written to {jp}")


if __name__ == "__main__":
    main()
