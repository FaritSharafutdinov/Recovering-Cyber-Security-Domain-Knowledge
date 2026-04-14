"""
Compare refusal rates across multiple response files.
Example:
python scripts/compare_refusal_reports.py --inputs outputs/prompt_ablation/*.json --out outputs/prompt_ablation/summary.md

PowerShell note: globs are not always expanded before Python sees them. Prefer:
  python scripts/compare_refusal_reports.py --input_dir outputs/prompt_ablation --out ...
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lib.refusal import classify_refusal


def compute_rate(path: str, manual: Optional[Dict[int, str]] = None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    labels = []
    for item in data:
        qid = int(item.get("id", -1))
        if manual and qid in manual:
            labels.append(manual[qid])
        else:
            labels.append(classify_refusal(item.get("response", "")))
    n = len(labels)
    hard = sum(1 for x in labels if x == "hard")
    soft = sum(1 for x in labels if x == "soft")
    refusals = hard + soft
    rate = 100.0 * refusals / n if n else 0.0
    return n, hard, soft, refusals, rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="One or more response JSON paths. On Windows PowerShell, pass explicit paths or use --input_dir.",
    )
    p.add_argument(
        "--input_dir",
        default=None,
        help="Directory containing *.json runs (e.g. outputs/prompt_ablation). Used when --inputs is empty or to avoid shell glob issues.",
    )
    p.add_argument("--out", default=None)
    p.add_argument(
        "--manual_labels",
        default=None,
        help="Optional JSON mapping query id -> {hard|soft|none} for consistent labeling across files.",
    )
    p.add_argument("--out_json", default=None, help="Optional JSON for plotting / dashboards.")
    args = p.parse_args()

    input_paths: list[str] = list(args.inputs)
    if args.input_dir:
        d = Path(args.input_dir)
        found = sorted(d.glob("*.json"))
        for p in found:
            # Avoid accidentally including aggregate summaries / run configs from the same folder.
            if p.name == "summary.json" or p.name.endswith("_inference_config.json"):
                continue
            input_paths.append(str(p))
    # De-dupe while preserving order
    seen = set()
    deduped: list[str] = []
    for p in input_paths:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    input_paths = deduped

    if not input_paths:
        raise SystemExit("No inputs: pass --inputs path1 path2 ... and/or --input_dir DIR with *.json files.")

    manual = None
    if args.manual_labels:
        with open(args.manual_labels, encoding="utf-8") as f:
            manual = {int(k): v for k, v in json.load(f).items()}

    rows = []
    for input_path in input_paths:
        n, hard, soft, refusals, rate = compute_rate(input_path, manual=manual)
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

    if args.out_json:
        jpath = Path(args.out_json)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"file": file_name, "n": n, "hard": hard, "soft": soft, "refusals": refusals, "refusal_rate_pct": rate}
            for file_name, n, hard, soft, refusals, rate in rows
        ]
        jpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON summary written to {jpath}")


if __name__ == "__main__":
    main()
