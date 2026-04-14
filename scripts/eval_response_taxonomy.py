"""
Broader response taxonomy for error / failure analysis (extends hard vs soft refusal).

Categories per row:
  - refusal: hard | soft | none (same heuristics as eval_refusal_rate.py)
  - answer_shape: refusal | brief | substantive (length / structure proxy)
  - usefulness_proxy: 0..3 rule-based score (NOT a human rating; for scaling plots only)

For rigorous usefulness and hallucination labels, plug in an external LLM judge or manual review.

Example:
  python scripts/eval_response_taxonomy.py --input data/baseline_outputs.json --out outputs/taxonomy_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lib.refusal import classify_refusal


def usefulness_proxy(refusal: str, text: str) -> int:
    if refusal == "hard":
        return 0
    if refusal == "soft":
        return 1
    t = (text or "").strip()
    wc = len(t.split())
    if wc < 40:
        return 2
    if "```" in t or t.count("\n") >= 6:
        return 3
    return 2


def answer_shape(refusal: str, text: str) -> str:
    if refusal in {"hard", "soft"}:
        return "refusal"
    t = (text or "").strip()
    if len(t.split()) < 35:
        return "brief"
    return "substantive"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/baseline_outputs.json")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    counts = {"hard": 0, "soft": 0, "none": 0}
    shapes = {"refusal": 0, "brief": 0, "substantive": 0}
    for item in data:
        resp = item.get("response", "")
        lab = classify_refusal(resp)
        counts[lab] = counts.get(lab, 0) + 1
        sh = answer_shape(lab, resp)
        shapes[sh] = shapes.get(sh, 0) + 1

    n = len(data)
    print(f"Total: {n}")
    print(f"Refusal hard/soft/none: {counts['hard']}/{counts['soft']}/{counts['none']}")
    print(f"Shape refusal/brief/substantive: {shapes['refusal']}/{shapes['brief']}/{shapes['substantive']}")

    if args.out:
        lines = [
            "# Response taxonomy (automated)",
            "",
            f"- Input: `{args.input}`",
            f"- Total: **{n}**",
            "",
            "## Aggregate counts",
            "",
            "| Refusal | Count |",
            "|---|---:|",
        ]
        for k in ("hard", "soft", "none"):
            lines.append(f"| {k} | {counts[k]} |")
        lines.extend(["", "| Answer shape | Count |", "|---|---:|"])
        for k in ("refusal", "brief", "substantive"):
            lines.append(f"| {k} | {shapes[k]} |")
        lines.extend(["", "## Per-item table", "", "| id | refusal | shape | usefulness_proxy | wc |", "|---|---|:---:|---:|---:|"])
        for item in data:
            qid = item.get("id", "")
            resp = item.get("response", "")
            lab = classify_refusal(resp)
            sh = answer_shape(lab, resp)
            up = usefulness_proxy(lab, resp)
            wc = len(resp.split())
            lines.append(f"| {qid} | {lab} | {sh} | {up} | {wc} |")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
