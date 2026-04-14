"""
Score multiple-choice probe accuracy from model responses (forgetting / general capability).

Workflow (reuses inference.py, no duplicate model code):
  python scripts/mcq_to_queries.py --mcq data/general_capability_mcq.json --out outputs/mcq_queries.json
  python scripts/inference.py --queries outputs/mcq_queries.json --out outputs/mcq_responses.json
  python scripts/eval_mcq_probe.py --mcq data/general_capability_mcq.json --responses outputs/mcq_responses.json

Run the same pipeline after LoRA to compare accuracy drop (catastrophic forgetting hypothesis).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional


_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def extract_choice(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.search(r"(?:answer|choice)\s*:\s*([ABCD])\b", ln, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    allm = _LETTER_RE.findall(text)
    if not allm:
        return None
    return allm[-1].upper()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mcq", default="data/general_capability_mcq.json")
    p.add_argument("--responses", required=True)
    p.add_argument("--save_json", default=None)
    args = p.parse_args()

    with open(args.mcq, encoding="utf-8") as f:
        mcq = json.load(f)
    gold: Dict[int, str] = {int(x["id"]): x["answer"].upper() for x in mcq}

    with open(args.responses, encoding="utf-8") as f:
        rows = json.load(f)

    correct = 0
    total = 0
    details = []
    for row in rows:
        qid = int(row.get("id", -1))
        if qid not in gold:
            continue
        pred = extract_choice(row.get("response", ""))
        g = gold[qid]
        ok = pred == g
        if ok:
            correct += 1
        total += 1
        details.append({"id": qid, "gold": g, "pred": pred, "correct": ok})

    acc = 100.0 * correct / total if total else 0.0
    print(f"MCQ probe: {correct}/{total} correct ({acc:.1f}%)")
    wrong = [d for d in details if not d["correct"]]
    if wrong:
        print("Mismatches (id, gold, pred):")
        for d in wrong[:20]:
            print(f"  - {d['id']}: gold={d['gold']} pred={d['pred']}")
        if len(wrong) > 20:
            print(f"  ... and {len(wrong) - 20} more")

    if args.save_json:
        outp = Path(args.save_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        payload = {"accuracy_pct": acc, "correct": correct, "total": total, "details": details}
        outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON summary to {outp}")


if __name__ == "__main__":
    main()
