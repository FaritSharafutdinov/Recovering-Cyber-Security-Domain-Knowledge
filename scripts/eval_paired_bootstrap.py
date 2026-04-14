"""
Paired bootstrap on refusal-indicator differences between two response files (same query ids).

Use case: baseline vs LoRA (or prompt A vs prompt B) on identical items — uncertainty for the
*change* in refusal rate, not only each run in isolation.

Example:
  python scripts/eval_paired_bootstrap.py \\
    --baseline data/baseline_outputs.json \\
    --candidate outputs/lora_outputs.json \\
    --manual_labels data/baseline_refusal_labels.json \\
    --seed 42
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lib.bootstrap import bootstrap_mean_ci
from lib.refusal import classify_refusal, refusal_binary
from lib.responses import load_manual_labels, load_responses_by_id


def label_for(
    qid: int,
    response: str,
    manual: Dict[int, str],
    manual_scope: str,
) -> str:
    if manual_scope == "baseline_only":
        if qid in manual:
            return manual[qid]
        return classify_refusal(response)
    if manual_scope == "both":
        return manual[qid] if qid in manual else classify_refusal(response)
    return classify_refusal(response)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="Responses JSON (reference run, e.g. baseline).")
    p.add_argument("--candidate", required=True, help="Responses JSON (comparison run).")
    p.add_argument("--manual_labels", default=None)
    p.add_argument(
        "--manual_scope",
        choices=("baseline_only", "both", "none"),
        default="baseline_only",
        help="How to apply manual labels: only for baseline ids, for both files when key exists, or never.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--save_json", default=None)
    args = p.parse_args()

    manual = load_manual_labels(args.manual_labels) if args.manual_labels else {}
    base = load_responses_by_id(args.baseline)
    cand = load_responses_by_id(args.candidate)
    ids = sorted(set(base.keys()) & set(cand.keys()))
    if not ids:
        raise SystemExit("No overlapping query ids between baseline and candidate.")

    deltas: List[float] = []
    rows: List[Tuple[int, str, str, int, int, float]] = []
    for qid in ids:
        la = label_for(qid, base[qid].get("response", ""), manual, args.manual_scope)
        lb = label_for(qid, cand[qid].get("response", ""), manual, args.manual_scope)
        ra = refusal_binary(la)
        rb = refusal_binary(lb)
        # Positive => candidate refused more than baseline
        d = float(rb - ra)
        deltas.append(d)
        rows.append((qid, la, lb, ra, rb, d))

    n = len(deltas)
    mean_delta = sum(deltas) / n
    lo, hi = bootstrap_mean_ci(deltas, seed=args.seed, iterations=args.iterations)

    rate_base = 100.0 * sum(r[3] for r in rows) / n
    rate_cand = 100.0 * sum(r[4] for r in rows) / n

    print(f"Paired ids: {n}")
    print(f"Refusal rate baseline: {rate_base:.1f}%")
    print(f"Refusal rate candidate: {rate_cand:.1f}%")
    print(f"Mean paired delta (candidate - baseline): {100 * mean_delta:.2f} pp on binary refusal")
    print(
        f"95% bootstrap CI on mean delta: [{100 * lo:.2f} pp, {100 * hi:.2f} pp] "
        f"(B={args.iterations}, negative is fewer refusals in candidate)"
    )

    if args.save_json:
        outp = Path(args.save_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n": n,
            "refusal_rate_baseline_pct": rate_base,
            "refusal_rate_candidate_pct": rate_cand,
            "mean_delta_candidate_minus_baseline_pp": 100 * mean_delta,
            "bootstrap_ci_delta_pp": [100 * lo, 100 * hi],
            "per_id": [
                {"id": qid, "label_baseline": la, "label_candidate": lb, "delta": d}
                for qid, la, lb, ra, rb, d in rows
            ],
        }
        outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
