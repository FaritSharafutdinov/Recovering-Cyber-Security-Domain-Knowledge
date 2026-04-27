"""
Plot refusal rates from compare_refusal_reports.py --out_json output (evidence for reports).

Example:
  python scripts/compare_refusal_reports.py --inputs outputs/prompt_ablation/*.json \\
    --out_json outputs/prompt_ablation/summary.json
  python scripts/plot_refusal_rates.py --summary_json outputs/prompt_ablation/summary.json \\
    --out outputs/prompt_ablation/refusal_rates.png

Or build a three-bar comparison from experiment_suite responses (baseline / RAG TF-IDF / RAG FAISS):

  python scripts/plot_refusal_rates.py --experiment_dir outputs/experiment_suite \\
    --manual_labels data/baseline_refusal_labels.json --out figures/refusal_comparison.png
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compare_refusal_reports import compute_rate


def _rows_from_experiment_dir(
    experiment_dir: Path,
    manual_labels: Optional[Path],
) -> List[Dict[str, Any]]:
    manual = None
    if manual_labels and manual_labels.exists():
        manual = {int(k): v for k, v in json.loads(manual_labels.read_text(encoding="utf-8")).items()}
    triple = [
        ("responses_queries_baseline.json", "baseline"),
        ("responses_rag_tfidf.json", "rag_tfidf"),
        ("responses_rag_faiss.json", "rag_faiss"),
    ]
    rows: List[Dict[str, Any]] = []
    for fname, label in triple:
        path = experiment_dir / fname
        if not path.exists():
            continue
        n, hard, soft, refusals, rate = compute_rate(str(path), manual=manual)
        rows.append(
            {
                "file": label,
                "n": n,
                "hard": hard,
                "soft": soft,
                "refusals": refusals,
                "refusal_rate_pct": rate,
            }
        )
    if not rows:
        raise SystemExit(f"No response JSONs found under {experiment_dir} (expected baseline / RAG files).")
    return rows


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--summary_json", help="JSON from compare_refusal_reports.py --out_json")
    g.add_argument(
        "--experiment_dir",
        "--input_dir",
        dest="experiment_dir",
        help="Directory containing responses_queries_baseline.json, responses_rag_tfidf.json, responses_rag_faiss.json",
    )
    p.add_argument("--manual_labels", default=None, help="Used with --experiment_dir (same as eval_refusal_rate).")
    p.add_argument("--out", default="outputs/refusal_rates.png")
    args = p.parse_args()

    import matplotlib.pyplot as plt

    if args.summary_json:
        with open(args.summary_json, encoding="utf-8") as f:
            rows = json.load(f)
    else:
        exp = Path(args.experiment_dir)
        if not exp.is_absolute():
            exp = Path.cwd() / exp
        ml = Path(args.manual_labels) if args.manual_labels else None
        rows = _rows_from_experiment_dir(exp, ml)
    labels = [str(r["file"]).replace(".json", "") for r in rows]
    rates = [r["refusal_rate_pct"] for r in rows]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.45), 4))
    ax.bar(labels, rates, color="#3b5bdb")
    ax.set_ylabel("Refusal rate (%)")
    ax.set_title("Refusal rate by run / profile")
    ax.set_ylim(0, max(100, max(rates) * 1.1 if rates else 100))
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {outp}")
    # Mirror only the report bar chart (avoid copying smoke_ci scratch PNGs into reports/).
    try:
        if outp.name == "refusal_comparison.png":
            mirror = Path("reports") / "figures" / outp.name
            if mirror.resolve() != outp.resolve():
                mirror.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(outp, mirror)
                print(f"Also copied to {mirror}")
    except OSError as e:
        print(f"(skip mirror to reports/figures: {e})")


if __name__ == "__main__":
    main()
