"""
Plot refusal rates from compare_refusal_reports.py --out_json output (evidence for reports).

Example:
  python scripts/compare_refusal_reports.py --inputs outputs/prompt_ablation/*.json \\
    --out_json outputs/prompt_ablation/summary.json
  python scripts/plot_refusal_rates.py --summary_json outputs/prompt_ablation/summary.json \\
    --out outputs/prompt_ablation/refusal_rates.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_json", required=True)
    p.add_argument("--out", default="outputs/refusal_rates.png")
    args = p.parse_args()

    import matplotlib.pyplot as plt

    with open(args.summary_json, encoding="utf-8") as f:
        rows = json.load(f)
    labels = [r["file"].replace(".json", "") for r in rows]
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


if __name__ == "__main__":
    main()
