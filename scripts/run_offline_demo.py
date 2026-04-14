"""
Offline/no-GPU end-to-end demo.

Goal: produce evidence artifacts (metrics, tables, plots) from in-repo data without
running heavyweight model inference. This keeps the project "non-raw" on a laptop,
and the same evaluation scripts can be re-used later on GPU runs by swapping the
response JSONs.

Run from repo root:
  python scripts/run_offline_demo.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_dir", default="outputs/offline_demo")
    p.add_argument("--baseline", default="data/baseline_outputs.json")
    p.add_argument("--queries", default="data/queries.json")
    p.add_argument("--tiers", default="data/query_tiers.json")
    p.add_argument("--manual_labels", default="data/baseline_refusal_labels.json")
    p.add_argument("--triggers", default="data/trigger_tokens.json")
    p.add_argument("--rag_corpus", default="data/rag_corpus.json")
    p.add_argument("--mcq", default="data/general_capability_mcq.json")
    args = p.parse_args()

    out_dir = Path(args.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Refusal metrics (baseline) + tier breakdown + bootstrap CI
    _run(
        [
            sys.executable,
            "scripts/eval_refusal_rate.py",
            "--input",
            args.baseline,
            "--tiers",
            args.tiers,
            "--manual_labels",
            args.manual_labels,
            "--save_markdown",
            str(out_dir / "baseline_refusal_report.md"),
        ]
    )

    # 2) Trigger-token sensitivity report (baseline)
    _run(
        [
            sys.executable,
            "scripts/analyze_trigger_tokens.py",
            "--queries",
            args.queries,
            "--responses",
            args.baseline,
            "--triggers",
            args.triggers,
            "--manual_labels",
            args.manual_labels,
            "--out",
            str(out_dir / "baseline_trigger_report.md"),
        ]
    )

    # 3) Automated response taxonomy table (baseline)
    _run(
        [
            sys.executable,
            "scripts/eval_response_taxonomy.py",
            "--input",
            args.baseline,
            "--manual_labels",
            args.manual_labels,
            "--out",
            str(out_dir / "baseline_taxonomy_report.md"),
        ]
    )

    # 4) RAG augmentation file (no model needed)
    rag_queries = out_dir / "queries_rag_augmented.json"
    _run(
        [
            sys.executable,
            "scripts/retrieval_augment_queries.py",
            "--queries",
            args.queries,
            "--corpus",
            args.rag_corpus,
            "--out",
            str(rag_queries),
            "--top_k",
            "3",
        ]
    )

    # 5) MCQ pipeline preparation (queries only; inference happens later on GPU)
    mcq_queries = out_dir / "mcq_queries.json"
    _run([sys.executable, "scripts/mcq_to_queries.py", "--mcq", args.mcq, "--out", str(mcq_queries)])

    # 6) Compare-report + plot: create a tiny "comparison set" from existing files
    #    so plotting/reporting is always demonstrated even offline.
    #    We build: baseline + a "masked" variant that blanks out responses (simulated worst-case).
    baseline_copy = out_dir / "baseline_copy.json"
    masked = out_dir / "candidate_masked.json"
    data = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    baseline_copy.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    masked_rows = []
    for row in data:
        masked_rows.append(
            {
                "id": row.get("id"),
                "instruction": row.get("instruction", ""),
                "response": "",
            }
        )
    masked.write_text(json.dumps(masked_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md = out_dir / "comparison_summary.md"
    summary_json = out_dir / "comparison_summary.json"
    _run(
        [
            sys.executable,
            "scripts/compare_refusal_reports.py",
            "--inputs",
            str(baseline_copy),
            str(masked),
            "--out",
            str(summary_md),
            "--out_json",
            str(summary_json),
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/plot_refusal_rates.py",
            "--summary_json",
            str(summary_json),
            "--out",
            str(out_dir / "refusal_rates.png"),
        ]
    )

    print("\nOffline demo finished.")
    print(f"Artifacts written under: {out_dir}")


if __name__ == "__main__":
    main()

