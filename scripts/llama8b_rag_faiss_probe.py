"""
Minimal GPU probe: first N queries ->
  (1) baseline inference on the slice,
  (2) dense RAG augment + inference,
  (3) eval_refusal_rate on the RAG file,
  (4) paired bootstrap (baseline vs RAG) with --manual_scope none so deltas reflect
      response-based lexical classification (static per-id manual maps trivialize paired deltas).

Requires an existing FAISS index (e.g. from run_experiment_suite --preset full).

Example (repo root):
  python scripts/llama8b_rag_faiss_probe.py --n_queries 25 \\
    --out outputs/llama3_probe.json \\
    --baseline_out outputs/llama3_probe_baseline.json \\
    --metrics_out outputs/llama3_probe_refusal_metrics.json \\
    --paired_out outputs/llama3_probe_paired.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_queries", type=int, default=25)
    p.add_argument("--corpus", default="data/rag_corpus.json")
    p.add_argument("--index_dir", default="outputs/rag_index")
    p.add_argument("--queries", default="data/queries.json")
    p.add_argument("--out", default="outputs/llama3_probe.json")
    p.add_argument("--baseline_out", default="outputs/llama3_probe_baseline.json")
    p.add_argument("--metrics_out", default="outputs/llama3_probe_refusal_metrics.json")
    p.add_argument("--paired_out", default="outputs/llama3_probe_paired.json")
    p.add_argument(
        "--infer_extra",
        default="--model_id unsloth/llama-3-8b-instruct-bnb-4bit --load_in_4bit --device cuda --max_new_tokens 384",
        help="Whitespace-separated args forwarded to inference.py (quoted string).",
    )
    p.add_argument("--manual_labels", default="data/baseline_refusal_labels.json")
    args = p.parse_args()

    qpath = REPO_ROOT / args.queries
    raw = json.loads(qpath.read_text(encoding="utf-8"))
    if len(raw) < args.n_queries:
        raise SystemExit(f"Need at least {args.n_queries} queries in {qpath}, found {len(raw)}")
    subset = raw[: args.n_queries]
    tmp_q = REPO_ROOT / "outputs" / "_probe_queries_subset.json"
    tmp_q.parent.mkdir(parents=True, exist_ok=True)
    tmp_q.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")

    infer_tokens = __import__("shlex").split(args.infer_extra.strip(), posix=sys.platform != "win32")
    baseline_path = REPO_ROOT / args.baseline_out
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "inference.py"),
            "--queries",
            str(tmp_q),
            "--out",
            str(baseline_path),
            "--seed",
            "42",
            "--save_run_config",
            *infer_tokens,
        ]
    )

    tmp_aug = REPO_ROOT / "outputs" / "_probe_rag_faiss_queries.json"
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "retrieval_augment_queries.py"),
            "--mode",
            "embedding_faiss",
            "--queries",
            str(tmp_q),
            "--index_dir",
            str(REPO_ROOT / args.index_dir),
            "--out",
            str(tmp_aug),
            "--top_k",
            "3",
        ]
    )

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "inference.py"),
            "--queries",
            str(tmp_aug),
            "--out",
            str(out_path),
            "--seed",
            "42",
            "--save_run_config",
            *infer_tokens,
        ]
    )

    metrics = REPO_ROOT / args.metrics_out
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "eval_refusal_rate.py"),
            "--input",
            str(out_path),
            "--manual_labels",
            str(REPO_ROOT / args.manual_labels),
            "--max_items",
            str(args.n_queries),
            "--save_json",
            str(metrics),
        ]
    )

    paired = REPO_ROOT / args.paired_out
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "eval_paired_bootstrap.py"),
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(out_path),
            "--manual_scope",
            "none",
            "--seed",
            "42",
            "--save_json",
            str(paired),
        ]
    )
    print(f"\nProbe complete.\n  Baseline: {baseline_path}\n  RAG FAISS: {out_path}\n  Refusal metrics: {metrics}\n  Paired (heuristic): {paired}")


if __name__ == "__main__":
    main()
