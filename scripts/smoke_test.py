"""
Fast health checks for the repo (no full LLM inference).

Runs lightweight subprocess steps: imports, offline demo, tiny RAG index,
TF-IDF + FAISS augment on 2 queries, NVD converter on a minimal feed, CTF builder (small),
eval scripts, compare/plot, experiment suite dry path, paired bootstrap smoke,
and ``py_compile`` on probe/pipeline drivers (no full LLM inference).

Usage (from repo root):
  python scripts/smoke_test.py

Exit code non-zero on first failure.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    smoke = REPO_ROOT / "outputs" / "smoke_ci"
    smoke.mkdir(parents=True, exist_ok=True)

    # 1) Core imports (no GPU model load)
    run(
        [
            PY,
            "-c",
            "import torch, transformers, datasets, peft; "
            "import faiss, ijson, sentence_transformers; "
            "import matplotlib; "
            "print('imports_ok', torch.__version__)",
        ]
    )

    # 2) Two tiny queries for augment / index smoke
    two_q = smoke / "two_queries.json"
    two_q.write_text(
        json.dumps(
            [
                {"id": 0, "instruction": "What is SQL injection and how do prepared statements help?", "system": None},
                {"id": 1, "instruction": "Summarize TLS 1.3 goals in one sentence.", "system": None},
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    # 3) Offline demo (eval + compare + plot, no model)
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "run_offline_demo.py"),
            "--outputs_dir",
            str(smoke / "offline_demo"),
        ]
    )

    # 4) Refusal JSON export
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "eval_refusal_rate.py"),
            "--input",
            str(REPO_ROOT / "data" / "baseline_outputs.json"),
            "--manual_labels",
            str(REPO_ROOT / "data" / "baseline_refusal_labels.json"),
            "--save_json",
            str(smoke / "baseline_refusal_metrics.json"),
        ]
    )

    # 5) TF-IDF augment (small corpus)
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "retrieval_augment_queries.py"),
            "--mode",
            "tfidf",
            "--queries",
            str(two_q),
            "--corpus",
            str(REPO_ROOT / "data" / "rag_corpus_education.json"),
            "--out",
            str(smoke / "queries_rag_tfidf.json"),
            "--top_k",
            "2",
        ]
    )

    # 6) Tiny FAISS index + dense augment
    idx_dir = smoke / "rag_index_mini"
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "build_rag_index.py"),
            "--corpus",
            str(REPO_ROOT / "data" / "rag_corpus_education.json"),
            "--output_dir",
            str(idx_dir),
        ]
    )
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "retrieval_augment_queries.py"),
            "--mode",
            "embedding_faiss",
            "--queries",
            str(two_q),
            "--index_dir",
            str(idx_dir),
            "--out",
            str(smoke / "queries_rag_faiss.json"),
            "--top_k",
            "2",
        ]
    )

    # 7) Minimal NVD-shaped JSON → corpus converter
    mini_nvd = smoke / "mini_nvdcve.json"
    mini_nvd.write_text(
        json.dumps(
            {
                "resultsPerPage": 1,
                "startIndex": 0,
                "totalResults": 1,
                "format": "NVD_CVE",
                "version": "2.0",
                "timestamp": "2026-01-01T00:00:00",
                "vulnerabilities": [
                    {
                        "cve": {
                            "id": "CVE-2099-0001",
                            "published": "2026-01-01T00:00:00",
                            "lastModified": "2026-01-01T00:00:00",
                            "vulnStatus": "Analyzed",
                            "references": [],
                            "descriptions": [
                                {
                                    "lang": "en",
                                    "value": "Synthetic test CVE for smoke_test.py only (not a real vulnerability record).",
                                }
                            ],
                        }
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "nvdcve_to_rag_corpus.py"),
            "--inputs",
            str(mini_nvd),
            "--out",
            str(smoke / "nvd_corpus.jsonl"),
            "--format",
            "jsonl",
            "--max-nvd",
            "5",
        ]
    )

    # 8) CTF subset (network, 3 items)
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "build_ctf_eval_subset.py"),
            "--out",
            str(smoke / "ctf_3.json"),
            "--manifest_out",
            str(smoke / "ctf_3_manifest.json"),
            "--total",
            "3",
            "--seed",
            "0",
        ]
    )

    # 9) Compare + plot from offline demo artifacts
    od = smoke / "offline_demo"
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "compare_refusal_reports.py"),
            "--inputs",
            str(od / "baseline_copy.json"),
            str(od / "candidate_masked.json"),
            "--out",
            str(smoke / "compare.md"),
            "--out_json",
            str(smoke / "compare.json"),
        ]
    )
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "plot_refusal_rates.py"),
            "--summary_json",
            str(smoke / "compare.json"),
            "--out",
            str(smoke / "rates.png"),
        ]
    )

    # 10) MCQ eval on synthetic responses
    mcq = json.loads((REPO_ROOT / "data" / "general_capability_mcq.json").read_text(encoding="utf-8"))
    syn = [{"id": x["id"], "instruction": "", "response": f"Answer: {x['answer']}"} for x in mcq[:5]]
    syn_path = smoke / "mcq_syn.json"
    syn_path.write_text(json.dumps(syn, indent=2), encoding="utf-8")
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "eval_mcq_probe.py"),
            "--mcq",
            str(REPO_ROOT / "data" / "general_capability_mcq.json"),
            "--responses",
            str(syn_path),
            "--save_json",
            str(smoke / "mcq_scores.json"),
        ]
    )

    # 11) experiment_suite dry (reuse mini index)
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "run_experiment_suite.py"),
            "--out_dir",
            str(smoke / "suite_dry"),
            "--rag_index_dir",
            str(idx_dir),
            "--corpus",
            str(REPO_ROOT / "data" / "rag_corpus_education.json"),
            "--skip_inference",
            "--modes",
            "rag_faiss",
            "--manifest_path",
            str(smoke / "experiment_manifest.json"),
            "--experiments_md",
            str(smoke / "EXPERIMENTS_SMOKE.md"),
        ]
    )

    # 12) Paired bootstrap (response-based labels; no GPU)
    run(
        [
            PY,
            str(REPO_ROOT / "scripts" / "eval_paired_bootstrap.py"),
            "--baseline",
            str(od / "baseline_copy.json"),
            "--candidate",
            str(od / "candidate_masked.json"),
            "--manual_scope",
            "none",
            "--seed",
            "42",
            "--save_json",
            str(smoke / "paired_smoke.json"),
        ]
    )

    # 13) Syntax-check orchestration / probe scripts (no execution of GPU probe)
    run(
        [
            PY,
            "-m",
            "py_compile",
            str(REPO_ROOT / "scripts" / "llama8b_rag_faiss_probe.py"),
            str(REPO_ROOT / "scripts" / "run_full_report_pipeline.py"),
            str(REPO_ROOT / "scripts" / "eval_paired_bootstrap.py"),
        ]
    )

    print("\nAll smoke checks passed. Artifacts under:", smoke.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
