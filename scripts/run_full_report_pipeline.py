"""
Single GPU-oriented entry point for thesis-style evidence: experiment suite (RAG + benchmarks),
prompt ablation, trigger-token analysis, LoRA sweeps (rank / modules / dataset size), optional
MMLU with adapter (forgetting), RAG queries + adapter, and a consolidated REPORT.md.

Does not write or regenerate data/rag_corpus.json (only reads it for index / TF-IDF).

Example (Windows PowerShell, from repo root):

  python scripts/run_full_report_pipeline.py ^
    --infer_extra "--model_id unsloth/llama-3-8b-instruct-bnb-4bit --load_in_4bit --device cuda"

Requires CUDA. Pass the same flags you use for inference.py (including --load_in_4bit when using 4-bit).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: List[str], cwd: Path = REPO_ROOT) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _parse_infer_extra(infer_extra: str) -> Tuple[str, List[str], bool]:
    """Return (model_id, full_token_list, load_in_4bit)."""
    tokens = shlex.split(infer_extra.strip(), posix=os.name != "nt")
    mid = ""
    use_4bit = False
    i = 0
    while i < len(tokens):
        if tokens[i] == "--model_id" and i + 1 < len(tokens):
            mid = tokens[i + 1]
            i += 2
            continue
        if tokens[i] == "--load_in_4bit":
            use_4bit = True
            i += 1
            continue
        i += 1
    if not mid:
        raise SystemExit(
            "--infer_extra must include --model_id <hf_id> (same as inference.py). "
            "For 8B runs add --load_in_4bit when using a bnb-4bit checkpoint."
        )
    return mid, tokens, use_4bit


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sweep_jobs(matrix_path: Path) -> List[Dict[str, Any]]:
    raw = _load_json(matrix_path)
    if not isinstance(raw, list):
        raise SystemExit("Matrix JSON must be a list of objects.")
    return raw


def _train_cmd(py: str, job: Dict[str, Any], model_id: str, load_in_4bit: bool) -> List[str]:
    name = job.get("name")
    if not name:
        raise SystemExit("Each sweep job needs a string 'name'.")
    out_dir = REPO_ROOT / "outputs" / "full_report" / "lora_train" / name
    parts = [
        py,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--model_name",
        model_id,
        "--data_path",
        str(REPO_ROOT / "data" / "baseline_outputs.json"),
        "--output_dir",
        str(out_dir),
        "--save_adapter",
    ]
    if load_in_4bit:
        parts.append("--load_in_4bit")
    skip = {"name"}
    for k, v in sorted(job.items()):
        if k in skip or v is None:
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                parts.append(flag)
        else:
            parts.extend([flag, str(v)])
    return parts


def _infer_cmd(
    py: str,
    queries: Path,
    out_json: Path,
    infer_tokens: List[str],
    lora_adapter: Optional[Path] = None,
) -> List[str]:
    cmd = [
        py,
        str(REPO_ROOT / "scripts" / "inference.py"),
        "--queries",
        str(queries),
        "--out",
        str(out_json),
        "--seed",
        "42",
        "--save_run_config",
    ]
    if lora_adapter is not None:
        cmd.extend(["--lora_adapter", str(lora_adapter)])
    cmd.extend(infer_tokens)
    return cmd


def _eval_refusal_cmd(py: str, inp: Path, out_json: Path, manual: Optional[Path]) -> List[str]:
    cmd = [
        py,
        str(REPO_ROOT / "scripts" / "eval_refusal_rate.py"),
        "--input",
        str(inp),
        "--tiers",
        str(REPO_ROOT / "data" / "query_tiers.json"),
        "--save_json",
        str(out_json),
        "--seed",
        "42",
    ]
    if manual is not None:
        cmd.extend(["--manual_labels", str(manual)])
    return cmd


def _write_report(
    out_md: Path,
    model_id: str,
    infer_extra: str,
    primary_job: str,
    sweep_rows: List[Dict[str, Any]],
    manifest_path: Path,
    skip_lora: bool,
) -> None:
    lines = [
        "# Full research pipeline report (auto-generated)",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Environment",
        "",
        f"- Base / inference model: `{model_id}`",
        f"- `inference.py` extra flags: `{infer_extra}`",
        f"- Primary LoRA job (forgetting / RAG+LoRA): `{primary_job}`",
        "",
        "## What this run covers (for your write-up)",
        "",
        "| Topic | Evidence in this repo | Artifact paths |",
        "|---|---|---|",
        "| **RAG (TF-IDF + FAISS)** vs baseline | Refusal rates + paired bootstrap baseline vs dense RAG | `outputs/experiment_suite/`, `EXPERIMENTS.md` |",
        "| **RAG vs fine-tuning** | RAG = retrieval + same base weights; fine-tuning = LoRA adapters — compare refusal/MMLU across conditions | LoRA JSON under `outputs/full_report/lora_infer/`; RAG responses `responses_rag_*.json` |",
        "| **Prompt engineering vs LoRA** | Prompt profiles (`run_prompt_ablation`) vs adapter-tuned generations on same `queries.json` | `outputs/prompt_ablation/`, `outputs/full_report/lora_infer/` |",
        "| **LoRA rank ablation** | `lora_rank_r8`, `r16`, `r32` in sweep | `outputs/full_report/lora_train/lora_rank_*` |",
        "| **LoRA module ablation** | Attention-only vs wider targets | `lora_modules_*` train dirs |",
        "| **Dataset size (LoRA)** | `n_train` 10 / 30 / 50 from `baseline_outputs.json` | `lora_data_n*` |",
        "| **Catastrophic forgetting** | MMLU accuracy base run (suite) vs MMLU with adapter | `outputs/experiment_suite/mmlu_scores.json` vs `outputs/full_report/mmlu_lora_scores.json` |",
        "| **Trigger-token analysis** | Conditional refusal by trigger keywords | `outputs/full_report/trigger_report.md` |",
        "| **Response taxonomy** | Rule-based answer-shape / usefulness proxy | `outputs/full_report/taxonomy_baseline.md` |",
        "| **Benchmarks** | MMLU subset + CTF-style prompts (manifest) | `data/mmlu_eval_100.json`, `data/ctf_eval_50.json`, suite responses |",
        "",
        "## Consolidated LoRA sweep (refusal on cyber queries)",
        "",
        "| Job | Refusal rate % | JSON metrics |",
        "|---|---:|---|",
    ]
    if skip_lora:
        lines.append("| *(LoRA sweep skipped)* | — | — |")
    else:
        for row in sweep_rows:
            rate = row.get("refusal_rate_pct")
            rate_s = f"{rate:.1f}" if isinstance(rate, (int, float)) else ""
            lines.append(
                f"| {row.get('job', '')} | {rate_s} | `{row.get('metrics_path', '')}` |"
            )
    lines.extend(
        [
            "",
            "## Experiment manifest (suite + eval)",
            "",
            f"See `{manifest_path.relative_to(REPO_ROOT)}` and root `EXPERIMENTS.md`.",
            "",
            "## Prompt ablation",
            "",
            "- Summary: `outputs/prompt_ablation/summary.md`",
            "- Plot: `outputs/prompt_ablation/refusal_rates.png`",
            "",
            "## Catastrophic forgetting (MMLU)",
            "",
            "- Base model (no adapter): `outputs/experiment_suite/mmlu_scores.json` (from the suite).",
        ]
    )
    if not skip_lora:
        lines.extend(
            [
                "- Same MMLU prompts with primary LoRA adapter: `outputs/full_report/mmlu_lora_scores.json`.",
                "",
                "## RAG + same LoRA adapter",
                "",
                "- Dense RAG prompts with adapter: `outputs/full_report/responses_rag_faiss_lora.json` and `outputs/full_report/rag_faiss_lora_refusal.json`.",
                "",
            ]
        )
    lines.extend(
        [
            "- Optional extra probe (not run automatically): `data/general_capability_mcq.json` → `mcq_to_queries.py` → `inference.py` → `eval_mcq_probe.py`.",
            "",
            "## Notes",
            "",
            "- `data/rag_corpus.json` is **read-only** in this pipeline (index + TF-IDF). It is never regenerated.",
            "- Re-run with a larger `--matrix` or edited `configs/report_full_sweep.json` for extra LoRA points.",
            "",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_md}")


def main() -> None:
    p = argparse.ArgumentParser(description="GPU-only full report pipeline (suite + LoRA + REPORT.md).")
    p.add_argument(
        "--infer_extra",
        required=True,
        help='Quoted args for inference.py, e.g. --model_id unsloth/llama-3-8b-instruct-bnb-4bit --load_in_4bit --device cuda',
    )
    p.add_argument("--corpus", default="data/rag_corpus.json", help="Read-only corpus for RAG index + TF-IDF.")
    p.add_argument("--matrix", default="configs/report_full_sweep.json", help="LoRA sweep job list.")
    p.add_argument(
        "--primary_lora_job",
        default="lora_rank_r16",
        help="Job name from matrix used for MMLU-with-adapter and RAG+LoRA inference.",
    )
    p.add_argument("--skip_lora_sweep", action="store_true", help="Only suite + reports (no train/infer LoRA grid).")
    p.add_argument(
        "--out_report",
        default="outputs/full_report/REPORT.md",
        help="Markdown index for your thesis/report.",
    )
    args = p.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. This pipeline is intended for GPU runs. "
            "Use a machine with an NVIDIA driver + CUDA-enabled PyTorch, then re-run."
        )

    model_id, infer_tokens, load_in_4bit = _parse_infer_extra(args.infer_extra)
    if load_in_4bit is False and ("8b" in model_id.lower() or "70b" in model_id.lower() or "13b" in model_id.lower()):
        print(
            "Warning: large model without --load_in_4bit in --infer_extra may OOM. "
            "Consider adding --load_in_4bit for training and inference.",
            file=sys.stderr,
        )

    py = sys.executable
    infer_extra_str = args.infer_extra.strip()
    suite_cmd = [
        py,
        str(REPO_ROOT / "scripts" / "run_experiment_suite.py"),
        "--preset",
        "full",
        "--with_prompt_ablation",
        "--corpus",
        args.corpus,
        "--infer_extra",
        infer_extra_str,
    ]
    _run(suite_cmd)

    ablation_dir = REPO_ROOT / "outputs" / "prompt_ablation"
    summary_md = ablation_dir / "summary.md"
    summary_json = ablation_dir / "summary.json"
    manual = REPO_ROOT / "data" / "baseline_refusal_labels.json"
    _run(
        [
            py,
            str(REPO_ROOT / "scripts" / "compare_refusal_reports.py"),
            "--input_dir",
            str(ablation_dir),
            "--manual_labels",
            str(manual),
            "--out",
            str(summary_md),
            "--out_json",
            str(summary_json),
        ]
    )
    _run(
        [
            py,
            str(REPO_ROOT / "scripts" / "plot_refusal_rates.py"),
            "--summary_json",
            str(summary_json),
            "--out",
            str(ablation_dir / "refusal_rates.png"),
        ]
    )

    exp_dir = REPO_ROOT / "outputs" / "experiment_suite"
    baseline_resp = exp_dir / "responses_queries_baseline.json"
    trig_out = REPO_ROOT / "outputs" / "full_report" / "trigger_report.md"
    if baseline_resp.exists():
        _run(
            [
                py,
                str(REPO_ROOT / "scripts" / "analyze_trigger_tokens.py"),
                "--queries",
                str(REPO_ROOT / "data" / "queries.json"),
                "--responses",
                str(baseline_resp),
                "--triggers",
                str(REPO_ROOT / "data" / "trigger_tokens.json"),
                "--manual_labels",
                str(manual),
                "--out",
                str(trig_out),
            ]
        )
        tax_out = REPO_ROOT / "outputs" / "full_report" / "taxonomy_baseline.md"
        _run(
            [
                py,
                str(REPO_ROOT / "scripts" / "eval_response_taxonomy.py"),
                "--input",
                str(baseline_resp),
                "--out",
                str(tax_out),
            ]
        )

    sweep_rows: List[Dict[str, Any]] = []
    primary_adapter: Optional[Path] = None
    matrix_path = (REPO_ROOT / args.matrix).resolve() if not Path(args.matrix).is_absolute() else Path(args.matrix)

    if not args.skip_lora_sweep:
        if not matrix_path.exists():
            raise SystemExit(f"Missing matrix file: {matrix_path}")
        jobs = _sweep_jobs(matrix_path)
        for job in jobs:
            _run(_train_cmd(py, job, model_id, load_in_4bit))
        infer_root = REPO_ROOT / "outputs" / "full_report" / "lora_infer"
        metrics_root = REPO_ROOT / "outputs" / "full_report" / "lora_metrics"
        infer_root.mkdir(parents=True, exist_ok=True)
        metrics_root.mkdir(parents=True, exist_ok=True)
        queries_cyber = REPO_ROOT / "data" / "queries.json"
        for job in jobs:
            name = job["name"]
            adapter = REPO_ROOT / "outputs" / "full_report" / "lora_train" / name / "adapter"
            out_j = infer_root / f"{name}_queries.json"
            _run(_infer_cmd(py, queries_cyber, out_j, infer_tokens, lora_adapter=adapter))
            met_j = metrics_root / f"{name}_refusal.json"
            _run(_eval_refusal_cmd(py, out_j, met_j, manual))
            data = _load_json(met_j)
            sweep_rows.append(
                {
                    "job": name,
                    "refusal_rate_pct": data.get("refusal_rate_pct"),
                    "metrics_path": str(met_j.relative_to(REPO_ROOT)),
                }
            )
            if name == args.primary_lora_job:
                primary_adapter = adapter

        if primary_adapter is None or not primary_adapter.exists():
            raise SystemExit(
                f"Primary LoRA job {args.primary_lora_job!r} not found after sweep. "
                f"Check --primary_lora_job matches a 'name' in {matrix_path}."
            )

        mmlu_queries = exp_dir / "mmlu_queries_for_inference.json"
        mmlu_mcq = REPO_ROOT / "data" / "mmlu_eval_100.json"
        if mmlu_queries.exists():
            mmlu_lora_resp = REPO_ROOT / "outputs" / "full_report" / "mmlu_responses_lora.json"
            _run(_infer_cmd(py, mmlu_queries, mmlu_lora_resp, infer_tokens, lora_adapter=primary_adapter))
            mmlu_scores = REPO_ROOT / "outputs" / "full_report" / "mmlu_lora_scores.json"
            _run(
                [
                    py,
                    str(REPO_ROOT / "scripts" / "eval_mcq_probe.py"),
                    "--mcq",
                    str(mmlu_mcq),
                    "--responses",
                    str(mmlu_lora_resp),
                    "--save_json",
                    str(mmlu_scores),
                ]
            )

        rag_q = exp_dir / "queries_rag_faiss.json"
        if rag_q.exists():
            rag_lora_out = REPO_ROOT / "outputs" / "full_report" / "responses_rag_faiss_lora.json"
            _run(_infer_cmd(py, rag_q, rag_lora_out, infer_tokens, lora_adapter=primary_adapter))
            rag_met = REPO_ROOT / "outputs" / "full_report" / "rag_faiss_lora_refusal.json"
            _run(_eval_refusal_cmd(py, rag_lora_out, rag_met, manual))

    manifest_path = REPO_ROOT / "outputs" / "experiment_manifest.json"
    out_report = REPO_ROOT / args.out_report if not Path(args.out_report).is_absolute() else Path(args.out_report)
    _write_report(
        out_report,
        model_id,
        infer_extra_str,
        args.primary_lora_job,
        sweep_rows,
        manifest_path,
        args.skip_lora_sweep,
    )


if __name__ == "__main__":
    main()
