"""
Orchestrate RAG index build, dataset generation, GPU inference, and evaluation exports.

Designed for one or two commands from repo root, writing:
  - outputs/rag_index/          (optional --build_rag_index)
  - data/mmlu_eval_100.json     (via build_mmlu_subset.py)
  - data/ctf_eval_50.json       (+ manifest via build_ctf_eval_subset.py)
  - outputs/experiment_manifest.json
  - EXPERIMENTS.md              (regenerated summary table)

GPU inference requires explicit model flags (same pitfall as run_prompt_ablation.py on Windows):
  python scripts/run_experiment_suite.py --infer_extra "--model_id unsloth/llama-3-8b-instruct-bnb-4bit --load_in_4bit --device cuda"

If --infer_extra is omitted, the suite still builds datasets/index and writes a manifest with inference skipped.

Full feature set (all retrieval modes + all benchmark JSONs) in one flag:

  python scripts/run_experiment_suite.py --preset full --build_rag_index \\
    --infer_extra "--model_id unsloth/llama-3-8b-instruct-bnb-4bit --load_in_4bit --device cuda"

`--preset full` implies: build FAISS index, `--ensure_all_datasets`, and
`--modes baseline_queries,rag_tfidf,rag_faiss,mmlu,ctf`. Add `--with_prompt_ablation` for prompt profiles (slow).
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
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(
    cmd: List[str],
    cwd: Path = REPO_ROOT,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    print("\n$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _git_commit() -> Optional[str]:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return p.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _render_experiments_md(manifest: Dict[str, Any]) -> str:
    lines = [
        "# Experiment log (auto-generated)",
        "",
        "Regenerate with `python scripts/run_experiment_suite.py` (see README).",
        "",
        "| UTC timestamp | Host | GPU | Commit | Run name | Command | Output JSON | Refusal % | 95% CI | paired Δ pp | MMLU acc % |",
        "|---|---|---|---|---|---|---|---:|---|---:|---:|",
    ]
    for run in manifest.get("runs", []):
        m = run.get("metrics") or {}
        ci = ""
        if m.get("refusal_ci_low_pct") is not None:
            ci = f"{m.get('refusal_ci_low_pct'):.1f}–{m.get('refusal_ci_high_pct'):.1f}%"
        lines.append(
            "| {ts} | {host} | {gpu} | `{commit}` | {name} | `{cmd}` | `{outp}` | {rate} | {ci} | {pd} | {mmlu} |".format(
                ts=run.get("timestamp_utc", ""),
                host=run.get("hostname", ""),
                gpu=run.get("gpu_name", ""),
                commit=(run.get("git_commit") or "")[:8],
                name=run.get("name", ""),
                cmd=(run.get("command", "") or "").replace("|", "\\|")[:120],
                outp=run.get("output_json", ""),
                rate="" if m.get("refusal_rate_pct") is None else f"{m['refusal_rate_pct']:.1f}",
                ci=ci,
                pd="" if m.get("paired_delta_pp") is None else f"{m['paired_delta_pp']:.2f}",
                mmlu="" if m.get("mmlu_accuracy_pct") is None else f"{m['mmlu_accuracy_pct']:.1f}",
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="outputs/experiment_suite", help="Per-run JSON/Markdown artifacts.")
    p.add_argument("--build_rag_index", action="store_true", help="Run build_rag_index.py first.")
    p.add_argument("--corpus", default="data/rag_corpus.json")
    p.add_argument("--rag_index_dir", default="outputs/rag_index")
    p.add_argument("--force_datasets", action="store_true", help="Rebuild MMLU/CTF JSON even if present.")
    p.add_argument(
        "--ensure_all_datasets",
        action="store_true",
        help="Build MMLU + CTF files even if those modes are disabled (default: only build datasets needed for --modes).",
    )
    p.add_argument("--skip_inference", action="store_true", help="Only datasets + index + dry manifest.")
    p.add_argument(
        "--infer_extra",
        default="",
        help='Extra args for inference.py (quoted), e.g. --model_id ... --load_in_4bit --device cuda',
    )
    p.add_argument("--lora_adapter", default=None, help="Optional path passed to inference.py.")
    p.add_argument(
        "--modes",
        default="baseline_queries,rag_faiss,mmlu,ctf",
        help="Comma list: baseline_queries, rag_tfidf, rag_faiss, mmlu, ctf",
    )
    p.add_argument("--with_prompt_ablation", action="store_true", help="Also run run_prompt_ablation.py (slow).")
    p.add_argument("--manifest_path", default="outputs/experiment_manifest.json")
    p.add_argument("--experiments_md", default="EXPERIMENTS.md")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--preset",
        choices=("none", "full"),
        default="none",
        help="full = enable --build_rag_index, --ensure_all_datasets, and all default modes (incl. rag_tfidf).",
    )
    args = p.parse_args()

    if args.preset == "full":
        args.build_rag_index = True
        args.ensure_all_datasets = True
        args.modes = "baseline_queries,rag_tfidf,rag_faiss,mmlu,ctf"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    infer_tokens: List[str] = []
    if args.infer_extra.strip():
        infer_tokens = shlex.split(args.infer_extra, posix=os.name != "nt")

    modes = {x.strip() for x in args.modes.split(",") if x.strip()}

    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME", ""),
        "git_commit": _git_commit(),
        "python": py,
        "gpu_name": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "runs": [],
    }

    def add_run(
        name: str,
        command: str,
        output_json: Optional[str],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        manifest["runs"].append(
            {
                "name": name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "hostname": manifest["hostname"],
                "gpu_name": manifest.get("gpu_name", ""),
                "git_commit": manifest.get("git_commit"),
                "command": command,
                "output_json": output_json or "",
                "metrics": metrics or {},
            }
        )

    # --- RAG index ---
    if args.build_rag_index:
        cmd = [
            py,
            str(REPO_ROOT / "scripts" / "build_rag_index.py"),
            "--corpus",
            args.corpus,
            "--output_dir",
            args.rag_index_dir,
            "--seed",
            str(args.seed),
        ]
        _run(cmd)
        add_run("build_rag_index", " ".join(cmd), None, {"index_dir": str(Path(args.rag_index_dir).resolve())})

    # --- Datasets (only when needed, unless --ensure_all_datasets) ---
    mmlu_path = REPO_ROOT / "data" / "mmlu_eval_100.json"
    need_mmlu = "mmlu" in modes or args.ensure_all_datasets
    if need_mmlu and (args.force_datasets or not mmlu_path.exists()):
        cmd = [py, str(REPO_ROOT / "scripts" / "build_mmlu_subset.py"), "--out", str(mmlu_path), "--seed", str(args.seed)]
        _run(cmd)
    if need_mmlu:
        add_run("ensure_mmlu_eval_100", f"{py} scripts/build_mmlu_subset.py", str(mmlu_path), {})

    ctf_path = REPO_ROOT / "data" / "ctf_eval_50.json"
    ctf_man = REPO_ROOT / "data" / "ctf_eval_50_manifest.json"
    need_ctf = "ctf" in modes or args.ensure_all_datasets
    if need_ctf and (args.force_datasets or not ctf_path.exists()):
        cmd = [
            py,
            str(REPO_ROOT / "scripts" / "build_ctf_eval_subset.py"),
            "--out",
            str(ctf_path),
            "--manifest_out",
            str(ctf_man),
            "--seed",
            str(args.seed),
        ]
        _run(cmd)
    if need_ctf:
        add_run("ensure_ctf_eval_50", f"{py} scripts/build_ctf_eval_subset.py", str(ctf_path), {})

    mmlu_queries = out_dir / "mmlu_queries_for_inference.json"
    cmd = [
        py,
        str(REPO_ROOT / "scripts" / "mcq_to_queries.py"),
        "--mcq",
        str(mmlu_path),
        "--out",
        str(mmlu_queries),
    ]
    if "mmlu" in modes:
        if not mmlu_path.exists():
            raise SystemExit(f"Missing {mmlu_path}; enable mmlu in --modes and/or run build_mmlu_subset.py first.")
        _run(cmd)
        add_run("mcq_to_queries_mmlu", " ".join(cmd), str(mmlu_queries), {})

    rag_tfidf_out = out_dir / "queries_rag_tfidf.json"
    rag_faiss_out = out_dir / "queries_rag_faiss.json"
    if "rag_tfidf" in modes:
        cmd = [
            py,
            str(REPO_ROOT / "scripts" / "retrieval_augment_queries.py"),
            "--mode",
            "tfidf",
            "--queries",
            str(REPO_ROOT / "data" / "queries.json"),
            "--corpus",
            args.corpus,
            "--out",
            str(rag_tfidf_out),
            "--top_k",
            "3",
        ]
        _run(cmd)
        add_run("retrieval_augment_tfidf", " ".join(cmd), str(rag_tfidf_out), {})
    if "rag_faiss" in modes:
        idx = Path(args.rag_index_dir)
        if not (idx / "index.faiss").exists():
            raise SystemExit(
                f"FAISS index missing under {idx}. Run with --build_rag_index or build manually first."
            )
        cmd = [
            py,
            str(REPO_ROOT / "scripts" / "retrieval_augment_queries.py"),
            "--mode",
            "embedding_faiss",
            "--queries",
            str(REPO_ROOT / "data" / "queries.json"),
            "--index_dir",
            str(idx),
            "--out",
            str(rag_faiss_out),
            "--top_k",
            "3",
        ]
        _run(cmd)
        add_run("retrieval_augment_faiss", " ".join(cmd), str(rag_faiss_out), {})

    if args.skip_inference or not infer_tokens:
        if not infer_tokens and not args.skip_inference:
            print("Note: --infer_extra empty; skipping all inference. Pass model/GPU flags to run generation.")
        Path(args.manifest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        Path(args.experiments_md).write_text(_render_experiments_md(manifest), encoding="utf-8")
        print(f"Wrote {args.manifest_path} and {args.experiments_md}")
        return

    def run_inference(name: str, queries_path: Path, out_json: Path) -> None:
        cmd = [
            py,
            str(REPO_ROOT / "scripts" / "inference.py"),
            "--queries",
            str(queries_path),
            "--out",
            str(out_json),
            "--seed",
            str(args.seed),
            "--save_run_config",
        ]
        if args.lora_adapter:
            cmd.extend(["--lora_adapter", args.lora_adapter])
        cmd.extend(infer_tokens)
        _run(cmd)
        add_run(name, " ".join(cmd), str(out_json), {})

    manual = REPO_ROOT / "data" / "baseline_refusal_labels.json"
    tiers = REPO_ROOT / "data" / "query_tiers.json"
    triggers = REPO_ROOT / "data" / "trigger_tokens.json"
    cyber_queries = REPO_ROOT / "data" / "queries.json"

    resp_baseline = out_dir / "responses_queries_baseline.json"
    resp_rag_tfidf = out_dir / "responses_rag_tfidf.json"
    resp_rag_faiss = out_dir / "responses_rag_faiss.json"
    resp_mmlu = out_dir / "responses_mmlu.json"
    resp_ctf = out_dir / "responses_ctf.json"

    if "baseline_queries" in modes:
        run_inference("inference_baseline_queries", cyber_queries, resp_baseline)
    if "rag_tfidf" in modes:
        run_inference("inference_rag_tfidf", rag_tfidf_out, resp_rag_tfidf)
    if "rag_faiss" in modes:
        run_inference("inference_rag_faiss", rag_faiss_out, resp_rag_faiss)
    if "mmlu" in modes:
        run_inference("inference_mmlu", mmlu_queries, resp_mmlu)
    if "ctf" in modes:
        run_inference("inference_ctf", ctf_path, resp_ctf)

    # --- Evaluations ---
    def eval_refusal_json(name: str, inp: Path, use_manual: bool) -> Dict[str, Any]:
        metrics_path = out_dir / f"{inp.stem}_refusal_metrics.json"
        cmd = [
            py,
            str(REPO_ROOT / "scripts" / "eval_refusal_rate.py"),
            "--input",
            str(inp),
            "--tiers",
            str(tiers),
            "--save_json",
            str(metrics_path),
            "--seed",
            str(args.seed),
        ]
        if use_manual:
            cmd.extend(["--manual_labels", str(manual)])
        _run(cmd)
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        add_run(name, " ".join(cmd), str(inp), data)
        return data

    if "baseline_queries" in modes and resp_baseline.exists():
        eval_refusal_json("eval_refusal_baseline", resp_baseline, True)
        trig_out = out_dir / "trigger_baseline.md"
        cmd = [
            py,
            str(REPO_ROOT / "scripts" / "analyze_trigger_tokens.py"),
            "--queries",
            str(cyber_queries),
            "--responses",
            str(resp_baseline),
            "--triggers",
            str(triggers),
            "--manual_labels",
            str(manual),
            "--out",
            str(trig_out),
        ]
        _run(cmd)
        add_run("analyze_trigger_baseline", " ".join(cmd), str(resp_baseline), {})

    if "rag_tfidf" in modes and resp_rag_tfidf.exists():
        eval_refusal_json("eval_refusal_rag_tfidf", resp_rag_tfidf, True)
    if "rag_faiss" in modes and resp_rag_faiss.exists():
        eval_refusal_json("eval_refusal_rag_faiss", resp_rag_faiss, True)
        if "baseline_queries" in modes and resp_baseline.exists():
            paired_json = out_dir / "paired_baseline_vs_rag_faiss.json"
            cmd = [
                py,
                str(REPO_ROOT / "scripts" / "eval_paired_bootstrap.py"),
                "--baseline",
                str(resp_baseline),
                "--candidate",
                str(resp_rag_faiss),
                "--manual_labels",
                str(manual),
                "--manual_scope",
                "both",
                "--save_json",
                str(paired_json),
            ]
            _run(cmd)
            paired_payload = json.loads(paired_json.read_text(encoding="utf-8"))
            ci = paired_payload.get("bootstrap_ci_delta_pp") or [None, None]
            metrics = {
                "paired_delta_pp": paired_payload.get("mean_delta_candidate_minus_baseline_pp"),
                "paired_ci_low_pp": ci[0],
                "paired_ci_high_pp": ci[1] if len(ci) > 1 else None,
            }
            add_run("eval_paired_baseline_vs_rag_faiss", " ".join(cmd), str(paired_json), metrics)

    if "mmlu" in modes and resp_mmlu.exists():
        mmlu_eval_json = out_dir / "mmlu_scores.json"
        cmd = [
            py,
            str(REPO_ROOT / "scripts" / "eval_mcq_probe.py"),
            "--mcq",
            str(mmlu_path),
            "--responses",
            str(resp_mmlu),
            "--save_json",
            str(mmlu_eval_json),
        ]
        _run(cmd)
        scores = json.loads(mmlu_eval_json.read_text(encoding="utf-8"))
        add_run("eval_mcq_mmlu", " ".join(cmd), str(resp_mmlu), {"mmlu_accuracy_pct": scores.get("accuracy_pct")})

    if "ctf" in modes and resp_ctf.exists():
        eval_refusal_json("eval_refusal_ctf_heuristic", resp_ctf, False)

    if args.with_prompt_ablation:
        extra = args.infer_extra.strip()
        if not extra:
            raise SystemExit("--with_prompt_ablation requires --infer_extra for GPU/model flags.")
        cmd = [py, str(REPO_ROOT / "scripts" / "run_prompt_ablation.py"), "--infer_extra", extra]
        _run(cmd)
        add_run("run_prompt_ablation", " ".join(cmd), str(REPO_ROOT / "outputs" / "prompt_ablation"), {})

    Path(args.manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Path(args.experiments_md).write_text(_render_experiments_md(manifest), encoding="utf-8")
    print(f"\nDone. Wrote {args.manifest_path} and {args.experiments_md}")


if __name__ == "__main__":
    main()
