# Recovering-Cyber-Security-Domain-Knowledge

Project: Recovering Cyber-Security Domain Knowledge via LoRA Fine-Tuning.

**Layout reference:** [STRUCTURE.md](STRUCTURE.md) (directories, `scripts/` index, `scripts/lib/`).

## Repository layout

- **scripts/** — CLIs (see [STRUCTURE.md](STRUCTURE.md)); shared code in **`scripts/lib/`** (`refusal`, `bootstrap`, `tiers`, `responses`).
- **configs/** — JSON manifests for LoRA/command sweeps (`train_sweep.example.json`, `seeds.example.json`).
- **data/** — `queries.json`, `baseline_outputs.json`, … **`rag_corpus.json`** is built from **NIST NVD** CVE JSON feeds (`nvdcve-2.0-*.json`) via `scripts/nvdcve_to_rag_corpus.py` (English description + CVE id + published date; heavy fields like configurations/CVSS JSON are dropped). Short **tutorial** snippets live in `data/rag_corpus_education.json` and are prepended when using `--merge-education`. For a **full** multi-year corpus use `--format jsonl` and point `--corpus` at that file. **Other generated files:** `mmlu_eval_100.json`, `ctf_eval_50.json` + manifest (see respective builders).
- **notebooks/** — `baseline_testing.ipynb` (if run from repo root, use paths like \texttt{data/queries.json})
- **reports/** — baseline report (PDF + LaTeX). To rebuild PDF: `pdflatex reports/baseline_report.tex`

## Contributions

- **Farit Sharafutdinov**: dataset, LoRA training pipeline, experiments
- **Grigorii Belayev**: baseline evaluation, inference scripts, report

## Dataset pipeline

- **data/queries.json**: 50 instruction-style prompts (cyber-security and sysadmin). Fields: `id`, `instruction`, `system` (optional).
- **data/baseline_outputs.json**: baseline model responses for the same queries (`id`, `instruction`, `response`). Produced by running inference on Llama-3-8B-Instruct.
- Preprocessing: no extra download step; data is in-repo. For full LoRA training we will use instruction–response pairs (e.g. from Cyber-Security-Instruct or from baseline_outputs as seed).

## How to run

From the **repository root**:

### Windows / RTX troubleshooting (read this first)

- **`outputs/` is gitignored** (see `.gitignore`). New files **still appear on disk**, but many IDEs hide ignored folders from the file tree. Check with Explorer or: `Get-ChildItem outputs -Recurse` (PowerShell) / `dir outputs /s` (cmd).
- **PowerShell glob pitfall**: `outputs/prompt_ablation/*.json` is **not always expanded** before Python sees it. Prefer:
  - `python scripts/compare_refusal_reports.py --input_dir outputs/prompt_ablation --out ... --out_json ...`
- **`run_prompt_ablation.py` must forward GPU flags** to `inference.py` (it calls inference as a subprocess). Example:

```powershell
python scripts/run_prompt_ablation.py --infer_extra "--model_id unsloth/llama-3-8b-instruct-bnb-4bit --load_in_4bit --device cuda"
```

- **End-to-end experiment suite (master script):** `python scripts/run_experiment_suite.py` writes `outputs/experiment_manifest.json` and refreshes [EXPERIMENTS.md](EXPERIMENTS.md). **`--preset full`** turns on index build, `--ensure_all_datasets`, and all modes (`baseline_queries`, `rag_tfidf`, `rag_faiss`, `mmlu`, `ctf`). Optional **`--with_prompt_ablation`** (slow). Typical GPU run:

```powershell
python scripts/run_experiment_suite.py --preset full --infer_extra "--model_id unsloth/llama-3-8b-instruct-bnb-4bit --load_in_4bit --device cuda"
```

- **Quick health check (no LLM inference):** `python scripts/smoke_test.py` — artifacts under `outputs/smoke_ci/`.

Dry run (index + datasets + augment, **skip** inference): add `--skip_inference`. To build `data/mmlu_eval_100.json` / `data/ctf_eval_50.json` without listing those modes: `--ensure_all_datasets`.

- **Rebuild RAG corpus from NVD feeds (streaming, large files safe):**
  ```powershell
  python scripts/nvdcve_to_rag_corpus.py --inputs data/nvdcve-2.0-2024.json data/nvdcve-2.0-2025.json data/nvdcve-2.0-2026.json `
    --out data/rag_corpus.json --format json --max-nvd 12000 --merge-education data/rag_corpus_education.json
  ```
  Use `--format jsonl` and omit `--max-nvd` to emit **all** CVE rows to e.g. `data/rag_corpus_nvd.jsonl` (recommended for full feeds; then pass that path to `build_rag_index.py` / `retrieval_augment_queries.py --corpus`).
- **Dense RAG (FAISS CPU + embeddings):** `python scripts/build_rag_index.py --corpus data/rag_corpus.json --output_dir outputs/rag_index` then augment with `python scripts/retrieval_augment_queries.py --mode embedding_faiss --index_dir outputs/rag_index --queries data/queries.json --out outputs/queries_rag_faiss.json`. Legacy sparse retrieval remains `--mode tfidf`.

- **MMLU / CTF evaluation JSON:** `python scripts/build_mmlu_subset.py` (100 items, stratified across `cais/mmlu` subjects; requires Hugging Face download) and `python scripts/build_ctf_eval_subset.py` (50 defensive prompts; metadata from [NYU CTF Bench](https://github.com/NYU-LLM-CTF/NYU_CTF_Bench) `test_dataset.json`, **GPL-2.0** — see manifest). These are **not** full CTF agent runs (no Docker solves).

- **If `python` behaves oddly** (Store alias / wrong interpreter), use the same interpreter explicitly, e.g. `py -3 ...` or your conda `python.exe` path, and verify with `where.exe python` / `python -c "import sys; print(sys.executable)"`.

- **Offline/no-GPU demo (recommended on a laptop)**: `python scripts/run_offline_demo.py` — generates evidence artifacts in `outputs/offline_demo/` without downloading or running a model:
  - refusal report with bootstrap CI + difficulty tiers
  - trigger-token sensitivity table
  - response taxonomy table
  - RAG-augmented queries JSON (for later GPU inference)
  - MCQ probe queries JSON (for later GPU inference)
  - a comparison plot (baseline vs a masked worst-case candidate) to demonstrate reporting/plotting flow

- **LoRA sanity check / small ablations**: `python scripts/train.py` — default: TinyLlama + LoRA on 5 examples, ~20 steps. Loss is logged each step. Knobs for proposal-style studies:
  - `--lora_r`, `--lora_alpha`, `--target_modules` (comma-separated, e.g. `q_proj,v_proj` vs `q_proj,k_proj,v_proj,o_proj`), `--n_train` (subset size from `data/baseline_outputs.json`), `--max_steps`, `--seed`, `--output_dir`.
  - To save an adapter for later inference: add `--save_adapter` (writes `<output_dir>/adapter/`).
- **Baseline inference**: `python scripts/inference.py` — runs inference on `data/queries.json`, writes `outputs/inference_outputs.json`.
  - Reproducibility controls: `--seed`, deterministic mode by default (`--do_sample` enables sampling), and explicit generation settings (`--temperature`, `--top_p`, `--max_new_tokens`).
  - Prompt controls: `--default_system` and `--override_system` for controlled prompt-engineering experiments.
  - Config snapshot: `--save_run_config` (omit value to write `<out_stem>_inference_config.json` next to `--out`).
  - Laptop/offline notes: use `--offline` to avoid any HuggingFace network calls (requires the model to be already cached locally).
  - To apply a trained LoRA adapter: pass `--lora_adapter <output_dir>/adapter`.

- **Manual evaluation protocol (no API keys)**: see `MANUAL_EVALUATION_PROTOCOL.md`.
  - Export a labeling sheet: `python scripts/export_manual_eval_sheet.py --queries data/queries.json --responses <RUN.json> --out outputs/manual_eval/sheet.json`
- **Paired refusal delta + CI**: `python scripts/eval_paired_bootstrap.py --baseline data/baseline_outputs.json --candidate outputs/candidate.json --manual_labels data/baseline_refusal_labels.json` — bootstrap on per-id refusal differences.
- **Train sweep plan (no GPU)**: `python scripts/render_train_sweep.py --matrix configs/train_sweep.example.json` — prints `train.py` command lines from a JSON job list.
- **LLM judge stub (optional / not required for grading)**: `python scripts/llm_judge.py ...` — placeholder only; prefer `MANUAL_EVALUATION_PROTOCOL.md` if you have no API budget.
- **Refusal + tier evaluation**: `python scripts/eval_refusal_rate.py` — reads a responses JSON and prints:
  - total / hard / soft refusals;
  - refusal rate with 95% bootstrap CI;
  - breakdown by difficulty tiers from `data/query_tiers.json`.
  - Optional Markdown export: `--save_markdown outputs/eval_report.md`.
  - Optional manual labels from baseline report: `--manual_labels data/baseline_refusal_labels.json`.
- **Prompt-engineering ablation**: `python scripts/run_prompt_ablation.py` — runs inference for all profiles in `data/prompt_profiles.json`, stores outputs in `outputs/prompt_ablation/`. On GPU machines pass model flags via `--infer_extra` (see troubleshooting above).
- **Compare prompt runs**: `python scripts/compare_refusal_reports.py --input_dir outputs/prompt_ablation --out outputs/prompt_ablation/summary.md --out_json outputs/prompt_ablation/summary.json` — ranking by refusal rate; JSON feeds plotting. (You can still pass explicit `--inputs` paths if you prefer.)
  - Optional `--manual_labels data/baseline_refusal_labels.json` for consistent labels across files.
- **Trigger token sensitivity**: `python scripts/analyze_trigger_tokens.py --queries data/queries.json --responses data/baseline_outputs.json --manual_labels data/baseline_refusal_labels.json --out outputs/trigger_token_report.md` — estimates which keywords are most associated with refusals.
- **RAG-style augmentation (prototype)**: `python scripts/retrieval_augment_queries.py --queries data/queries.json --corpus data/rag_corpus.json --out outputs/queries_rag_augmented.json` then `python scripts/inference.py --queries outputs/queries_rag_augmented.json --out outputs/rag_outputs.json`. Uses offline TF–IDF over the in-repo corpus (no FAISS/Chroma required).
- **General MCQ probe (forgetting-style protocol)**: `python scripts/mcq_to_queries.py --mcq data/general_capability_mcq.json --out outputs/mcq_queries.json` → run `inference.py` on that file → `python scripts/eval_mcq_probe.py --mcq data/general_capability_mcq.json --responses outputs/mcq_responses.json --save_json outputs/mcq_scores.json`. Re-run after a LoRA adapter is integrated into inference to compare accuracy.
- **Larger evaluation sets from Hugging Face**: `python scripts/hf_dataset_to_queries.py --dataset yahma/alpaca-cleaned --split train --n 150 --field instruction --keyword "security|vulnerability|encryption" --out data/hf_security_queries.json` (downloads on first use).
- **Automated failure-shape taxonomy**: `python scripts/eval_response_taxonomy.py --input data/baseline_outputs.json --out outputs/taxonomy_report.md` — extends refusal labels with brief vs substantive proxies (not a substitute for human or LLM judging).
- **Plots for reports**: `python scripts/plot_refusal_rates.py --summary_json outputs/prompt_ablation/summary.json --out outputs/prompt_ablation/refusal_rates.png`.

## Implemented improvements (post-baseline)

1. **Reproducibility protocol strengthened**: deterministic inference by default + fixed seed and explicit generation hyperparameters.
2. **Prompt-engineering baseline added**: profile-based system prompts to test alternatives before fine-tuning.
3. **Refusal taxonomy implemented**: separate hard vs soft refusal counts instead of a single aggregate number.
4. **Difficulty-tier analysis introduced**: per-tier refusal breakdown using `data/query_tiers.json`.
5. **Statistical robustness added**: bootstrap 95% confidence interval for refusal rate.
6. **Cross-run comparison added**: multi-file refusal comparison report for prompt ablation outputs (Markdown + optional JSON).
7. **Trigger-token sensitivity analysis added**: ranks keywords/phrases by refusal association (hard/soft split + examples).
8. **RAG prototype**: retrieval + context injection pipeline with a checked-in cyber corpus (`data/rag_corpus.json`).
9. **LoRA ablation hooks**: CLI on `train.py` for LoRA rank, target modules, and training subset size (same script remains a lightweight sanity trainer on TinyLlama).
10. **Catastrophic-forgetting protocol (v1)**: general-domain MCQ export + scoring, designed to be run before/after domain adaptation.
11. **Benchmark scaling helper**: sample filtered instructions from public HF datasets into the same JSON format as `queries.json`.
12. **Failure analysis table**: `eval_response_taxonomy.py` aggregates answer shape and a transparent usefulness proxy.
13. **Evidence plots**: bar chart helper driven by evaluation JSON.
14. **Shared library layout**: refusal + bootstrap + tiers under `scripts/lib/` for consistent imports from any CLI.
15. **Paired bootstrap** for A/B refusal comparisons; **inference/train config snapshots** for reproducibility hardening.
16. **Sweep manifests** in `configs/` + `render_train_sweep.py`; **LLM judge** module with stub backend.

## Proposal extensions vs repository (for grading / report alignment)

This maps the numbered research extensions from the project proposal (e.g. report\_3) to concrete artifacts. Status: **done** = implemented end-to-end in-repo; **partial** = scaffolding or subset in place; **planned** = documented next step.

| # | Extension | Status | Where / how |
|---|-----------|--------|-------------|
| 1 | Prompt engineering vs LoRA | partial | Prompt side: `prompt_profiles.json`, `run_prompt_ablation.py`, `compare_refusal_reports.py`, same metrics via `eval_refusal_rate.py`. LoRA side: run `train.py` / full trainer, export responses, compare on identical queries. |
| 2 | RAG vs fine-tuning | partial | `retrieval_augment_queries.py`, `data/rag_corpus.json`, then same evaluation stack. Swap in FAISS/Chroma + larger corpus when resources allow. |
| 3 | Dataset size ablation (LoRA) | partial | `train.py --n_train …` varies subset size; scaling to 100–5000 pairs needs the full training dataset + GPU budget. |
| 4 | Catastrophic forgetting | partial | `mcq_to_queries.py`, `general_capability_mcq.json`, `eval_mcq_probe.py` + optional `lm-evaluation-harness` / MMLU for a stronger version. |
| 5 | LoRA layer / module ablation | partial | `train.py --target_modules …` (e.g. attention-only vs add MLP if supported by base model keys). |
| 6 | LoRA rank ablation | partial | `train.py --lora_r …` |
| 7 | Trigger-token analysis | done | `analyze_trigger_tokens.py`, `trigger_tokens.json` |
| 8 | LLM-as-judge evaluation | partial | Heuristic + manual labels today; `eval_response_taxonomy.py` adds structured proxies. Plug in a remote judge model behind a small wrapper when API keys are available. |
| 9 | Standard / larger benchmark | partial | `hf_dataset_to_queries.py`; core paper still uses `queries.json` (50) for controlled comparison. |
| 10 | Difficulty tier analysis | done | `query_tiers.json`, `eval_refusal_rate.py` tier table |
| 11 | Statistical significance | done | Bootstrap CI in `eval_refusal_rate.py` |
| 12 | Reproducibility protocol | done | Seeds, deterministic decoding default, documented CLI |
| 13 | Error / response taxonomy | partial | Hard/soft + `eval_response_taxonomy.py`; hallucination requires expert or LLM judge |

Install: `pip install -r requirements.txt`

- **peft** — Hugging Face library for LoRA and other parameter-efficient fine-tuning ([PyPI](https://pypi.org/project/peft/)).
- **datasets** — Hugging Face datasets ([PyPI](https://pypi.org/project/datasets/)).
