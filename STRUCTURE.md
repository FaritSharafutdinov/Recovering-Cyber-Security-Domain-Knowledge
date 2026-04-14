# Repository structure

## Top level

| Path | Role |
|------|------|
| `data/` | Fixed prompts, labels, tiers, RAG corpus, MCQ probes (versioned inputs). |
| `scripts/` | Runnable CLIs; shared helpers live in `scripts/lib/`. |
| `scripts/lib/` | Importable modules: refusal heuristics, bootstrap CIs, tier loading, response indexing. |
| `configs/` | JSON manifests for sweeps (e.g. LoRA jobs) — consumed by `render_train_sweep.py`. |
| `notebooks/` | Exploratory work (paths assume repo root as cwd). |
| `reports/` | LaTeX / PDF write-ups (rebuild with `pdflatex` where applicable). |
| `outputs/` | Generated runs (gitignored except `.gitkeep`). |

## `scripts/` index

| Script | Role |
|--------|------|
| `inference.py` | Llama-3-8B-Instruct 4-bit generation; optional `--save_run_config`. |
| `train.py` | TinyLlama LoRA sanity + rank/module/dataset-size knobs; writes `run_config.json`. |
| `eval_refusal_rate.py` | Refusal counts, bootstrap CI, tier table, Markdown export. |
| `eval_paired_bootstrap.py` | Paired bootstrap on refusal deltas (baseline vs candidate JSON). |
| `compare_refusal_reports.py` | Multi-file refusal ranking; Markdown + JSON. Supports `--input_dir` (PowerShell-friendly; avoids broken `*.json` glob expansion). |
| `plot_refusal_rates.py` | Bar chart from compare JSON. |
| `run_prompt_ablation.py` | Subprocess driver over `prompt_profiles.json` (uses `sys.executable`; forward GPU flags via `--infer_extra`). |
| `analyze_trigger_tokens.py` | Trigger-token conditional refusal rates. |
| `retrieval_augment_queries.py` | TF–IDF RAG-style query augmentation. |
| `hf_dataset_to_queries.py` | Sample HF dataset → `queries.json` format. |
| `mcq_to_queries.py` / `eval_mcq_probe.py` | General MCQ probe pipeline. |
| `eval_response_taxonomy.py` | Automated failure-shape / usefulness proxy table. |
| `render_train_sweep.py` | Print bash lines from `configs/*.json` job list (no training). |
| `llm_judge.py` | Judge API contract + stub backend (wire real LLM later). |

## `scripts/lib/` index

| Module | Role |
|--------|------|
| `refusal.py` | `classify_refusal`, `refusal_binary`. |
| `bootstrap.py` | `bootstrap_ci`, `bootstrap_mean_ci`. |
| `tiers.py` | `load_query_tiers`. |
| `responses.py` | Align JSON response files by `id`. |

## `configs/` index

| File | Role |
|------|------|
| `train_sweep.example.json` | Example jobs for `render_train_sweep.py`. |
| `seeds.example.json` | Example seed list for reproducibility planning. |
