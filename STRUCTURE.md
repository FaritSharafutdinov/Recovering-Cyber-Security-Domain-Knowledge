# Repository structure

## Top level

| Path | Role |
|------|------|
| `data/` | Fixed prompts, labels, tiers, RAG corpus, MCQ probes (versioned inputs). |
| `scripts/` | Runnable CLIs; shared helpers live in `scripts/lib/`. |
| `scripts/lib/` | Importable modules: refusal heuristics, bootstrap CIs, tier loading, response indexing, RAG corpus loader. |
| `configs/` | JSON manifests for sweeps (e.g. LoRA jobs) — consumed by `render_train_sweep.py`. |
| `notebooks/` | Exploratory work (paths assume repo root as cwd). |
| `reports/` | LaTeX / PDF write-ups (rebuild with `pdflatex` where applicable). |
| `EXPERIMENTS.md` | Auto-generated experiment table + commands (see `run_experiment_suite.py`). |
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
| `retrieval_augment_queries.py` | RAG-style query augmentation: `--mode tfidf` (legacy) or `--mode embedding_faiss` (CPU FAISS + sentence-transformers index). |
| `build_rag_index.py` | Build `outputs/rag_index/` (FAISS `IndexFlatIP`, `meta.json`, `embeddings.npy`) from `rag_corpus.json` or `.jsonl`. |
| `nvdcve_to_rag_corpus.py` | Stream NVD `nvdcve-2.0-*.json` feeds into `id`/`title`/`text` chunks (ijson); optional `--merge-education`. |
| `build_mmlu_subset.py` | Stratified MMLU subset (`data/mmlu_eval_100.json`) via Hugging Face `cais/mmlu`. |
| `build_ctf_eval_subset.py` | NYU CTF Bench–derived defensive prompts + manifest (metadata from public `test_dataset.json`, GPL-2.0). |
| `run_experiment_suite.py` | **Master pipeline:** optional FAISS index, MMLU/CTF dataset builders, retrieval (tfidf + faiss), inference, eval, manifest + [EXPERIMENTS.md](EXPERIMENTS.md). Use `--preset full` for all modes; `--skip_inference` for dry runs. |
| `smoke_test.py` | Fast CI-style checks (imports, offline demo, tiny RAG index, NVD/CTF scripts, no full LLM inference). |
| `hf_dataset_to_queries.py` | Sample HF dataset → `queries.json` format. |
| `mcq_to_queries.py` / `eval_mcq_probe.py` | General MCQ probe pipeline. |
| `eval_response_taxonomy.py` | Automated failure-shape / usefulness proxy table. |
| `render_train_sweep.py` | Print bash lines from `configs/*.json` job list (no training). |
| `llm_judge.py` | Judge API contract + stub backend (wire real LLM later). |

## `scripts/lib/` index

| Module | Role |
|--------|------|
| `refusal.py` | `classify_refusal`, `refusal_binary`. |
| `rag_corpus.py` | Load RAG corpus JSON / JSONL as normalized chunk dicts. |
| `bootstrap.py` | `bootstrap_ci`, `bootstrap_mean_ci`. |
| `tiers.py` | `load_query_tiers`. |
| `responses.py` | Align JSON response files by `id`. |

## `configs/` index

| File | Role |
|------|------|
| `train_sweep.example.json` | Example jobs for `render_train_sweep.py`. |
| `seeds.example.json` | Example seed list for reproducibility planning. |
