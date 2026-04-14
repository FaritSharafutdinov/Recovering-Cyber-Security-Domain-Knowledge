# Experiment log (auto-generated)

The table is **regenerated** on each `run_experiment_suite.py` run. Machine-readable log: `outputs/experiment_manifest.json`.

**Full GPU pipeline (index + default modes + inference + eval):**

```powershell
python scripts/run_experiment_suite.py --build_rag_index --infer_extra "--model_id unsloth/llama-3-8b-instruct-bnb-4bit --load_in_4bit --device cuda"
```

**Dry run (index + augment only, no inference):** `--skip_inference`

**Pre-build MMLU/CTF JSON without enabling those inference modes:** `--ensure_all_datasets`

See [README.md](README.md) for details.

| UTC timestamp | Host | GPU | Commit | Run name | Command | Output JSON | Refusal % | 95% CI | paired Δ pp | MMLU acc % |
|---|---|---|---|---|---|---|---:|---|---:|---:|
| 2026-04-14T20:47:52.040133+00:00 | IMMATER |  | `1ee4abb9` | retrieval_augment_faiss | `C:\Users\Ivenho\miniconda3\python.exe C:\Users\Ivenho\Recovering-Cyber-Security-Domain-Knowledge\scripts\retrieval_augme` | `outputs\exp_smoke\queries_rag_faiss.json` |  |  |  |  |
