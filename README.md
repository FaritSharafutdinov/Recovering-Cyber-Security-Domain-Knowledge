# Recovering-Cyber-Security-Domain-Knowledge

Project: Recovering Cyber-Security Domain Knowledge via LoRA Fine-Tuning.

## Repository layout

- **scripts/** — `train.py` (sanity check), `inference.py` (baseline), `eval_refusal_rate.py` (refusal metric)
- **data/** — `queries.json`, `baseline_outputs.json`, `statistics.xlsx`
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

- **Sanity check (learning curve)**: `python scripts/train.py` — LoRA overfit on 5 examples, ~20 steps. Loss is logged each step. Writes to `outputs/sanity_check_output/`.
- **Baseline inference**: `python scripts/inference.py` — runs 4-bit Llama-3-8B-Instruct on `data/queries.json`, writes `outputs/inference_outputs.json`. Options: `--queries`, `--out`, `--max_queries`, `--seed` (default 42).
- **Refusal rate (quantitative metric)**: `python scripts/eval_refusal_rate.py` — reads a responses JSON and prints refusal rate. Default input: `data/baseline_outputs.json`. Use `--input` for another file.

Install: `pip install -r requirements.txt`

- **peft** — Hugging Face library for LoRA and other parameter-efficient fine-tuning ([PyPI](https://pypi.org/project/peft/)).
- **datasets** — Hugging Face datasets ([PyPI](https://pypi.org/project/datasets/)).
