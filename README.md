# Recovering-Cyber-Security-Domain-Knowledge

Project: Recovering Cyber-Security Domain Knowledge via LoRA Fine-Tuning.

## Repository layout

- **scripts/** — `train.py` (sanity check), `inference.py` (baseline), `eval_refusal_rate.py` (refusal metric)
- **data/** — `queries.json`, `baseline_outputs.json`, `statistics.xlsx`, `query_tiers.json`, `prompt_profiles.json`, `baseline_refusal_labels.json`, `trigger_tokens.json`
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
- **Baseline inference**: `python scripts/inference.py` — runs 4-bit Llama-3-8B-Instruct on `data/queries.json`, writes `outputs/inference_outputs.json`.
  - Reproducibility controls: `--seed`, deterministic mode by default (`--do_sample` enables sampling), and explicit generation settings (`--temperature`, `--top_p`, `--max_new_tokens`).
  - Prompt controls: `--default_system` and `--override_system` for controlled prompt-engineering experiments.
- **Refusal + tier evaluation**: `python scripts/eval_refusal_rate.py` — reads a responses JSON and prints:
  - total / hard / soft refusals;
  - refusal rate with 95% bootstrap CI;
  - breakdown by difficulty tiers from `data/query_tiers.json`.
  - Optional Markdown export: `--save_markdown outputs/eval_report.md`.
  - Optional manual labels from baseline report: `--manual_labels data/baseline_refusal_labels.json`.
- **Prompt-engineering ablation**: `python scripts/run_prompt_ablation.py` — runs inference for all profiles in `data/prompt_profiles.json`, stores outputs in `outputs/prompt_ablation/`.
- **Compare prompt runs**: `python scripts/compare_refusal_reports.py --inputs outputs/prompt_ablation/*.json --out outputs/prompt_ablation/summary.md` — builds a ranking by refusal rate.
- **Trigger token sensitivity**: `python scripts/analyze_trigger_tokens.py --queries data/queries.json --responses data/baseline_outputs.json --manual_labels data/baseline_refusal_labels.json --out outputs/trigger_token_report.md` — estimates which keywords are most associated with refusals.

## Implemented improvements (post-baseline)

1. **Reproducibility protocol strengthened**: deterministic inference by default + fixed seed and explicit generation hyperparameters.
2. **Prompt-engineering baseline added**: profile-based system prompts to test alternatives before fine-tuning.
3. **Refusal taxonomy implemented**: separate hard vs soft refusal counts instead of a single aggregate number.
4. **Difficulty-tier analysis introduced**: per-tier refusal breakdown using `data/query_tiers.json`.
5. **Statistical robustness added**: bootstrap 95% confidence interval for refusal rate.
6. **Cross-run comparison added**: multi-file refusal comparison report for prompt ablation outputs.
7. **Trigger-token sensitivity analysis added**: ranks keywords/phrases by refusal association (hard/soft split + examples).

Install: `pip install -r requirements.txt`

- **peft** — Hugging Face library for LoRA and other parameter-efficient fine-tuning ([PyPI](https://pypi.org/project/peft/)).
- **datasets** — Hugging Face datasets ([PyPI](https://pypi.org/project/datasets/)).
