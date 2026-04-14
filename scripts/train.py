"""
LoRA sanity check and small ablation harness (rank, target modules, training subset size).

Examples:
  python scripts/train.py
  python scripts/train.py --lora_r 16 --target_modules q_proj,k_proj,v_proj,o_proj --n_train 8 --max_steps 30
  python scripts/train.py --lora_r 64 --n_train 20 --output_dir outputs/lora_rank64_n20
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def load_subset(data_path: str, n: int):
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    subset = data[:n]
    texts = []
    for item in subset:
        inst = item["instruction"]
        resp = item["response"][:400]
        texts.append(f"Instruction: {inst}\nResponse: {resp}")
    return Dataset.from_dict({"text": texts})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/baseline_outputs.json")
    p.add_argument("--n_train", type=int, default=5, help="Number of instruction-response pairs from the start of the file.")
    p.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument(
        "--save_adapter",
        action="store_true",
        help="Save the trained LoRA adapter to <output_dir>/adapter so it can be loaded in inference.",
    )
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument(
        "--target_modules",
        default="q_proj,v_proj",
        help="Comma-separated module names for LoRA (e.g. q_proj,v_proj or q_proj,k_proj,v_proj,o_proj).",
    )
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--output_dir", default="outputs/sanity_check_output")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_subset(args.data_path, args.n_train)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train_data = train_data.map(tokenize, batched=True, remove_columns=["text"])
    train_data.set_format("torch", columns=["input_ids", "attention_mask"])

    def add_labels(examples):
        labels = examples["input_ids"].clone()
        labels[examples["attention_mask"] == 0] = -100
        return {"labels": labels}

    train_data = train_data.map(add_labels, batched=True)

    target_list = [s.strip() for s in args.target_modules.split(",") if s.strip()]
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_list,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_data)
    trainer.train()

    adapter_dir = out_dir / "adapter"
    if args.save_adapter:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        # Save only the LoRA adapter weights/config (not the full base model).
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

    versions = {"python": sys.version.split()[0], "torch": torch.__version__}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import peft

        versions["peft"] = peft.__version__
    except Exception:
        pass

    meta = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "n_train": args.n_train,
        "save_adapter": bool(args.save_adapter),
        "adapter_dir": str(adapter_dir) if args.save_adapter else None,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_list,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "seed": args.seed,
        "versions": versions,
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nTraining finished.")
    print(f"Config written to {out_dir / 'run_config.json'}")
    if args.save_adapter:
        print(f"Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
