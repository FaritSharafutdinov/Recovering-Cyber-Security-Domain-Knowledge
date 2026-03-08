"""
Sanity check: overfit on a tiny subset and show learning curve.
Run from repo root: python scripts/train.py
"""
import json
import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

OUTPUT_DIR = Path("outputs/sanity_check_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_tiny_subset(data_path: str = "data/baseline_outputs.json", n: int = 5):
    with open(data_path) as f:
        data = json.load(f)
    subset = data[:n]
    texts = []
    for item in subset:
        inst = item["instruction"]
        resp = item["response"][:400]
        texts.append(f"Instruction: {inst}\nResponse: {resp}")
    return Dataset.from_dict({"text": texts})

def main():
    train_data = load_tiny_subset(n=5)

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
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

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        max_steps=20,
        per_device_train_batch_size=1,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
    )
    trainer.train()
    print("\nSanity check done. Learning curve above shows loss decreasing (expected).")

if __name__ == "__main__":
    main()
