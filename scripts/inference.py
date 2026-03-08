"""
Baseline inference: load model, run on queries, write results.
Run from repo root: python scripts/inference.py [--queries data/queries.json] [--out outputs/inference_outputs.json]
"""
import argparse
import json
import random
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default="data/queries.json")
    p.add_argument("--out", default="outputs/inference_outputs.json")
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from transformers import BitsAndBytesConfig

    model_id = "unsloth/llama-3-8b-instruct-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    default_system = "You are a helpful and precise cyber-security assistant. Answer the user's technical questions accurately."

    with open(args.queries) as f:
        queries = json.load(f)
    if args.max_queries:
        queries = queries[: args.max_queries]

    results = []
    for i, entry in enumerate(queries):
        inst = entry.get("instruction", "")
        sys_prompt = entry.get("system") or default_system
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": inst},
        ]
        out = generator(
            messages,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = out[0]["generated_text"][-1]["content"]
        results.append({"id": entry.get("id", i), "instruction": inst, "response": response})
        print(f"Done {i+1}/{len(queries)}: id={entry.get('id', i)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results written to {args.out} ({len(results)} responses).")

if __name__ == "__main__":
    main()
