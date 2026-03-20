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
    p.add_argument("--model_id", default="unsloth/llama-3-8b-instruct-bnb-4bit")
    p.add_argument(
        "--default_system",
        default="You are a helpful and precise cyber-security assistant. Answer the user's technical questions accurately.",
    )
    p.add_argument("--override_system", default=None)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling. If omitted, generation is deterministic.",
    )
    args = p.parse_args()

    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from transformers import BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb, device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    with open(args.queries) as f:
        queries = json.load(f)
    if args.max_queries:
        queries = queries[: args.max_queries]

    results = []
    for i, entry in enumerate(queries):
        inst = entry.get("instruction", "")
        if args.override_system is not None:
            sys_prompt = args.override_system
        else:
            sys_prompt = entry.get("system") or args.default_system
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": inst},
        ]
        out = generator(
            messages,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=terminators,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        response = out[0]["generated_text"][-1]["content"]
        results.append(
            {
                "id": entry.get("id", i),
                "instruction": inst,
                "system_prompt": sys_prompt,
                "response": response,
            }
        )
        print(f"Done {i+1}/{len(queries)}: id={entry.get('id', i)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results written to {args.out} ({len(results)} responses).")

if __name__ == "__main__":
    main()
