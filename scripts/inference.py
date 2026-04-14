"""
Baseline inference: load model, run on queries, write results.
Run from repo root: python scripts/inference.py [--queries data/queries.json] [--out outputs/inference_outputs.json]
"""
import argparse
import json
import random
import sys
from pathlib import Path


def _try_versions():
    out = {"python": sys.version.split()[0]}
    try:
        import torch

        out["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import transformers

        out["transformers"] = transformers.__version__
    except Exception:
        pass
    return out


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
    p.add_argument(
        "--save_run_config",
        nargs="?",
        const="__auto__",
        default=None,
        help="Write JSON of CLI args (+ library versions). "
        "Omit value to write alongside --out as <stem>_inference_config.json, or pass a file path.",
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
        row = {
            "id": entry.get("id", i),
            "instruction": inst,
            "system_prompt": sys_prompt,
            "response": response,
        }
        if "rag_chunk_ids" in entry:
            row["rag_chunk_ids"] = entry["rag_chunk_ids"]
        results.append(row)
        print(f"Done {i+1}/{len(queries)}: id={entry.get('id', i)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results written to {args.out} ({len(results)} responses).")

    if args.save_run_config:
        cfg_path = (
            out_path.with_name(out_path.stem + "_inference_config.json")
            if args.save_run_config == "__auto__"
            else Path(args.save_run_config)
        )
        cfg = {k: getattr(args, k) for k in vars(args) if k != "save_run_config"}
        cfg["do_sample"] = bool(args.do_sample)
        cfg["versions"] = _try_versions()
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"Run config written to {cfg_path}")


if __name__ == "__main__":
    main()
