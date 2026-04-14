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


def _format_chat(tokenizer, messages):
    """
    Return a single text prompt from chat-style messages.
    Uses the tokenizer chat template when available; otherwise uses a simple fallback.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        parts.append(f"{role.upper()}:\n{content}\n")
    parts.append("ASSISTANT:\n")
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default="data/queries.json")
    p.add_argument("--out", default="outputs/inference_outputs.json")
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument(
        "--lora_adapter",
        default=None,
        help="Optional path to a LoRA adapter directory (saved by train.py --save_adapter).",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode (local_files_only). Useful on laptops without HF network access.",
    )
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Enable 4-bit loading (requires bitsandbytes + typically CUDA). If omitted, uses standard weights.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device preference. 'auto' picks cuda if available else cpu.",
    )
    p.add_argument(
        "--default_system",
        default="You are a helpful and precise cyber-security assistant. Answer the user's technical questions accurately.",
    )
    p.add_argument("--override_system", default=None)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument(
        "--max_input_tokens",
        type=int,
        default=768,
        help="Truncate the formatted prompt to this many tokens to keep CPU runs manageable.",
    )
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

    from transformers import AutoTokenizer, AutoModelForCausalLM

    use_cuda = torch.cuda.is_available()
    if args.device == "cuda" and not use_cuda:
        raise RuntimeError("--device cuda requested, but CUDA is not available.")
    device = "cuda" if (args.device == "auto" and use_cuda) or args.device == "cuda" else "cpu"

    # If the machine is offline / HF blocked, avoid any network calls.
    local_only = bool(args.offline)
    if local_only:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=local_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = None
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
        except Exception as e:
            raise RuntimeError("4-bit requested but BitsAndBytesConfig is not available/working.") from e

    # Use device_map only for CUDA; for CPU it can lead to confusing behavior on Windows.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            local_files_only=local_only,
        )
    except Exception as e:
        # Common failure mode on laptops: HF network/DNS issues mid-run.
        # If the model is already cached locally, retry in strict offline mode.
        if not local_only:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" else None,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                local_files_only=True,
            )
            local_only = True
        else:
            raise e
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    if args.lora_adapter:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.lora_adapter, is_trainable=False)
            model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA adapter from {args.lora_adapter!r}") from e

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
        prompt = _format_chat(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": bool(args.do_sample),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        gen = model.generate(
            **inputs,
            **gen_kwargs,
        )
        out_text = tokenizer.decode(gen[0], skip_special_tokens=True)
        # Heuristic: response is the suffix after the prompt when possible.
        response = out_text[len(prompt) :].strip() if out_text.startswith(prompt) else out_text.strip()
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
