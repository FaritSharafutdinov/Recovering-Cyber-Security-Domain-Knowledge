"""
Run prompt-engineering ablation on the same query set.
Example:
python scripts/run_prompt_ablation.py --queries data/queries.json --profiles data/prompt_profiles.json
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default="data/queries.json")
    p.add_argument("--profiles", default="data/prompt_profiles.json")
    p.add_argument("--out_dir", default="outputs/prompt_ablation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument(
        "--infer_extra",
        default="",
        help="Extra args forwarded to scripts/inference.py (quoted string). "
        "Example (Windows): --infer_extra \"--model_id unsloth/llama-3-8b-instruct-bnb-4bit --load_in_4bit --device cuda\"",
    )
    args = p.parse_args()

    profiles_path = Path(args.profiles)
    if not profiles_path.is_absolute():
        profiles_path = REPO_ROOT / profiles_path
    with open(profiles_path, encoding="utf-8") as f:
        profiles = json.load(f)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    extra: list[str] = []
    if args.infer_extra.strip():
        extra = shlex.split(args.infer_extra, posix=os.name != "nt")

    for profile in profiles:
        name = profile["name"]
        sys_prompt = profile["system_prompt"]
        out_file = out_dir / f"{name}.json"
        queries_path = Path(args.queries)
        if not queries_path.is_absolute():
            queries_path = REPO_ROOT / queries_path
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "inference.py"),
            "--queries",
            str(queries_path),
            "--out",
            str(out_file),
            "--seed",
            str(args.seed),
            "--override_system",
            sys_prompt,
        ]
        if args.max_queries is not None:
            cmd.extend(["--max_queries", str(args.max_queries)])
        if args.do_sample:
            cmd.append("--do_sample")
        cmd.extend(extra)
        print(f"\nRunning profile: {name}")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    print(f"\nPrompt ablation finished. Outputs saved in {out_dir}")


if __name__ == "__main__":
    main()
