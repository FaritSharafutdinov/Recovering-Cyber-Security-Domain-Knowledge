"""
Run prompt-engineering ablation on the same query set.
Example:
python scripts/run_prompt_ablation.py --queries data/queries.json --profiles data/prompt_profiles.json
"""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default="data/queries.json")
    p.add_argument("--profiles", default="data/prompt_profiles.json")
    p.add_argument("--out_dir", default="outputs/prompt_ablation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--do_sample", action="store_true")
    args = p.parse_args()

    with open(args.profiles, encoding="utf-8") as f:
        profiles = json.load(f)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for profile in profiles:
        name = profile["name"]
        sys_prompt = profile["system_prompt"]
        out_file = out_dir / f"{name}.json"
        cmd = [
            "python",
            "scripts/inference.py",
            "--queries",
            args.queries,
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
        print(f"\nRunning profile: {name}")
        subprocess.run(cmd, check=True)

    print(f"\nPrompt ablation finished. Outputs saved in {out_dir}")


if __name__ == "__main__":
    main()
