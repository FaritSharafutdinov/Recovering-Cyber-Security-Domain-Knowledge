"""
Export a compact JSON sheet for manual evaluation.

Creates a list of rows:
  - id
  - instruction
  - response

Optionally includes system_prompt and rag_chunk_ids if present in the responses file.

Example:
  python scripts/export_manual_eval_sheet.py --queries data/queries.json --responses outputs/run.json --out outputs/manual_eval/sheet.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default="data/queries.json")
    p.add_argument("--responses", required=True, help="Run outputs JSON (list of {id, response, ...}).")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    responses = json.loads(Path(args.responses).read_text(encoding="utf-8"))

    q_by_id = {int(q["id"]): q for q in queries}
    r_by_id = {int(r["id"]): r for r in responses}
    ids = sorted(set(q_by_id.keys()) & set(r_by_id.keys()))

    rows = []
    for qid in ids:
        q = q_by_id[qid]
        r = r_by_id[qid]
        row = {
            "id": qid,
            "instruction": q.get("instruction", ""),
            "response": r.get("response", ""),
        }
        if "system_prompt" in r:
            row["system_prompt"] = r.get("system_prompt")
        if "rag_chunk_ids" in r:
            row["rag_chunk_ids"] = r.get("rag_chunk_ids")
        rows.append(row)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {outp}")


if __name__ == "__main__":
    main()

