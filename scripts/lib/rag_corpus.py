"""Load RAG corpus from JSON array or JSONL (one JSON object per line)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_corpus_chunks(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return _normalize_chunks(rows)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return _normalize_chunks(data)
    raise ValueError(f"Expected JSON list or .jsonl, got {type(data)}")


def _normalize_chunks(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, row in enumerate(rows):
        if "text" not in row:
            raise KeyError(f"Chunk {i} missing 'text' field")
        cid = row.get("id", i)
        out.append(
            {
                "id": cid,
                "title": row.get("title", f"chunk_{i}"),
                "text": str(row["text"]),
            }
        )
    return out
