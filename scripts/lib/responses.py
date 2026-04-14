"""Helpers for aligning model output JSON files by query id."""
import json
from typing import Dict, List, Optional


def load_responses_by_id(path: str) -> Dict[int, dict]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {int(r["id"]): r for r in rows}


def paired_ids(paths: List[str]) -> List[int]:
    """Query ids present in every file."""
    sets = []
    for p in paths:
        by_id = load_responses_by_id(p)
        sets.append(set(by_id.keys()))
    if not sets:
        return []
    inter = set.intersection(*sets)
    return sorted(inter)


def load_manual_labels(path: Optional[str]) -> Dict[int, str]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}
