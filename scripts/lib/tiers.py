"""Difficulty tier metadata keyed by query id."""
import json
from typing import Dict


def load_query_tiers(path: str) -> Dict[int, str]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}
