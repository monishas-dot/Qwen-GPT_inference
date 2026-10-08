"""src/evaluation/eval_utils.py — shared evaluation helpers.

Provides three small utilities imported by both evaluate_detection.py and
evaluate_extraction.py.  Keeping them here avoids duplicating identical
code across the two evaluate scripts.
"""

import json
import yaml
from typing import Any, Dict, List


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration from the specified path."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_split(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load all records from a JSONL file into a list.

    Skips blank lines silently; callers should validate the result is non-empty.
    """
    records: List[Dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dedupe_by_id(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate records sharing the same 'id' key, keeping the first
    occurrence.  Records without an 'id' key are always kept.
    """
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for r in records:
        rid = r.get("id")
        if rid is not None and rid in seen:
            continue
        if rid is not None:
            seen.add(rid)
        out.append(r)
    return out
