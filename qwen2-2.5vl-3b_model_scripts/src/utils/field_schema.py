"""utils/field_schema.py — shared helpers for the field-extraction task: building the
extraction prompt from a config-defined field schema, and validating/reconciling
field dicts (ground-truth at data-prep time, predictions at inference time) against
that schema. Single source of truth so prepare_dataset_extraction.py and
run_infer_extract.py can never drift apart on what fields are expected.
"""

from typing import Any, Dict, List, Optional


def build_field_extraction_prompt(fields: List[Dict[str, Any]], task_description: str) -> str:
    """Build the extraction prompt text from the config's field schema. Called both at
    data-prep time (stored per training row) and at inference time, so the trained
    model and the inference prompt always agree on the exact field list/order."""
    field_lines = "\n".join(
        f'- "{f["name"]}"' + (f': {f["hint"]}' if f.get("hint") else "")
        for f in fields
    )
    return (
        f"{task_description.strip()}\n\n"
        f"Extract exactly these fields from the document:\n{field_lines}\n\n"
        f'Respond with ONLY a JSON object whose keys are exactly the field names above. '
        f'If a field is not present on the page, use null as its value. '
        f'Do not include any conversational text, explanations, or markdown formatting '
        f'outside the JSON object.'
    )


def validate_fields(fields: Any, schema: List[Dict[str, Any]]) -> Optional[str]:
    """Validate a ground-truth field dict (from the annotation file) has all required
    keys and is well-formed. Returns an error string, or None if valid."""
    if not isinstance(fields, dict):
        return f"fields is not a dict: {fields!r}"

    required = [f["name"] for f in schema if f.get("required", False)]
    missing = [k for k in required if k not in fields]
    if missing:
        return f"fields missing required keys: {missing}"

    schema_names = {f["name"] for f in schema}
    unexpected = [k for k in fields if k not in schema_names]
    if unexpected:
        return f"fields has keys not in schema: {unexpected}"

    return None


def reconcile_predicted_fields(pred: Any, schema: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure every schema field key is present in a model prediction (fill with None
    if the model omitted it), so downstream consumers/metrics can rely on a fixed key
    set even when the model's JSON output is incomplete."""
    out = dict(pred) if isinstance(pred, dict) else {}
    for f in schema:
        out.setdefault(f["name"], None)
    return out

# ---------------------------------------------------------------------------
# Tolerant parsing of model output (truncated / looping / wrongly-shaped JSON)
# ---------------------------------------------------------------------------
import json
import re


def _repair_truncated_json(text: str):
    """Recover the longest valid prefix of truncated JSON by cutting at a structural
    boundary and closing any open brackets. Returns a parsed object or None."""
    stack, cuts = [], []
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
            cuts.append((i + 1, "".join(reversed(stack))))
        elif ch in "}]":
            if stack:
                stack.pop()
            cuts.append((i + 1, "".join(reversed(stack))))
        elif ch == ",":
            cuts.append((i, "".join(reversed(stack))))
    if not in_str:
        cuts.append((len(text), "".join(reversed(stack))))
    for pos, closers in reversed(cuts):
        try:
            return json.loads(text[:pos] + closers)
        except json.JSONDecodeError:
            continue
    return None


_CODE_OK = re.compile(r"^[A-Za-z][0-9A-Za-z.]{1,9}$")


def _clean_diagnoses(items):
    out = []
    for it in (items if isinstance(items, list) else []):
        code = it.get("diagnosisCode") if isinstance(it, dict) else it
        if isinstance(code, str):
            code = code.strip()
            # drop placeholders ("(ICD-10)"), runaway repeats and codes with no digit
            if _CODE_OK.match(code) and any(c.isdigit() for c in code):
                out.append({"diagnosisCode": code})
    return out


def parse_model_json(raw: str, schema: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse raw model text into a schema-reconciled field dict. Strips code fences,
    repairs truncated JSON, converts diagnoses given as plain strings into objects,
    and drops placeholder/runaway diagnosis codes. Raises ValueError only if nothing
    usable can be recovered."""
    clean = re.sub(r"```(?:json)?|```", "", raw or "").strip()
    start = clean.find("{")
    if start > 0:
        clean = clean[start:]
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        data = _repair_truncated_json(clean)
    if not isinstance(data, dict):
        raise ValueError(f"Cannot parse output as JSON: {(raw or '')[:200]}")

    if "diagnoses" in data:
        data["diagnoses"] = _clean_diagnoses(data["diagnoses"])
    procs = data.get("procedures")
    if isinstance(procs, list):
        data["procedures"] = [p for p in procs if isinstance(p, dict) and any(v not in (None, "") for v in p.values())]
    return reconcile_predicted_fields(data, schema)