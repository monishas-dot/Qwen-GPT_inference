"""
evaluate_extraction_metrics.py  (single-file version)

Purpose:
    Compare Qwen's and GPT's extracted fields against human-annotated ground
    truth (gold_set_annotations*.jsonl), and report accuracy broken down by
    the GT's own per-field blur severity ("none"/"low"/"medium"/"high").

    This is a self-contained version: field_matching.py's logic is inlined
    below, and the file paths are hardcoded constants (see "PATHS TO EDIT"
    right below the imports) instead of CLI args. Edit those four paths,
    then just run:

        python evaluate_extraction_metrics.py

Key rules (confirmed with the team):
  - Join all three files on "id" only (ignore image_path/source_pool/
    overall_blur_class -- those live in GT but aren't used for matching).
  - field_blur_severity is GT-only annotation, never touched on the model
    side -- it's purely the lens metrics get sliced through.
  - Any field with GT severity == "high" is EXCLUDED from scoring entirely
    (for every model), since GT itself is considered unreliable there.
  - "diagnoses" and "procedures" are list-of-object fields; instead of one
    collapsed pass/fail per list, they're decomposed into one metric per
    sub-field (e.g. "procedures.dateOfServiceStart", "procedures.
    numberOfVisits") via field_level_scores_partial, so a wrong
    numberOfVisits doesn't drag down the procedureCode/date metrics. Every
    sub-field inherits its parent field's single GT severity bucket
    (severity is only annotated at the parent-field level).
  - id present in GT but missing from a model's file, OR present in a
    model's file but missing from GT -> that id is skipped for that
    model's metrics; count + ids reported at the end, never silently
    dropped without a trace.
  - Qwen rows with status != "OK", or GPT rows with a non-empty "errors"
    list, are excluded entirely from that model's metrics (not penalized,
    not credited) -- but counted and reported at the end.

Reporting, per model (Qwen and GPT separately):
  1. Total files scored / excluded (missing id in model file, extra id not
     in GT, bad status) -- with reasons.
  2. Field-instance counts by GT severity bucket (none/low/medium/high),
     computed once from GT alone (severity distribution is a GT property,
     not a per-model one).
  3. Overall accuracy per severity bucket (high excluded), pooling all
     fields together -- both a strict "% of instances scored exactly 1.0"
     and a "weighted" score (mean of partial-credit scores; identical to
     strict for scalar fields, meaningfully different for list sub-fields
     when a file has multiple diagnoses/procedures).
  4. The same breakdown per individual field (including procedures.* /
     diagnoses.* sub-fields) x severity bucket.

Output:
    <output_dir>/summary.txt                  -- human-readable console-style report
    <output_dir>/per_field_severity_qwen.csv  -- full breakdown, one row per (field, severity)
    <output_dir>/per_field_severity_gpt.csv
    <output_dir>/exclusions.json              -- exactly which ids were skipped/excluded and why
"""

import csv
import json
import re
import sys
import difflib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# PATHS TO EDIT -- change these four before running, then just:
#     python evaluate_extraction_metrics.py
# ---------------------------------------------------------------------------
GT_JSONL = "/Users/monishasp/Downloads/prompt_ops_reqd/gold_set_annotations_final_repaired.jsonl"
QWEN_JSONL = "/Users/monishasp/Downloads/prompt_ops_reqd/runs/prompt_v5/qwen/per_file_report.jsonl"
GPT_JSONL = "/Users/monishasp/Downloads/prompt_ops_reqd/gpt output/gpt.jsonl"
OUTPUT_DIR = "/Users/monishasp/Downloads/prompt_ops_reqd/runs/prompt_v5"

# ===========================================================================
# field_matching.py, inlined
# ===========================================================================
# Why this exists: raw string-equality on JSON field values counts formatting
# noise as failures -- punctuation, casing, "null" vs None, decimal-less
# diagnosis codes, differently-ordered date separators, and shuffled list
# order. This distinguishes that noise from real extraction errors, so
# reported metrics reflect what the model actually got wrong.
#
# `match_type` is set per-field in the config's `fields:` list:
#   - "exact" / "text" (default) : normalized-string match -- case,
#                                   whitespace, punctuation, and null-alias
#                                   insensitive.
#   - "fuzzy_text"                : same as above, plus a similarity-ratio
#                                   fallback (FUZZY_TEXT_THRESHOLD) for
#                                   free-text fields prone to minor
#                                   OCR/spelling variance (names, addresses,
#                                   service descriptions).
#   - "code"                      : alphanumeric-only match -- for diagnosis/
#                                   procedure codes and NPIs. Ignores
#                                   decimals, dashes, spaces:
#                                   "F329" == "F32.9" == "f32-9".
#   - "phone"                     : digits-only match, for phone/fax numbers.
#   - "date"                      : calendar-date match across separator/
#                                   digit-count variants: "3/16/26" ==
#                                   "03-16-2026" == "03.16.2026".
#   - "numeric"                   : numeric match ignoring formatting
#                                   (commas, trailing .0, leading zeros).
#
# List-of-object fields (field_type: "list", e.g. diagnoses/procedures) are
# scored as an order-independent multiset via greedy matching. Sub-field
# partial credit within list items (field_level_scores_partial): each gt
# item is paired with its best-overlap predicted item (by fraction of
# sub-fields matching), and each sub-field of the item_schema is then scored
# independently across the whole list. document_exact_match still requires a
# *complete* match, unaffected by partial credit.

_NULL_ALIASES = {"", "null", "none", "n/a", "na", "nil"}
FUZZY_TEXT_THRESHOLD = 0.90
_DATE_SPLIT_RE = re.compile(r"[\-/.\s]+")


def _is_null(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in _NULL_ALIASES)


def _clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _norm_text(v: Any) -> str:
    if _is_null(v):
        return ""
    return _clean_whitespace(str(v)).lower()


def _norm_text_no_punct(v: Any) -> str:
    t = _norm_text(v)
    return _clean_whitespace(re.sub(r"[^\w\s]", " ", t))


def _norm_alnum(v: Any) -> str:
    if _is_null(v):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def _norm_digits(v: Any) -> str:
    if _is_null(v):
        return ""
    return re.sub(r"\D", "", str(v))


def _norm_numeric(v: Any) -> str:
    if _is_null(v):
        return ""
    s = re.sub(r"[,\s]", "", str(v))
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s.lower()


def _norm_date(v: Any) -> str:
    if _is_null(v):
        return ""
    parts = [p for p in _DATE_SPLIT_RE.split(str(v).strip()) if p]
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return _norm_text_no_punct(v)
    mm, dd, yy = parts
    if len(yy) == 2:
        yy = ("20" if int(yy) < 50 else "19") + yy
    try:
        return datetime(int(yy), int(mm), int(dd)).strftime("%Y-%m-%d")
    except ValueError:
        return _norm_text_no_punct(v)


def _values_match(v_pred: Any, v_gt: Any, match_type: str) -> bool:
    if _is_null(v_pred) and _is_null(v_gt):
        return True
    if match_type == "code":
        return _norm_alnum(v_pred) == _norm_alnum(v_gt)
    if match_type == "phone":
        return _norm_digits(v_pred) == _norm_digits(v_gt)
    if match_type == "date":
        return _norm_date(v_pred) == _norm_date(v_gt)
    if match_type == "numeric":
        return _norm_numeric(v_pred) == _norm_numeric(v_gt)
    if match_type == "fuzzy_text":
        a, b = _norm_text_no_punct(v_pred), _norm_text_no_punct(v_gt)
        if a == b:
            return True
        if not a or not b:
            return False
        return difflib.SequenceMatcher(None, a, b).ratio() >= FUZZY_TEXT_THRESHOLD
    return _norm_text_no_punct(v_pred) == _norm_text_no_punct(v_gt)


def _item_schema_lookup(field_def: Dict[str, Any]) -> Dict[str, str]:
    return {s["name"]: s.get("match_type", "text") for s in field_def.get("item_schema", [])}


def _list_item_matches(pred_item: Any, gt_item: Any, sub_types: Dict[str, str]) -> bool:
    if not isinstance(pred_item, dict) or not isinstance(gt_item, dict):
        return pred_item == gt_item
    for key, gt_val in gt_item.items():
        if not _values_match(pred_item.get(key), gt_val, sub_types.get(key, "text")):
            return False
    return True


def _score_list_field(pred_val: Any, gt_val: Any, field_def: Dict[str, Any]) -> float:
    gt_list = gt_val if isinstance(gt_val, list) else ([] if _is_null(gt_val) else [gt_val])
    pred_list = pred_val if isinstance(pred_val, list) else ([] if _is_null(pred_val) else [pred_val])

    if not gt_list and not pred_list:
        return 1.0
    if not gt_list or not pred_list:
        return 0.0

    sub_types = _item_schema_lookup(field_def)
    remaining_pred = list(pred_list)
    matched = 0
    for gt_item in gt_list:
        hit_idx = next(
            (i for i, p in enumerate(remaining_pred) if _list_item_matches(p, gt_item, sub_types)),
            None,
        )
        if hit_idx is not None:
            matched += 1
            remaining_pred.pop(hit_idx)

    return matched / max(len(gt_list), len(pred_list))


def _pair_list_items(
    gt_list: List[Any], pred_list: List[Any], sub_types: Dict[str, str]
) -> List[Any]:
    def _overlap(gt_item: Any, pred_item: Any) -> float:
        if not isinstance(gt_item, dict) or not isinstance(pred_item, dict):
            return 1.0 if gt_item == pred_item else 0.0
        keys = list(sub_types.keys()) or list(gt_item.keys())
        if not keys:
            return 0.0
        hits = sum(_values_match(pred_item.get(k), gt_item.get(k), sub_types.get(k, "text")) for k in keys)
        return hits / len(keys)

    candidates = []
    for gi, g in enumerate(gt_list):
        for pi, p in enumerate(pred_list):
            candidates.append((_overlap(g, p), gi, pi))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    gt_paired = [None] * len(gt_list)
    used_pred = set()
    used_gt = set()
    for score, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        gt_paired[gi] = pi
        used_gt.add(gi)
        used_pred.add(pi)

    return gt_paired


def _score_list_field_by_subfield(
    pred_val: Any, gt_val: Any, field_def: Dict[str, Any]
) -> Dict[str, float]:
    gt_list = gt_val if isinstance(gt_val, list) else ([] if _is_null(gt_val) else [gt_val])
    pred_list = pred_val if isinstance(pred_val, list) else ([] if _is_null(pred_val) else [pred_val])

    sub_types = _item_schema_lookup(field_def)
    sub_keys = list(sub_types.keys()) or ["value"]

    if not gt_list and not pred_list:
        return {k: 1.0 for k in sub_keys}
    if not gt_list or not pred_list:
        return {k: 0.0 for k in sub_keys}

    denom = max(len(gt_list), len(pred_list))
    pairing = _pair_list_items(gt_list, pred_list, sub_types)

    scores = {}
    for key in sub_keys:
        matched = 0
        for gi, gt_item in enumerate(gt_list):
            pi = pairing[gi]
            if pi is None:
                continue
            pred_item = pred_list[pi]
            if not isinstance(gt_item, dict) or not isinstance(pred_item, dict):
                matched += int(gt_item == pred_item)
                continue
            if _values_match(pred_item.get(key), gt_item.get(key), sub_types.get(key, "text")):
                matched += 1
        scores[key] = matched / denom

    return scores


def field_level_scores_partial(
    pred_fields: Dict[str, Any], gt_fields: Dict[str, Any], schema: List[Dict[str, Any]]
) -> Dict[str, float]:
    by_name = {f["name"]: f for f in schema}
    scores: Dict[str, float] = {}
    for key, gt_val in gt_fields.items():
        field_def = by_name.get(key, {"name": key})
        pred_val = pred_fields.get(key) if isinstance(pred_fields, dict) else None
        if field_def.get("field_type") == "list":
            sub_scores = _score_list_field_by_subfield(pred_val, gt_val, field_def)
            for sub_key, sub_score in sub_scores.items():
                scores[f"{key}.{sub_key}"] = sub_score
        else:
            scores[key] = score_field(pred_val, gt_val, field_def)
    return scores


def score_field(pred_val: Any, gt_val: Any, field_def: Dict[str, Any]) -> float:
    if field_def.get("field_type") == "list":
        return _score_list_field(pred_val, gt_val, field_def)
    return 1.0 if _values_match(pred_val, gt_val, field_def.get("match_type", "text")) else 0.0


def field_level_scores(
    pred_fields: Dict[str, Any], gt_fields: Dict[str, Any], schema: List[Dict[str, Any]]
) -> Dict[str, float]:
    by_name = {f["name"]: f for f in schema}
    scores = {}
    for key, gt_val in gt_fields.items():
        pred_val = pred_fields.get(key) if isinstance(pred_fields, dict) else None
        scores[key] = score_field(pred_val, gt_val, by_name.get(key, {"name": key}))
    return scores


def document_exact_match(
    pred_fields: Dict[str, Any], gt_fields: Dict[str, Any], schema: List[Dict[str, Any]]
) -> float:
    scores = field_level_scores(pred_fields, gt_fields, schema)
    return 1.0 if scores and all(v == 1.0 for v in scores.values()) else 0.0


# ===========================================================================
# Field schema -- single source of truth, mirrors the team's real
# field_schema.py "fields:" config exactly (verified field-by-field against
# sample GT/Qwen/GPT records: 19 top-level fields, diagnoses/procedures as
# list-of-object with their declared item_schema).
# ===========================================================================
FIELD_SCHEMA = [
    {"name": "diagnoses", "field_type": "list", "item_schema": [
        {"name": "diagnosisCode", "match_type": "code"},
    ]},
    {"name": "procedures", "field_type": "list", "item_schema": [
        {"name": "procedureCode", "match_type": "code"},
        {"name": "dateOfServiceStart", "match_type": "date"},
        {"name": "dateOfServiceEnd", "match_type": "date"},
        {"name": "numberOfVisits", "match_type": "numeric"},
    ]},
    {"name": "servicing_provider_npi", "match_type": "code"},
    {"name": "type_of_service", "match_type": "fuzzy_text"},
    {"name": "member_id", "match_type": "code"},
    {"name": "requesting_provider_phone", "match_type": "phone"},
    {"name": "acuity", "match_type": "exact"},
    {"name": "date_of_birth", "match_type": "date"},
    {"name": "servicing_provider_name", "match_type": "fuzzy_text"},
    {"name": "requesting_provider_name", "match_type": "fuzzy_text"},
    {"name": "member_name", "match_type": "fuzzy_text"},
    {"name": "place_of_service", "match_type": "fuzzy_text"},
    {"name": "requesting_provider_npi", "match_type": "code"},
    {"name": "requesting_provider_fax", "match_type": "phone"},
    {"name": "lob", "match_type": "fuzzy_text"},
    {"name": "servicing_provider_fax", "match_type": "phone"},
    {"name": "servicing_provider_phone", "match_type": "phone"},
    {"name": "servicing_provider_address", "match_type": "fuzzy_text"},
    {"name": "requesting_provider_address", "match_type": "fuzzy_text"},
]

SEVERITY_LEVELS = ["none", "low", "medium", "high"]
EXCLUDED_SEVERITY = "high"


def parent_field_of(score_key: str) -> str:
    """'procedures.dateOfServiceStart' -> 'procedures'; 'lob' -> 'lob'."""
    return score_key.split(".", 1)[0]


def load_jsonl(path: Path):
    rows = []
    n_bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                n_bad += 1
                print(f"WARNING: {path}: could not parse line {line_no}: {e}", file=sys.stderr)
    if n_bad:
        print(f"WARNING: {path}: {n_bad} unparseable lines skipped", file=sys.stderr)
    return rows


def load_gt(path: Path):
    """id -> {"gt_fields": ..., "field_blur_severity": ...}"""
    out = {}
    for row in load_jsonl(path):
        rid = row.get("id")
        if not rid:
            continue
        out[rid] = {
            "gt_fields": row.get("gt_fields") or {},
            "field_blur_severity": row.get("field_blur_severity") or {},
        }
    return out


def load_qwen(path: Path):
    """id -> {"predicted_fields": ..., "status": ...}"""
    out = {}
    for row in load_jsonl(path):
        rid = row.get("id")
        if not rid:
            continue
        out[rid] = {
            "predicted_fields": row.get("predicted_fields") or {},
            "status": row.get("status"),
        }
    return out


def load_gpt(path: Path):
    """id -> {"predicted_fields": ..., "errors": [...]}"""
    out = {}
    for row in load_jsonl(path):
        rid = row.get("id")
        if not rid:
            continue
        pred = ((row.get("prediction") or {}).get("fields")) or {}
        out[rid] = {
            "predicted_fields": pred,
            "errors": row.get("errors") or [],
        }
    return out


class Accumulator:
    """Collects (score, severity, field_key) triples and turns them into the
    reporting tables. One instance per model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        # field_key -> severity -> list of scores (each in [0,1])
        self.scores = defaultdict(lambda: defaultdict(list))
        self.files_scored = 0
        self.excluded_missing_id = []      # ids in GT, absent from this model's file
        self.excluded_extra_id = []        # ids in this model's file, absent from GT
        self.excluded_bad_status = []      # ids present but status/errors excluded them
        self.missing_severity_fields = []  # (id, field) where GT had no severity entry

    def add_file(self, rid, gt_fields, severities, pred_fields):
        self.files_scored += 1
        raw_scores = field_level_scores_partial(pred_fields, gt_fields, FIELD_SCHEMA)
        for score_key, score in raw_scores.items():
            parent = parent_field_of(score_key)
            sev = severities.get(parent)
            if sev is None:
                self.missing_severity_fields.append((rid, parent))
                continue
            if sev not in SEVERITY_LEVELS:
                print(f"WARNING: id={rid} field={parent} has unrecognized severity "
                      f"'{sev}', skipping", file=sys.stderr)
                continue
            if sev == EXCLUDED_SEVERITY:
                continue  # high-blur GT is unreliable -- excluded from scoring by design
            self.scores[score_key][sev].append(score)

    def report_rows(self):
        """Yields dicts, one per (field_key, severity) with n / strict_correct /
        strict_pct / weighted_pct, sorted for readability."""
        all_field_keys = sorted(self.scores.keys())
        for field_key in all_field_keys:
            for sev in ["none", "low", "medium"]:
                vals = self.scores[field_key].get(sev, [])
                if not vals:
                    continue
                n = len(vals)
                strict_correct = sum(1 for v in vals if v == 1.0)
                weighted_sum = sum(vals)
                yield {
                    "model": self.model_name,
                    "field": field_key,
                    "severity": sev,
                    "n": n,
                    "strict_correct": strict_correct,
                    "strict_pct": round(100 * strict_correct / n, 1),
                    "weighted_pct": round(100 * weighted_sum / n, 1),
                }

    def overall_by_severity(self):
        """Pooled across ALL fields, per severity bucket."""
        pooled = defaultdict(list)
        for field_key, by_sev in self.scores.items():
            for sev, vals in by_sev.items():
                pooled[sev].extend(vals)
        out = {}
        for sev in ["none", "low", "medium"]:
            vals = pooled.get(sev, [])
            if not vals:
                out[sev] = None
                continue
            n = len(vals)
            strict_correct = sum(1 for v in vals if v == 1.0)
            out[sev] = {
                "n": n,
                "strict_correct": strict_correct,
                "strict_pct": round(100 * strict_correct / n, 1),
                "weighted_pct": round(100 * sum(vals) / n, 1),
            }
        return out


def run_model(model_name, gt_by_id, pred_by_id, status_ok_fn, get_pred_fields_fn):
    acc = Accumulator(model_name)

    # ids that exist in this model's file but not in GT at all -- "vice versa"
    # case from the spec. Reported, never silently dropped.
    acc.excluded_extra_id = sorted(set(pred_by_id.keys()) - set(gt_by_id.keys()))

    for rid, gt_row in gt_by_id.items():
        pred_row = pred_by_id.get(rid)
        if pred_row is None:
            acc.excluded_missing_id.append(rid)
            continue
        if not status_ok_fn(pred_row):
            acc.excluded_bad_status.append(rid)
            continue
        acc.add_file(rid, gt_row["gt_fields"], gt_row["field_blur_severity"],
                     get_pred_fields_fn(pred_row))
    return acc


def gt_severity_distribution(gt_by_id):
    """GT-only, model-independent: for each of the 19 top-level fields, count
    how many files have each severity bucket. This is the 'total fields /
    none / low / medium / high' breakdown, computed once."""
    counts = defaultdict(lambda: defaultdict(int))
    total_files = len(gt_by_id)
    for rid, gt_row in gt_by_id.items():
        sev_map = gt_row["field_blur_severity"]
        for field_def in FIELD_SCHEMA:
            fname = field_def["name"]
            sev = sev_map.get(fname)
            if sev in SEVERITY_LEVELS:
                counts[fname][sev] += 1
            else:
                counts[fname]["__missing__"] += 1
    return total_files, counts


def write_csv(rows, path: Path):
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expanded_field_count(schema=FIELD_SCHEMA) -> int:
    """Number of distinct *scoring keys* per fully-populated file. Top-level
    scalar fields count as 1 each; list fields (diagnoses/procedures) are
    expanded by field_level_scores_partial into one key per item_schema
    sub-field, so they count as len(item_schema) instead of 1. This is the
    number that actually matches the 'n' denominators seen in the per-field
    and overall accuracy tables -- len(FIELD_SCHEMA) alone (19) undercounts
    it because it doesn't account for that expansion (22 with the current
    schema: 17 scalar + 1 diagnoses sub-field + 4 procedures sub-fields)."""
    return sum(
        len(f["item_schema"]) if f.get("field_type") == "list" else 1
        for f in schema
    )


def format_summary(gt_by_id, qwen_acc: Accumulator, gpt_acc: Accumulator) -> str:
    lines = []
    total_files, gt_dist = gt_severity_distribution(gt_by_id)
    n_fields = len(FIELD_SCHEMA)
    n_expanded = expanded_field_count()

    lines.append("=" * 78)
    lines.append("GROUND TRUTH OVERVIEW")
    lines.append("=" * 78)
    lines.append(f"Total annotated files (GT):              {total_files}")
    lines.append(f"Top-level fields (schema):                {n_fields}")
    lines.append(f"Scoring keys per file (sub-fields expanded): {n_expanded}")
    lines.append(f"Total (file x top-level field) instances: {total_files * n_fields}")
    lines.append(f"Total (file x scoring key) instances:     {total_files * n_expanded}  "
                  f"(matches the 'n' denominators below)")

    sev_totals = defaultdict(int)
    for fname, by_sev in gt_dist.items():
        for sev, cnt in by_sev.items():
            sev_totals[sev] += cnt
    lines.append("")
    lines.append("Field-instance counts by GT severity (all fields pooled):")
    for sev in SEVERITY_LEVELS:
        lines.append(f"  {sev:8s}: {sev_totals.get(sev, 0)}")
    if sev_totals.get("__missing__"):
        lines.append(f"  {'MISSING':8s}: {sev_totals['__missing__']}  "
                      f"(no severity recorded -- excluded from scoring)")

    for acc in [qwen_acc, gpt_acc]:
        lines.append("")
        lines.append("=" * 78)
        lines.append(f"MODEL: {acc.model_name}")
        lines.append("=" * 78)
        lines.append(f"Files scored:                  {acc.files_scored}")
        lines.append(f"Excluded (id missing in file): {len(acc.excluded_missing_id)}")
        lines.append(f"Excluded (id not in GT):       {len(acc.excluded_extra_id)}")
        lines.append(f"Excluded (bad status/errors):  {len(acc.excluded_bad_status)}")
        if acc.missing_severity_fields:
            lines.append(f"Field instances w/ no GT severity (skipped): "
                          f"{len(acc.missing_severity_fields)}")

        lines.append("")
        lines.append("-- Overall accuracy, all fields pooled, by severity --")
        overall = acc.overall_by_severity()
        for sev in ["none", "low", "medium"]:
            stat = overall[sev]
            if stat is None:
                lines.append(f"  {sev:8s}: (no instances)")
                continue
            lines.append(
                f"  {sev:8s}: {stat['strict_correct']}/{stat['n']} correct "
                f"({stat['strict_pct']}%)   [weighted: {stat['weighted_pct']}%]"
            )

        lines.append("")
        lines.append("-- Per-field accuracy by severity --")
        lines.append(f"  {'field':38s} {'sev':8s} {'n':>4s}  {'correct':>7s}  "
                      f"{'strict%':>8s}  {'weighted%':>9s}")
        for row in acc.report_rows():
            lines.append(
                f"  {row['field']:38s} {row['severity']:8s} {row['n']:>4d}  "
                f"{row['strict_correct']:>7d}  {row['strict_pct']:>7.1f}%  "
                f"{row['weighted_pct']:>8.1f}%"
            )

    return "\n".join(lines)


def _md_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    """Build a GitHub-flavored markdown table as a list of lines."""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return out


def format_summary_markdown(gt_by_id, qwen_acc: Accumulator, gpt_acc: Accumulator) -> str:
    """Same content as format_summary(), rendered as markdown with tables
    instead of fixed-width text -- for readability in GitHub/docs viewers."""
    lines = []
    total_files, gt_dist = gt_severity_distribution(gt_by_id)
    n_fields = len(FIELD_SCHEMA)
    n_expanded = expanded_field_count()

    lines.append("# Extraction Metrics Report")
    lines.append("")
    lines.append("## Ground Truth Overview")
    lines.append("")
    lines.extend(_md_table(
        ["Metric", "Value"],
        [
            ["Total annotated files (GT)", total_files],
            ["Top-level fields (schema)", n_fields],
            ["Scoring keys per file (sub-fields expanded)", n_expanded],
            ["Total (file x top-level field) instances", total_files * n_fields],
            ["Total (file x scoring key) instances", total_files * n_expanded],
        ],
    ))

    sev_totals = defaultdict(int)
    for fname, by_sev in gt_dist.items():
        for sev, cnt in by_sev.items():
            sev_totals[sev] += cnt
    lines.append("")
    lines.append("### Field-instance counts by GT severity (all top-level fields pooled)")
    lines.append("")
    sev_rows = [[sev, sev_totals.get(sev, 0)] for sev in SEVERITY_LEVELS]
    if sev_totals.get("__missing__"):
        sev_rows.append(["MISSING (no severity recorded)", sev_totals["__missing__"]])
    lines.extend(_md_table(["Severity", "Count"], sev_rows))

    for acc in [qwen_acc, gpt_acc]:
        lines.append("")
        lines.append(f"## Model: {acc.model_name}")
        lines.append("")
        lines.extend(_md_table(
            ["Metric", "Value"],
            [
                ["Files scored", acc.files_scored],
                ["Excluded (id missing in file)", len(acc.excluded_missing_id)],
                ["Excluded (id not in GT)", len(acc.excluded_extra_id)],
                ["Excluded (bad status/errors)", len(acc.excluded_bad_status)],
                ["Field instances w/ no GT severity (skipped)", len(acc.missing_severity_fields)],
            ],
        ))

        lines.append("")
        lines.append("### Overall accuracy, all fields pooled, by severity")
        lines.append("")
        overall = acc.overall_by_severity()
        overall_rows = []
        for sev in ["none", "low", "medium"]:
            stat = overall[sev]
            if stat is None:
                overall_rows.append([sev, "-", "-", "-", "-"])
            else:
                overall_rows.append([
                    sev, stat["n"], stat["strict_correct"],
                    f"{stat['strict_pct']}%", f"{stat['weighted_pct']}%",
                ])
        lines.extend(_md_table(
            ["Severity", "n", "Strict correct", "Strict %", "Weighted %"],
            overall_rows,
        ))

        lines.append("")
        lines.append("### Per-field accuracy by severity")
        lines.append("")
        field_rows = [
            [row["field"], row["severity"], row["n"], row["strict_correct"],
             f"{row['strict_pct']}%", f"{row['weighted_pct']}%"]
            for row in acc.report_rows()
        ]
        lines.extend(_md_table(
            ["Field", "Severity", "n", "Strict correct", "Strict %", "Weighted %"],
            field_rows,
        ))

    return "\n".join(lines)


def main():
    gt_path = Path(GT_JSONL)
    qwen_path = Path(QWEN_JSONL)
    gpt_path = Path(GPT_JSONL)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_by_id = load_gt(gt_path)
    qwen_by_id = load_qwen(qwen_path)
    gpt_by_id = load_gpt(gpt_path)

    print(f"GT: {len(gt_by_id)} files")
    print(f"Qwen: {len(qwen_by_id)} files")
    print(f"GPT: {len(gpt_by_id)} files")

    qwen_acc = run_model(
        "Qwen", gt_by_id, qwen_by_id,
        status_ok_fn=lambda row: row.get("status") == "OK",
        get_pred_fields_fn=lambda row: row["predicted_fields"],
    )
    gpt_acc = run_model(
        "GPT", gt_by_id, gpt_by_id,
        status_ok_fn=lambda row: len(row.get("errors") or []) == 0,
        get_pred_fields_fn=lambda row: row["predicted_fields"],
    )

    # 1) Terminal
    summary = format_summary(gt_by_id, qwen_acc, gpt_acc)
    print("\n" + summary)

    # 2) Markdown report
    summary_md = format_summary_markdown(gt_by_id, qwen_acc, gpt_acc)

    (out_dir / "summary.txt").write_text(summary)
    (out_dir / "summary.md").write_text(summary_md)
    # 3) CSV (per-field x severity breakdown, one file per model)
    write_csv(list(qwen_acc.report_rows()), out_dir / "per_field_severity_qwen.csv")
    write_csv(list(gpt_acc.report_rows()), out_dir / "per_field_severity_gpt.csv")

    exclusions = {
        "qwen": {
            "excluded_missing_id": qwen_acc.excluded_missing_id,
            "excluded_extra_id_not_in_gt": qwen_acc.excluded_extra_id,
            "excluded_bad_status": qwen_acc.excluded_bad_status,
            "missing_severity_field_instances": qwen_acc.missing_severity_fields,
        },
        "gpt": {
            "excluded_missing_id": gpt_acc.excluded_missing_id,
            "excluded_extra_id_not_in_gt": gpt_acc.excluded_extra_id,
            "excluded_bad_status": gpt_acc.excluded_bad_status,
            "missing_severity_field_instances": gpt_acc.missing_severity_fields,
        },
    }
    (out_dir / "exclusions.json").write_text(json.dumps(exclusions, indent=2))

    print(f"\nWrote: {out_dir/'summary.txt'}")
    print(f"Wrote: {out_dir/'summary.md'}")
    print(f"Wrote: {out_dir/'per_field_severity_qwen.csv'}")
    print(f"Wrote: {out_dir/'per_field_severity_gpt.csv'}")
    print(f"Wrote: {out_dir/'exclusions.json'}")


if __name__ == "__main__":
    main()