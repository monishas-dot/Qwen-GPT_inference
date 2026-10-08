"""field_matching.py — schema-aware, format-tolerant field comparison for the
extraction task's evaluation.

Why this exists: raw string-equality on JSON field values counts formatting
noise as failures — punctuation, casing, "null" vs None, decimal-less
diagnosis codes, differently-ordered date separators, and shuffled list
order. This module distinguishes that noise from real extraction errors, so
reported metrics reflect what the model actually got wrong.

`match_type` is set per-field in the config's `fields:` list:
  - "exact" / "text" (default) : normalized-string match — case, whitespace,
                                  punctuation, and null-alias insensitive.
  - "fuzzy_text"                : same as above, plus a similarity-ratio
                                  fallback (FUZZY_TEXT_THRESHOLD) for free-text
                                  fields prone to minor OCR/spelling variance
                                  (names, addresses, service descriptions).
  - "code"                      : alphanumeric-only match — for diagnosis/
                                  procedure codes and NPIs. Ignores decimals,
                                  dashes, spaces: "F329" == "F32.9" == "f32-9".
  - "phone"                     : digits-only match, for phone/fax numbers.
  - "date"                      : calendar-date match across separator/
                                  digit-count variants: "3/16/26" ==
                                  "03-16-2026" == "03.16.2026".
  - "numeric"                   : numeric match ignoring formatting (commas,
                                  trailing .0, leading zeros).

List-of-object fields (field_type: "list", e.g. diagnoses/procedures) are
scored as an order-independent multiset via greedy matching: each
ground-truth item is matched against the best unclaimed predicted item using
each sub-field's own match_type (from the field's item_schema), so shuffled
procedure order or a sub-field formatting quirk no longer fails the whole
list. Score = matched_count / max(len(gt), len(pred)); document_exact_match
still requires a *complete* match (score == 1.0) to count fields/lists as
passing, so partial-credit scoring never masks a genuinely incomplete
extraction at the document level.

For list fields whose items are plain scalars rather than objects (no
item_schema declared), the field's own top-level `match_type` is used to
compare items instead of falling back to raw Python equality — otherwise a
bare scalar list (e.g. a list of diagnosis codes) would silently lose all the
format-tolerance this module exists to provide.

Sub-field partial credit within list items (field_level_scores_partial):
instead of an all-or-nothing per-item match, each gt item is paired with its
best-overlap predicted item (by fraction of sub-fields matching), and each
sub-field of the item_schema is then scored independently across the whole
list, e.g. "procedures.procedureCode", "procedures.dateOfServiceStart",
"procedures.dateOfServiceEnd", "procedures.numberOfVisits" as four separate
per-list metrics instead of one collapsed "procedures" score. This means a
procedure entry with a right code/dates but wrong numberOfVisits contributes
to the code/date metrics being higher while only numberOfVisits absorbs the
penalty. The item *pairing* itself is still computed once per list (best
overall sub-field overlap, greedy, order-independent) and that same pairing
is reused across all of that list's sub-field metrics. This is used only for
field-level accuracy reporting; document_exact_match (below) is unaffected
and still requires every list item to fully match on every sub-field.
"""

import re
import difflib
from datetime import datetime
from typing import Any, Dict, List

_NULL_ALIASES = {"", "null", "none", "n/a", "na", "nil"}
FUZZY_TEXT_THRESHOLD = 0.90
_DATE_SPLIT_RE = re.compile(r"[\-/.\s]+")


def _is_null(v: Any) -> bool:
    """True for None and common string aliases of null ('null', 'n/a', ''...)."""
    return v is None or (isinstance(v, str) and v.strip().lower() in _NULL_ALIASES)


def _clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _norm_text(v: Any) -> str:
    """Casing/whitespace/null-alias normalized string (punctuation kept)."""
    if _is_null(v):
        return ""
    return _clean_whitespace(str(v)).lower()


def _norm_text_no_punct(v: Any) -> str:
    """_norm_text with punctuation stripped — the base for exact/fuzzy text
    comparisons, so dash/comma/period formatting differences don't count."""
    t = _norm_text(v)
    return _clean_whitespace(re.sub(r"[^\w\s]", " ", t))


def _norm_alnum(v: Any) -> str:
    """Alphanumeric-only, lowercased — for code/id-like fields. Strips
    decimals, dashes, spaces so 'F32.9' / 'F329' / 'f32-9' compare equal."""
    if _is_null(v):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def _norm_digits(v: Any) -> str:
    """Digits-only — for phone/fax numbers."""
    if _is_null(v):
        return ""
    return re.sub(r"\D", "", str(v))


def _norm_numeric(v: Any) -> str:
    """Numeric value stripped of thousands separators / trailing .0 /
    leading zeros, e.g. '048' == '48' == '48.0' -> '48'."""
    if _is_null(v):
        return ""
    s = re.sub(r"[,\s]", "", str(v))
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s.lower()


def _norm_date(v: Any) -> str:
    """Canonicalize to YYYY-MM-DD regardless of separator (-, /, ., space) or
    2- vs 4-digit year, assuming US month/day/year convention (matches the
    source SRF documents). Falls back to normalized text if it isn't a
    3-part numeric date, rather than raising."""
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
    """Compare a single pred/gt scalar pair under the given match_type."""
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
    # "exact" / "text" / unset -> punctuation/case/whitespace-tolerant exact match
    return _norm_text_no_punct(v_pred) == _norm_text_no_punct(v_gt)


def _item_schema_lookup(field_def: Dict[str, Any]) -> Dict[str, str]:
    """match_type per sub-key for a list-of-object field, from its item_schema."""
    return {s["name"]: s.get("match_type", "text") for s in field_def.get("item_schema", [])}


def _scalar_item_match_type(field_def: Dict[str, Any]) -> str:
    """match_type to use when a list field's items are plain scalars (no
    item_schema) — falls back to the field's own top-level match_type, e.g. a
    bare list of diagnosis codes declaring match_type: "code" at the field
    level rather than under item_schema."""
    return field_def.get("match_type", "text")


def _list_item_matches(
    pred_item: Any, gt_item: Any, sub_types: Dict[str, str], scalar_match_type: str
) -> bool:
    """A list item counts as a match only if every ground-truth sub-key matches
    under its declared match_type. Plain-scalar items (no item_schema) are
    compared via _values_match under the field's own match_type, not raw `==`,
    so format-tolerance still applies to scalar lists."""
    if not isinstance(pred_item, dict) or not isinstance(gt_item, dict):
        return _values_match(pred_item, gt_item, scalar_match_type)
    for key, gt_val in gt_item.items():
        if not _values_match(pred_item.get(key), gt_val, sub_types.get(key, "text")):
            return False
    return True


def _score_list_field(pred_val: Any, gt_val: Any, field_def: Dict[str, Any]) -> float:
    """Order-independent greedy matching for list-of-object fields (diagnoses,
    procedures). 1.0 if both empty/absent; otherwise matched_count / max(len)."""
    gt_list = gt_val if isinstance(gt_val, list) else ([] if _is_null(gt_val) else [gt_val])
    pred_list = pred_val if isinstance(pred_val, list) else ([] if _is_null(pred_val) else [pred_val])

    if not gt_list and not pred_list:
        return 1.0
    if not gt_list or not pred_list:
        return 0.0

    sub_types = _item_schema_lookup(field_def)
    scalar_match_type = _scalar_item_match_type(field_def)
    remaining_pred = list(pred_list)
    matched = 0
    for gt_item in gt_list:
        hit_idx = next(
            (i for i, p in enumerate(remaining_pred)
             if _list_item_matches(p, gt_item, sub_types, scalar_match_type)),
            None,
        )
        if hit_idx is not None:
            matched += 1
            remaining_pred.pop(hit_idx)

    return matched / max(len(gt_list), len(pred_list))


def _pair_list_items(
    gt_list: List[Any], pred_list: List[Any], sub_types: Dict[str, str], scalar_match_type: str
) -> List[Any]:
    """Pair each gt item with its best-overlap unclaimed pred item (greedy, by
    descending overlap score), for later per-sub-field scoring. Overlap score
    for a (gt_item, pred_item) pair is the fraction of the gt item's declared
    sub-fields (per sub_types) that match under their own match_type -- not
    an all-or-nothing item match. For plain-scalar items (no item_schema),
    overlap is 1.0/0.0 based on _values_match under scalar_match_type. Ties
    broken by earliest gt/pred index.

    Returns a list, one entry per gt item, of the paired pred item's index in
    the ORIGINAL pred_list, or None if no pred item was left to pair with it.
    """
    def _overlap(gt_item: Any, pred_item: Any) -> float:
        if not isinstance(gt_item, dict) or not isinstance(pred_item, dict):
            return 1.0 if _values_match(pred_item, gt_item, scalar_match_type) else 0.0
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
    """Partial-credit, per-sub-field scores for a list-of-object field, e.g.
    {"procedureCode": 0.92, "dateOfServiceStart": 0.87, "dateOfServiceEnd":
    0.87, "numberOfVisits": 0.61}. See module docstring for the algorithm.
    Empty-schema (plain scalar list) fields fall back to a single "value" key,
    scored under the field's own top-level match_type.
    """
    gt_list = gt_val if isinstance(gt_val, list) else ([] if _is_null(gt_val) else [gt_val])
    pred_list = pred_val if isinstance(pred_val, list) else ([] if _is_null(pred_val) else [pred_val])

    sub_types = _item_schema_lookup(field_def)
    scalar_match_type = _scalar_item_match_type(field_def)
    sub_keys = list(sub_types.keys()) or ["value"]

    if not gt_list and not pred_list:
        return {k: 1.0 for k in sub_keys}
    if not gt_list or not pred_list:
        return {k: 0.0 for k in sub_keys}

    denom = max(len(gt_list), len(pred_list))
    pairing = _pair_list_items(gt_list, pred_list, sub_types, scalar_match_type)

    scores = {}
    for key in sub_keys:
        matched = 0
        for gi, gt_item in enumerate(gt_list):
            pi = pairing[gi]
            if pi is None:
                continue
            pred_item = pred_list[pi]
            if not isinstance(gt_item, dict) or not isinstance(pred_item, dict):
                matched += int(_values_match(pred_item, gt_item, scalar_match_type))
                continue
            if _values_match(pred_item.get(key), gt_item.get(key), sub_types.get(key, "text")):
                matched += 1
        scores[key] = matched / denom

    return scores


def field_level_scores_partial(
    pred_fields: Dict[str, Any], gt_fields: Dict[str, Any], schema: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Like field_level_scores, but list-of-object fields (diagnoses,
    procedures) are expanded into one metric per sub-field instead of one
    collapsed score, e.g. "procedures" becomes "procedures.procedureCode",
    "procedures.dateOfServiceStart", "procedures.dateOfServiceEnd",
    "procedures.numberOfVisits" -- each with independent partial credit.
    Scalar fields are scored exactly as in field_level_scores (unchanged).

    This is the function to use for field-level accuracy REPORTING under the
    new partial-credit rules. document_exact_match is unaffected by this and
    should keep using field_level_scores/score_field (strict, unchanged).
    """
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


def all_score_keys(schema: List[Dict[str, Any]]) -> List[str]:
    """Full list of field_level_scores_partial's output keys for a given
    schema, independent of any particular gt/pred document. List-of-object
    fields expand to "name.subkey" per item_schema entry (or "name.value" if
    item_schema is absent, i.e. a bare scalar list); scalar fields keep their
    bare "name".

    Used to keep failed-parse rows weighted identically to successfully-parsed
    rows when pooling per-field scores over a split/batch (e.g. in
    get_compute_metrics_extraction and evaluate_extraction.py) -- without
    this, a list field would silently contribute fewer score entries on
    failure than on success, under-weighting failures in the aggregate.
    """
    keys: List[str] = []
    for f in schema:
        if f.get("field_type") == "list":
            sub_keys = list(_item_schema_lookup(f).keys()) or ["value"]
            keys.extend(f"{f['name']}.{sk}" for sk in sub_keys)
        else:
            keys.append(f["name"])
    return keys


def score_field(pred_val: Any, gt_val: Any, field_def: Dict[str, Any]) -> float:
    """Score a single top-level field per its schema-declared field_type/match_type."""
    if field_def.get("field_type") == "list":
        return _score_list_field(pred_val, gt_val, field_def)
    return 1.0 if _values_match(pred_val, gt_val, field_def.get("match_type", "text")) else 0.0


def field_level_scores(
    pred_fields: Dict[str, Any], gt_fields: Dict[str, Any], schema: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Schema-aware per-field score in [0.0, 1.0] (list fields can get partial
    credit for partial overlap). One entry per key present in gt_fields; a
    predicted dict missing a key still scores 0.0 for that key, never skipped."""
    by_name = {f["name"]: f for f in schema}
    scores = {}
    for key, gt_val in gt_fields.items():
        pred_val = pred_fields.get(key) if isinstance(pred_fields, dict) else None
        scores[key] = score_field(pred_val, gt_val, by_name.get(key, {"name": key}))
    return scores


def document_exact_match(
    pred_fields: Dict[str, Any], gt_fields: Dict[str, Any], schema: List[Dict[str, Any]]
) -> float:
    """1.0 only if every field scores a full 1.0 (list fields must be a
    *complete* match, not just partial overlap), else 0.0."""
    scores = field_level_scores(pred_fields, gt_fields, schema)
    return 1.0 if scores and all(v == 1.0 for v in scores.values()) else 0.0