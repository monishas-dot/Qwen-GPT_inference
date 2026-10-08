"""compare_qwen_gpt.py — field-by-field comparison of GPT-5.2 vs. Qwen
predictions on the same full-page extraction task, using the same match_type
normalization rules as config_extractor_full_page.yaml.

Usage:
    python compare_qwen_gpt.py <qwen_per_file_report.jsonl> <gpt_predictions.jsonl> <output_dir>

Outputs (all plain JSON, not JSONL, so they open nicely in a JSON tree
viewer / VS Code's built-in JSON folding):
    per_record_comparison.json   full detail for every commonly-id'd record,
                                  sorted worst-first (most mismatches on top)
    mismatches_only.json         same records, but ONLY the fields that
                                  disagree -- side-by-side qwen/gpt values.
                                  This is the one to eyeball first.
    summary.json                 aggregate per-field agreement rates
"""

import sys
import json
import re
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Schema: name -> match_type, mirrors config_extractor_full_page.yaml exactly.
# --------------------------------------------------------------------------
SCALAR_FIELDS = {
    "servicing_provider_npi": "code",
    "type_of_service": "fuzzy_text",
    "member_id": "code",
    "requesting_provider_phone": "phone",
    "acuity": "exact",
    "date_of_birth": "date",
    "servicing_provider_name": "fuzzy_text",
    "requesting_provider_name": "fuzzy_text",
    "member_name": "fuzzy_text",
    "place_of_service": "fuzzy_text",
    "requesting_provider_npi": "code",
    "requesting_provider_fax": "phone",
    "lob": "fuzzy_text",
    "servicing_provider_fax": "phone",
    "servicing_provider_phone": "phone",
    "servicing_provider_address": "fuzzy_text",
    "requesting_provider_address": "fuzzy_text",
}

# List fields: field name -> (item key -> match_type). First key in each dict
# is used as the "primary key" for greedy order-independent item matching.
LIST_FIELDS = {
    "diagnoses": {
        "diagnosisCode": "code",
    },
    "procedures": {
        "procedureCode": "code",
        "dateOfServiceStart": "date",
        "dateOfServiceEnd": "date",
        "numberOfVisits": "numeric",
    },
}

NULL_ALIASES = {"", "none", "null", "n/a", "na", "nil", "not found", "not specified", "unknown"}

# --------------------------------------------------------------------------
# Normalization / matching per match_type
# --------------------------------------------------------------------------
def _is_null(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in NULL_ALIASES:
        return True
    return False


def norm_text(v: Any) -> str:
    if _is_null(v):
        return ""
    s = str(v).strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_code(v: Any) -> str:
    if _is_null(v):
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(v)).upper()


def norm_phone(v: Any) -> str:
    if _is_null(v):
        return ""
    digits = re.sub(r"\D", "", str(v))
    return digits[-10:] if len(digits) >= 10 else digits


def norm_date(v: Any) -> Optional[Tuple[int, int, int]]:
    if _is_null(v):
        return None
    s = str(v).strip()
    s = re.sub(r"[.\-]", "/", s)
    parts = s.split("/")
    if len(parts) != 3:
        return None
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    # Assumes M/D/Y ordering (matches "3/18/2026" style seen in the data).
    m, d, y = nums
    if y < 100:
        y += 2000
    return (y, m, d)


def norm_numeric(v: Any) -> Optional[float]:
    if _is_null(v):
        return None
    s = str(v).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def values_match(a: Any, b: Any, match_type: str) -> bool:
    a_null, b_null = _is_null(a), _is_null(b)
    if a_null and b_null:
        return True
    if a_null != b_null:
        return False

    if match_type == "fuzzy_text":
        na, nb = norm_text(a), norm_text(b)
        if na == nb:
            return True
        return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.85
    elif match_type == "code":
        return norm_code(a) == norm_code(b)
    elif match_type == "phone":
        return norm_phone(a) == norm_phone(b)
    elif match_type == "date":
        return norm_date(a) == norm_date(b)
    elif match_type == "numeric":
        na, nb = norm_numeric(a), norm_numeric(b)
        return na is not None and nb is not None and na == nb
    else:  # exact / text
        return norm_text(a) == norm_text(b)


# --------------------------------------------------------------------------
# List-field comparison (order-independent, greedy match on primary key)
# --------------------------------------------------------------------------
def compare_list_field(list_a: Any, list_b: Any, item_schema: Dict[str, str]) -> Dict[str, Any]:
    primary_key = next(iter(item_schema))
    primary_type = item_schema[primary_key]

    # Coerce non-list values (like error strings) to empty lists safely
    if not isinstance(list_a, list):
        list_a = []
    if not isinstance(list_b, list):
        list_b = []

    used_b = set()
    matched_pairs = []
    for item_a in list_a:
        if not isinstance(item_a, dict):
            continue
        best_j = None
        for jb, item_b in enumerate(list_b):
            if jb in used_b or not isinstance(item_b, dict):
                continue
            if values_match(item_a.get(primary_key), item_b.get(primary_key), primary_type):
                best_j = jb
                break
        if best_j is not None:
            used_b.add(best_j)
            matched_pairs.append((item_a, list_b[best_j]))

    matched_a_indices = {id(pair[0]) for pair in matched_pairs}
    unmatched_a = [item for item in list_a if isinstance(item, dict) and id(item) not in matched_a_indices]
    unmatched_b = [item for j, item in enumerate(list_b) if isinstance(item, dict) and j not in used_b]

    per_subfield_agree = {k: [] for k in item_schema}
    subfield_mismatches = []  # detail, only for matched pairs with a disagreement
    for item_a, item_b in matched_pairs:
        row_mismatches = {}
        for key, mtype in item_schema.items():
            match = values_match(item_a.get(key), item_b.get(key), mtype)
            per_subfield_agree[key].append(match)
            if not match:
                row_mismatches[key] = {"qwen": item_a.get(key), "gpt": item_b.get(key)}
        if row_mismatches:
            subfield_mismatches.append({
                "matched_on": item_a.get(primary_key),
                "mismatched_subfields": row_mismatches,
            })

    per_subfield_rate = {
        k: (sum(v) / len(v) if v else None) for k, v in per_subfield_agree.items()
    }

    denom = max(len(list_a), len(list_b))
    item_level_agreement = (len(matched_pairs) / denom) if denom else 1.0

    n_mismatches = len(unmatched_a) + len(unmatched_b) + len(subfield_mismatches)

    return {
        "item_level_agreement": item_level_agreement,
        "n_items_qwen": len(list_a),
        "n_items_gpt": len(list_b),
        "n_matched_items": len(matched_pairs),
        "per_subfield_agreement": per_subfield_rate,
        "n_mismatches": n_mismatches,
        "detail": {
            "unmatched_in_qwen_only": unmatched_a,
            "unmatched_in_gpt_only": unmatched_b,
            "subfield_mismatches_in_matched_items": subfield_mismatches,
        } if n_mismatches else None,
    }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_qwen(path: str) -> Dict[str, Dict]:
    """predict_extraction.py output: fields at row['predicted_fields']."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "OK" and row.get("predicted_fields") is not None:
                out[row["id"]] = row["predicted_fields"]
    return out


def load_gpt(path: str) -> Dict[str, Dict]:
    """run_openai_full_page.py output: fields at row['prediction']['fields']."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("errors") and row.get("prediction", {}).get("fields") is not None:
                out[row["id"]] = row["prediction"]["fields"]
    return out


# --------------------------------------------------------------------------
# Main comparison
# --------------------------------------------------------------------------
def compare_record(qwen_fields: Dict, gpt_fields: Dict) -> Dict[str, Any]:
    scalar_detail = {}
    n_scalar_mismatches = 0
    for name, mtype in SCALAR_FIELDS.items():
        qv, gv = qwen_fields.get(name), gpt_fields.get(name)
        match = values_match(qv, gv, mtype)
        scalar_detail[name] = {"match": match, "qwen": qv, "gpt": gv}
        if not match:
            n_scalar_mismatches += 1

    list_detail = {}
    n_list_mismatches = 0
    for name, item_schema in LIST_FIELDS.items():
        result = compare_list_field(qwen_fields.get(name) or [], gpt_fields.get(name) or [], item_schema)
        list_detail[name] = result
        n_list_mismatches += result["n_mismatches"]

    scalar_matches = [v["match"] for v in scalar_detail.values()]
    scalar_agreement = sum(scalar_matches) / len(scalar_matches) if scalar_matches else None

    return {
        "scalar_fields": scalar_detail,
        "list_fields": list_detail,
        "scalar_agreement_rate": scalar_agreement,
        "total_mismatches": n_scalar_mismatches + n_list_mismatches,
    }


def build_mismatch_only_view(record_id: str, comparison: Dict) -> Optional[Dict]:
    """Strip a full comparison record down to just the fields that disagree,
    for fast eyeballing. Returns None if there are no mismatches at all."""
    if comparison["total_mismatches"] == 0:
        return None

    mismatched_scalars = {
        name: {"qwen": v["qwen"], "gpt": v["gpt"]}
        for name, v in comparison["scalar_fields"].items() if not v["match"]
    }
    mismatched_lists = {
        name: v["detail"]
        for name, v in comparison["list_fields"].items() if v["n_mismatches"]
    }

    return {
        "id": record_id,
        "total_mismatches": comparison["total_mismatches"],
        "mismatched_scalar_fields": mismatched_scalars,
        "mismatched_list_fields": mismatched_lists,
    }


def main():
    if len(sys.argv) != 4:
        print("Usage: python compare_qwen_gpt.py <qwen.jsonl> <gpt.jsonl> <output_dir>")
        sys.exit(1)

    qwen_path, gpt_path, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qwen = load_qwen(qwen_path)
    gpt = load_gpt(gpt_path)

    qwen_ids, gpt_ids = set(qwen), set(gpt)
    common_ids = sorted(qwen_ids & gpt_ids)
    only_qwen = sorted(qwen_ids - gpt_ids)
    only_gpt = sorted(gpt_ids - qwen_ids)

    print(f"Qwen records: {len(qwen_ids)}")
    print(f"GPT records:  {len(gpt_ids)}")
    print(f"Common ids:   {len(common_ids)}")
    if only_qwen:
        print(f"  In Qwen only ({len(only_qwen)}): {only_qwen[:5]}{' ...' if len(only_qwen) > 5 else ''}")
    if only_gpt:
        print(f"  In GPT only ({len(only_gpt)}): {only_gpt[:5]}{' ...' if len(only_gpt) > 5 else ''}")

    full_records = []
    mismatch_records = []
    scalar_field_totals = {name: [] for name in SCALAR_FIELDS}
    list_field_totals = {name: [] for name in LIST_FIELDS}

    for rid in common_ids:
        comparison = compare_record(qwen[rid], gpt[rid])
        full_records.append({"id": rid, **comparison})

        mismatch_view = build_mismatch_only_view(rid, comparison)
        if mismatch_view:
            mismatch_records.append(mismatch_view)

        for name in SCALAR_FIELDS:
            scalar_field_totals[name].append(comparison["scalar_fields"][name]["match"])
        for name in LIST_FIELDS:
            list_field_totals[name].append(comparison["list_fields"][name]["item_level_agreement"])

    # Worst-first: most total mismatches on top, for fast eyeballing.
    full_records.sort(key=lambda r: r["total_mismatches"], reverse=True)
    mismatch_records.sort(key=lambda r: r["total_mismatches"], reverse=True)

    with open(output_dir / "per_record_comparison.json", "w") as f:
        json.dump(full_records, f, indent=2, default=str)

    with open(output_dir / "mismatches_only.json", "w") as f:
        json.dump(mismatch_records, f, indent=2, default=str)

    summary = {
        "n_common_records": len(common_ids),
        "n_records_with_any_mismatch": len(mismatch_records),
        "n_records_perfect_match": len(common_ids) - len(mismatch_records),
        "n_only_qwen": len(only_qwen),
        "n_only_gpt": len(only_gpt),
        "only_qwen_ids": only_qwen,
        "only_gpt_ids": only_gpt,
        "scalar_field_agreement_rate": {
            name: (sum(v) / len(v) if v else None) for name, v in scalar_field_totals.items()
        },
        "list_field_item_level_agreement_rate": {
            name: (sum(v) / len(v) if v else None) for name, v in list_field_totals.items()
        },
        "overall_scalar_agreement_rate": (
            sum(r["scalar_agreement_rate"] for r in full_records if r["scalar_agreement_rate"] is not None)
            / len(full_records) if full_records else None
        ),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY — Qwen vs GPT-5.2")
    print("=" * 70)
    print(f"  Common records compared: {summary['n_common_records']}")
    print(f"  Perfect matches:         {summary['n_records_perfect_match']}")
    print(f"  Records with mismatches: {summary['n_records_with_any_mismatch']}")
    if summary["overall_scalar_agreement_rate"] is not None:
        print(f"  Overall scalar-field agreement: {summary['overall_scalar_agreement_rate']:.4f}")
    print("\n  Per scalar field agreement rate (lowest first):")
    for name, rate in sorted(summary["scalar_field_agreement_rate"].items(), key=lambda kv: (kv[1] is None, kv[1])):
        rate_str = f"{rate:.4f}" if rate is not None else "n/a"
        print(f"    {name:32s} {rate_str}")
    print("\n  List-field item-level agreement rate:")
    for name, rate in summary["list_field_item_level_agreement_rate"].items():
        rate_str = f"{rate:.4f}" if rate is not None else "n/a"
        print(f"    {name:32s} {rate_str}")

    print(f"\nEyeball this first: {output_dir / 'mismatches_only.json'}")
    print(f"Full detail:        {output_dir / 'per_record_comparison.json'}")
    print(f"Summary:            {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
