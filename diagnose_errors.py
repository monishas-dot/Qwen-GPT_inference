"""diagnose_errors.py - classify every scalar-field error of Qwen (and GPT) against GOLD.
Run from the project root:  python diagnose_errors.py
Prints COUNTS ONLY (no patient values), so the output is safe to paste.

Per field it reports:
  missed    model said null, gold has a value
  spurious  model gave a value, gold is null
  near      both present, wrong, but within 2 edits (digit misread -> resolution problem)
  swapped   wrong value equals ANOTHER field's gold (requesting<->servicing, phone<->fax) -> prompt problem
  other     both present, wrong, none of the above
"""
import json
import re
import sys

import yaml

GT_JSONL = "gold_set_annotations_final_repaired.jsonl"
QWEN_JSONL = "runs/prompt_v5/qwen/per_file_report.jsonl"
GPT_JSONL = "gpt output/gpt_prompt_v5_output.jsonl"
CONFIG = "qwen2-2.5vl-3b_model_scripts/configs/config_extraction_v5.yaml"

sys.path.insert(0, "qwen2-2.5vl-3b_model_scripts")
from src.evaluation import field_matching as fm  # noqa: E402

NULLS = {"", "null", "none", "n/a", "na", "nan"}


def is_null(v):
    if v is None:
        return True
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return str(v).strip().lower() in NULLS


def norm(v, mt):
    s = str(v)
    if mt == "phone":
        return re.sub(r"\D", "", s)
    return re.sub(r"[^0-9A-Za-z]", "", s).lower()


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load(path):
    rows = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if r.get("id"):
                rows[r["id"]] = r
    return rows


def partners(name):
    """other scalar fields a value could have been confused with"""
    out = []
    m = re.match(r"(requesting|servicing)_provider_(\w+)$", name)
    if not m:
        return out
    side, kind = m.groups()
    other = "servicing" if side == "requesting" else "requesting"
    out.append(f"{other}_provider_{kind}")
    if kind in ("phone", "fax"):
        out.append(f"{side}_provider_{'fax' if kind == 'phone' else 'phone'}")
    return out



from collections import Counter, defaultdict


def date_parts(v):
    nums = [int(x) for x in re.findall(r"\d+", str(v))]
    if len(nums) != 3:
        return None
    a, b, c = nums
    if a > 31:                      # YYYY-MM-DD
        y, m, d = a, b, c
    else:                           # MM/DD/YY(YY)
        m, d, y = a, b, c
    if y < 100:
        y += 2000
    return (m, d, y)


def struct(mt, pv, gv):
    """label the shape of a wrong answer (both values present)"""
    if mt == "date":
        p, g = date_parts(pv), date_parts(gv)
        if not p or not g:
            return "unparsable"
        if sorted(p[:2]) == sorted(g[:2]) and p[2] == g[2]:
            return "day/month swapped"
        same = sum(a == b for a, b in zip(p, g))
        return {2: "one part off", 1: "two parts off", 0: "all parts off"}.get(same, "other")
    tp = set(re.findall(r"[0-9a-z]+", str(pv).lower()))
    tg = set(re.findall(r"[0-9a-z]+", str(gv).lower()))
    if not tp or not tg:
        return "empty tokens"
    if tp == tg:
        return "same words, different order/format"
    if tp < tg:
        return "pred is subset of gold (too short)"
    if tg < tp:
        return "pred is superset of gold (too long)"
    return "some words shared" if tp & tg else "no words shared"


def char_pairs(a, b):
    return [(x, y) for x, y in zip(a, b) if x != y] if len(a) == len(b) else []

gt = load(GT_JSONL)
schema = yaml.safe_load(open(CONFIG))["fields"]
mtype = {f["name"]: f.get("match_type", "exact") for f in schema}
scalars = [f["name"] for f in schema if f.get("field_type") != "list"]

models = {
    "QWEN": {i: r.get("predicted_fields") or {} for i, r in load(QWEN_JSONL).items() if r.get("status") == "OK"},
    "GPT": {i: ((r.get("prediction") or {}).get("fields") or {}) for i, r in load(GPT_JSONL).items() if not r.get("errors")},
}

for label, preds in models.items():
    structs, confusions = defaultdict(Counter), Counter()
    print(f"\n{'=' * 100}\n{label}  ({len(preds)} predictions)\n{'=' * 100}")
    print(f"{'field':30s} {'n':>4s} {'ok':>4s} {'missed':>7s} {'spur.':>6s} {'near':>5s} {'swap':>5s} {'other':>6s}  swapped-with")
    for name in scalars:
        mt = mtype[name]
        c = dict(n=0, ok=0, missed=0, spur=0, near=0, swap=0, other=0)
        swaps = {}
        for rid, g in gt.items():
            if rid not in preds:
                continue
            gv, pv = (g.get("gt_fields") or {}).get(name), preds[rid].get(name)
            c["n"] += 1
            try:
                ok = fm._values_match(pv, gv, mt)
            except Exception:
                ok = False
            if ok or (is_null(pv) and is_null(gv)):
                c["ok"] += 1
            elif is_null(pv):
                c["missed"] += 1
            elif is_null(gv):
                c["spur"] += 1
            else:
                hit = None
                for p in partners(name):
                    ov = (g.get("gt_fields") or {}).get(p)
                    if not is_null(ov):
                        try:
                            if fm._values_match(pv, ov, mt):
                                hit = p
                                break
                        except Exception:
                            pass
                if hit:
                    c["swap"] += 1
                    swaps[hit] = swaps.get(hit, 0) + 1
                elif mt in ("code", "phone") and lev(norm(pv, mt), norm(gv, mt)) <= 2:
                    c["near"] += 1
                    confusions.update(char_pairs(norm(gv, mt), norm(pv, mt)))
                else:
                    c["other"] += 1
                    structs[name][struct(mt, pv, gv)] += 1
        sw = ", ".join(f"{k}x{v}" for k, v in swaps.items())
        print(f"{name:30s} {c['n']:4d} {c['ok']:4d} {c['missed']:7d} {c['spur']:6d} {c['near']:5d} {c['swap']:5d} {c['other']:6d}  {sw}")

    print("\nShape of the 'other' errors (both values present, wrong):")
    for name, cnt in structs.items():
        print(f"  {name:30s} " + ", ".join(f"{k}: {v}" for k, v in cnt.most_common(5)))
    print("\nDigit/character confusions in near-misses (gold -> model), top 12:")
    print("  " + ", ".join(f"{g}->{p} x{n}" for (g, p), n in confusions.most_common(12)))

    print("\nList fields: predicted row count vs gold row count (per document)")
    for lf in [f["name"] for f in schema if f.get("field_type") == "list"]:
        fewer = same = more = 0
        for rid, g in gt.items():
            if rid not in preds:
                continue
            gl = (g.get("gt_fields") or {}).get(lf) or []
            pl = preds[rid].get(lf) or []
            gl = gl if isinstance(gl, list) else []
            pl = pl if isinstance(pl, list) else []
            fewer += len(pl) < len(gl)
            same += len(pl) == len(gl)
            more += len(pl) > len(gl)
        print(f"  {lf:12s} fewer={fewer}  same={same}  more={more}")