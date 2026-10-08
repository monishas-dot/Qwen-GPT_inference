"""diagnose_fields.py - per-field accuracy of Qwen and GPT against GOLD, plus the
value vocabulary for category fields. Run from the project root:

    python diagnose_fields.py

Edit the 4 paths below. Output has no patient identifiers for the category fields
(only values like Outpatient / Medicaid). Check before sharing it.
"""
import json
import sys
from collections import Counter, defaultdict

import yaml

GT_JSONL = "/Users/monishasp/Downloads/prompt_ops_reqd/gold_set_annotations_final_repaired.jsonl"
QWEN_JSONL = "/Users/monishasp/Downloads/prompt_ops_reqd/runs/prompt_v3/qwen/per_file_report.jsonl"   # predicted_fields, status
GPT_JSONL = "/Users/monishasp/Downloads/prompt_ops_reqd/gpt output/gpt.jsonl"                              # prediction.fields, errors
CONFIG = "/Users/monishasp/Downloads/prompt_ops_reqd/qwen2-2.5vl-3b_model_scripts/configs/config_extraction_v3.yaml"

sys.path.insert(0, "qwen2-2.5vl-3b_model_scripts")
from src.evaluation import field_matching as fm  # noqa: E402

CATEGORY_FIELDS = ["acuity", "lob", "type_of_service", "place_of_service"]


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("id"):
                    rows[r["id"]] = r
    return rows


gt = load(GT_JSONL)
qwen = load(QWEN_JSONL)
gpt = load(GPT_JSONL)
schema = yaml.safe_load(open(CONFIG))["fields"]
scalars = [f for f in schema if f.get("field_type") != "list"]

q_pred = {i: (r.get("predicted_fields") or {}) for i, r in qwen.items() if r.get("status") == "OK"}
g_pred = {i: ((r.get("prediction") or {}).get("fields") or {}) for i, r in gpt.items() if not r.get("errors")}

print(f"gold={len(gt)} qwen_ok={len(q_pred)} gpt_ok={len(g_pred)}\n")
print(f"{'field':32s} {'n':>4s} {'qwen':>7s} {'gpt':>7s}")
for f in scalars:
    name, mt = f["name"], f.get("match_type", "exact")
    res = {"q": [0, 0], "g": [0, 0]}
    for rid, g in gt.items():
        gold_val = (g.get("gt_fields") or {}).get(name)
        for key, preds in (("q", q_pred), ("g", g_pred)):
            if rid not in preds:
                continue
            res[key][1] += 1
            try:
                ok = fm._values_match(preds[rid].get(name), gold_val, mt)
            except Exception:
                ok = False
            res[key][0] += bool(ok)
    pct = lambda a: f"{100 * a[0] / a[1]:6.1f}%" if a[1] else "    n/a"
    print(f"{name:32s} {res['q'][1]:4d} {pct(res['q'])} {pct(res['g'])}")

print("\n=== Category vocabulary (gold value -> what each model said) ===")
for name in CATEGORY_FIELDS:
    gold_counts = Counter()
    confusions = {"qwen": defaultdict(Counter), "gpt": defaultdict(Counter)}
    for rid, g in gt.items():
        gv = (g.get("gt_fields") or {}).get(name)
        gold_counts[str(gv)] += 1
        for label, preds in (("qwen", q_pred), ("gpt", g_pred)):
            if rid in preds:
                confusions[label][str(gv)][str(preds[rid].get(name))] += 1
    print(f"\n[{name}] gold values: {gold_counts.most_common(12)}")
    for label in ("qwen", "gpt"):
        print(f"  {label} top wrong answers:")
        wrong = Counter()
        for gv, c in confusions[label].items():
            for pv, n in c.items():
                if pv.strip().lower() != gv.strip().lower():
                    wrong[(gv, pv)] += n
        for (gv, pv), n in wrong.most_common(8):
            print(f"    gold={gv!r} -> {pv!r}  x{n}")