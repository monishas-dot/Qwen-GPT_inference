import re, sys

versions = sys.argv[1:] or sys.exit("usage: python compare_gpt_versions.py v3 v4 v5")
LINE = re.compile(r"^\s+(\S+)\s+(none|low|medium)\s+(\d+)\s+(\d+)\s+([\d.]+)%")
OVER = re.compile(r"^\s+(none|low|medium)\s*:\s*(\d+)/(\d+) correct")

def load(v):
    txt = open(f"runs/gpt_{v}/metrics/summary.txt").read()
    gpt = txt.split("MODEL: GPT", 1)[1]
    fields, overall = {}, {}
    for line in gpt.splitlines():
        m = LINE.match(line)
        if m:
            f, sev, n, c = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            if sev in ("none", "low"):
                a = fields.setdefault(f, [0, 0]); a[0] += c; a[1] += n
            continue
        m = OVER.match(line)
        if m and m.group(1) not in overall:
            overall[m.group(1)] = 100 * int(m.group(2)) / int(m.group(3))
    return fields, overall

data = {v: load(v) for v in versions}
print("Overall accuracy by blur level (%)")
print(f"{'':10s}" + "".join(f"{v:>8s}" for v in versions))
for sev in ("none", "low", "medium"):
    print(f"{sev:10s}" + "".join(f"{data[v][1].get(sev, float('nan')):8.1f}" for v in versions))
print("\nPer field, 'none' + 'low' blur pooled (%).  <<< / >>> = 10+ points below / above the first version")
print(f"{'field':34s}" + "".join(f"{v:>8s}" for v in versions) + "    n")
base = data[versions[0]][0]
for f in sorted(base):
    vals = [100 * data[v][0][f][0] / data[v][0][f][1] if f in data[v][0] and data[v][0][f][1] else float("nan") for v in versions]
    flags = ""
    for x in vals[1:]:
        flags += " >>>" if x - vals[0] >= 10 else (" <<<" if vals[0] - x >= 10 else "     ")
    print(f"{f:34s}" + "".join(f"{x:8.1f}" for x in vals) + f"  {base[f][1]:4d}{flags}")
