import os, re, subprocess, sys

v = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: python score_gpt_version.py v4")
src = open("scripts/evaluate_extraction_metrics.py").read()
gpt_file = f"gpt output/gpt_{v}.jsonl"
if not os.path.exists(gpt_file):
    sys.exit(f"Missing {gpt_file}. Run the GPT version first.")
for name, val in (("GPT_JSONL", gpt_file), ("OUTPUT_DIR", f"runs/gpt_{v}/metrics")):
    src, n = re.subn(rf'^{name}\s*=.*$', f'{name} = "{val}"', src, count=1, flags=re.M)
    assert n == 1, name
tmp = f"scripts/_eval_{v}.py"
open(tmp, "w").write(src)
try:
    out = subprocess.run([sys.executable, tmp], capture_output=True, text=True)
finally:
    os.remove(tmp)
if out.returncode:
    sys.exit(out.stderr[-1500:])
gpt = out.stdout.split("MODEL: GPT", 1)[-1]
print("MODEL: GPT", gpt.split("-- Per-field")[0])
print(f"Full report: runs/gpt_{v}/metrics/summary.md")
