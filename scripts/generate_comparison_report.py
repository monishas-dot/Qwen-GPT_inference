"""generate_comparison_report.py — renders mismatches_only.json + summary.json
(from compare_qwen_gpt.py) into a single self-contained HTML report for
visual review. No external dependencies, works fully offline.

Usage:
    python generate_comparison_report.py <comparison_output_dir>

Reads:
    <dir>/mismatches_only.json
    <dir>/summary.json
Writes:
    <dir>/comparison_report.html
"""

import sys
import json
from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Qwen vs GPT-5.2 — Field Comparison</title>
<style>
  :root {
    --bg: #faf9f7;
    --panel: #ffffff;
    --ink: #1c1b1a;
    --ink-soft: #5c5852;
    --rule: #e2ddd4;
    --qwen: #2b5d7c;
    --qwen-bg: #eaf1f5;
    --gpt: #8a4a2b;
    --gpt-bg: #f6ece4;
    --warn: #b8863a;
    --ok: #3f7a4d;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: -apple-system, "SF Pro Text", "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.5;
  }
  .mono { font-family: "SF Mono", "Menlo", "Consolas", monospace; }
  header {
    padding: 32px 40px 20px;
    border-bottom: 1px solid var(--rule);
    background: var(--panel);
  }
  header h1 {
    margin: 0 0 4px;
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.01em;
  }
  header p {
    margin: 0;
    color: var(--ink-soft);
    font-size: 13px;
  }
  .layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 0;
    min-height: calc(100vh - 90px);
  }
  aside {
    border-right: 1px solid var(--rule);
    padding: 24px;
    background: var(--panel);
  }
  aside h2 {
    font-size: 11px;
    text-transform: none;
    color: var(--ink-soft);
    margin: 0 0 12px;
    font-weight: 600;
  }
  .stat-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--rule);
    font-size: 13px;
  }
  .stat-row .n { font-weight: 600; }
  .field-bar-wrap { margin: 4px 0 10px; }
  .field-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--ink-soft);
    margin-bottom: 3px;
  }
  .field-bar-track {
    height: 5px;
    background: var(--rule);
    border-radius: 3px;
    overflow: hidden;
  }
  .field-bar-fill {
    height: 100%;
    background: var(--ok);
  }
  .field-bar-fill.low { background: var(--warn); }
  .field-bar-fill.vlow { background: var(--gpt); }
  main { padding: 24px 40px 60px; }
  #search {
    width: 100%;
    max-width: 420px;
    padding: 9px 12px;
    border: 1px solid var(--rule);
    border-radius: 6px;
    font-size: 13px;
    margin-bottom: 20px;
    background: var(--panel);
    color: var(--ink);
  }
  #search:focus { outline: 2px solid var(--qwen); outline-offset: 1px; }
  .record-card {
    background: var(--panel);
    border: 1px solid var(--rule);
    border-radius: 8px;
    margin-bottom: 14px;
    overflow: hidden;
  }
  .record-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    cursor: pointer;
    user-select: none;
  }
  .record-head:hover { background: #f5f3ef; }
  .record-id {
    font-size: 13px;
    font-weight: 600;
  }
  .badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
    background: var(--gpt-bg);
    color: var(--gpt);
    font-weight: 600;
  }
  .record-body {
    display: none;
    border-top: 1px solid var(--rule);
    padding: 4px 16px 12px;
  }
  .record-card.open .record-body { display: block; }
  .record-card.open .chevron { transform: rotate(90deg); }
  .chevron {
    display: inline-block;
    transition: transform 0.15s ease;
    color: var(--ink-soft);
    margin-right: 8px;
    font-size: 11px;
  }
  table { width: 100%; border-collapse: collapse; margin: 10px 0; }
  th {
    text-align: left;
    font-size: 11px;
    color: var(--ink-soft);
    font-weight: 600;
    padding: 4px 8px;
    border-bottom: 1px solid var(--rule);
  }
  td {
    padding: 6px 8px;
    border-bottom: 1px solid var(--rule);
    vertical-align: top;
    font-size: 12.5px;
  }
  td.field-name { color: var(--ink-soft); white-space: nowrap; }
  td.val-qwen { background: var(--qwen-bg); color: var(--qwen); }
  td.val-gpt { background: var(--gpt-bg); color: var(--gpt); }
  .null-val { color: #b0aaa0; font-style: italic; }
  .list-sub-heading {
    font-size: 11.5px;
    font-weight: 600;
    color: var(--ink-soft);
    margin: 12px 0 4px;
  }
  .item-json {
    font-size: 11.5px;
    padding: 6px 8px;
    background: #f5f3ef;
    border-radius: 4px;
    margin: 4px 0;
    white-space: pre-wrap;
  }
  .empty-state {
    padding: 60px 20px;
    text-align: center;
    color: var(--ink-soft);
  }
  .legend {
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
    font-size: 12px;
    color: var(--ink-soft);
  }
  .legend span.swatch {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 2px;
    margin-right: 5px;
    vertical-align: -1px;
  }
</style>
</head>
<body>

<header>
  <h1>Qwen vs GPT-5.2 &mdash; Field Comparison</h1>
  <p id="header-sub"></p>
</header>

<div class="layout">
  <aside>
    <h2>Records</h2>
    <div class="stat-row"><span>Compared</span><span class="n" id="stat-compared"></span></div>
    <div class="stat-row"><span>Perfect match</span><span class="n" id="stat-perfect"></span></div>
    <div class="stat-row"><span>With mismatches</span><span class="n" id="stat-mismatch"></span></div>
    <div class="stat-row"><span>Qwen-only ids</span><span class="n" id="stat-only-qwen"></span></div>
    <div class="stat-row"><span>GPT-only ids</span><span class="n" id="stat-only-gpt"></span></div>

    <h2 style="margin-top:24px;">Scalar field agreement</h2>
    <div id="field-bars"></div>

    <h2 style="margin-top:24px;">List field agreement</h2>
    <div id="list-field-bars"></div>
  </aside>

  <main>
    <input id="search" type="text" placeholder="Filter by record id or field name...">
    <div class="legend">
      <span><span class="swatch" style="background:var(--qwen-bg); border:1px solid var(--qwen);"></span>Qwen value</span>
      <span><span class="swatch" style="background:var(--gpt-bg); border:1px solid var(--gpt);"></span>GPT-5.2 value</span>
    </div>
    <div id="records"></div>
    <div class="empty-state" id="empty-state" style="display:none;">No records match this filter.</div>
  </main>
</div>

<script>
const SUMMARY = __SUMMARY_JSON__;
const MISMATCHES = __MISMATCHES_JSON__;

function esc(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function valCell(v, cls) {
  if (v === null || v === undefined || v === '') {
    return `<td class="${cls}"><span class="null-val">null</span></td>`;
  }
  return `<td class="${cls} mono">${esc(v)}</td>`;
}

function barClass(rate) {
  if (rate === null || rate === undefined) return '';
  if (rate < 0.7) return 'vlow';
  if (rate < 0.9) return 'low';
  return '';
}

function renderBars(containerId, ratesObj) {
  const el = document.getElementById(containerId);
  const entries = Object.entries(ratesObj).sort((a, b) => {
    const av = a[1] === null ? 1 : a[1], bv = b[1] === null ? 1 : b[1];
    return av - bv;
  });
  el.innerHTML = entries.map(([name, rate]) => {
    const pct = rate === null ? 0 : Math.round(rate * 100);
    const label = rate === null ? 'n/a' : pct + '%';
    return `<div class="field-bar-wrap">
      <div class="field-bar-label"><span>${esc(name)}</span><span>${label}</span></div>
      <div class="field-bar-track"><div class="field-bar-fill ${barClass(rate)}" style="width:${pct}%;"></div></div>
    </div>`;
  }).join('');
}

function renderScalarTable(mismatches) {
  if (!mismatches || Object.keys(mismatches).length === 0) return '';
  const rows = Object.entries(mismatches).map(([name, v]) => `
    <tr>
      <td class="field-name">${esc(name)}</td>
      ${valCell(v.qwen, 'val-qwen')}
      ${valCell(v.gpt, 'val-gpt')}
    </tr>
  `).join('');
  return `<table>
    <thead><tr><th>Field</th><th>Qwen</th><th>GPT-5.2</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderListMismatches(listFields) {
  if (!listFields || Object.keys(listFields).length === 0) return '';
  let out = '';
  for (const [fieldName, detail] of Object.entries(listFields)) {
    if (!detail) continue;
    out += `<div class="list-sub-heading">${esc(fieldName)}</div>`;

    if (detail.subfield_mismatches_in_matched_items && detail.subfield_mismatches_in_matched_items.length) {
      const rows = detail.subfield_mismatches_in_matched_items.map(item => {
        const subRows = Object.entries(item.mismatched_subfields).map(([k, v]) => `
          <tr>
            <td class="field-name">${esc(item.matched_on)} &rsaquo; ${esc(k)}</td>
            ${valCell(v.qwen, 'val-qwen')}
            ${valCell(v.gpt, 'val-gpt')}
          </tr>
        `).join('');
        return subRows;
      }).join('');
      out += `<table><thead><tr><th>Item &rsaquo; Field</th><th>Qwen</th><th>GPT-5.2</th></tr></thead><tbody>${rows}</tbody></table>`;
    }

    if (detail.unmatched_in_qwen_only && detail.unmatched_in_qwen_only.length) {
      out += `<div class="field-name" style="margin-top:6px;">In Qwen only:</div>`;
      detail.unmatched_in_qwen_only.forEach(item => {
        out += `<div class="item-json mono val-qwen">${esc(JSON.stringify(item))}</div>`;
      });
    }
    if (detail.unmatched_in_gpt_only && detail.unmatched_in_gpt_only.length) {
      out += `<div class="field-name" style="margin-top:6px;">In GPT-5.2 only:</div>`;
      detail.unmatched_in_gpt_only.forEach(item => {
        out += `<div class="item-json mono val-gpt">${esc(JSON.stringify(item))}</div>`;
      });
    }
  }
  return out;
}

function recordSearchText(rec) {
  let parts = [rec.id];
  parts.push(...Object.keys(rec.mismatched_scalar_fields || {}));
  parts.push(...Object.keys(rec.mismatched_list_fields || {}));
  return parts.join(' ').toLowerCase();
}

function renderRecords(filterText) {
  const container = document.getElementById('records');
  const filtered = MISMATCHES.filter(r => !filterText || recordSearchText(r).includes(filterText));

  document.getElementById('empty-state').style.display = filtered.length ? 'none' : 'block';

  container.innerHTML = filtered.map((rec, i) => `
    <div class="record-card" data-idx="${i}">
      <div class="record-head" onclick="this.parentElement.classList.toggle('open')">
        <span><span class="chevron">&#9656;</span><span class="record-id mono">${esc(rec.id)}</span></span>
        <span class="badge">${rec.total_mismatches} mismatch${rec.total_mismatches === 1 ? '' : 'es'}</span>
      </div>
      <div class="record-body">
        ${renderScalarTable(rec.mismatched_scalar_fields)}
        ${renderListMismatches(rec.mismatched_list_fields)}
      </div>
    </div>
  `).join('');
}

document.getElementById('header-sub').textContent =
  `${SUMMARY.n_common_records} records compared \u2022 ${SUMMARY.n_records_with_any_mismatch} with at least one mismatch`;
document.getElementById('stat-compared').textContent = SUMMARY.n_common_records;
document.getElementById('stat-perfect').textContent = SUMMARY.n_records_perfect_match;
document.getElementById('stat-mismatch').textContent = SUMMARY.n_records_with_any_mismatch;
document.getElementById('stat-only-qwen').textContent = SUMMARY.n_only_qwen;
document.getElementById('stat-only-gpt').textContent = SUMMARY.n_only_gpt;

renderBars('field-bars', SUMMARY.scalar_field_agreement_rate);
renderBars('list-field-bars', SUMMARY.list_field_item_level_agreement_rate);
renderRecords('');

document.getElementById('search').addEventListener('input', (e) => {
  renderRecords(e.target.value.trim().toLowerCase());
});
</script>

</body>
</html>
"""


def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_comparison_report.py <comparison_output_dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    mismatches_path = output_dir / "mismatches_only.json"
    summary_path = output_dir / "summary.json"

    if not mismatches_path.exists() or not summary_path.exists():
        print(f"Expected both {mismatches_path} and {summary_path} to exist. "
              f"Run compare_qwen_gpt.py first.")
        sys.exit(1)

    with open(mismatches_path) as f:
        mismatches = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)

    html = HTML_TEMPLATE.replace("__SUMMARY_JSON__", json.dumps(summary))
    html = html.replace("__MISMATCHES_JSON__", json.dumps(mismatches))

    out_path = output_dir / "comparison_report.html"
    with open(out_path, "w") as f:
        f.write(html)

    print(f"Report written to: {out_path}")
    print(f"Open it directly in a browser: file://{out_path.resolve()}")


if __name__ == "__main__":
    main()