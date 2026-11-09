"""A self-contained HTML report of the evaluation."""

from __future__ import annotations

import html

from .evaluate import MODES, PSI_ALERT, Result
from .features import FEATURES

CSS = """
:root { --bg:#0b0e13; --panel:#121720; --line:#232c3a; --ink:#e8ecf2; --ink2:#9aa6b8; --s1:#3987e5; --s2:#d95926;
  --good:#3fcf8e; --bad:#ff6b6b; --mid:#ffb454; font-family: Inter, system-ui, sans-serif; }
* { box-sizing: border-box; } body { margin:0; background:var(--bg); color:var(--ink); line-height:1.5; }
main { max-width: 1180px; margin: 0 auto; padding: 40px 28px 64px; }
h1 { font-size: 30px; letter-spacing:-0.02em; margin: 0 0 6px; } h2 { font-size: 19px; margin: 34px 0 14px; }
.sub, .muted { color: var(--ink2); } .sub { margin: 0 0 24px; }
.tiles { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.tile { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }
.tile b { display:block; font-size: 28px; letter-spacing:-0.02em; } .tile span { color: var(--ink2); font-size: 13px; }
table { width:100%; border-collapse: collapse; background:var(--panel); border:1px solid var(--line);
  border-radius:12px; overflow:hidden; font-size: 13.5px; }
th, td { padding: 8px 12px; border-bottom:1px solid var(--line); text-align:right; }
th:first-child, td:first-child { text-align:left; } th { color: var(--ink2); font-weight:500; font-size:12px; }
td.good { color: var(--good); font-weight:600; } td.bad { color: var(--bad); font-weight:600; } td.mid { color: var(--mid); font-weight:600; }
.pair { display:inline-block; height: 8px; border-radius: 0 4px 4px 0; vertical-align: middle; margin-right: 6px; }
.s1 { background: var(--s1); } .s2 { background: var(--s2); }
.legend span { margin-right: 18px; font-size: 13px; color: var(--ink2); }
"""


def render(r: Result) -> str:
    pit = r.auc[MODES["point_in_time"]]
    lab = r.auc[MODES["label_date"]]
    bug = r.skew["serving with amounts in pence"]
    tiles = [
        (
            f"{r.consistency[MODES['point_in_time']]:.0%}",
            "of point-in-time training rows differ from what serving returned",
        ),
        (f"{lab['offline']:.2f} → {lab['production']:.2f}", "AUC of a label-date join, offline then in production"),
        (f"{pit['production']:.3f}", "production AUC of the point-in-time model, exactly as measured offline"),
        (
            f"{sum(v >= PSI_ALERT for v in bug.values())} / {len(bug)}",
            "features flagged when serving sends pence instead of pounds",
        ),
    ]
    t = "".join(f'<div class="tile"><b>{html.escape(a)}</b><span>{html.escape(b)}</span></div>' for a, b in tiles)
    auc = "".join(
        f'<tr><td>{html.escape(k)}</td><td><span class="pair s1" style="width:{int((v["offline"] - 0.5) * 300)}px"></span>{v["offline"]:.3f}</td>'
        f'<td><span class="pair s2" style="width:{int((v["production"] - 0.5) * 300)}px"></span>{v["production"]:.3f}</td>'
        f'<td class="{"good" if abs(v["offline"] - v["production"]) < 0.005 else "bad"}">{v["offline"] - v["production"]:+.3f}</td></tr>'
        for k, v in r.auc.items()
    )
    cons = "".join(
        f'<tr><td>{html.escape(k)}</td><td class="{"good" if v == 0 else "bad"}">{v:.2%}</td></tr>'
        for k, v in r.consistency.items()
    )
    skew = "".join(
        f"<tr><td>{f.name}</td>"
        + "".join(
            f'<td class="{"bad" if v[f.name] >= PSI_ALERT else "good"}">{v[f.name]:.3f}</td>' for v in r.skew.values()
        )
        + "</tr>"
        for f in FEATURES
    )
    o = r.online_ms
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Feature Store: evaluation</title><style>{CSS}</style></head><body><main>
<h1>Training that sees what serving sees</h1>
<p class="sub">{r.events:,} card transactions, {r.labels:,} labelled moments ("will this customer buy in the next 14 days?"),
{r.late_share:.1%} of events arriving a day or more late. The online store is replayed in arrival order and compared
with every offline training row.</p>
<div class="tiles">{t}</div>
<h2>What a leaky training set costs (AUC on later moments)</h2>
<p class="legend"><span><i class="pair s1" style="width:14px"></i>offline, what the team sees</span>
<span><i class="pair s2" style="width:14px"></i>production, what customers get</span></p>
<table><tr><th>training set built from</th><th>offline</th><th>production</th><th>gap</th></tr>{auc}</table>
<h2>Rows that differ from serving</h2>
<table><tr><th>training set built from</th><th>rows that differ from what serving returned</th></tr>{cons}</table>
<h2>Training/serving skew (PSI, alert at {PSI_ALERT})</h2>
<table><tr><th>feature</th>{"".join(f"<th>{html.escape(k)}</th>" for k in r.skew)}</tr>{skew}</table>
<p class="muted">Online lookup p50 {o["p50"]:.3f} ms, p95 {o["p95"]:.3f} ms. Offline point-in-time join
{r.offline_rows_per_s:,.0f} rows per second.</p>
</main></body></html>
"""
