"""Product-grade Streamlit dashboard for the AI Compute Optimizer.

Framing principle: the headline numbers are the **validated benchmark** (QQP
eval + benchmark run), NOT the current server's tiny live session. A demo with
three test queries must never be mistaken for the real result, so:

* the hero savings figure and the KPI cards show the validated benchmark, with
  live-session values as a small secondary readout;
* "Savings at your scale" renders the full sensitivity grid so a viewer finds
  their own number (the headline artifact);
* a clearly-labeled, visually distinct "Live session" panel holds the running
  server's actual metrics.

It reads the gateway's ``/metrics`` and ``/health`` endpoints and refreshes on
demand. Dollar projections come from the shared model in ``app/savings.py`` (via
``eval/savings_model.py``) so the hero and the grid can never drift.

Run::

    streamlit run dashboard.py

Point at a non-default gateway via the sidebar or ``GATEWAY_URL`` env var.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

# --- Defaults from the app's own config, with safe fallbacks --------------
try:
    from app.config import get_settings

    _S = get_settings()
    APP_NAME: str = _S.app_name
    STAGE1_THRESHOLD: float = _S.stage1_threshold
    RERANKER_MODEL: str = _S.reranker_model
    RERANKER_THRESHOLD: float = _S.reranker_threshold
    RERANKER_ENABLED: bool = _S.enable_reranker
except Exception:  # dashboard can run standalone, detached from the package
    APP_NAME = "AI Compute Optimizer"
    STAGE1_THRESHOLD = 0.70
    RERANKER_MODEL = "cross-encoder/quora-distilroberta-base"
    RERANKER_THRESHOLD = 0.943
    RERANKER_ENABLED = True

# --- Validated savings model (shared single source of truth) --------------
# Dollar figures use app/savings.py via eval/savings_model.py so the hero and
# the sensitivity grid can never diverge from the offline analysis.
try:
    from eval.savings_model import (
        AVG_INPUT_TOKENS,
        AVG_OUTPUT_TOKENS,
        INPUT_PRICE_PER_1M,
        MEASURED_HIT_RATE,
        OUTPUT_PRICE_PER_1M,
        SCENARIO_HIT_RATES,
        SCENARIO_VOLUMES,
        project_savings,
        sensitivity_grid,
    )
except Exception:  # standalone fallback: identical validated constants + math
    MEASURED_HIT_RATE = 0.444
    AVG_INPUT_TOKENS, AVG_OUTPUT_TOKENS = 43.9, 542.5
    INPUT_PRICE_PER_1M, OUTPUT_PRICE_PER_1M = 1.50, 9.00
    SCENARIO_VOLUMES = [10_000, 100_000, 500_000, 1_000_000]
    SCENARIO_HIT_RATES = [0.20, 0.30, MEASURED_HIT_RATE, 0.60]

    def project_savings(mv, hr, ai, ao, ip, op):  # type: ignore[no-redef]
        pq = (ai * ip + ao * op) / 1_000_000.0
        monthly = mv * hr * pq
        return {
            "per_query_remote_cost_usd": pq,
            "monthly_savings_usd": monthly,
            "annual_savings_usd": monthly * 12,
        }

    def sensitivity_grid(volumes, hit_rates, **_):  # type: ignore[no-redef]
        return [
            {"monthly_volume": v, **{h: project_savings(
                v, h, AVG_INPUT_TOKENS, AVG_OUTPUT_TOKENS,
                INPUT_PRICE_PER_1M, OUTPUT_PRICE_PER_1M)["annual_savings_usd"]
                for h in hit_rates}}
            for v in volumes
        ]

# Headline benchmark KPIs (from the validated two-stage run at 95% precision).
BENCH_VOLUME = 100_000
BENCH_HIT_RATE = MEASURED_HIT_RATE      # 44.4%
BENCH_COST_REDUCTION = 0.694            # 69.4% vs an all-remote gateway
BENCH_HIT_LATENCY_MS = 48.0            # 48 ms on a cache hit
BENCH_MISS_LATENCY_MS = 10_000.0       # ~10 s on an LLM miss

# Defaults to the deployed Render backend (for the Streamlit Cloud deploy);
# override via the GATEWAY_URL env var or the sidebar input (e.g. for local runs
# point it at http://localhost:8000).
DEFAULT_URL = os.environ.get("GATEWAY_URL", "https://ai-compute-optimizer.onrender.com")

# Cohesive palette (route colors reused across every chart).
C_CACHE = "#059669"   # green  — served from cache
C_LOCAL = "#2563eb"   # blue   — local model
C_REMOTE = "#d97706"  # amber  — remote model
C_BASELINE = "#94a3b8"  # slate — "all remote" baseline

CSS = """
:root {
  --bg: #f5f6f8; --card: #ffffff; --ink: #0f172a; --muted: #64748b;
  --border: #e6e8ec; --accent: #4f46e5; --green: #059669; --amber: #d97706;
  --shadow: 0 1px 2px rgba(16,24,40,.06), 0 4px 12px rgba(16,24,40,.05);
}
.stApp { background: var(--bg); }
#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px; }
html, body, [class*="css"] {
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: var(--ink);
}
.brand { display:flex; align-items:center; gap:.65rem; margin-bottom:.25rem; }
.brand-mark { font-size:1.5rem; }
.brand-name { font-size:1.35rem; font-weight:700; letter-spacing:-.01em; }
.brand-sub { color: var(--muted); font-size:.9rem; margin:-.15rem 0 1.1rem 2.15rem; }

.section-title {
  font-size:.78rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase;
  color: var(--muted); margin:1.9rem 0 .85rem;
}

/* Hero */
.hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #253449 100%);
  color:#fff; border-radius:18px; padding:2.1rem 2.3rem; box-shadow: var(--shadow);
  position:relative; overflow:hidden;
}
.hero::after {
  content:""; position:absolute; right:-60px; top:-60px; width:240px; height:240px;
  background: radial-gradient(circle, rgba(5,150,105,.28), transparent 70%);
}
.hero-label { font-size:.82rem; font-weight:600; letter-spacing:.09em;
  text-transform:uppercase; color:#9fb3c8; }
.hero-value { font-size:3.5rem; font-weight:800; letter-spacing:-.03em; line-height:1.05;
  margin:.3rem 0 .1rem; }
.hero-monthly { font-size:1.15rem; color:#cbd5e1; font-weight:500; }
.hero-monthly b { color:#fff; font-weight:700; }
.hero-caption { margin-top:1rem; font-size:.86rem; color:#94a3b8;
  border-top:1px solid rgba(255,255,255,.1); padding-top:.8rem; }
.hero-caption b { color:#cbd5e1; }
.hero-callout { margin-top:.85rem; background:var(--card); border:1px solid var(--border);
  border-left:3px solid var(--amber); border-radius:12px; padding:.85rem 1.15rem;
  box-shadow:var(--shadow); font-size:.92rem; color:#334155; }
.hero-callout b { color:var(--ink); font-weight:750; }
.callout-tag { display:inline-block; font-size:.68rem; font-weight:800; letter-spacing:.05em;
  text-transform:uppercase; color:#b45309; background:#fdf1e3; padding:.16rem .5rem;
  border-radius:6px; margin-right:.6rem; }
.callout-note { color:var(--muted); font-size:.85rem; }

/* KPI grid */
.kpi-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:1rem; margin-top:1rem; }
.kpi-card { background:var(--card); border:1px solid var(--border); border-radius:14px;
  padding:1.1rem 1.25rem; box-shadow: var(--shadow); }
.kpi-label { font-size:.78rem; color:var(--muted); font-weight:600; letter-spacing:.02em; }
.kpi-value { font-size:1.9rem; font-weight:750; letter-spacing:-.02em; margin:.2rem 0 .1rem; }
.kpi-sub { font-size:.82rem; color:var(--muted); }
.kpi-live { margin-top:.6rem; padding-top:.5rem; border-top:1px dashed var(--border);
  font-size:.76rem; color:var(--muted); }
.kpi-live b { color:#334155; font-weight:700; }

/* Cards + charts */
.grid-2 { display:grid; grid-template-columns: 1fr 1fr; gap:1rem; }
.card { background:var(--card); border:1px solid var(--border); border-radius:14px;
  padding:1.3rem 1.45rem; box-shadow: var(--shadow); height:100%; }
.card-title { font-size:.98rem; font-weight:700; margin:0 0 .3rem; }
.card-note { font-size:.8rem; color:var(--muted); margin:0 0 1rem; }

.bar-row { display:grid; grid-template-columns: 128px 1fr 92px; align-items:center;
  gap:.7rem; margin:.55rem 0; }
.bar-label { font-size:.86rem; color:#334155; font-weight:500; }
.bar-track { background:#eef1f5; border-radius:999px; height:16px; overflow:hidden; }
.bar-fill { height:100%; border-radius:999px; }
.bar-value { font-size:.86rem; font-weight:600; text-align:right;
  font-variant-numeric: tabular-nums; }

/* Donut */
.donut-wrap { display:flex; align-items:center; gap:1.6rem; }
.donut { width:150px; height:150px; border-radius:50%; position:relative; flex:none; }
.donut::after { content:""; position:absolute; inset:24px; background:var(--card);
  border-radius:50%; }
.legend { display:flex; flex-direction:column; gap:.5rem; }
.legend-item { display:flex; align-items:center; gap:.55rem; font-size:.88rem; }
.dot { width:11px; height:11px; border-radius:3px; flex:none; }
.legend-val { color:var(--muted); font-variant-numeric: tabular-nums; }

/* Sensitivity grid */
.table-wrap { overflow-x:auto; }
.sens-table { width:100%; border-collapse:separate; border-spacing:0; font-size:.92rem; }
.sens-table th, .sens-table td { padding:.7rem .9rem; text-align:right;
  border-bottom:1px solid #e6e8ec; font-variant-numeric: tabular-nums; white-space:nowrap; }
.sens-table td { color:#0f172a; background:#fff; font-weight:600; }  /* all cells legible */
.sens-table th { color:#475569; font-weight:700; font-size:.78rem;
  text-transform:uppercase; letter-spacing:.03em; }
.sens-table th:first-child, .sens-table td:first-child { text-align:left; }
.sens-table td:first-child { color:#334155; }
.sens-measured-h { color:#047857; }
/* Only the validated 100K x 44.4% cell is highlighted (more specific than td). */
.sens-table td.sens-hero-cell { background:#059669; color:#fff; font-weight:800; }

/* Live session */
.live-head { display:flex; align-items:center; gap:.6rem; margin-bottom:.2rem; }
.live-badge { font-size:.68rem; font-weight:800; letter-spacing:.06em;
  background:#fdeaea; color:#b91c1c; padding:.18rem .55rem; border-radius:6px; }
.live-badge::before { content:"● "; }
.live-title { font-weight:700; }
.live-disclaimer { font-size:.82rem; color:var(--muted); margin:.15rem 0 1rem; }
.live-disclaimer b { color:#334155; }
.stat-row { display:grid; grid-template-columns: repeat(5, 1fr); gap:.85rem; margin-bottom:1.1rem; }
.stat-tile { background:#fafbfc; border:1px solid var(--border); border-radius:12px;
  padding:.8rem .95rem; }
.stat-tile .t-label { font-size:.72rem; color:var(--muted); font-weight:600; }
.stat-tile .t-value { font-size:1.3rem; font-weight:750; letter-spacing:-.01em;
  font-variant-numeric: tabular-nums; }

/* Trust & Safety */
.trust-grid { display:grid; grid-template-columns: repeat(2, 1fr); gap:1rem; }
.trust-card { background:var(--card); border:1px solid var(--border); border-radius:14px;
  padding:1.15rem 1.3rem; box-shadow: var(--shadow); }
.trust-card.warn { border-left:3px solid var(--amber); }
.trust-card.ok { border-left:3px solid var(--green); }
.trust-h { font-size:.92rem; font-weight:700; display:flex; align-items:center; gap:.5rem; }
.trust-b { font-size:.85rem; color:#475569; margin-top:.35rem; line-height:1.5; }
.stat-inline { font-weight:750; color:var(--ink); }

/* Badges */
.badge { display:inline-block; padding:.15rem .55rem; border-radius:999px;
  font-size:.72rem; font-weight:700; letter-spacing:.02em; }
.b-green { background:#e7f6ef; color:#047857; }
.b-red { background:#fdeaea; color:#b91c1c; }
.b-blue { background:#e8effd; color:#1d4ed8; }
.b-amber { background:#fdf1e3; color:#b45309; }
.b-gray { background:#eef1f5; color:#475569; }
.b-measured { background:#e7f6ef; color:#047857; }
.b-assumed { background:#fdf1e3; color:#b45309; }

.assump-row { display:flex; align-items:center; gap:.7rem; padding:.45rem 0;
  border-bottom:1px solid #f1f3f6; font-size:.88rem; }
.assump-row:last-child { border-bottom:none; }
.assump-key { min-width:190px; color:#334155; font-weight:600; }
.assump-val { color:var(--muted); }

/* Query tester response */
.qr { background:var(--card); border:1px solid var(--border); border-radius:14px;
  padding:1.2rem 1.35rem; box-shadow: var(--shadow); }
.qr-meta { display:flex; flex-wrap:wrap; gap:.5rem; margin-bottom:.9rem; }
.qr-chip { font-size:.78rem; background:#f4f6f9; border:1px solid var(--border);
  border-radius:8px; padding:.28rem .6rem; color:#334155; }
.qr-chip b { color:var(--ink); font-variant-numeric: tabular-nums; }
.qr-answer { font-size:.92rem; line-height:1.55; color:#1f2937; white-space:pre-wrap;
  background:#f8fafc; border:1px solid var(--border); border-radius:10px; padding:.85rem 1rem; }

.conn { font-size:.83rem; }
.conn .dot { display:inline-block; margin-right:.4rem; }
.unreachable { background:var(--card); border:1px solid var(--border);
  border-left:3px solid var(--amber); border-radius:14px; padding:1.5rem 1.7rem;
  box-shadow: var(--shadow); }
.unreachable h3 { margin:0 0 .5rem; font-size:1.05rem; }
.unreachable code { background:#f1f3f6; padding:.12rem .4rem; border-radius:6px; }
"""

# Provenance of the validated model, shown in the Assumptions expander.
VALIDATED_BASIS: list[tuple[str, str, str]] = [
    ("MEASURED", "token prices",
     "$1.50 in / $9.00 out per 1M — Gemini 3.5 Flash published list price"),
    ("MEASURED", "avg tokens / query", "43.9 in / 542.5 out — benchmark.py workload"),
    ("MEASURED", "hit rate", "44.4% — QQP-validated two-stage eval + live benchmark"),
    ("MEASURED", "cost reduction", "69.4% vs all-remote — benchmark"),
    ("ASSUMED", "monthly query volume", "the scenario columns — pick your own in the grid"),
    ("ASSUMED", "local route cost", "$0 — optimistic; self-hosted inference still costs compute"),
    ("ASSUMED", "traffic distribution", "real traffic resembles the benchmark / QQP mix"),
]


# ---------------------------------------------------------------------------
# Data access (all failures handled gracefully — never a stack trace)
# ---------------------------------------------------------------------------
def get_json(url: str, path: str, timeout: float = 5.0) -> tuple[dict | None, str | None]:
    """GET ``url+path`` -> (json, None) or (None, error message)."""
    try:
        response = httpx.get(f"{url.rstrip('/')}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:  # noqa: BLE001 - surface any failure as a message
        return None, str(exc)


def post_query(url: str, prompt: str, timeout: float = 60.0) -> tuple[dict | None, str | None]:
    """POST a prompt to /query -> (json, None) or (None, error message)."""
    try:
        response = httpx.post(
            f"{url.rstrip('/')}/query", json={"prompt": prompt}, timeout=timeout
        )
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def money(value: float, decimals: int = 0) -> str:
    return f"${value:,.{decimals}f}"


def money_small(value: float) -> str:
    """Adaptive precision for small dollar amounts."""
    if value == 0:
        return "$0.00"
    if abs(value) < 1:
        return f"${value:,.4f}"
    return f"${value:,.2f}"


def compact(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return f"{n:.0f}"


def pct(numer: float, denom: float) -> float:
    return (numer / denom * 100.0) if denom else 0.0


def ms(value: float) -> str:
    return f"{value:,.0f} ms" if value >= 10 else f"{value:.1f} ms"


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------
def kpi_card(label: str, value: str, sub: str, live: str = "") -> str:
    live_html = f'<div class="kpi-live">{live}</div>' if live else ""
    return (
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div>'
        f'{live_html}</div>'
    )


def stat_tiles(items: list[tuple[str, str]]) -> str:
    tiles = "".join(
        f'<div class="stat-tile"><div class="t-label">{label}</div>'
        f'<div class="t-value">{value}</div></div>'
        for label, value in items
    )
    return f'<div class="stat-row">{tiles}</div>'


def bar_chart(rows: list[tuple[str, float, str, str]]) -> str:
    """rows = (label, magnitude_for_width, display_value, color)."""
    peak = max((r[1] for r in rows), default=0.0) or 1.0
    html = ['<div>']
    for label, mag, disp, color in rows:
        width = max(2.0, mag / peak * 100.0)
        html.append(
            f'<div class="bar-row"><div class="bar-label">{label}</div>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{width:.1f}%;background:{color}"></div></div>'
            f'<div class="bar-value">{disp}</div></div>'
        )
    html.append('</div>')
    return "".join(html)


def donut(segments: list[tuple[str, float, str]]) -> str:
    """segments = (label, count, color). Renders a conic donut + legend."""
    total = sum(s[1] for s in segments)
    if total <= 0:
        gradient = "#eef1f5 0 100%"
        legend_rows = '<div class="legend-item"><span class="legend-val">No requests yet</span></div>'
    else:
        stops, cursor, legend = [], 0.0, []
        for label, count, color in segments:
            start = cursor / total * 100.0
            cursor += count
            end = cursor / total * 100.0
            stops.append(f"{color} {start:.2f}% {end:.2f}%")
            legend.append(
                f'<div class="legend-item"><span class="dot" style="background:{color}"></span>'
                f'{label} <span class="legend-val">{int(count):,} '
                f'({pct(count, total):.0f}%)</span></div>'
            )
        gradient = ", ".join(stops)
        legend_rows = "".join(legend)
    return (
        f'<div class="donut-wrap"><div class="donut" '
        f'style="background:conic-gradient({gradient})"></div>'
        f'<div class="legend">{legend_rows}</div></div>'
    )


def card(title: str, note: str, body: str, extra_class: str = "") -> str:
    note_html = f'<p class="card-note">{note}</p>' if note else ""
    return f'<div class="card {extra_class}"><div class="card-title">{title}</div>{note_html}{body}</div>'


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def render_executive(data: dict) -> None:
    """Hero + KPIs — VALIDATED benchmark headline, live session as secondary."""
    bench = project_savings(
        BENCH_VOLUME, BENCH_HIT_RATE, AVG_INPUT_TOKENS, AVG_OUTPUT_TOKENS,
        INPUT_PRICE_PER_1M, OUTPUT_PRICE_PER_1M,
    )
    annual = bench["annual_savings_usd"]
    monthly = bench["monthly_savings_usd"]

    st.markdown('<div class="section-title">Executive summary</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-label">Projected annual savings</div>
          <div class="hero-value">{money(annual)}</div>
          <div class="hero-monthly"><b>{money(monthly)}</b> / month</div>
          <div class="hero-caption">Projected at <b>{compact(BENCH_VOLUME)} queries/mo, {BENCH_HIT_RATE * 100:.1f}%
            validated hit rate</b> (QQP-validated two-stage benchmark, 95% precision).
            Local routes modeled at $0 (optimistic). Find your own number in
            <b>Savings at your scale</b> below.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Secondary upside scenario — clearly labeled as a ceiling, NOT the headline.
    enterprise = project_savings(
        1_000_000, 0.60, AVG_INPUT_TOKENS, AVG_OUTPUT_TOKENS,
        INPUT_PRICE_PER_1M, OUTPUT_PRICE_PER_1M,
    )["annual_savings_usd"]
    st.markdown(
        f'<div class="hero-callout"><span class="callout-tag">Upper-bound scenario</span>'
        f'At enterprise scale (1M queries/mo, 60% hit rate): <b>~{money(enterprise)}/yr</b> '
        f'<span class="callout-note">— illustrative ceiling, not the headline.</span></div>',
        unsafe_allow_html=True,
    )

    # Live-session secondary values (this running server only).
    live_hr = data.get("hit_rate", 0.0)
    live_req = data.get("total_requests", 0)
    live_red = pct(data.get("total_savings_usd", 0.0), data.get("total_baseline_cost_usd", 0.0))
    live_hit = data.get("avg_hit_latency_ms", 0.0)
    live_miss = data.get("avg_miss_latency_ms", 0.0)

    cards = [
        kpi_card(
            "Cache hit rate", f"{BENCH_HIT_RATE * 100:.1f}%", "validated at 95% precision",
            live=f"Live session: <b>{live_hr * 100:.1f}%</b> ({live_req:,} req)",
        ),
        kpi_card(
            "Cost reduction", f"{BENCH_COST_REDUCTION * 100:.1f}%", "vs an all-remote gateway",
            live=f"Live session: <b>{live_red:.0f}%</b> this run",
        ),
        kpi_card(
            "Cache-hit latency", ms(BENCH_HIT_LATENCY_MS),
            f"vs ~{BENCH_MISS_LATENCY_MS / 1000:.0f}s on an LLM miss",
            live=f"Live: <b>{ms(live_hit)}</b> hit / {ms(live_miss)} miss",
        ),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_savings_grid() -> None:
    """The headline artifact: annual savings across volume x hit rate."""
    st.markdown('<div class="section-title">Savings at your scale</div>', unsafe_allow_html=True)
    grid = sensitivity_grid(SCENARIO_VOLUMES, SCENARIO_HIT_RATES)

    heads = ['<th>Monthly volume</th>']
    for hr in SCENARIO_HIT_RATES:
        measured = abs(hr - MEASURED_HIT_RATE) < 1e-9
        cls = ' class="sens-measured-h"' if measured else ""
        label = f"{hr * 100:.1f}%" + (" (measured)" if measured else "")
        heads.append(f"<th{cls}>{label}</th>")

    body_rows = []
    for row in grid:
        vol = row["monthly_volume"]
        cells = [f"<td>{compact(vol)}/mo</td>"]
        for hr in SCENARIO_HIT_RATES:
            measured_col = abs(hr - MEASURED_HIT_RATE) < 1e-9
            hero_cell = measured_col and vol == BENCH_VOLUME
            attr = ' class="sens-hero-cell"' if hero_cell else ""
            cells.append(f"<td{attr}>{money(row[hr])}</td>")
        body_rows.append(f'<tr>{"".join(cells)}</tr>')

    table = (
        '<div class="table-wrap"><table class="sens-table">'
        f'<thead><tr>{"".join(heads)}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table></div>'
        '<p class="card-note" style="margin-top:.9rem">Annual savings (USD) by monthly '
        "query volume &times; cache hit rate. The highlighted cell is the validated "
        "100K &times; 44.4% scenario shown above. Same model as eval/savings_model.py.</p>"
    )
    st.markdown(card("Find your own number", "", table), unsafe_allow_html=True)


def render_trust() -> None:
    st.markdown('<div class="section-title">Trust &amp; safety</div>', unsafe_allow_html=True)
    rr = "enabled" if RERANKER_ENABLED else "disabled"
    cards = f"""
    <div class="trust-card ok">
      <div class="trust-h">🔎 Two-stage retrieval</div>
      <div class="trust-b">Stage 1 filters candidates by bi-encoder cosine
        &ge; <span class="stat-inline">{STAGE1_THRESHOLD:.2f}</span> (tuned for recall).
        Stage 2 reranks with a cross-encoder ({rr}), accepting a hit only above
        <span class="stat-inline">{RERANKER_THRESHOLD:.3f}</span>. A single cosine
        threshold could not separate the classes; the reranker can.</div>
    </div>
    <div class="trust-card ok">
      <div class="trust-h">🎯 95% precision target</div>
      <div class="trust-b">The threshold is set to a <span class="stat-inline">95%
        precision</span> operating point, so at most ~5% of cache hits return an
        answer to a different question. Precision is the guardrail, not raw hit rate.</div>
    </div>
    <div class="trust-card ok">
      <div class="trust-h">📊 Validated on held-out data</div>
      <div class="trust-b">On the QQP-validation split, two-stage retrieval reaches
        <span class="stat-inline">68.8% recall at 95% precision</span> — vs ~7.7% for a
        bi-encoder alone. Contamination checked: train 66.8% vs validation 68.8%
        (no memorization drop).</div>
    </div>
    <div class="trust-card warn">
      <div class="trust-h">⚠️ Documented limitation (PAWS)</div>
      <div class="trust-b">On PAWS — adversarial paraphrases with near-identical
        wording but inverted meaning — precision drops. Adversarial or
        entity-swapped traffic would need a third-tier LLM judge or a
        domain-fine-tuned reranker. Not hidden: measured and disclosed.</div>
    </div>
    """
    st.markdown(f'<div class="trust-grid">{cards}</div>', unsafe_allow_html=True)


def render_live_session(data: dict) -> None:
    """Actual metrics from the running server — clearly separated from the benchmark."""
    st.markdown('<div class="section-title">Live session — this server</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="live-head"><span class="live-badge">LIVE</span>'
        '<span class="live-title">Actual metrics from this running gateway</span></div>'
        '<div class="live-disclaimer">Reflects only the queries sent to this server '
        "since it started — <b>not</b> the validated benchmark above. With a handful "
        "of demo queries these numbers are noisy by design.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        stat_tiles([
            ("Hit rate (session)", f"{data.get('hit_rate', 0.0) * 100:.1f}%"),
            ("Requests", f"{data.get('total_requests', 0):,}"),
            ("Hit latency", ms(data.get("avg_hit_latency_ms", 0.0))),
            ("Miss latency", ms(data.get("avg_miss_latency_ms", 0.0))),
            ("Saved (session)", money_small(data.get("total_savings_usd", 0.0))),
        ]),
        unsafe_allow_html=True,
    )

    # Row 1: routing donut + latency bars (this session).
    routing = donut([
        ("Cache", data.get("cache_hits", 0), C_CACHE),
        ("Local", data.get("local_routes", 0), C_LOCAL),
        ("Remote", data.get("remote_routes", 0), C_REMOTE),
    ])
    latency = bar_chart([
        ("Cache hit", data.get("avg_hit_latency_ms", 0.0), ms(data.get("avg_hit_latency_ms", 0.0)), C_CACHE),
        ("LLM (miss)", data.get("avg_miss_latency_ms", 0.0), ms(data.get("avg_miss_latency_ms", 0.0)), C_REMOTE),
    ])
    st.markdown(
        f'<div class="grid-2">'
        f'{card("Where requests were served", "Cache hit vs local vs remote model", routing)}'
        f'{card("Latency", "Cache hit vs LLM call on a miss (local + remote)", latency)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Row 2: cost breakdown + token usage (this session).
    actual = data.get("total_cost_usd", 0.0)
    baseline = data.get("total_baseline_cost_usd", 0.0)
    cost = bar_chart([
        ("All-remote", baseline, money_small(baseline), C_BASELINE),
        ("Actual", actual, money_small(actual), C_CACHE),
    ])
    avg_in = data.get("avg_input_tokens", 0.0)
    avg_out = data.get("avg_output_tokens", 0.0)
    tokens = bar_chart([
        ("Input", avg_in, f"{avg_in:.0f} tok", C_LOCAL),
        ("Output", avg_out, f"{avg_out:.0f} tok", C_REMOTE),
    ])
    st.markdown(
        f'<div class="grid-2" style="margin-top:1rem">'
        f'{card("Spend vs baseline", "Actual cost vs an all-remote gateway", cost)}'
        f'{card("Avg tokens per query", "Output is priced ~6x input", tokens)}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_assumptions() -> None:
    with st.expander("Assumptions & basis — what is MEASURED vs ASSUMED"):
        rows = []
        for tag, key, detail in VALIDATED_BASIS:
            klass = "b-measured" if tag == "MEASURED" else "b-assumed"
            rows.append(
                f'<div class="assump-row"><span class="badge {klass}">{tag}</span>'
                f'<span class="assump-key">{key}</span>'
                f'<span class="assump-val">{detail}</span></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.caption(
            "Local (Ollama) routes are modeled at $0 — optimistic; self-hosted "
            "inference still costs compute. The 44.4% hit rate is from a synthetic "
            "workload; real traffic determines the real number. For the full "
            "volume × hit-rate model, see eval/savings_model.py."
        )


def render_query_tester(url: str) -> None:
    st.markdown('<div class="section-title">Live query tester</div>', unsafe_allow_html=True)
    st.caption(
        "Send a prompt to the gateway. Submit a question, then submit a reworded "
        "version — watch the second one come back as a cache hit."
    )
    prompt = st.text_area(
        "Prompt", key="tester_prompt", height=90,
        placeholder="e.g. What is the capital of France?",
        label_visibility="collapsed",
    )
    if st.button("Send to gateway", type="primary"):
        if not prompt.strip():
            st.warning("Enter a prompt first.")
            return
        with st.spinner("Querying the gateway..."):
            result, error = post_query(url, prompt.strip())
        if error is not None:
            st.error(f"Query failed: {error}")
            return
        _render_query_result(result)


def _render_query_result(r: dict) -> None:
    hit = r.get("cache_hit", False)
    route = r.get("route", "?")
    hit_badge = (
        '<span class="badge b-green">CACHE HIT</span>' if hit
        else '<span class="badge b-red">CACHE MISS</span>'
    )
    route_klass = {"cache": "b-green", "local": "b-blue", "remote": "b-amber"}.get(route, "b-gray")
    sim = r.get("similarity")
    rr = r.get("reranker_score")
    sim_txt = f"{sim:.3f}" if isinstance(sim, (int, float)) else "n/a"
    rr_txt = f"{rr:.3f}" if isinstance(rr, (int, float)) else "n/a"
    answer = (r.get("answer") or "").strip() or "(empty response)"

    st.markdown(
        f"""
        <div class="qr">
          <div class="qr-meta">
            {hit_badge}
            <span class="badge {route_klass}">route: {route}</span>
            <span class="qr-chip">model <b>{r.get("model", "?")}</b></span>
            <span class="qr-chip">stage-1 sim <b>{sim_txt}</b></span>
            <span class="qr-chip">reranker <b>{rr_txt}</b></span>
            <span class="qr-chip">latency <b>{r.get("latency_ms", 0):.0f} ms</b></span>
          </div>
          <div class="qr-answer">{answer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render_unreachable(url: str, error: str) -> None:
    st.markdown(
        f"""
        <div class="unreachable">
          <h3>⚠️ Can't reach the gateway</h3>
          <p>No response from <code>{url}</code>.</p>
          <p style="color:#64748b;font-size:.9rem;"><b>If this is the hosted
          backend:</b> it runs on Render's free tier and sleeps when idle, so the
          first request can take <b>~50&nbsp;s</b> to wake it — wait ~50&nbsp;s and
          press <b>Refresh</b>.</p>
          <p style="color:#64748b;font-size:.9rem;">Running locally? Start it with
          <code>uvicorn app.main:app</code> or <code>docker compose up</code>,
          then check the URL in the sidebar and press <b>Refresh</b>.</p>
          <p style="color:#94a3b8;font-size:.82rem;margin-top:.6rem;">Details: {error}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} — FinOps Dashboard", page_icon="⚡", layout="wide")
    st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Gateway")
        url = st.text_input("URL", value=DEFAULT_URL, label_visibility="collapsed")
        refresh = st.button("↻ Refresh", use_container_width=True)
        health, _ = get_json(url, "/health")
        if health is not None:
            st.markdown(
                f'<div class="conn"><span class="dot" style="background:{C_CACHE}"></span>'
                f'Connected · v{health.get("version", "?")} · '
                f'{health.get("cache_size", 0):,} cached</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="conn"><span class="dot" style="background:{C_REMOTE}"></span>'
                f'Offline</div>', unsafe_allow_html=True,
            )
        st.info(
            "⏳ The backend runs on Render's free tier and **sleeps when idle** — "
            "the first request can take **~50 s** to wake it. If it shows offline, "
            "wait ~50 s and press Refresh.",
            icon="⏳",
        )
        st.caption("Headline numbers are the validated benchmark. The Live session "
                   "panel reads /metrics from this server. Refreshes on demand.")
    if refresh:
        st.rerun()

    st.markdown(
        f'<div class="brand"><span class="brand-mark">⚡</span>'
        f'<span class="brand-name">{APP_NAME}</span></div>'
        f'<div class="brand-sub">LLM cost-optimization gateway · live FinOps view</div>',
        unsafe_allow_html=True,
    )

    data, error = get_json(url, "/metrics")

    # The validated headline does NOT depend on the live server — always show it.
    render_executive(data or {})
    render_savings_grid()
    render_trust()

    st.markdown('<div class="section-title">&nbsp;</div>', unsafe_allow_html=True)
    if data is None:
        render_unreachable(url, error or "unknown error")
    else:
        render_live_session(data)
        render_query_tester(url)

    render_assumptions()


if __name__ == "__main__":
    main()
