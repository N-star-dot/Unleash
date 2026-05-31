"""hud_theme.py — Tactical HUD "Command Center" theme for the Unleash service-dog Streamlit app.

Single source of truth for the look is design_export/tokens.css + design_export/hud.css.
This module mirrors those tokens & component classes so the Streamlit shell and the
HTML helpers below render the same cyan-#00D4FF / teal-#34E0CE tactical HUD aesthetic.

Usage
-----
    import hud_theme as hud
    hud.inject_theme()                                  # once, at top of the app
    st.markdown(hud.stat_card("12", "DETECTIONS", "LIVE", "success"),
                unsafe_allow_html=True)

Every helper returns an HTML string; pass it to st.markdown(..., unsafe_allow_html=True).
Only token colors are used. No external JS — icons are inline SVG / unicode.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Variant -> token color map. Keys are the public variant names used across the
# helper functions; values resolve to CSS custom properties from tokens.css.
# ---------------------------------------------------------------------------
_VARIANTS = {
    "success": ("--hud-success", "--hud-success-12"),
    "warning": ("--hud-warning", "--hud-warning-12"),
    "danger": ("--hud-danger", "--hud-danger-12"),
    "memory": ("--hud-memory", "--hud-memory-12"),
    "cyan": ("--hud-cyan", "--hud-cyan-12"),
    "live": ("--hud-success", "--hud-success-12"),
}


def _vcolor(variant: str | None) -> tuple[str, str]:
    """Resolve (accent_var, tint_var) for a variant, defaulting to cyan."""
    return _VARIANTS.get(variant or "cyan", _VARIANTS["cyan"])


# A handful of tiny inline icons (stroke = currentColor so variants tint them).
def _icon(name: str) -> str:
    paths = {
        "eye": '<circle cx="12" cy="12" r="3"/><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/>',
        "shield": '<path d="M12 2 4 5v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V5l-8-3Z"/>',
        "alert": '<path d="M12 2 1 21h22L12 2Z"/><path d="M12 9v5M12 17h.01"/>',
        "brain": '<path d="M9 3a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 5 2V3Z"/>',
        "pulse": '<path d="M3 12h4l2-6 4 12 2-6h6"/>',
        "dog": '<path d="M10 5 8 3v4l-3 2 1 9h10l1-9-3-2V3l-2 2"/>',
    }
    inner = paths.get(name, paths["pulse"])
    return (
        f'<svg viewBox="0 0 24 24" width="13" height="13" fill="none" '
        f'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true" focusable="false">{inner}</svg>'
    )


# ===========================================================================
# 1. THEME INJECTION
# ===========================================================================
def inject_theme() -> None:
    """Inject the full Command Center HUD stylesheet into the Streamlit app."""
    st.markdown(_STYLE, unsafe_allow_html=True)


_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');

/* ---- Tokens (mirror of design_export/tokens.css) ---- */
:root {
  --hud-bg:#080C14; --hud-card:#0C1220; --hud-inset:#0A0F1C; --hud-border:#00D4FF33;
  --hud-cyan:#00D4FF; --hud-cyan-08:#00D4FF14; --hud-cyan-12:#00D4FF1F;
  --hud-cyan-20:#00D4FF33; --hud-cyan-40:#00D4FF66;
  /* --hud-muted raised to an opaque cyan-grey so small informational text
     (sub-lines, honest-status notes, empty-state copy) clears WCAG 1.4.3 AA
     4.5:1 over the card/deep bg. Was #00D4FF73 (~3.0:1, large-text only). */
  --hud-muted:#7FB8CE;
  --hud-label:#8FB8C8; --hud-text:#E0F2FE;
  --hud-success:#00E676; --hud-success-12:#00E6761F;
  --hud-warning:#FFB300; --hud-warning-12:#FFB3001F;
  --hud-danger:#FF5252;  --hud-danger-12:#FF52521F;
  --hud-memory:#A78BFA;  --hud-memory-12:#A78BFA22;
  /* Command Center / marketing palette */
  --accent:#34E0CE; --accent-blue:#5390FF; --accent-dim:#34E0CE26;
  --bg-base:#0C111A; --bg-deep:#080B11; --bg-panel:#111824;
  --bg-elevated:#161F2E; --bg-inset:#0A0F17; --bg-glow:#161F2E;
  --border-strong:#2C3D54; --border-subtle:#1E2A3C;
  --text-primary:#EAF1FA; --text-secondary:#8C9BB2; --text-muted:#566276;
  --safe:#3FE08A; --safe-dim:#3FE08A22; --warn:#FFB23E; --danger:#FF4D5E;
  --font-mono:"JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
  --font-body:"Inter", system-ui, -apple-system, sans-serif;
  --r-sm:3px; --r-md:4px; --r-lg:6px;
}

/* ---- Streamlit shell -> Command Center dark HUD ---- */
.stApp, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 600px at 50% -10%, var(--bg-glow) 0%, transparent 60%),
    var(--bg-deep);
  color: var(--text-primary);
  font-family: var(--font-body);
}
.main .block-container { padding-top: 2.2rem; max-width: 1180px; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1rem; }

/* Headings -> mono, tracked, teal/cyan accents */
h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-mono); color: var(--text-primary);
  letter-spacing: 2px; text-transform: uppercase; font-weight: 700;
}
h1 { font-size: 1.55rem; }
h2 { font-size: 1.15rem; color: var(--accent); letter-spacing: 2.5px; }
h3 { font-size: .95rem; color: var(--hud-cyan); letter-spacing: 2px; }
p, li, span, label, .stMarkdown { color: var(--text-secondary); }
a, a:visited { color: var(--accent); text-decoration: none; }
hr { border: none; border-top: 1px solid var(--border-subtle); margin: 1.1rem 0; }
code { background: var(--bg-inset); color: var(--hud-cyan);
       border: 1px solid var(--hud-border); border-radius: var(--r-sm); padding: .05rem .35rem; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-base) 100%);
  border-right: 1px solid var(--border-subtle);
}
[data-testid="stSidebar"] * { color: var(--text-secondary); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--accent); }

/* ---- Sidebar radio nav -> tactical tab treatment ----
   The primary nav is an st.radio; without this it renders as default Streamlit
   radio dots + body text. Give each option mono/tracked labels with hover +
   selected cyan accent, and color the selected dot with --accent so selection
   is perceptible on the dark sidebar (not a washed-out browser default). */
[data-testid="stSidebar"] [role="radiogroup"] label {
  font-family: var(--font-mono); font-size: .74rem; letter-spacing: 2px;
  text-transform: uppercase; color: var(--hud-label);
  padding: .3rem .2rem; border-left: 2px solid transparent; transition: all .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { color: var(--text-primary); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  color: var(--hud-cyan); border-left: 2px solid var(--hud-cyan);
  text-shadow: 0 0 8px var(--hud-cyan-40);
}
[data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] div[aria-checked="true"] {
  background: var(--accent) !important; border-color: var(--accent) !important;
}

/* ---- Tabs (active tab = cyan underline glow) ---- */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px; border-bottom: 1px solid var(--border-subtle); background: transparent;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--font-mono); font-size: .72rem; letter-spacing: 2px;
  text-transform: uppercase; color: var(--hud-label);
  background: transparent; border-radius: 0; padding: .55rem .9rem;
  border-bottom: 2px solid transparent;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-primary); }
.stTabs [aria-selected="true"] {
  color: var(--hud-cyan) !important;
  border-bottom: 2px solid var(--hud-cyan);
  box-shadow: 0 2px 6px -2px var(--hud-cyan);
}
.stTabs [data-baseweb="tab-highlight"] { background: var(--hud-cyan); }

/* ---- Metric ---- */
[data-testid="stMetric"] {
  background: var(--hud-card); border: 1px solid var(--hud-border);
  border-radius: var(--r-md); padding: .85rem 1rem;
}
[data-testid="stMetricLabel"] {
  font-family: var(--font-mono); font-size: .72rem !important; letter-spacing: 1.5px;
  text-transform: uppercase; color: var(--hud-label);
}
[data-testid="stMetricValue"] {
  font-family: var(--font-mono); color: var(--hud-cyan);
  text-shadow: 0 0 8px var(--hud-cyan); letter-spacing: 2px;
}
[data-testid="stMetricDelta"] { font-family: var(--font-mono); }

/* ---- Buttons (HUD bordered) ---- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  font-family: var(--font-mono); font-size: .72rem; letter-spacing: 1.5px;
  text-transform: uppercase; font-weight: 700;
  color: var(--hud-cyan); background: var(--hud-cyan-08);
  border: 1px solid var(--hud-cyan-40); border-radius: var(--r-md);
  padding: .5rem 1.1rem; transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
  color: var(--hud-bg); background: var(--hud-cyan);
  border-color: var(--hud-cyan); box-shadow: 0 0 12px var(--hud-cyan-40);
}
/* Focus indicator: full-opacity cyan (#00D4FF ~11:1 on the deep bg) at >=2px,
   never nulling the outline without a visible replacement (WCAG 2.4.7 / 1.4.11). */
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
.stFormSubmitButton > button:focus-visible {
  outline: 2px solid var(--hud-cyan); outline-offset: 2px;
  box-shadow: 0 0 0 4px var(--hud-cyan-40);
}

/* ---- Toggle / checkbox ---- */
[data-baseweb="checkbox"] [data-testid="stWidgetLabel"] p,
.stToggle label { font-family: var(--font-mono); letter-spacing: 1px; }
[data-baseweb="checkbox"] div[role="presentation"] { background: var(--bg-inset); }
.stCheckbox [aria-checked="true"] > div,
[data-baseweb="checkbox"] [aria-checked="true"] { background: var(--accent) !important; }
[data-baseweb="toggle"] div[aria-checked="true"] { background: var(--accent) !important; }
[data-baseweb="toggle"] { background: var(--border-strong); }

/* ---- Inputs / selects ---- */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="select"] > div {
  background: var(--bg-inset) !important; color: var(--text-primary) !important;
  border: 1px solid var(--border-subtle) !important; border-radius: var(--r-md) !important;
  font-family: var(--font-mono);
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus,
[data-baseweb="select"] > div:focus-within {
  border-color: var(--hud-cyan) !important;
  box-shadow: 0 0 0 2px var(--hud-cyan) !important;
}

/* Visible focus for the sidebar radio nav, toggle, and checkbox — these had no
   explicit focus style and relied on browser defaults the dark theme can wash
   out (WCAG 2.4.7 / 1.4.11). Full-cyan ring, never color-on-color sub-3:1. */
[data-testid="stSidebar"] [role="radiogroup"] label:focus-within,
[data-baseweb="radio"]:focus-within,
.stCheckbox label:focus-within, [data-baseweb="checkbox"]:focus-within,
.stToggle label:focus-within, [data-baseweb="toggle"]:focus-within {
  outline: 2px solid var(--hud-cyan); outline-offset: 2px; border-radius: var(--r-sm);
}
[data-testid="stSidebar"] [role="radiogroup"] input:focus-visible,
.stCheckbox input:focus-visible, .stToggle input:focus-visible {
  outline: 2px solid var(--hud-cyan); outline-offset: 2px;
}

/* ---- DataFrame / tables ---- */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  background: var(--hud-card); border: 1px solid var(--hud-border);
  border-radius: var(--r-md); font-family: var(--font-mono);
}
[data-testid="stDataFrame"] [role="columnheader"] {
  background: var(--bg-elevated); color: var(--hud-cyan);
  letter-spacing: 1px; text-transform: uppercase; font-size: .68rem;
}
[data-testid="stDataFrame"] [role="gridcell"] { color: var(--text-primary); }

/* ---- Audio container ---- */
[data-testid="stAudio"] {
  background: var(--hud-card); border: 1px solid var(--hud-cyan-40);
  border-radius: var(--r-md); box-shadow: 0 0 10px var(--hud-cyan-20);
  padding: .55rem; }
audio { width: 100%; filter: hue-rotate(160deg) saturate(1.1); }

/* ---- Expander / alerts ---- */
[data-testid="stExpander"] {
  background: var(--hud-card); border: 1px solid var(--hud-border);
  border-radius: var(--r-md); }
[data-testid="stExpander"] summary { font-family: var(--font-mono); letter-spacing: 1px; }
.stAlert { border-radius: var(--r-md); border: 1px solid var(--hud-border);
           background: var(--hud-card); font-family: var(--font-mono); }

/* ===================================================================
   HUD component classes (mirror of design_export/hud.css). Used by the
   HTML helper functions below.
   =================================================================== */
.hud-sep { display:flex; align-items:center; gap:8px; margin:.6rem 0;
  font-size:inherit; text-transform:none; }
.hud-sep::before, .hud-sep::after { content:""; flex:1; height:1px; background:var(--hud-cyan-20); }
.hud-sep span { font-family:var(--font-mono); font-size:12px; letter-spacing:2px;
  color:var(--hud-label); white-space:nowrap; }

.stat-card { position:relative; display:flex; flex-direction:column; gap:6px;
  padding:14px; min-height:98px; background:var(--hud-card);
  border:1px solid var(--hud-border); border-radius:var(--r-md);
  font-family:var(--font-mono); }
.stat-card .top { display:flex; align-items:center; justify-content:space-between; }
.stat-card .ib { width:26px; height:26px; display:flex; align-items:center; justify-content:center;
  border-radius:var(--r-md); background:var(--hud-cyan-12);
  border:1px solid var(--hud-cyan); color:var(--hud-cyan); }
.stat-card .cb { padding:3px 7px; border-radius:var(--r-sm); background:var(--hud-cyan-12);
  border:1px solid var(--hud-cyan); color:var(--hud-cyan);
  font-size:8px; letter-spacing:1px; }
.stat-card .num { font-size:30px; font-weight:700; letter-spacing:2px; line-height:1;
  color:var(--hud-cyan); text-shadow:0 0 8px var(--hud-cyan); }
.stat-card .lbl { font-size:12px; letter-spacing:1.5px; color:var(--hud-label); }
.corner { position:absolute; background:var(--hud-cyan); }
.corner.tl-h{top:0;left:0;width:14px;height:1.5px} .corner.tl-v{top:0;left:0;width:1.5px;height:14px}
.corner.tr-h{top:0;right:0;width:14px;height:1.5px;opacity:.5} .corner.tr-v{top:0;right:0;width:1.5px;height:14px;opacity:.5}
.corner.bl-h{bottom:0;left:0;width:14px;height:1.5px;opacity:.35} .corner.bl-v{bottom:0;left:0;width:1.5px;height:14px;opacity:.35}
.corner.br-h{bottom:0;right:0;width:14px;height:1.5px;opacity:.35} .corner.br-v{bottom:0;right:0;width:1.5px;height:14px;opacity:.35}

.panel { position:relative; background:var(--hud-card); border:1px solid var(--hud-border);
  border-radius:var(--r-md); overflow:hidden; font-family:var(--font-mono); margin:.4rem 0; }
.panel-head { display:flex; align-items:center; gap:8px; padding:11px 12px;
  border-bottom:1px solid var(--hud-border); }
.panel-head .dot { width:6px; height:6px; border-radius:50%; background:var(--hud-cyan);
  box-shadow:0 0 6px var(--hud-cyan); }
.panel-head .pht { display:flex; flex-direction:column; gap:2px; flex:1; }
.panel-head .t { font-size:11px; letter-spacing:2px; color:var(--hud-cyan);
  margin:0; text-transform:uppercase; font-weight:700; }
.panel-head .s { font-size:12px; letter-spacing:1.5px; color:var(--hud-label); }
.panel-body { display:flex; flex-direction:column; gap:10px; padding:12px;
  font-family:var(--font-body); color:var(--text-secondary); font-size:13px; }

.chips { display:flex; flex-wrap:wrap; gap:6px; margin:.3rem 0; }
.chip { display:inline-flex; align-items:center; gap:5px; padding:5px 8px;
  border-radius:var(--r-sm); background:var(--hud-inset); border:1px solid var(--hud-cyan);
  color:var(--hud-text); font-family:var(--font-mono); font-size:9px; letter-spacing:.5px; }
.chip .d { width:5px; height:5px; border-radius:50%; background:var(--hud-cyan); }
.chip.is-danger{border-color:var(--hud-danger)} .chip.is-danger .d{background:var(--hud-danger)}
.chip.is-warning{border-color:var(--hud-warning)} .chip.is-warning .d{background:var(--hud-warning)}
.chip.is-success{border-color:var(--hud-success)} .chip.is-success .d{background:var(--hud-success)}

.pill { display:inline-flex; align-items:center; gap:6px; padding:5px 9px;
  border-radius:var(--r-sm); font-family:var(--font-mono); font-size:9px; letter-spacing:1.5px; }
.pill .dot { width:5px; height:5px; border-radius:50%; }
.pill--live { background:var(--hud-success-12); border:1px solid var(--hud-success); color:var(--hud-success); }
.pill--live .dot { background:var(--hud-success); box-shadow:0 0 5px var(--hud-success); }
.pill--danger { background:var(--hud-danger-12); border:1px solid var(--hud-danger); color:var(--hud-danger); }
.pill--danger .dot { background:var(--hud-danger); }
.pill--cyan { background:var(--hud-cyan-12); border:1px solid var(--hud-cyan); color:var(--hud-cyan); }
.pill--cyan .dot { background:var(--hud-cyan); }
.pill--warning { background:var(--hud-warning-12); border:1px solid var(--hud-warning); color:var(--hud-warning); }
.pill--warning .dot { background:var(--hud-warning); }

.hud-reactor-wrap { display:flex; align-items:center; gap:14px; padding:6px 0 12px;
  font-family:var(--font-mono); }
.hud-reactor { position:relative; width:40px; height:40px; flex:none; }
.hud-reactor .ring1,.hud-reactor .ring2,.hud-reactor .core { position:absolute; border-radius:50%; }
.hud-reactor .ring1 { width:40px; height:40px; border:1px solid var(--hud-cyan-20); }
.hud-reactor .ring2 { width:28px; height:28px; inset:6px; border:1.5px solid var(--hud-cyan-40); }
.hud-reactor .core { width:16px; height:16px; inset:12px; display:flex; align-items:center;
  justify-content:center; background:var(--hud-cyan-20); border:1.5px solid var(--hud-cyan);
  box-shadow:0 0 8px var(--hud-cyan); color:var(--hud-cyan); }
.hud-htxt { display:flex; flex-direction:column; gap:3px; flex:1; }
.hud-title { font-size:15px; font-weight:700; letter-spacing:2px; color:var(--text-primary); }
.hud-sub { font-size:12px; letter-spacing:1.5px; color:var(--hud-label); }

.claim-row { display:flex; align-items:center; gap:10px; padding:9px 11px; margin:5px 0;
  background:var(--bg-inset); border:1px solid var(--border-subtle);
  border-left:3px solid var(--hud-cyan); border-radius:var(--r-sm);
  font-family:var(--font-mono); }
.claim-row .src { font-size:12px; letter-spacing:1px; color:var(--hud-label);
  min-width:74px; text-transform:uppercase; }
.claim-row .typ { font-size:11px; letter-spacing:.5px; color:var(--text-primary); flex:1; }
.claim-row .sev { font-size:9px; font-weight:700; letter-spacing:1px; padding:2px 6px;
  border-radius:var(--r-sm); text-transform:uppercase; }
.claim-row .val { font-size:11px; font-weight:700; letter-spacing:1px; }

.agent-chip { display:inline-flex; align-items:center; gap:7px; padding:5px 9px; margin:3px;
  background:var(--hud-card); border:1px solid var(--border-subtle); border-radius:var(--r-sm);
  font-family:var(--font-mono); font-size:9px; letter-spacing:1px; color:var(--text-secondary); }
.agent-chip .adot { width:6px; height:6px; border-radius:50%; background:var(--text-muted); }
.agent-chip.is-online { border-color:var(--hud-success); color:var(--text-primary); }
.agent-chip.is-online .adot { background:var(--hud-success); box-shadow:0 0 5px var(--hud-success); }

/* visually-hidden: present for screen readers, off-screen visually */
.vis-hidden { position:absolute; width:1px; height:1px; padding:0; margin:-1px;
  overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; border:0; }
</style>
"""


# ===========================================================================
# 2. HTML HELPER FUNCTIONS  (return strings -> st.markdown(..., unsafe_allow_html=True))
# ===========================================================================
def stat_card(value: str, label: str, badge: str = "", variant: str | None = None) -> str:
    """Corner-bracketed metric card. variant in {success,warning,danger,memory}
    recolors the number glow, icon box, and badge via inline styles."""
    accent, tint = _vcolor(variant)
    icon_name = {"success": "shield", "warning": "alert",
                 "danger": "alert", "memory": "brain"}.get(variant or "", "eye")
    ib = f"background:var({tint});border-color:var({accent});color:var({accent})"
    num = f"color:var({accent});text-shadow:0 0 8px var({accent})"
    cb = f"background:var({tint});border-color:var({accent});color:var({accent})"
    badge_html = f'<span class="cb" style="{cb}">{badge}</span>' if badge else ""
    corners = "".join(
        f'<span class="corner {c}"></span>'
        for c in ("tl-h", "tl-v", "tr-h", "tr-v", "bl-h", "bl-v", "br-h", "br-v")
    )
    return (
        f'<div class="stat-card">{corners}'
        f'<div class="top"><span class="ib" style="{ib}">{_icon(icon_name)}</span>{badge_html}</div>'
        f'<div class="num" style="{num}">{value}</div>'
        f'<div class="lbl">{label}</div></div>'
    )


def panel(title: str, subtitle: str = "", body_html: str = "", pill_html: str = "") -> str:
    """Bordered .panel with a cyan-dot head, optional subtitle + status pill."""
    sub = f'<div class="s">{subtitle}</div>' if subtitle else ""
    pill = f'<div style="margin-left:auto">{pill_html}</div>' if pill_html else ""
    return (
        f'<div class="panel"><div class="panel-head"><span class="dot"></span>'
        f'<div class="pht"><h3 class="t">{title}</h3>{sub}</div>{pill}</div>'
        f'<div class="panel-body">{body_html}</div></div>'
    )


def chips(items: list[dict]) -> str:
    """Row of HUD chips. items = [{"text": str, "variant": danger|warning|success|None}]."""
    out = []
    for it in items:
        variant = it.get("variant")
        cls = f" is-{variant}" if variant in ("danger", "warning", "success") else ""
        out.append(f'<span class="chip{cls}"><span class="d"></span>{it.get("text", "")}</span>')
    return f'<div class="chips">{"".join(out)}</div>'


def pill(text: str, variant: str = "cyan") -> str:
    """Status pill. variant in {live, danger, cyan, warning}."""
    variant = variant if variant in ("live", "danger", "cyan", "warning") else "cyan"
    return f'<span class="pill pill--{variant}"><span class="dot"></span>{text}</span>'


def hud_sep(label: str) -> str:
    """The ─── LABEL ─── section separator, emitted as a real <h2> so screen
    readers can navigate by heading (WCAG 1.3.1 / 2.4.6)."""
    return f'<h2 class="hud-sep"><span>{label}</span></h2>'


def reactor_header(title: str, subtitle: str = "", pill_html: str = "") -> str:
    """Arc-reactor mark + title block; optional trailing status pill."""
    sub = f'<div class="hud-sub">{subtitle}</div>' if subtitle else ""
    pill = f'<div>{pill_html}</div>' if pill_html else ""
    return (
        '<div class="hud-reactor-wrap"><div class="hud-reactor">'
        '<span class="ring1"></span><span class="ring2"></span>'
        f'<span class="core">{_icon("dog")}</span></div>'
        f'<div class="hud-htxt"><div class="hud-title">{title}</div>{sub}</div>{pill}</div>'
    )


def claim_row(source: str, ctype: str, value: str, severity: str = "cyan") -> str:
    """Conflict-bus claim row: inset row with a colored left border by severity
    (danger / warning / cyan / memory). Severity is also stated as text + a
    badge so it never relies on color alone (WCAG 1.4.1)."""
    accent, tint = _vcolor(severity)
    border = f"border-left-color:var({accent})"
    val = f"color:var({accent})"
    sev_text = {"danger": "ALERT", "warning": "HEADS-UP",
                "memory": "MEMORY"}.get(severity, "INFO")
    sev = f"background:var({tint});border:1px solid var({accent});color:var({accent})"
    return (
        f'<div class="claim-row" style="{border}">'
        f'<span class="sev" style="{sev}">{sev_text}</span>'
        f'<span class="src">{source}</span>'
        f'<span class="typ">{ctype}</span>'
        f'<span class="val" style="{val}">{value}</span></div>'
    )


def agent_status(name: str, state: str = "ONLINE") -> str:
    """Small agent chip for the Agents row. ONLINE = success dot, else muted."""
    online = (state or "").upper() == "ONLINE"
    cls = " is-online" if online else ""
    return (
        f'<span class="agent-chip{cls}"><span class="adot"></span>'
        f'{name} · {state}</span>'
    )
