"""Unleash — Companion HUD (Streamlit embed).

Renders the 390x844 cyan-on-black HUD HOME screen inside Streamlit via
st.components.v1.html, wired to live values. The iframe cannot see the parent
page CSS, so tokens.css + hud.css are inlined into an embedded <style> block
and the lucide CDN script is injected so icons render inside the iframe.

Usage:
    from companion_hud import render_companion_hud
    render_companion_hud(state)   # state is an optional dict of live values

All state keys are optional; each falls back to the mockup default from
design_export/hud_home.html.
"""

from __future__ import annotations

import html
import os

import streamlit.components.v1 as components

_HERE = os.path.dirname(os.path.abspath(__file__))
_DESIGN_DIR = os.path.join(_HERE, "design_export")


def _read_css(name: str) -> str:
    """Read a CSS file from design_export/, returning '' if missing."""
    path = os.path.join(_DESIGN_DIR, name)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _esc(value: object) -> str:
    """HTML-escape a value for safe injection into text/attribute spots."""
    return html.escape(str(value), quote=True)


# Tone -> pill class. URGENT is a danger pill; CALM/SAFE are cyan.
_TONE_DANGER = {"URGENT", "ALERT", "DANGER", "WARNING"}


def _tone_pill_class(tone: str) -> str:
    return "pill--danger" if tone.strip().upper() in _TONE_DANGER else "pill--cyan"


def _stat_card(
    *,
    icon: str,
    icon_color: str,
    badge: str,
    badge_style: str,
    value: object,
    label: str,
    variant: str = "",
) -> str:
    """Build one stat card with corner brackets."""
    variant_cls = f" is-{variant}" if variant else ""
    corners = (
        '<span class="corner tl-h"></span><span class="corner tl-v"></span>'
        '<span class="corner tr-h"></span><span class="corner tr-v"></span>'
        '<span class="corner bl-h"></span><span class="corner bl-v"></span>'
        '<span class="corner br-h"></span><span class="corner br-v"></span>'
    )
    return f"""
        <div class="stat-card{variant_cls}">
          <div class="top">
            <span class="ib"><i data-lucide="{icon}" aria-hidden="true" style="width:14px;height:14px;color:{icon_color}"></i></span>
            <span class="cb"{badge_style}>{_esc(badge)}</span>
          </div>
          <div class="num">{_esc(value)}</div><div class="lbl">{_esc(label)}</div>
          {corners}
        </div>"""


def _chips(detections: list[dict]) -> str:
    """Build the detection chip row from state.detections."""
    out = []
    for det in detections:
        text = det.get("text", "")
        variant = det.get("variant", "")
        variant_cls = f" is-{variant}" if variant else ""
        out.append(
            f'<span class="chip{variant_cls}"><span class="d"></span>{_esc(text)}</span>'
        )
    return "".join(out)


def render_companion_hud(state: dict | None = None) -> None:
    """Render the HUD HOME screen inside Streamlit, wired to live values.

    Args:
        state: Optional dict of live values. Recognised keys (all optional):
            agents_online (int)      default 6
            confidence (int 0-100)   default 91
            heart_rate (int)         default 72
            hazards_near (int)       default 2
            heart_variant (str)      success/warning/danger; default "success"
            hazards_variant (str)    success/warning/danger; default "warning"
            detections (list[{text, variant}])
            spoken (str)             default "RED LIGHT. TWO CARS ON YOUR RIGHT. WAIT."
            tone (str)               default "URGENT"
            frame_info (str)         default "FRAME 4210 // 30FPS"
    """
    state = state or {}

    # --- live values w/ mockup defaults ---
    agents_online = state.get("agents_online", 6)
    confidence = state.get("confidence", 91)
    heart_rate = state.get("heart_rate", 72)
    hazards_near = state.get("hazards_near", 2)
    heart_variant = state.get("heart_variant", "success")
    hazards_variant = state.get("hazards_variant", "warning")
    detections = state.get(
        "detections",
        [
            {"text": "RED LIGHT", "variant": "danger"},
            {"text": "CROSSWALK"},
            {"text": "CAR x2", "variant": "warning"},
            {"text": "CURB", "variant": "success"},
        ],
    )
    spoken = state.get("spoken", "RED LIGHT. TWO CARS ON YOUR RIGHT. WAIT.")
    tone = state.get("tone", "URGENT")
    frame_info = state.get("frame_info", "FRAME 4210 // 30FPS")

    # --- variant-driven colors for the two semantic stat cards ---
    _var_color = {
        "success": "var(--hud-success)",
        "warning": "var(--hud-warning)",
        "danger": "var(--hud-danger)",
    }
    _var_tint = {
        "success": "var(--hud-success-12)",
        "warning": "var(--hud-warning-12)",
        "danger": "var(--hud-danger-12)",
    }
    heart_color = _var_color.get(heart_variant, "var(--hud-success)")
    heart_tint = _var_tint.get(heart_variant, "var(--hud-success-12)")
    hazards_color = _var_color.get(hazards_variant, "var(--hud-warning)")
    hazards_tint = _var_tint.get(hazards_variant, "var(--hud-warning-12)")

    # --- stat grid ---
    card_agents = _stat_card(
        icon="cpu",
        icon_color="var(--hud-cyan)",
        badge="OK",
        badge_style="",
        value=agents_online,
        label="AGENTS ONLINE",
    )
    card_confidence = _stat_card(
        icon="target",
        icon_color="var(--hud-cyan)",
        badge="HI",
        badge_style="",
        value=confidence,
        label="CONFIDENCE",
    )
    card_heart = _stat_card(
        icon="heart-pulse",
        icon_color=heart_color,
        badge="BPM",
        badge_style=f' style="background:{heart_tint};border-color:{heart_color}"',
        value=heart_rate,
        label="HEART RATE",
        variant=heart_variant,
    )
    card_hazards = _stat_card(
        icon="triangle-alert",
        icon_color=hazards_color,
        badge="NEAR",
        badge_style=f' style="background:{hazards_tint};border-color:{hazards_color}"',
        value=hazards_near,
        label="HAZARDS NEAR",
        variant=hazards_variant,
    )

    chips_html = _chips(detections)
    tone_pill_cls = _tone_pill_class(tone)

    css = _read_css("tokens.css") + "\n" + _read_css("hud.css")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Companion HUD mirror (read-only)</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap" rel="stylesheet" />
<script src="https://unpkg.com/lucide@latest"></script>
<style>{css}</style>
<style>
  html, body {{ background:#05080e; margin:0; }}
  body {{ display:grid; place-items:start center; min-height:100vh; padding:8px 0; }}
</style>
</head>
<body>
  <div class="hud-screen">

    <!-- StatusBar -->
    <div class="hud-statusbar">
      <span class="time">09:41</span>
      <span class="sbr"><span class="net">5G</span><i data-lucide="battery-full" aria-hidden="true" style="color:var(--hud-cyan);width:16px;height:16px"></i></span>
    </div>

    <!-- Header: arc-reactor + LIVE pill -->
    <div class="hud-header">
      <div class="hud-reactor">
        <div class="ring1"></div><div class="ring2"></div>
        <div class="core"><i data-lucide="radio" aria-hidden="true" style="width:8px;height:8px"></i></div>
      </div>
      <div class="hud-htxt">
        <div class="hud-title">COMMAND</div>
        <div class="hud-sub">SERVICE DOG // ONLINE</div>
      </div>
      <span class="pill pill--live"><span class="dot"></span>LIVE</span>
    </div>

    <!-- Body -->
    <div class="hud-body">
      <div class="hud-sep"><span>SENSORS // ACTIVE</span></div>

      <!-- Stat grid -->
      <div class="stat-grid">
        <div class="stat-row">{card_agents}{card_confidence}
        </div>
        <div class="stat-row">{card_heart}{card_hazards}
        </div>
      </div>

      <!-- Perception panel -->
      <div class="panel">
        <div class="panel-head">
          <span class="dot"></span>
          <div class="pht"><span class="t">PERCEPTION</span><span class="s">YOLOV8 // CONTINUITY CAM</span></div>
          <span class="pill pill--danger"><span class="dot"></span>REC</span>
        </div>
        <div class="panel-body">
          <div class="cam cam--standby" role="img" aria-label="Camera mirror, no live frame; standby">
            <span class="cam-standby">NO SIGNAL // STANDBY</span>
            <span class="frame-tag"><i data-lucide="scan-eye" aria-hidden="true" style="width:10px;height:10px;color:var(--hud-cyan)"></i>{_esc(frame_info)}</span>
          </div>
          <div class="chips">{chips_html}</div>
        </div>
      </div>

      <!-- Voice output panel -->
      <div class="voice">
        <div class="voice-head">
          <span class="vhl"><i data-lucide="volume-2" aria-hidden="true" style="width:14px;height:14px"></i>VOICE OUTPUT</span>
          <span class="pill {tone_pill_cls}">{_esc(tone)}</span>
        </div>
        <div class="voice-body" role="status" aria-live="assertive" aria-atomic="true">
          <span class="vis-hidden">Spoken directive: </span>
          <p class="voice-spoken">&ldquo;{_esc(spoken)}&rdquo;</p>
        </div>
      </div>
    </div>

    <!-- Tab bar: decorative mirror only; hidden from AT so it does not expose dead controls -->
    <nav class="hud-tabbar" aria-hidden="true">
      <span class="hud-tab is-active"><span class="ul"></span><i data-lucide="house" aria-hidden="true" style="width:21px;height:21px"></i><span class="lbl">HOME</span></span>
      <span class="hud-tab"><span class="ul"></span><i data-lucide="activity" aria-hidden="true" style="width:21px;height:21px"></i><span class="lbl">DATA</span></span>
      <span class="hud-tab"><span class="ul"></span><i data-lucide="map" aria-hidden="true" style="width:21px;height:21px"></i><span class="lbl">MAP</span></span>
      <span class="hud-tab"><span class="ul"></span><i data-lucide="user" aria-hidden="true" style="width:21px;height:21px"></i><span class="lbl">UNIT</span></span>
    </nav>
  </div>
  <script>
    function _renderIcons() {{
      if (window.lucide && typeof window.lucide.createIcons === "function") {{
        window.lucide.createIcons();
      }} else {{
        setTimeout(_renderIcons, 60);
      }}
    }}
    _renderIcons();
  </script>
</body>
</html>"""

    components.html(page, height=880, scrolling=False)

# touched 2026-06-03
