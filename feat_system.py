"""feat_system.py — FEAT // SYSTEM tab for the Unleash service-dog dashboard.

The fifth FEAT tab. There was no designed frame for SYSTEM, so this surface is
composed in the same tactical-cyan HUD language as the rest of the app, using
only hud_theme helpers + token colors.

Scope (per FEAT_TABS_PRD.md §5):
  • SERVICES   — API-key / service presence (Groq, Gemini, Shaped, Weave/W&B, Bland, Calendar)
  • CONFIG     — runtime config (Weave project, camera index, runtime versions)
  • EMERGENCY  — the "ARM REAL CALLS" toggle gating live Bland AI phone calls

Hard rule: this module reads only the PRESENCE/ABSENCE of secrets via
``os.environ`` — it never reads, prints, or logs any secret VALUE.

Public surface:
    render() -> None   # draws the SYSTEM tab body into the current container.
                       # Assumes hud.inject_theme() already ran in feat_ui.py.
"""

from __future__ import annotations

import html
import os
import platform

import streamlit as st

import hud_theme as hud

# ---------------------------------------------------------------------------
# Namespaced session-state keys (so reruns persist + tabs don't collide).
# ---------------------------------------------------------------------------
_K_CAMERA = "feat_system_camera_index"
_K_ARM = "arm_real_calls"  # shared gate name used by ui.py / run_pipeline.


def _esc(value: object) -> str:
    """HTML-escape any value before it touches an HTML helper string."""
    return html.escape(str(value), quote=True)


def _init_state() -> None:
    """Seed defaults so the tab renders fully before any interaction."""
    if _K_CAMERA not in st.session_state:
        st.session_state[_K_CAMERA] = 1
    if _K_ARM not in st.session_state:
        st.session_state[_K_ARM] = False  # default OFF — calls simulated.


# ---------------------------------------------------------------------------
# Service-presence checks. Each returns a (state_text, variant) WITHOUT ever
# reading a secret value — only bool(os.environ.get(...)) / file existence.
# variant maps to a hud.pill style: live=connected, danger=missing-required,
# cyan=optional-present, warning/muted handled via text below.
# ---------------------------------------------------------------------------
def _env_present(var: str) -> bool:
    """True if the env var is set AND non-empty. Never returns the value."""
    return bool(os.environ.get(var, "").strip())


def _calendar_credentials_present() -> bool:
    """Google Calendar is wired via a local credentials.json (OAuth client)."""
    try:
        return os.path.isfile(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials.json")
        ) or os.path.isfile("credentials.json")
    except Exception:  # noqa: BLE001 — presence check must never crash the tab.
        return False


# A service row: (label, detail, status_text, variant)
# variant: "live" connected | "danger" missing-required | "cyan" optional-present
#          | "muted" optional-absent (rendered as warning pill + OPTIONAL text)
def _service_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []

    # GEMINI — required for voice STT + general chat / intent classification.
    gem = _env_present("GEMINI_API_KEY")
    rows.append((
        "GEMINI",
        "GEMINI_API_KEY · voice STT + general chat",
        "CONNECTED" if gem else "MISSING",
        "live" if gem else "danger",
    ))

    # GROQ — required for the orchestrator LLM (Llama-3.3-70b).
    groq = _env_present("GROQ_API_KEY")
    rows.append((
        "GROQ",
        "GROQ_API_KEY · orchestrator (Llama-3.3-70b)",
        "CONNECTED" if groq else "MISSING",
        "live" if groq else "danger",
    ))

    # SHAPED — required for ranked episodic memory recall.
    shaped = _env_present("SHAPED_API_KEY")
    rows.append((
        "SHAPED",
        "SHAPED_API_KEY · episodic memory recall",
        "CONNECTED" if shaped else "MISSING",
        "live" if shaped else "danger",
    ))

    # WEAVE / W&B — optional cloud trace. Needs key + project to be useful.
    wb_key = _env_present("WANDB_API_KEY")
    wv_proj = _env_present("WEAVE_PROJECT")
    if wb_key and wv_proj:
        rows.append((
            "WEAVE",
            "WANDB_API_KEY + WEAVE_PROJECT · cloud trace",
            "CONNECTED", "live",
        ))
    elif wb_key or wv_proj:
        rows.append((
            "WEAVE",
            "WANDB_API_KEY + WEAVE_PROJECT · cloud trace (partial)",
            "PARTIAL", "warning",
        ))
    else:
        rows.append((
            "WEAVE",
            "WANDB_API_KEY + WEAVE_PROJECT · cloud trace",
            "OPTIONAL", "muted",
        ))

    # BLAND AI emergency call — optional, needs key + caregiver phone target.
    bl_key = _env_present("BLAND_API_KEY")
    cg_phone = _env_present("CAREGIVER_PHONE")
    if bl_key and cg_phone:
        rows.append((
            "BLAND",
            "BLAND_API_KEY + CAREGIVER_PHONE · emergency call",
            "CONNECTED", "live",
        ))
    elif bl_key or cg_phone:
        rows.append((
            "BLAND",
            "BLAND_API_KEY + CAREGIVER_PHONE · emergency call (partial)",
            "PARTIAL", "warning",
        ))
    else:
        rows.append((
            "BLAND",
            "BLAND_API_KEY + CAREGIVER_PHONE · emergency call",
            "OPTIONAL", "muted",
        ))

    # GOOGLE CALENDAR — optional, via local credentials.json (events.json falls back).
    cal = _calendar_credentials_present()
    rows.append((
        "CALENDAR",
        "credentials.json · Google Calendar OAuth (events.json fallback)",
        "CONNECTED" if cal else "OPTIONAL",
        "live" if cal else "muted",
    ))

    return rows


# ---------------------------------------------------------------------------
# Status-row HTML. Never relies on color alone: every row pairs an icon glyph
# + status WORD + colored pill (WCAG 1.4.1). Whole list wrapped aria-live so a
# screen reader announces re-checks.
# ---------------------------------------------------------------------------
def _status_pill(state_text: str, variant: str) -> str:
    """Map an internal variant to a hud.pill style + safe text."""
    if variant == "live":
        return hud.pill(state_text, "live")
    if variant == "danger":
        return hud.pill(state_text, "danger")
    if variant == "warning":
        return hud.pill(state_text, "warning")
    # muted/optional -> cyan pill, text says OPTIONAL so it's not just color.
    return hud.pill(state_text, "cyan")


def _status_glyph(variant: str) -> str:
    """A text glyph paired with every row so meaning survives without color."""
    return {
        "live": "●",      # connected
        "danger": "✕",    # required + missing
        "warning": "◐",   # partial
        "muted": "○",     # optional + absent
    }.get(variant, "○")


def _service_row_html(label: str, detail: str, state_text: str, variant: str) -> str:
    glyph = _status_glyph(variant)
    color = {
        "live": "var(--hud-success)", "danger": "var(--hud-danger)",
        "warning": "var(--hud-warning)", "muted": "var(--hud-muted)",
    }.get(variant, "var(--hud-cyan)")
    return (
        '<div style="display:flex;align-items:center;gap:10px;padding:9px 11px;margin:5px 0;'
        'background:var(--bg-inset);border:1px solid var(--border-subtle);'
        f'border-left:3px solid {color};border-radius:var(--r-sm);font-family:var(--font-mono)">'
        f'<span aria-hidden="true" style="color:{color};font-size:13px;width:14px;text-align:center">{glyph}</span>'
        f'<span style="font-size:12px;letter-spacing:1px;color:var(--hud-cyan);min-width:84px;'
        f'text-transform:uppercase">{_esc(label)}</span>'
        f'<span style="font-size:11px;letter-spacing:.4px;color:var(--text-secondary);flex:1">{_esc(detail)}</span>'
        f'<span style="margin-left:auto">{_status_pill(state_text, variant)}</span>'
        '</div>'
    )


def _render_services() -> None:
    rows = _service_rows()
    n_ok = sum(1 for *_, v in rows if v == "live")
    body_rows = "".join(
        _service_row_html(label, detail, state_text, variant)
        for (label, detail, state_text, variant) in rows
    )
    # aria-live so re-checks (on rerun) are announced to assistive tech.
    body = (
        '<div role="status" aria-live="polite" aria-atomic="true">'
        '<span class="vis-hidden">Service connectivity status. </span>'
        f'{body_rows}</div>'
    )
    summary_pill = hud.pill(f"{n_ok}/{len(rows)} ONLINE", "live" if n_ok else "warning")
    st.markdown(
        hud.panel("SERVICES", "API KEYS // CONNECTIVITY", body, summary_pill),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# CONFIG panel — runtime config. Camera index is a real bound control; the
# rest are read-only runtime facts. No secrets shown.
# ---------------------------------------------------------------------------
def _runtime_versions() -> tuple[str, str]:
    py = platform.python_version()
    try:
        import streamlit as _st  # already imported; re-ref for version only.
        sl = getattr(_st, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        sl = "unknown"
    return py, sl


def _render_config() -> None:
    # WEAVE project: show that it's SET / NOT SET, never the value itself.
    weave_set = _env_present("WEAVE_PROJECT")
    weave_state = "SET" if weave_set else "NOT SET"
    weave_pill = hud.pill(weave_state, "cyan" if weave_set else "warning")
    py_ver, sl_ver = _runtime_versions()
    cam = int(st.session_state.get(_K_CAMERA, 1))

    info_html = (
        '<div style="display:flex;flex-direction:column;gap:8px;font-family:var(--font-mono);'
        'font-size:11px;letter-spacing:.5px">'
        '<div style="display:flex;align-items:center;justify-content:space-between">'
        '<span style="color:var(--text-secondary)">WEAVE PROJECT</span>'
        f'<span>{weave_pill}</span></div>'
        '<div style="display:flex;align-items:center;justify-content:space-between">'
        '<span style="color:var(--text-secondary)">CAMERA INDEX</span>'
        f'<span style="color:var(--hud-cyan)">{_esc(cam)}</span></div>'
        '<div style="display:flex;align-items:center;justify-content:space-between">'
        '<span style="color:var(--text-secondary)">PYTHON</span>'
        f'<span style="color:var(--hud-cyan)">{_esc(py_ver)}</span></div>'
        '<div style="display:flex;align-items:center;justify-content:space-between">'
        '<span style="color:var(--text-secondary)">STREAMLIT</span>'
        f'<span style="color:var(--hud-cyan)">{_esc(sl_ver)}</span></div>'
        '</div>'
    )
    st.markdown(
        hud.panel("CONFIG", "RUNTIME // PARAMETERS", info_html),
        unsafe_allow_html=True,
    )

    # Real bound control: camera index, persisted to session_state.
    st.number_input(
        "Camera index (YOLO perception source)",
        min_value=0,
        max_value=8,
        step=1,
        key=_K_CAMERA,
        help="Which local camera device the live YOLOv8 feed should open. "
             "0 is usually the built-in webcam.",
    )


# ---------------------------------------------------------------------------
# EMERGENCY CONTROL — the ARM REAL CALLS toggle. Default OFF. ON = a REAL
# outbound Bland AI phone call may fire when the safety pipeline escalates.
# ---------------------------------------------------------------------------
def _render_emergency() -> None:
    armed = bool(st.session_state.get(_K_ARM, False))

    # Danger warning is text + icon + color (never color alone). aria-live so the
    # state flip is announced — this is a safety-critical control (WCAG 4.1.3).
    if armed:
        warn_html = (
            '<div role="alert" aria-live="assertive" aria-atomic="true" '
            'style="display:flex;gap:10px;align-items:flex-start;padding:11px 12px;'
            'background:var(--hud-danger-12);border:1px solid var(--hud-danger);'
            'border-radius:var(--r-sm);font-family:var(--font-mono)">'
            '<span aria-hidden="true" style="color:var(--hud-danger);font-size:15px">⚠</span>'
            '<div><div style="color:var(--hud-danger);font-size:12px;letter-spacing:1px;'
            'font-weight:700">LIVE CALLS ARMED</div>'
            '<div style="color:var(--text-primary);font-size:11px;letter-spacing:.4px;margin-top:4px">'
            'A real outbound Bland AI phone call to the caregiver MAY be placed when the '
            'safety pipeline escalates to an emergency. Turn OFF to keep all calls simulated.'
            '</div></div></div>'
        )
        status_pill = hud.pill("LIVE CALLS ARMED", "danger")
    else:
        warn_html = (
            '<div role="status" aria-live="polite" aria-atomic="true" '
            'style="display:flex;gap:10px;align-items:flex-start;padding:11px 12px;'
            'background:var(--hud-cyan-08);border:1px solid var(--hud-cyan-40);'
            'border-radius:var(--r-sm);font-family:var(--font-mono)">'
            '<span aria-hidden="true" style="color:var(--hud-cyan);font-size:15px">●</span>'
            '<div><div style="color:var(--hud-cyan);font-size:12px;letter-spacing:1px;'
            'font-weight:700">CALLS SAFE // SIMULATED</div>'
            '<div style="color:var(--text-secondary);font-size:11px;letter-spacing:.4px;margin-top:4px">'
            'Emergency dispatch is simulated only — no real phone call is placed. '
            'Arming below enables a real Bland AI call to the caregiver on escalation.'
            '</div></div></div>'
        )
        status_pill = hud.pill("CALLS SAFE / SIMULATED", "cyan")

    st.markdown(
        hud.panel("EMERGENCY CONTROL", "BLAND AI // OUTBOUND CALL GATE", warn_html, status_pill),
        unsafe_allow_html=True,
    )

    # The bound toggle. Default OFF; writes st.session_state["arm_real_calls"].
    st.toggle(
        "ARM REAL CALLS",
        key=_K_ARM,
        help="OFF (default): the emergency path is ALWAYS simulated — no live call. "
             "ON: a real Bland AI phone call to the caregiver may be placed when the "
             "pipeline escalates to an emergency.",
    )
    st.caption(
        "Safety gate. Leave OFF for demos. Requires BLAND_API_KEY + CAREGIVER_PHONE "
        "to actually place a call."
    )


# ===========================================================================
# PUBLIC ENTRYPOINT
# ===========================================================================
def render() -> None:
    """Render the SYSTEM tab body into the current Streamlit container.

    Assumes hud.inject_theme() already ran (in feat_ui.py). Reads only secret
    PRESENCE, never values. Every backend/IO touch is guarded so a missing key
    or unreadable file degrades gracefully instead of crashing the tab.
    """
    _init_state()

    # TitleBlock
    st.markdown(hud.hud_sep("SYSTEM // STATUS"), unsafe_allow_html=True)
    st.markdown(
        hud.reactor_header(
            "SYSTEM CONTROL",
            "SERVICES · CONFIG · EMERGENCY GATE",
            hud.pill("OPERATOR", "cyan"),
        ),
        unsafe_allow_html=True,
    )

    # SERVICES — full width (most important: what's wired vs missing).
    try:
        _render_services()
    except Exception as exc:  # noqa: BLE001 — a status surface must never crash.
        st.warning(f"Service status unavailable: {exc}")

    # CONFIG + EMERGENCY side by side on wide layouts.
    col_cfg, col_emrg = st.columns(2)
    with col_cfg:
        st.markdown(hud.hud_sep("RUNTIME // CONFIG"), unsafe_allow_html=True)
        try:
            _render_config()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Config panel unavailable: {exc}")
    with col_emrg:
        st.markdown(hud.hud_sep("EMERGENCY // GATE"), unsafe_allow_html=True)
        try:
            _render_emergency()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Emergency control unavailable: {exc}")


# Allow `streamlit run feat_system.py` for isolated preview of this tab.
if __name__ == "__main__":  # pragma: no cover
    hud.inject_theme()
    st.title("SYSTEM // PREVIEW")
    render()
