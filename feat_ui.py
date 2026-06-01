"""feat_ui.py — Unleash FEATURE STORYBOARD shell.

Thin compositor for the five FEAT tab modules. Owns nothing but the chrome:
a reactor-branded sidebar nav and a title bar; every tab's content lives in its
own module (each exposes ``def render() -> None``).

    SAFETY → MEMORY → VOICE → ASSISTANT → SYSTEM

Run with:  streamlit run feat_ui.py

inject_theme() is called FIRST so every helper string and Streamlit widget
renders in the cyan/teal tactical HUD. Each module's render() is dispatched
inside try/except so a single tab failing degrades to an st.error instead of a
blank page.
"""

from __future__ import annotations

import streamlit as st

import hud_theme as hud
import feat_safety
import feat_memory
import feat_voice
import feat_assistant
import feat_system

# ---------------------------------------------------------------------------
# Page config + theme — MUST run before any other Streamlit output.
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Unleash — Features", page_icon="🐕")
hud.inject_theme()

# Tab registry: label -> module exposing render(). Order is the nav order.
_TABS: dict[str, object] = {
    "SAFETY": feat_safety,
    "MEMORY": feat_memory,
    "VOICE": feat_voice,
    "ASSISTANT": feat_assistant,
    "SYSTEM": feat_system,
}


# ---------------------------------------------------------------------------
# Sidebar — reactor brand + radio nav + running status. Selection persists in
# st.session_state via the radio's key.
# ---------------------------------------------------------------------------
def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            hud.reactor_header("UNLEASH", "FEATURE STORYBOARD"),
            unsafe_allow_html=True,
        )
        st.markdown(hud.hud_sep("NAVIGATION"), unsafe_allow_html=True)

        active = st.radio(
            "Feature tab",
            options=list(_TABS.keys()),
            key="feat_ui_tab",
            label_visibility="collapsed",
        )

        st.markdown(
            '<div style="margin-top:1.1rem">'
            + hud.pill("RUNNING // 6 AGENTS", "live")
            + "</div>",
            unsafe_allow_html=True,
        )
    return active


# ---------------------------------------------------------------------------
# Main chrome — title bar (active tab + LIVE pill), then dispatch render().
# ---------------------------------------------------------------------------
def _render_chrome(active: str) -> None:
    bar_l, bar_r = st.columns([4, 1])
    with bar_l:
        st.markdown(
            f'<h1 style="margin-bottom:2px">FEAT // {active}</h1>'
            '<p style="font-family:var(--font-mono);font-size:11px;letter-spacing:2px;'
            'color:var(--hud-muted);margin-top:0">UNLEASH · FEATURE STORYBOARD</p>',
            unsafe_allow_html=True,
        )
    with bar_r:
        st.markdown(
            '<div style="text-align:right;padding-top:14px">'
            + hud.pill("LIVE", "live")
            + "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        '<hr style="margin:.3rem 0 1rem;border-top:1px solid var(--hud-cyan-20)">',
        unsafe_allow_html=True,
    )


def _dispatch(active: str) -> None:
    # Release the SAFETY tab's webcam when navigating away — it keeps an OS camera
    # handle open across reruns (for a flicker-free live feed), so free it here.
    if active != "SAFETY" and hasattr(feat_safety, "on_deactivate"):
        try:
            feat_safety.on_deactivate()
        except Exception:
            pass
    module = _TABS.get(active)
    if module is None:  # unreachable via the radio, but never blank.
        st.error(f"Unknown tab: {active}")
        return
    try:
        module.render()
    except Exception as exc:  # noqa: BLE001 — a failing tab must never blank the app.
        st.error(f"The {active} tab failed to render: {type(exc).__name__}: {exc}")


def main() -> None:
    active = _render_sidebar()
    _render_chrome(active)
    _dispatch(active)


if __name__ == "__main__":
    main()
