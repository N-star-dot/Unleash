"""feat_assistant.py — FEAT // Assistant Location+Calendar tab.

Renders the "ask it anything practical" surface for the Unleash service-dog HUD:
a TitleBlock, a metrics row (current place + today's event count), a LOCATION
panel (where_am_i + straight-line/estimated directions), a CALENDAR Q&A panel
(CalendarAgent.answer), and a GENERAL CHAT FALLBACK panel (Assistant.handle /
general Gemini chat).

Public contract:  render() -> None  — draws this tab's body into the current
Streamlit container. inject_theme() is assumed to have already run in feat_ui.py.

Honest status (see FEAT_TABS_PRD.md): directions are straight-line / estimated
walk time, NOT turn-by-turn routing. Every backend / network / Gemini call is
guarded so a missing GEMINI_API_KEY, no network, or an offline routing service
degrades gracefully instead of crashing. Accessibility is the product: every
control is labelled, all agent output is shown on-screen as text (never
audio-only), severity is paired with text (never color alone), and live agent
output is wrapped in aria-live regions.
"""

from __future__ import annotations

import html

import streamlit as st

import hud_theme as hud


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _esc(value: object) -> str:
    """HTML-escape any value for safe interpolation into our HUD markup."""
    return html.escape(str(value), quote=True)


def _init_state() -> None:
    """Seed namespaced defaults so the tab renders before any interaction."""
    st.session_state.setdefault("feat_place", "")          # cached where_am_i() text
    st.session_state.setdefault("feat_directions", "")     # last directions reply
    st.session_state.setdefault("feat_dest", "")           # last directions destination
    st.session_state.setdefault("feat_calendar_a", "")     # last calendar answer
    st.session_state.setdefault("feat_calendar_q", "")     # last calendar question
    st.session_state.setdefault("feat_chat_a", "")         # last general-chat answer
    st.session_state.setdefault("feat_chat_q", "")         # last general-chat question
    st.session_state.setdefault("feat_event_count", None)  # today's event count


# ---- cached agents (one instance each, reused across reruns) ---------------
def _location_agent():
    """Return a cached LocationAgent, or None if it can't be constructed."""
    if "feat_location_agent" not in st.session_state:
        try:
            from location_agent import LocationAgent
            st.session_state.feat_location_agent = LocationAgent()
        except Exception as exc:  # noqa: BLE001 — never crash the tab
            st.session_state.feat_location_agent = None
            st.warning(f"Location agent unavailable: {exc}")
    return st.session_state.feat_location_agent


def _calendar_agent():
    """Return a cached CalendarAgent, or None if it can't be constructed."""
    if "feat_calendar_agent" not in st.session_state:
        try:
            from calendar_agent import CalendarAgent
            st.session_state.feat_calendar_agent = CalendarAgent()
        except Exception as exc:  # noqa: BLE001
            st.session_state.feat_calendar_agent = None
            st.warning(f"Calendar agent unavailable: {exc}")
    return st.session_state.feat_calendar_agent


def _assistant():
    """Return a cached Assistant (general chat router), or None on failure."""
    if "feat_assistant" not in st.session_state:
        try:
            from assistant import Assistant
            st.session_state.feat_assistant = Assistant()
        except Exception as exc:  # noqa: BLE001
            st.session_state.feat_assistant = None
            st.warning(f"Assistant unavailable: {exc}")
    return st.session_state.feat_assistant


# ---- guarded backend wrappers ---------------------------------------------
def _safe_where_am_i() -> str:
    """LocationAgent.where_am_i() with a network/IP/geocode guard."""
    agent = _location_agent()
    if agent is None:
        return "Location service is offline right now."
    try:
        return agent.where_am_i() or "I can't tell where you are just now."
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Couldn't resolve your location: {exc}")
        return "I can't reach the location service right now."


def _safe_directions(dest: str) -> str:
    """LocationAgent.directions(dest) with a routing-service guard."""
    agent = _location_agent()
    if agent is None:
        return "Location service is offline right now."
    try:
        return agent.directions(dest) or "Where would you like to go?"
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Couldn't compute directions: {exc}")
        return "I couldn't reach the routing service right now."


def _safe_calendar_answer(q: str) -> str:
    """CalendarAgent.answer(q) with a backend guard."""
    agent = _calendar_agent()
    if agent is None:
        return "Calendar service is offline right now."
    try:
        return agent.answer(q) or "I couldn't read your schedule just now."
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Couldn't read your schedule: {exc}")
        return "I couldn't read your schedule just now."


def _safe_event_count() -> int:
    """Count today's events for the metrics row. 0 on any failure."""
    agent = _calendar_agent()
    if agent is None:
        return 0
    try:
        return len(agent.events_today())
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Couldn't load today's events: {exc}")
        return 0


def _safe_general_chat(q: str) -> str:
    """Assistant.handle(q) — general Gemini chat (degrades to a canned reply)."""
    agent = _assistant()
    if agent is None:
        return "My reasoning service is offline right now."
    try:
        return agent.handle(q) or "I'm here with you."
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Chat unavailable (check GEMINI_API_KEY): {exc}")
        return "I'm here with you, but I couldn't reach my reasoning service just now."


def _short_place(text: str) -> str:
    """Trim a spoken 'You're near …' sentence to a compact metric value."""
    t = (text or "").strip()
    for lead in ("You're near ", "You're at ", "You are near ", "You are at "):
        if t.startswith(lead):
            t = t[len(lead):]
            break
    t = t.rstrip(".")
    if " — " in t:                       # drop the "— near …" tail of preset answers
        t = t.split(" — ")[0]
    if "," in t:                         # keep just the first locality component
        t = t.split(",")[0]
    t = t.strip()
    return (t[:22] + "…") if len(t) > 23 else (t or "UNKNOWN")


def _live_answer_html(label_sr: str, text: str, ready: bool) -> str:
    """Agent reply in an aria-live region so assistive tech announces updates.

    `ready` flags a real answer vs. an idle/placeholder state; the state is
    conveyed as a text pill (not color alone) for WCAG 1.4.1.
    """
    pill = hud.pill("ANSWER", "live") if ready else hud.pill("IDLE", "cyan")
    body = (
        f'<div role="status" aria-live="polite" aria-atomic="true" '
        f'style="display:flex;flex-direction:column;gap:8px">'
        f'<span class="vis-hidden">{_esc(label_sr)}</span>'
        f'<div style="font-size:14px;color:var(--text-primary);line-height:1.5">'
        f'{_esc(text)}</div>'
        f'<div>{pill}</div></div>'
    )
    return body


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _render_title_block() -> None:
    """TitleBlock + a status pill, matching the COMMAND CENTER header rhythm."""
    head_l, head_r = st.columns([4, 1])
    with head_l:
        # The authoritative <h1> is the chrome's "FEAT // {active}" in feat_ui.py;
        # this tab title is an <h2> so the page has one h1 (WCAG 1.3.1).
        st.markdown(
            '<h2 style="margin-bottom:2px">ASSISTANT</h2>'
            '<p style="font-family:var(--font-mono);font-size:11px;letter-spacing:2px;'
            'color:var(--hud-muted);margin-top:0">'
            'LOCATION + CALENDAR // ASK ANYTHING PRACTICAL</p>',
            unsafe_allow_html=True,
        )
    with head_r:
        st.markdown(
            f'<div style="text-align:right;padding-top:14px">'
            f'{hud.pill("ON CALL", "live")}</div>',
            unsafe_allow_html=True,
        )


def _render_metrics() -> None:
    """Second / metrics row: current place + today's event count + mode."""
    place_text = st.session_state.feat_place or _safe_where_am_i()
    st.session_state.feat_place = place_text
    place_short = _short_place(place_text)

    count = st.session_state.feat_event_count
    if count is None:
        count = _safe_event_count()
        st.session_state.feat_event_count = count

    import os
    gemini_on = bool(os.environ.get("GEMINI_API_KEY"))

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            hud.stat_card(_esc(place_short), "CURRENT PLACE", "GPS / IP", "cyan"),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            hud.stat_card(str(count), "EVENTS TODAY", "CAL",
                          "success" if count else "cyan"),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            hud.stat_card(
                "LIVE" if gemini_on else "LOCAL",
                "CHAT MODE",
                "GEMINI" if gemini_on else "OFFLINE",
                "success" if gemini_on else "warning",
            ),
            unsafe_allow_html=True,
        )


def _render_location_panel() -> None:
    """LOCATION panel: where_am_i() + a 'directions to' input (estimated)."""
    st.markdown(hud.hud_sep("LOCATION // PLACE"), unsafe_allow_html=True)

    place_text = st.session_state.feat_place or _safe_where_am_i()
    st.session_state.feat_place = place_text

    if st.button("Refresh my location", key="feat_loc_refresh",
                 help="Re-resolve your position via GPS / IP and reverse-geocode it.",
                 use_container_width=True):
        st.session_state.feat_place = _safe_where_am_i()
        place_text = st.session_state.feat_place

    st.markdown(
        hud.panel(
            "WHERE AM I",
            "GPS / IP → OPENSTREETMAP",
            _live_answer_html("Your current location:", place_text, ready=True),
            hud.pill("RESOLVED", "live"),
        ),
        unsafe_allow_html=True,
    )

    # Directions — honestly labelled as straight-line / estimated, not turn-by-turn.
    with st.form("feat_directions_form", clear_on_submit=False):
        dest = st.text_input(
            "Directions to",
            value=st.session_state.feat_dest,
            placeholder="e.g. Harvard Square",
            help="Estimated straight-line distance + walk time — not turn-by-turn navigation.",
        )
        submitted = st.form_submit_button("Get directions", use_container_width=True)
    if submitted and dest.strip():
        st.session_state.feat_dest = dest.strip()
        st.session_state.feat_directions = _safe_directions(dest.strip())

    if st.session_state.feat_directions:
        body = _live_answer_html(
            "Directions:", st.session_state.feat_directions, ready=True
        )
        body += (
            '<div style="font-family:var(--font-mono);font-size:10px;'
            'letter-spacing:1px;color:var(--hud-label);margin-top:8px">'
            '⚠ ESTIMATE · STRAIGHT-LINE DISTANCE + WALK TIME · NOT TURN-BY-TURN</div>'
        )
        st.markdown(
            hud.panel("DIRECTIONS", f"TO {_esc(st.session_state.feat_dest).upper()}",
                      body, hud.pill("ESTIMATED", "warning")),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            hud.panel(
                "DIRECTIONS", "AWAITING DESTINATION",
                '<span style="color:var(--hud-muted)">Enter a place above for an '
                'estimated distance and walk time.</span>',
                hud.pill("IDLE", "cyan"),
            ),
            unsafe_allow_html=True,
        )


def _render_calendar_panel() -> None:
    """CALENDAR Q&A panel: ask about your schedule → CalendarAgent.answer(q)."""
    st.markdown(hud.hud_sep("CALENDAR // SCHEDULE Q&A"), unsafe_allow_html=True)

    with st.form("feat_calendar_form", clear_on_submit=False):
        q = st.text_input(
            "Ask about your schedule",
            value=st.session_state.feat_calendar_q,
            placeholder="e.g. what's at 4 pm?",
            help="Asks the calendar agent about your day (e.g. \"what's at 4 pm?\").",
        )
        submitted = st.form_submit_button("Ask the calendar", use_container_width=True)
    if submitted and q.strip():
        st.session_state.feat_calendar_q = q.strip()
        st.session_state.feat_calendar_a = _safe_calendar_answer(q.strip())

    ready = bool(st.session_state.feat_calendar_a)
    answer = st.session_state.feat_calendar_a or (
        "Ask me about your day — for example, \"what's at 4 pm?\" or "
        "\"what's on my schedule today?\""
    )
    st.markdown(
        hud.panel(
            "CALENDAR Q&A",
            "GOOGLE CALENDAR → EVENTS.JSON FALLBACK",
            _live_answer_html("Calendar answer:", answer, ready=ready),
            hud.pill("ANSWER", "live") if ready else hud.pill("READY", "cyan"),
        ),
        unsafe_allow_html=True,
    )


def _render_chat_panel() -> None:
    """GENERAL CHAT FALLBACK panel: free-text → Assistant.handle(q) (Gemini)."""
    st.markdown(hud.hud_sep("GENERAL CHAT // FALLBACK"), unsafe_allow_html=True)

    with st.form("feat_chat_form", clear_on_submit=False):
        q = st.text_input(
            "Ask anything",
            value=st.session_state.feat_chat_q,
            placeholder="e.g. how should I cross a busy street?",
            help="Routes to the assistant — location, calendar, or general Gemini chat.",
        )
        submitted = st.form_submit_button("Ask the assistant", use_container_width=True)
    if submitted and q.strip():
        st.session_state.feat_chat_q = q.strip()
        st.session_state.feat_chat_a = _safe_general_chat(q.strip())

    ready = bool(st.session_state.feat_chat_a)
    answer = st.session_state.feat_chat_a or (
        "I can tell you where you are, what's on your schedule, or just talk. "
        "Ask me anything."
    )
    st.markdown(
        hud.panel(
            "GENERAL CHAT",
            "ASSISTANT ROUTER // GEMINI",
            _live_answer_html("Assistant reply:", answer, ready=ready),
            hud.pill("REPLY", "live") if ready else hud.pill("READY", "cyan"),
        ),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def render() -> None:
    """Render the ASSISTANT tab body into the current Streamlit container."""
    _init_state()
    _render_title_block()
    _render_metrics()

    left, right = st.columns(2)
    with left:
        _render_location_panel()
    with right:
        _render_calendar_panel()

    _render_chat_panel()


if __name__ == "__main__":  # pragma: no cover — standalone smoke test
    hud.inject_theme()
    render()

# touched 2026-06-03
