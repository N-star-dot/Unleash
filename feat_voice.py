"""
Unleash — FEAT // Voice & Conversation tab
==========================================

The conversational interface, rendered as a tactical-HUD storyboard:

    MIC PANEL   →  st.audio_input → VoiceAgent().transcribe_bytes → text
    ROUTER      →  Assistant().classify(text) → highlighted intent chips
    TRANSCRIPT  →  recognized text + a short rolling history (session_state)
    REPLY //    →  Assistant().handle(text) shown on-screen AND
    SPOKEN         VoiceAgent().synthesize(reply) → st.audio(autoplay)

Accessibility is the product: every control is labelled, the spoken reply is
always shown as text (never audio-only), the router never relies on colour
alone (chip carries an icon glyph + text), and live agent output is wrapped in
aria-live regions. Every backend call is guarded — a missing mic, a missing
GEMINI_API_KEY, or a network blip degrades to a warning + the text fallback,
never a crash.

Public surface: ``render() -> None`` renders this tab's body into the current
Streamlit container. Assume ``hud_theme.inject_theme()`` already ran upstream.
"""

from __future__ import annotations

import html

import streamlit as st

import hud_theme as hud
from assistant import Assistant, INTENTS
from voice_agent import VoiceAgent


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


# Per-intent glyph so the router NEVER relies on colour alone (WCAG 1.4.1):
# every chip pairs an icon + the literal intent word.
_INTENT_GLYPH = {
    "location": "◎",
    "directions": "➤",
    "calendar": "▦",
    "general": "✦",
}
_INTENT_HELP = {
    "location": "Where you are / your surroundings",
    "directions": "How to get somewhere",
    "calendar": "Your schedule & events",
    "general": "Anything else (chat)",
}


# ---------------------------------------------------------------------------
# State — namespaced so reruns persist; initialised so the tab renders cold.
# ---------------------------------------------------------------------------
def _init_state() -> None:
    if "feat_assistant" not in st.session_state:
        try:
            st.session_state.feat_assistant = Assistant()
        except Exception as exc:  # noqa: BLE001 — never crash the tab on import-time deps
            st.session_state.feat_assistant = None
            st.session_state.feat_assistant_error = str(exc)
    st.session_state.setdefault("feat_voice_text", "")     # last recognized utterance
    st.session_state.setdefault("feat_voice_intent", "")   # last classified intent
    st.session_state.setdefault("feat_voice_reply", "")    # last spoken-ready reply
    st.session_state.setdefault("feat_voice_history", [])  # [{role,text,intent}]
    st.session_state.setdefault("feat_voice_audio", None)  # (bytes, mime) for autoplay
    st.session_state.setdefault("feat_voice_last_mic", None)  # de-dupe mic reruns


def _assistant() -> Assistant | None:
    return st.session_state.get("feat_assistant")


# ---------------------------------------------------------------------------
# Core turn: classify → handle → synthesize, all guarded.
# Updates session_state so the storyboard panels re-render with live values.
# ---------------------------------------------------------------------------
def _process_turn(text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    assistant = _assistant()
    if assistant is None:
        st.warning("Assistant unavailable — check that backend agents import cleanly.")
        return

    # 1) Route — classify intent (Gemini if keyed, else keyword fallback).
    intent = "general"
    try:
        intent = assistant.classify(text) or "general"
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Router unavailable (defaulting to general): {exc}")

    # 2) Answer — spoken-ready reply text.
    reply = ""
    try:
        reply = assistant.handle(text)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Assistant could not answer (check GEMINI_API_KEY): {exc}")
        reply = "I'm here with you, but I couldn't reach my reasoning service just now."

    # 3) Speak — synthesize audio (always still shown as text below).
    audio = None
    try:
        data, mime = assistant.voice.synthesize(reply)
        if data:
            audio = (data, mime)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Voice synthesis unavailable (text shown instead): {exc}")

    # 4) Persist for the storyboard panels + rolling history.
    st.session_state.feat_voice_text = text
    st.session_state.feat_voice_intent = intent
    st.session_state.feat_voice_reply = reply
    st.session_state.feat_voice_audio = audio
    hist = st.session_state.feat_voice_history
    hist.append({"role": "you", "text": text, "intent": intent})
    hist.append({"role": "dog", "text": reply, "intent": intent})
    st.session_state.feat_voice_history = hist[-12:]  # keep it short


# ---------------------------------------------------------------------------
# Storyboard panels
# ---------------------------------------------------------------------------
def _render_mic_panel() -> None:
    """Glowing MIC PANEL: st.audio_input → transcribe_bytes → text."""
    st.markdown(hud.hud_sep("MIC // CAPTURE"), unsafe_allow_html=True)

    # Glowing reactor header to give the panel its "voice" HUD identity.
    st.markdown(
        hud.reactor_header(
            "VOICE LINK",
            "TAP TO SPEAK · GEMINI STT",
            hud.pill("LISTENING", "live"),
        ),
        unsafe_allow_html=True,
    )

    mic_audio = None
    if hasattr(st, "audio_input"):
        try:
            mic_audio = st.audio_input("Tap to speak")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Microphone widget unavailable: {exc}")
    else:
        st.info("Microphone capture needs Streamlit ≥ 1.31 — use the text input below.")

    if mic_audio is not None:
        # De-dupe: Streamlit replays the same recording across reruns.
        try:
            raw = mic_audio.getvalue()
        except Exception:  # noqa: BLE001
            raw = None
        if raw and raw != st.session_state.get("feat_voice_last_mic"):
            st.session_state.feat_voice_last_mic = raw
            assistant = _assistant()
            text = ""
            if assistant is not None:
                with st.spinner("Transcribing…"):
                    try:
                        text = assistant.voice.transcribe_bytes(raw, suffix=".wav")
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Transcription failed: {exc}")
            if text:
                _process_turn(text)
                st.rerun()
            else:
                st.warning(
                    "Couldn't transcribe that — check GEMINI_API_KEY, "
                    "or type your question below."
                )


def _render_router_panel() -> None:
    """ROUTER panel: Assistant().classify result as highlighted chips."""
    st.markdown(hud.hud_sep("ROUTER // INTENT"), unsafe_allow_html=True)
    active = st.session_state.get("feat_voice_intent", "")

    # All four possible intents as chips; the matched one is highlighted as
    # success (and still carries its glyph + word, so colour is never the
    # only signal).
    items = []
    for intent in INTENTS:
        glyph = _INTENT_GLYPH.get(intent, "✦")
        is_active = intent == active
        label = f"{glyph} {intent.upper()}"
        if is_active:
            label = f"{glyph} {intent.upper()} ◄ ROUTED"
        items.append({"text": label, "variant": "success" if is_active else None})
    chips_html = hud.chips(items)

    routed = (
        f'<div role="status" aria-live="polite" aria-atomic="true">'
        f'<span class="vis-hidden">Routed to intent: </span>'
        f'{chips_html}'
        f'<div style="font-family:var(--font-mono);font-size:11px;letter-spacing:1px;'
        f'color:var(--hud-label);margin-top:8px">'
        + (
            f'→ {_esc(active.upper())} · {_esc(_INTENT_HELP.get(active, ""))}'
            if active
            else "AWAITING UTTERANCE — speak or type to route."
        )
        + "</div></div>"
    )
    pill = hud.pill(active.upper(), "cyan") if active else hud.pill("IDLE", "cyan")
    st.markdown(
        hud.panel("AGENT ROUTER", "ASSISTANT // CLASSIFY", routed, pill),
        unsafe_allow_html=True,
    )


def _render_transcript_panel() -> None:
    """TRANSCRIPT column: recognized text + short rolling history."""
    st.markdown(hud.hud_sep("TRANSCRIPT // LOG"), unsafe_allow_html=True)
    text = st.session_state.get("feat_voice_text", "")

    recognized = (
        f'<div role="status" aria-live="polite" aria-atomic="true">'
        f'<span class="vis-hidden">Recognized speech: </span>'
        f'<div style="font-size:14px;color:var(--text-primary);line-height:1.4">'
        + (f"&ldquo;{_esc(text)}&rdquo;" if text else "—")
        + "</div></div>"
    )
    st.markdown(
        hud.panel(
            "RECOGNIZED",
            "LATEST UTTERANCE",
            recognized,
            hud.pill("HEARD" if text else "EMPTY", "live" if text else "cyan"),
        ),
        unsafe_allow_html=True,
    )

    history = st.session_state.get("feat_voice_history", [])
    if history:
        rows = "".join(
            hud.claim_row(
                "YOU" if turn["role"] == "you" else "UNLEASH",
                _esc(turn["text"]),
                (turn.get("intent", "") or "").upper(),
                "cyan" if turn["role"] == "you" else "memory",
            )
            for turn in reversed(history)
        )
        body = f'<div role="log" aria-label="Conversation history">{rows}</div>'
        st.markdown(
            hud.panel("HISTORY", f"{len(history)} LINE(S)", body),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            hud.panel(
                "HISTORY",
                "NO TURNS YET",
                '<span style="color:var(--hud-muted)">Your conversation will appear here.</span>',
            ),
            unsafe_allow_html=True,
        )


def _render_reply_panel() -> None:
    """REPLY // SPOKEN column: handle() text on-screen + synthesized audio."""
    st.markdown(hud.hud_sep("REPLY // SPOKEN"), unsafe_allow_html=True)
    reply = st.session_state.get("feat_voice_reply", "")

    body = (
        '<div role="status" aria-live="assertive" aria-atomic="true">'
        '<span class="vis-hidden">Unleash replies: </span>'
        '<div style="font-size:15px;color:var(--hud-text);font-style:italic;line-height:1.45">'
        + (f"&ldquo;{_esc(reply)}&rdquo;" if reply else "Ask me anything — I'll answer out loud.")
        + "</div></div>"
    )
    st.markdown(
        hud.panel(
            "SPOKEN REPLY",
            "ASSISTANT // HANDLE",
            body,
            hud.pill("AUDIO" if reply else "READY", "live" if reply else "cyan"),
        ),
        unsafe_allow_html=True,
    )

    # Persistent, explicit control over autoplay (WCAG 1.4.2). Default OFF so
    # screen-reader users aren't forced into speech-over-speech with the aria-live
    # "Unleash replies:" announcement above. Native audio controls always allow
    # manual playback regardless of this setting.
    speak_aloud = st.toggle(
        "Speak replies aloud",
        key="feat_voice_speak_aloud",
        help="When ON, synthesized replies play automatically. OFF: use the audio "
             "player's controls to listen (avoids talking over screen readers).",
    )

    # Play the synthesized audio (text above is the source of truth).
    audio = st.session_state.get("feat_voice_audio")
    if audio:
        data, mime = audio
        try:
            st.audio(data, format=mime, autoplay=bool(speak_aloud))
            st.caption("Transcript shown above (audio is never the only output).")
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Audio playback unavailable (reply shown above): {exc}")
    elif reply:
        st.caption("Reply shown above. Voice synthesis is unavailable on this host.")


# ---------------------------------------------------------------------------
# Text input fallback — works with no mic and no API key.
# ---------------------------------------------------------------------------
def _render_text_fallback() -> None:
    st.markdown(hud.hud_sep("TEXT // FALLBACK"), unsafe_allow_html=True)
    with st.form("feat_voice_form", clear_on_submit=True):
        typed = st.text_input(
            "Type your question",
            placeholder="e.g. Where am I right now?",
            label_visibility="visible",
        )
        c1, c2 = st.columns([1, 3])
        submitted = c1.form_submit_button("Send", use_container_width=True)
        if submitted and typed.strip():
            _process_turn(typed)

    st.caption("Quick demo questions")
    d1, d2, d3 = st.columns(3)
    if d1.button("Where am I?", use_container_width=True, key="feat_demo_where"):
        _process_turn("where am I right now?")
        st.rerun()
    if d2.button("What's at 4 pm?", use_container_width=True, key="feat_demo_4pm"):
        _process_turn("what do I have at 4 pm?")
        st.rerun()
    if d3.button("My schedule today", use_container_width=True, key="feat_demo_sched"):
        _process_turn("what's on my schedule today?")
        st.rerun()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def render() -> None:
    """Render the VOICE tab body into the current Streamlit container."""
    _init_state()

    # Title block. The authoritative <h1> is the chrome's "FEAT // {active}" in
    # feat_ui.py; this tab title is an <h2> so the page has one h1 (WCAG 1.3.1).
    st.markdown(
        '<h2 style="margin-bottom:2px">VOICE &amp; CONVERSATION</h2>'
        '<p style="font-family:var(--font-mono);font-size:11px;letter-spacing:2px;'
        'color:var(--hud-muted);margin-top:0">'
        'FEAT // SPEAK → ROUTE → ANSWER → SPOKEN REPLY</p>',
        unsafe_allow_html=True,
    )

    if _assistant() is None:
        st.warning(
            "Voice backend failed to initialise: "
            f"{st.session_state.get('feat_assistant_error', 'unknown error')}. "
            "The text fallback below still routes intents and shows replies."
        )

    # Top row: MIC (left) + ROUTER (right).
    mic_col, router_col = st.columns([3, 2])
    with mic_col:
        _render_mic_panel()
    with router_col:
        _render_router_panel()

    # Bottom row: TRANSCRIPT (left) + REPLY // SPOKEN (right).
    transcript_col, reply_col = st.columns(2)
    with transcript_col:
        _render_transcript_panel()
    with reply_col:
        _render_reply_panel()

    # Text fallback — always available, works with no mic / no key.
    _render_text_fallback()

    # Working console — the shipped voice_ui panel, kept as a live reference.
    st.markdown(hud.hud_sep("WORKING CONSOLE"), unsafe_allow_html=True)
    with st.expander("Open the full voice console (mic · GPS · calendar)", expanded=False):
        try:
            from voice_ui import render_voice_panel
            render_voice_panel()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Voice console unavailable: {exc}")


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Unleash — Voice", page_icon="🐕")
    hud.inject_theme()
    render()
