"""
Unleash — Streamlit Voice Assistant page
========================================

Drop-in Streamlit UI for the conversational assistant. Two ways to use it:

  A) Standalone:        streamlit run voice_ui.py
  B) Inside the main app: from voice_ui import render_voice_panel
                          render_voice_panel()   # call from a tab/page in ui.py

Features
  * 🎤 Browser microphone -> Gemini transcription -> answer -> spoken reply (st.audio)
  * ⌨️  Text box fallback (works with no mic and no API key)
  * 📍 Real browser GPS (HTML5 geolocation) wired into the Location agent
  * 🗓️  Google Calendar (with local events.json fallback)

Mic widget: uses st.audio_input (Streamlit >= 1.31). If unavailable, the page
still works in text mode.
"""

from __future__ import annotations

import streamlit as st

from assistant import Assistant
from voice_agent import VoiceAgent
from location_agent import LocationAgent


# ----------------------------------------------------------------------------
# Real browser GPS helper
# ----------------------------------------------------------------------------
def get_browser_location() -> tuple[float, float] | None:
    """Read the device's real GPS via the browser. Returns (lat, lon) or None.

    Uses streamlit-geolocation if installed (pip install streamlit-geolocation).
    The browser prompts the user for location permission on first use.
    """
    try:
        from streamlit_geolocation import streamlit_geolocation
        loc = streamlit_geolocation()
        if loc and loc.get("latitude") and loc.get("longitude"):
            return float(loc["latitude"]), float(loc["longitude"])
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
# Main render function
# ----------------------------------------------------------------------------
def render_voice_panel():
    st.header("🐕 Talk to your Unleash companion")
    st.caption("Ask things like “Where am I right now?” or “What do I have at 4 pm?”")

    # Build the assistant once and keep it in session.
    if "assistant" not in st.session_state:
        st.session_state.assistant = Assistant()
    assistant: Assistant = st.session_state.assistant
    voice: VoiceAgent = assistant.voice

    # --- Location: use real browser GPS if the user grants it ---------------
    with st.expander("📍 Location source", expanded=False):
        st.write("Click to share your real GPS location with the dog.")
        coords = get_browser_location()
        if coords:
            assistant.location.set_location(*coords)
            st.success(f"Live GPS: {coords[0]:.5f}, {coords[1]:.5f}")
        else:
            st.info("No live GPS yet — falling back to IP-based location. "
                    "Install `streamlit-geolocation` and allow the browser prompt for real GPS.")

    def answer_and_speak(user_text: str):
        if not user_text:
            return
        st.chat_message("user").write(user_text)
        reply = assistant.handle(user_text)
        st.chat_message("assistant").write(reply)
        data, mime = voice.synthesize(reply)
        if data:
            st.audio(data, format=mime, autoplay=True)

    # --- Microphone input ---------------------------------------------------
    mic_audio = None
    if hasattr(st, "audio_input"):
        mic_audio = st.audio_input("🎤 Tap to speak, then it will answer")
    if mic_audio is not None:
        with st.spinner("Transcribing…"):
            text = voice.transcribe_bytes(mic_audio.getvalue(), suffix=".wav")
        if text:
            answer_and_speak(text)
        else:
            st.warning("Couldn't transcribe that. Check GEMINI_API_KEY, or type below.")

    # --- Text fallback ------------------------------------------------------
    typed = st.chat_input("…or type your question")
    if typed:
        answer_and_speak(typed)

    # --- Quick demo buttons -------------------------------------------------
    st.write("---")
    st.caption("Quick demo questions")
    c1, c2, c3 = st.columns(3)
    if c1.button("Where am I?", use_container_width=True):
        answer_and_speak("where am I right now?")
    if c2.button("What's at 4 pm?", use_container_width=True):
        answer_and_speak("what do I have at 4 pm?")
    if c3.button("My schedule today", use_container_width=True):
        answer_and_speak("what's on my schedule today?")


if __name__ == "__main__":
    st.set_page_config(page_title="Unleash — Voice", page_icon="🐕")
    render_voice_panel()
