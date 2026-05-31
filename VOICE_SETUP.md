# Unleash — Voice, Location & Calendar agents

The conversational layer: the user talks to the robodog and it talks back.
Answers questions like **"Where am I right now?"** and **"What do I have at 4 pm?"**
Built to run **inside Streamlit** (browser mic + in-browser audio), with real
GPS and Google Calendar.

## Files

| File | What it does |
|------|--------------|
| `voice_ui.py` | **Streamlit page** — mic, text box, audio replies, GPS, demo buttons. |
| `voice_agent.py` | Speech in/out. STT via Gemini; TTS via gTTS→mp3 bytes for `st.audio`. |
| `location_agent.py` | "Where am I?" — real GPS / preset / IP → real address (OpenStreetMap). |
| `calendar_agent.py` | "What's at 4 pm?" — Google Calendar API, falls back to `events.json`. |
| `assistant.py` | Router: classifies intent → Location / Calendar / general chat. |
| `voice_demo.py` | CLI entry point (text or desktop-voice mode) for quick testing. |
| `events.json` | Local calendar fallback data (edit freely). |
| `.env.example` | Copy to `.env`, add `GEMINI_API_KEY`. |
| `requirements-voice.txt` | Dependencies for this feature. |

## How it works (one line)

`browser mic → Gemini transcribes → Assistant classifies intent → Location/Calendar/general → gTTS mp3 → st.audio plays the reply`

Intent routing tries fast keyword matching first, then Gemini if a key is set —
so it degrades to a fully working **text** assistant with no API key at all.

## Quick start

```bash
pip install -r requirements-voice.txt
cp .env.example .env            # add GEMINI_API_KEY for voice + smart routing

# Standalone Streamlit page:
streamlit run voice_ui.py

# Or CLI (no browser): text mode
python voice_demo.py
python voice_demo.py --ask "what do I have at 4 pm?"
```

## For the UI teammate — integrating into the main app

`voice_ui.py` exposes one function. Add a tab/page to `ui.py` and call it:

```python
from voice_ui import render_voice_panel
render_voice_panel()        # renders the whole mic + chat + GPS panel
```

Or call the logic directly without the prebuilt UI:

```python
from assistant import Assistant
bot = Assistant()
reply = bot.handle("what do I have at 4 pm?")     # text in, text out

from voice_agent import VoiceAgent
data, mime = VoiceAgent().synthesize(reply)        # mp3 bytes
st.audio(data, format=mime, autoplay=True)         # speak it in the browser
```

The mic uses `st.audio_input` (Streamlit ≥ 1.31). Transcribe its bytes with
`VoiceAgent().transcribe_bytes(mic.getvalue())`.

## Real GPS

`voice_ui.py` calls the browser's HTML5 geolocation via `streamlit-geolocation`
(the browser prompts for permission). The coordinates are pushed into the agent
with `LocationAgent.set_location(lat, lon)`, then reverse-geocoded to a street
name via OpenStreetMap Nominatim (free, no key). If the user denies permission
or the package isn't installed, it falls back to free IP-based location, so
"Where am I?" always answers something sensible.

## Google Calendar setup

The Calendar agent uses the **Google Calendar API** and falls back to
`events.json` if it isn't configured — so the demo works either way.

To enable real Google Calendar:
1. In Google Cloud Console, enable the **Google Calendar API** and create an
   **OAuth client ID** (Desktop app). Download it as `credentials.json` and put
   it next to `calendar_agent.py`.
2. First run opens a browser to authorize; a `token.json` is cached for reuse.
3. `calendar_agent.py` reads your **primary** calendar's events for today.

No credentials present → it silently uses `events.json`. (Both produce the same
event shape, so nothing else changes.)

## Notes / next steps
- **Barge-in** (stop playback when the user talks) — not implemented; nice polish.
- **Maps/routing** ("how do I get to X") — add a directions call in `location_agent.py`.
- **Tone-by-urgency voice** — gTTS is one voice; swap to ElevenLabs for emotional range.
