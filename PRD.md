# Unleash — Product Requirements Document

**Product:** Unleash — a multi-agent AI "service dog" for visually impaired and
anxiety-prone users
**One-liner:** Give anyone the capabilities of a $50,000 service dog. The dog
sees, the system remembers, the voice keeps you safe.
**Status:** Hackathon build (SundAI Hack & Learn / #BosTechWeek)
**Owner:** Phillips Le · **Repo:** github.com/N-star-dot/Unleash
**Last updated:** 2026-05-31

> Companion docs in this repo:
> `UI_DESIGN_PRD.md` (UI spec + code), `VOICE_SETUP.md` (voice/location/calendar
> setup), `CLAUDE_CODE_CONTEXT.md` (engineering handoff), `design_export/`
> (frontend code extracted from the Pencil design).

---

## 1. Problem & vision

A guide dog notices danger, remembers risky places, and steers its handler to
safety — but costs ~$50k, takes years to train, and is scarce. Unleash recreates
that capability in software: a phone/wearable camera sees the world, biometrics
sense distress, an LLM decides what matters right now, and a calm voice guides
the user — escalating to a human only in a real emergency.

**Primary user:** a visually impaired or anxiety-prone person navigating the
world. **Secondary user:** their caregiver. **Accessibility is the product**, not
a feature — every capability must be reachable by voice.

---

## 2. Goals & non-goals

**Goals**
- Detect physical hazards in real time and warn the user by voice.
- Detect physiological distress (e.g. panic-attack precursor) and intervene.
- Answer the user's spoken questions: where am I, directions, what's on my
  schedule.
- Remember incidents and learn which interventions worked.
- Escalate to a caregiver (real phone call) when a crisis is detected.
- Make the agent reasoning observable (for trust, and for demo judging).

**Non-goals (for the hackathon)**
- Production-grade safety guarantees or medical certification.
- Robodog hardware integration (designed for; stubbed in software).
- Full turn-by-turn navigation (basic distance/bearing directions only).

---

## 3. System overview

Unleash is a **multi-agent system**: specialized agents coordinate over a shared
state object (the "conflict bus"), and an LLM orchestrator makes the final
safety decision. There are **two interaction paths**:

1. **System-initiated (hazard pipeline):** sensors → claims → prediction →
   memory → orchestrator decision → action → memory write. Runs continuously.
2. **User-initiated (voice assistant):** user speaks → intent classified →
   routed to Location / Directions / Calendar / general → spoken answer.

Both paths share the same agents, voice, and memory.

```
            ┌──────────── SYSTEM-INITIATED (hazard) ────────────┐
 Camera ─▶ Perception ─┐                                         │
 HR ─────▶ Biometric ──┼▶ Pattern Detector ▶ Memory ▶ Orchestrator ▶ Action
                       │        (claims)    (recall)   (Gemini)    (speak/call)
                       └────────────── Conflict Bus (shared dict) ┘
                                              │
            ┌──────────── USER-INITIATED (voice) ─────────────┐ │
 Mic ▶ Voice(STT) ▶ Assistant.classify ▶ {Location|Directions|Calendar|General} ▶ Voice(TTS)
                                              │
                            Memory: Qdrant(:memory:) + mem0_history.db(SQLite) + W&B Weave
```

---

## 4. The agents (functional requirements)

### 4.1 Perception agent — `perception.py` ✅ built
Runs YOLOv8n on the iPhone Continuity Camera / webcam. Emits a structured JSON
payload per frame: scene (street/indoor/nature), ambient light, crowd count,
risk level (LOW/MED/HIGH), and per-object detections with side
(left/center/right), proximity, velocity, and a collision-risk score.
- **Requirement:** flag HIGH risk when a tracked object's collision risk > 0.6.
- Local, no API key.

### 4.2 Biometric agent — `process_biometrics` in `app.py` ✅ built
Evaluates heart-rate (and HRV) telemetry. Emits a `TachycardiaAlert` claim when
HR > 100.
- **Requirement:** thresholds configurable; emits claims onto the bus, never
  decides actions itself.
- *Note:* real Apple Health integration is a future hook; currently simulated /
  fed via the pipeline.

### 4.3 Pattern Detector — `pattern_detector` in `app.py` ✅ built
Turns raw claims into named `predictions` with confidence (e.g. Tachycardia →
"Panic Attack Precursor" 0.91; Vision → "Approaching Crowd" 0.88).
- **Requirement:** deterministic and fast; an LLM upgrade is optional (see §9).

### 4.4 Memory agent — `memory_agent` in `app.py` ✅ built
Semantic recall via Mem0 (Gemini embeddings): given the latest prediction,
retrieves past episodes ("what stabilized the user last time?").

### 4.5 Orchestrator — `behavior_orchestrator` in `app.py` ✅ built
**The one true LLM decision-maker.** Sends predictions + recalled memories to
**Gemini 2.5 Flash** and selects exactly one directive: grounding audio /
physical nudge / call emergency / stay passive.

### 4.6 Action agent — `action_dispatcher` in `app.py` ✅ built
Executes the directive. AirPods audio and robot-nudge are status strings
(hardware hooks). **Emergency is real:** `_trigger_bland_ai_call()` POSTs to
Bland AI to place a live caregiver phone call (simulated if keys absent).

### 4.7 Retrospective agent — `retrospective_agent` in `app.py` ✅ built
Writes the episode outcome back to memory so the system can learn.

### 4.8 Voice agent — `voice_agent.py` ✅ built (new)
The user's interface. **STT** via Gemini (`transcribe` / `transcribe_bytes` for
Streamlit mic bytes). **TTS** via gTTS → mp3 bytes for `st.audio` (browser),
with macOS `say`/pyttsx3 fallbacks for CLI.
- **Requirement:** degrade gracefully — text mode works with no mic and no key.

### 4.9 Assistant / router — `assistant.py` ✅ built (new)
Classifies a user utterance into **location / directions / calendar / general**
(keyword-first, Gemini fallback) and routes it. Entry points: `handle(text)`
(text in/out) and `converse_once()` (full mic→answer→speak turn).

### 4.10 Location / Place agent — `location_agent.py` ✅ built (new)
Answers "where am I?" Resolves position via manual/phone GPS → preset → free IP
geolocation, then reverse-geocodes to a street/place via OpenStreetMap (no key,
cached). Also provides **`directions(destination)`** — distance + rough walk time
to a named place. Real browser GPS is fed in from the Streamlit UI.

### 4.11 Calendar / Software agent — `calendar_agent.py` ✅ built (new)
Answers "what do I have at 4 pm?" Reads the **Google Calendar API** (OAuth) for
today's events, parses the queried time, and replies — falling back to a local
`events.json` so the demo never breaks.

---

## 5. Memory & persistence (triple-save)

Every significant decision is saved three ways (configured in `app.py`):
1. **Qdrant vector store (`:memory:`)** — in-RAM, enables semantic recall within
   a session. ⚠️ **Known limitation:** starts empty on restart; no rehydration
   from SQLite yet, so cross-session recall is not real today.
2. **`mem0_history.db` (SQLite)** — on-disk raw text history; persists across
   restarts.
3. **Weights & Biases Weave** — every agent function is `@weave.op()`-decorated;
   a cloud dashboard traces each decision (time, reasoning, action payload).
- **Requirement before claiming "learns over time":** make Qdrant persistent or
  rehydrate it from SQLite on startup (see §9).

---

## 6. User interface

Two surfaces (full spec + extracted frontend code in `UI_DESIGN_PRD.md` and
`design_export/`):

- **Companion HUD** (mobile, 390×844) — tactical cyan-on-black HUD, monospace.
  Tabs: **HOME · DATA · MAP · UNIT**. HOME (sensor/command view with live
  perception + voice-output line) and DATA (vitals + episode timeline) are
  designed; **MAP and UNIT are not yet designed.**
- **Operator / Demo dashboard** (Streamlit) — **Command Center**, **Live
  Dashboard** (pipeline stepper), **Episode Trace** (history + W&B link). The
  voice panel ships as `voice_ui.render_voice_panel()` (mic, text, GPS, demo
  asks) and drops into `ui.py`.

Design system: tokens in `design_export/tokens.css` (cyan `#00D4FF` accent,
state colors green/amber/red, JetBrains Mono, lucide icons). HUD HOME exists as
runnable `hud_home.html` and a React `HudHome.jsx`.

---

## 7. Key user flows

1. **"Where am I?"** → STT → location → `where_am_i()` → spoken street + MAP pin.
2. **"How do I get to Harvard Square?"** → STT → directions →
   `directions("Harvard Square")` → distance + walk time, spoken.
3. **"What do I have at 4 pm?"** → STT → calendar → `CalendarAgent.answer()` →
   spoken event.
4. **Hazard (automatic):** camera HIGH risk → pipeline → orchestrator decides →
   HUD flips to ALERT, dog speaks, episode logged, dashboard animates.
5. **Emergency:** orchestrator picks "call emergency" → Bland AI dials caregiver
   → HUD shows "CALLING…" (armed only when keys set; UNIT toggle default OFF).

---

## 8. External dependencies & keys

| Service | Var | Required? | Used for |
|---|---|---|---|
| Google Gemini | `GEMINI_API_KEY` | **Yes** | orchestrator, Mem0 embeddings, STT, intent routing, chat |
| Weights & Biases | `WANDB_API_KEY` | Recommended | Weave decision dashboard |
| Google Calendar | OAuth `credentials.json` | Optional | real calendar (else `events.json`) |
| Bland AI | `BLAND_API_KEY` + `CAREGIVER_PHONE` | Optional | real emergency call (else simulated) |
| YOLOv8 / OpenStreetMap / IP-geo | — | none | vision + location run key-less |

Deps: `requirements.txt` (CV) + `requirements-voice.txt` (streamlit, google-genai,
weave, gTTS, dotenv, calendar + geolocation libs). ⚠️ Base `requirements.txt`
is incomplete (missing streamlit/mem0ai/weave/google-genai/requests) — fix needed.

---

## 9. Known gaps & engineering backlog

Priority order for remaining work:
1. **Fix `requirements.txt`** so a clean machine can install and run.
2. **Cross-session memory** — persist/rehydrate Qdrant so recall survives restart.
3. **Add `.env.example` + dotenv loading** (partially done in voice path).
4. **Resolve the dead LangGraph path** — the `*_agent.py` + `test_*.py` skeleton
   is broken (`conflict_bus.py` lacks the `Claim` class it imports). Delete, or
   repair and unify on LangGraph. `app.py` is the source of truth.
5. **Design + build HUD MAP and HUD UNIT** screens; wire MAP to the live
   location agent.
6. **Optionally upgrade Pattern Detector to an LLM agent** so "multiple AI
   agents" is literally true.
7. **Rename internal "ARGUS" → "Unleash"** (Weave project + UI title).

---

## 10. Success criteria (demo-ready definition of done)

- Live camera detects a hazard and the dog speaks a warning.
- A panic-precursor scenario triggers a grounding intervention.
- Voice Q&A answers location, directions, and calendar questions out loud.
- An emergency scenario places (or convincingly simulates) a caregiver call.
- The W&B Weave dashboard shows the decision trace live.
- Honest framing maintained: "a multi-agent system coordinating over a shared
  conflict bus, orchestrated by an LLM" — not "six fully autonomous AIs."

---

## 11. Risks

- **Live emergency call fires on stage** if armed — point `CAREGIVER_PHONE` at
  your own phone, or leave keys unset.
- **Indoor GPS** is unreliable — presets / IP fallback cover the demo.
- **Restart wipes vector memory** — don't claim persistent learning until §9.2.
- **Theme fit** — frame Unleash as autonomous multi-agent for the event's
  "Autonomous Companies & Research" theme.

---

## Appendix — file map

| File | Role |
|---|---|
| `app.py` | Hazard pipeline + all core agents + Mem0/Weave/Bland config |
| `ui.py` | Streamlit operator dashboard + live YOLO loop |
| `perception.py` | YOLOv8 vision |
| `voice_agent.py` | STT (Gemini) + TTS (gTTS/say) |
| `assistant.py` | Voice intent router (location/directions/calendar/general) |
| `location_agent.py` | "Where am I" + directions, geocoding |
| `calendar_agent.py` | Google Calendar + events.json fallback |
| `voice_ui.py` | Streamlit voice panel (`render_voice_panel`) |
| `voice_demo.py` | CLI entry (text / voice modes) |
| `events.json` | Local calendar data |
| `design_export/` | Frontend code from the Pencil design (tokens, HUD HOME) |
| `run_simulation.py` | Headless pipeline run |
| `*_agent.py` + `test_*.py` | ⚠️ broken LangGraph skeleton (dead path) |
