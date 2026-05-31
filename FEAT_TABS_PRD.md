# Unleash — FEAT Tabs PRD

**Source:** the `FEAT // …` frames in `untitled.pen` — a desktop feature
storyboard (1440×1024) that walks through each capability area.
**Shared layout:** a 300px **Sidebar** (brand + nav + "RUNNING // 6 AGENTS"
status) and a **Main** column (Chrome bar + Body). The sidebar nav has **five
tabs — SAFETY · MEMORY · VOICE · ASSISTANT · SYSTEM** — and each FEAT frame
highlights one. Tactical cyan-on-black HUD, JetBrains Mono.
**Purpose:** for each tab, define what it shows and the exact backend it maps to,
with an honest build status.
**Last updated:** 2026-05-31

**Status legend:** ✅ backend built · 🟡 built with caveat · ⛔ not built

---

| FEAT tab | Frame | Maps to backend | Status |
|---|---|---|---|
| SAFETY — Safety Pipeline | `cC3Pp` | `app.py` pipeline + `perception.py` | ✅ |
| MEMORY — Memory & Persistence | `QYbj5` | Mem0 + SQLite + W&B Weave | 🟡 |
| VOICE — Voice & Conversation | `l8HzGm` | `voice_agent.py` + `assistant.py` | ✅ |
| ASSISTANT — Location + Calendar | `OEpGX` | `location_agent.py` + `calendar_agent.py` | 🟡 |
| SYSTEM | *(not designed)* | env/keys, services status | ⛔ |

---

## 1. SAFETY — "FEAT // Safety Pipeline" (`cC3Pp`) ✅
The hazard-detection story. Body has a **TitleBlock**, a **metrics row**
(HIGH risk / STREET scene / 104 bpm / GROUNDING action), a **Pipeline** panel
(the agent chain), a **camera** showing a live street scene with detection boxes,
and the **claims/decision** column.

**Backend (built):** `perception.py` (YOLOv8 → scene/risk/detections) →
`process_biometrics` → `pattern_detector` (claims → prediction) →
`behavior_orchestrator` (Gemini directive) → `action_dispatcher`.
**Done when:** metrics + camera + pipeline read live pipeline values.

## 2. MEMORY — "FEAT // Memory & Persistence" (`QYbj5`) 🟡
The "it remembers / it learns" story. Body has a **TitleBlock**, **metrics**, a
**TripleSave** row (the three persistence layers), and a bottom detail row.

**Backend:** `retrospective_agent` writes episodes; `memory_agent` recalls via
Mem0; persistence = Qdrant (`:memory:`) + `mem0_history.db` (SQLite, real rows) +
**W&B Weave** cloud trace.
**Caveat:** semantic recall is **in-session only** — Qdrant starts empty on
restart and isn't rehydrated from SQLite. The TripleSave panel should label this
honestly (history persists; vector recall rebuilds).

## 3. VOICE — "FEAT // Voice & Conversation" (`l8HzGm`) ✅
The conversational interface. Body has a **MicPanel** (glowing, with a waveform),
a **Router** panel (intent routing), a transcript/conversation column, and a
**REPLY // SPOKEN** column.

**Backend (built):** `voice_agent.py` — Gemini STT (`transcribe_bytes`), gTTS
TTS (`synthesize` → `st.audio`); `assistant.py` router (location / directions /
calendar / general); ships as `voice_ui.render_voice_panel()`. Works incl. text
fallback with no key.

## 4. ASSISTANT — "FEAT // Assistant Location+Calendar" (`OEpGX`) 🟡
The "ask it anything practical" story. Content has a **TitleBlock**, **metrics**,
a second row, a **CALENDAR Q&A** panel (1084px), and a **GENERAL CHAT FALLBACK**
panel.

**Backend (built):** `location_agent.py` — `where_am_i()` (real GPS / IP →
OpenStreetMap reverse-geocode) and `directions(destination)`; `calendar_agent.py`
— Google Calendar API with `events.json` fallback; general chat via Gemini.
**Caveat:** **directions are straight-line** distance + estimated walk time, not
a routed/turn-by-turn path. Verified replies: "You're near South Street in
Quincy."; "At 4 pm you have: Dentist appointment…".

## 5. SYSTEM — (nav item, no frame yet) ⛔
The fifth nav tab is present in every sidebar but **has no designed screen**.
Intended scope: API-key/service status (Gemini, Google Calendar, W&B, Bland AI),
the emergency-calling arm toggle, and runtime config — i.e. the same checks
`ui.py` does in its sidebar. **Design + build needed.**

---

## Cross-cutting gaps (affect multiple tabs)
- **Routed directions** (ASSISTANT) — straight-line only today.
- **Cross-session memory** (MEMORY) — Qdrant rehydration not done.
- **Live biometrics** (SAFETY metrics) — heart-rate input is simulated/fed.
- **SYSTEM tab** — not designed or built.
- **Clean install** — base `requirements.txt` is incomplete; also need
  `requirements-voice.txt`.

## Build order (highest demo value first)
1. **VOICE** + **ASSISTANT** — backends already green; wire panels to live data.
2. **SAFETY** — core safety narrative; connect camera + pipeline metrics.
3. **MEMORY** — label the TripleSave honestly; optionally fix rehydration first.
4. **SYSTEM** — design last; mostly a status/config surface.

## Backend hooks (same as the rest of Unleash)
| Tab need | Call |
|---|---|
| Voice Q&A turn | `Assistant().handle(text)` / `converse_once()` |
| Speak reply | `VoiceAgent().synthesize(text)` → `st.audio` |
| Where am I | `LocationAgent().where_am_i()` |
| Directions | `LocationAgent().directions(dest)` |
| Calendar | `CalendarAgent().answer(q)` |
| Safety pipeline | functions in `app.py` (see `ui.py`) |
