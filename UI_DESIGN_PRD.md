# Unleash — UI Design PRD

**Product:** Unleash — multi-agent AI service dog for visually impaired / anxiety-prone users
**Doc owner:** Phillips Le (UI)
**Design file:** `untitled.pen` (Pencil)
**Status:** Draft for hackathon build — grounded in the existing Pencil frames
**Last updated:** 2026-05-31

---

## 1. Purpose & scope

Defines the UI for Unleash across the two surfaces that already exist as frames
in `untitled.pen`:

1. **Companion HUD** (mobile, 390×844) — the wearer/caregiver view. Tactical HUD
   with four tabs: **HOME · DATA · MAP · UNIT**.
2. **Operator / Demo surfaces** (desktop) — the judge-facing real-time view:
   **Command Center** (1480×1040) and two **Streamlit** mockups (1440×1024):
   *Live Dashboard* and *Episode Trace*. Plus a working scratch frame,
   **UI ENHANCEMENTS // CONCEPTS**.

Current state of the frames:
- ✅ Designed: HUD HOME, HUD DATA, Command Center, Streamlit Live Dashboard,
  Streamlit Episode Trace (and a concepts board).
- ⛔ **Empty (to be designed): HUD MAP, HUD UNIT/PROFILE.**

Out of scope: backend agent logic (already built — `app.py`, `assistant.py`,
`voice_agent.py`, `location_agent.py`, `calendar_agent.py`) and robodog hardware.

---

## 2. Design language (as built)

The established aesthetic is a **tactical / mission-control HUD**:

- **Palette:** near-black background (`$hud-bg`), panels on `$hud-card`, hairline
  `$hud-border`. **Cyan is the primary accent** (`$hud-cyan`, with `-12/-20/-40`
  alpha tints for glows, rules, and fills). Green `$hud-success` = live/safe.
- **Type:** monospace (`$font-mono`) for everything — labels are UPPERCASE with
  wide letter-spacing (1–2). Tiny labels (8px) for section/eyebrow text; 12–14px
  for values and titles.
- **Motifs:** "reactor" ring icon, `LIVE` pill, corner-bracket framing on the
  Perception panel, separator rows like `——— SENSORS // ACTIVE ———`, subtle
  outer-glow shadows in cyan.
- **State color ramp (the only semantic colors):** green `$hud-success` = safe/
  live, amber = heads-up, red = alert/hazard. Never encode state by color alone —
  pair with an icon + uppercase word.

> Design rule for new screens (MAP, UNIT): reuse these exact tokens and the
> StatusBar + Header + Body + TabBar skeleton so they feel native. Do not
> introduce new colors or fonts.

---

## 3. Shared HUD chrome (all four mobile tabs)

Every HUD tab is a 390×844 vertical frame, `clip:true`, with:

1. **StatusBar** (h34) — `09:41`, `5G`, battery icon. `$hud-bg`.
2. **Header** (h64) — reactor ring icon + title block (`COMMAND` /
   `SERVICE DOG // ONLINE`) + a right-side pill (`LIVE` on HOME, `WEAVE` on DATA).
3. **Body** (`fill_container`) — tab-specific content, 16px padding, 14px gap.
4. **TabBar** (h70) — four items HOME / DATA / MAP / UNIT (lucide `house`,
   `activity`, `map`, `user`). Active tab = cyan icon + label + top underline glow;
   inactive = `$hud-muted`.

Acceptance: switching tabs only swaps the Body; chrome stays identical.

---

## 4. Companion HUD — per-tab spec

### 4.1 HUD HOME ✅ (refine) — "what's happening right now"
Already built as the sensor/command view. Contains:
- **Sensor stat grid** — four tiles: people/threat count, two metrics, and
  `HAZARDS AHEAD`. Wire to live values: crowd count + risk from `perception.py`,
  heart rate from `process_biometrics`.
- **Perception panel** — live camera with corner brackets and a red `RED LIGHT`
  overlay + badge row (`RED LIGHT / CROSSWALK / CARS / WALK`). Maps to YOLO
  detections + scene/risk.
- **VOICE OUTPUT panel** — the dog's last spoken line in quotes
  (e.g. "RED LIGHT. TWO CARS ON YOUR RIGHT. WAIT."). Maps to the latest
  `VoiceAgent` utterance / `Assistant` reply.

Refinements: add a visible **push-to-talk mic affordance** and listening state
(the voice panel is currently output-only); make the top-line status reflect the
orchestrator state (SAFE / HEADS UP / ALERT) via the color ramp.

### 4.2 HUD DATA ✅ (refine) — vitals & episode timeline
Already built. Contains a **ChartPanel** (telemetry, h236) and a **Timeline**
(report list, h336), with a `WEAVE` pill in the header.
- Wire ChartPanel to heart rate / HRV trend.
- Wire Timeline to the episode history written to memory (`mem0_history.db`):
  each row = time, type (Panic precursor / Crowd / Hazard), action taken.
- Empty state (good news): "NO INCIDENTS TODAY // ALL CALM".

### 4.3 HUD MAP ⛔ (build) — "where am I + what's nearby"
Empty frame today. Build using the shared chrome. Must contain:
- **Map area** with a current-location pin.
- **Location readout** — the human address from `LocationAgent.where_am_i()`
  (e.g. "HARVARD SQUARE"), plus a **fix-source chip**: `GPS / IP / PRESET`.
- **Known/risky zone markers** — preset pins for the demo (home, 5th & Main,
  Mass General, Harvard Sq); future: Place-agent risk scores.
- **"NAVIGATE TO…"** affordance (voice-triggerable stub).
- Voice hook: "Where am I?" updates this tab.

### 4.4 HUD UNIT / PROFILE ⛔ (build) — settings & safety net
Empty frame today. Build using the shared chrome. Must contain:
- **Unit identity** (user name, optional photo) in HUD style.
- **Caregiver / emergency contact** — name + number the Action agent dials via
  Bland AI, with a prominent **EMERGENCY CALLING: ON/OFF** toggle (default OFF).
- **Connected services** status list: GEMINI / GOOGLE CALENDAR / WEIGHTS &
  BIASES — each `CONNECTED` (green) or `OFFLINE` (muted), mirroring `ui.py`'s
  sidebar checks.
- **Voice settings** — speaking rate, voice.

---

## 5. Operator / Demo surfaces

### 5.1 Command Center ✅ (desktop, 1480×1040) — the overview to keep open
Header (brand + status pill) over a 3-column Body:
- **Col Left — Agents (400px):** the agents as live tiles (Perception,
  Biometric, Pattern Detector, Memory, Orchestrator, Action) with idle/active/
  firing states. This is the "multi-agent system" made visible.
- **Col Center (fill):** live camera + current decision/directive, large.
- **Col Right (392px):** claims/predictions feed and the voice panel
  (`render_voice_panel()`), plus trigger buttons (Safe / Crowd / Panic).

### 5.2 Streamlit // Live Dashboard ✅ (1440×1024) — pipeline in motion
Sidebar (300px: brand, nav, selectbox, slider, toggles, status) + Main (chrome +
content). Content must show the **pipeline stepper** Ingest → Claims → Pattern →
Memory → Orchestrator → Action (matches `execute_scenario`'s `st.status` steps),
the **claims & predictions** with confidence, **retrieved memories**, and the
**action result** string (incl. real Bland AI confirmation when armed).

### 5.3 Streamlit // Episode Trace ✅ (1440×1024) — history & W&B proof
**Episode timeline** (newest first) + a prominent **W&B Weave link/embed**
(the credibility flex) + an honest **memory-persistence indicator**
(in-session vector recall vs. on-disk SQLite; note the `:memory:` restart caveat).

---

## 6. Key interaction flows

1. **"Where am I?" (voice)** → STT → intent=location → `LocationAgent.where_am_i()`
   → spoken reply + HUD MAP pin/readout update.
2. **"What do I have at 4 pm?" (voice)** → STT → intent=calendar →
   `CalendarAgent.answer()` (Google Calendar, JSON fallback) → spoken reply.
3. **Hazard (system-initiated):** camera HIGH risk → pipeline runs → HUD HOME
   flips to ALERT, VOICE OUTPUT updates, dog speaks; operator dashboard animates
   the stepper; episode logged.
4. **Emergency escalation:** orchestrator picks "call emergency" → Action agent
   dials caregiver (Bland AI) → HUD shows "CALLING [caregiver]…"; UNIT toggle
   governs whether armed.

---

## 7. Accessibility (non-negotiable — primary users are visually impaired)

- Every screen operable **by voice** and **by screen reader**; visuals are the
  secondary/caregiver channel.
- The HUD's 8px mono labels are decorative eyebrows only — any **essential**
  text must also exist at ≥16px and meet **WCAG AA contrast (4.5:1)** against
  `$hud-bg`/`$hud-card`. Audit cyan-on-black small text specifically.
- State conveyed by **icon + word**, never color alone.
- Tap/voice targets ≥ 44px.

---

## 8. Acceptance criteria (per screen "done")

- All "must contain" items render with real or clearly-mocked data.
- Reachable/operable by voice + screen reader; AA contrast on essential text.
- State changes (safe→alert) reflected within ~1s of the backend event.
- Streamlit screens read live backend values (except deterministic demo triggers).
- New screens (MAP, UNIT) reuse the shared chrome + existing tokens exactly.

---

## 9. Build priority (hackathon clock)

1. **HUD MAP** (empty, high demo value — pairs with the live location agent).
2. **HUD UNIT/PROFILE** (empty — needed for the emergency-toggle + services story).
3. **Command Center** wiring (camera + voice panel + triggers + decision).
4. **HUD HOME** refinements (mic affordance + status color ramp).
5. **Streamlit Live Dashboard** stepper; then **Episode Trace + W&B link**, **HUD DATA** wiring.

---

## 10. Backend hooks (already built — UI just calls these)

| UI need | Call |
|---|---|
| Answer any user question | `Assistant().handle(text)` |
| Full mic→answer→speak turn | `Assistant().converse_once()` |
| Speak text in browser | `VoiceAgent().synthesize(text)` → `st.audio` |
| Transcribe mic bytes | `VoiceAgent().transcribe_bytes(bytes)` |
| Where am I | `LocationAgent().where_am_i()` / `.set_location(lat,lon)` |
| Calendar at a time | `CalendarAgent().answer("what's at 4pm?")` |
| Full voice UI panel (Streamlit) | `from voice_ui import render_voice_panel` |
| Hazard pipeline (per scenario) | functions in `app.py` (see `ui.py`) |

---

## 11. Design tokens in use (reuse these; don't invent new ones)

`$hud-bg` (page) · `$hud-card` (panels) · `$hud-border` (hairlines) ·
`$hud-cyan` + `$hud-cyan-12/-20/-40` (accent/glow/rules) ·
`$hud-success` + `$hud-success-12` (live/safe) · `$hud-muted` (inactive) ·
`$hud-text` (primary) · `$font-mono` (all type) · `$safe` / `$safe-dim`
(Command Center status). Icons: **lucide**.

---

## 12. Open questions

- Companion HUD final target: native app, or Streamlit mock for the hackathon?
  (Recommend Streamlit/Pencil mock now.)
- MAP "risky zones": preset pins acceptable for the demo (Place-agent scoring not
  built yet)?
- Emergency calling armed during judging? (UNIT toggle default: OFF.)
- DATA's `WEAVE` pill — link out to the live W&B dashboard?

---

## Appendix A — Frontend code (extracted from `untitled.pen`)

Pencil has no code-export feature (its exporter only outputs PNG/JPEG/WEBP/PDF
images). So the design was translated to production-ready frontend code by hand
from the resolved node tree + design variables. The code lives in
`/design_export/` and is the buildable starting point for this UI.

| File | Purpose |
|---|---|
| `design_export/tokens.css` | All design variables as CSS custom properties (exact hex from the `.pen` file). Import once at app root. |
| `design_export/hud.css` | Component styles for the HUD chrome + HOME (status bar, header, reactor, stat cards w/ corner brackets, perception panel, voice panel, tab bar). |
| `design_export/hud_home.html` | Standalone, runnable HUD HOME screen (open in a browser; uses lucide via CDN). 1:1 with frame `QyOvA`. |
| `design_export/HudHome.jsx` | The same screen as a React component (`lucide-react`). All values are props, ready to wire to the backend hooks in §10. |

### Design tokens (authoritative hex values)

```
HUD core:    --hud-bg #080C14 · --hud-card #0C1220 · --hud-inset #0A0F1C · --hud-border #00D4FF33
Cyan accent: --hud-cyan #00D4FF · -08 #00D4FF14 · -12 #00D4FF1F · -20 #00D4FF33 · -40 #00D4FF66
             --hud-muted #00D4FF73 · --hud-text #E0F2FE
State:       --hud-success #00E676 · --hud-warning #FFB300 · --hud-danger #FF5252 · --hud-memory #A78BFA
             (each has a -12 alpha fill variant)
Alt brand:   --accent #34E0CE · --accent-blue #5390FF · --bg-base #0C111A · --bg-panel #111824
             --text-primary #EAF1FA · --text-secondary #8C9BB2 · --text-muted #566276
             --safe #3FE08A · --warn #FFB23E · --danger #FF4D5E
Type:        --font-mono "JetBrains Mono" · --font-body "Inter"
Radii:       3 / 4 / 6 px        Icons: lucide
```

### How to run / integrate

```bash
# Preview the screen as-is:
open design_export/hud_home.html         # macOS (or just double-click it)

# React: copy the three CSS/JSX files into your app, then
#   import "./tokens.css"; import "./hud.css";
#   import HudHome from "./HudHome";
#   <HudHome active="home" onTab={setTab} spoken={lastSpoken} stats={liveStats} />
npm i lucide-react
```

### Status of generated screens

- **HUD HOME** — ✅ full code generated (`hud_home.html`, `HudHome.jsx`).
- **HUD DATA, Command Center, Streamlit frames** — design exists in `.pen`;
  reuse `tokens.css` + `hud.css` chrome to build them next (same patterns).
- **HUD MAP, HUD UNIT** — empty in `.pen`; design first (see §9), then generate.

> To regenerate or extend: read the frame's node tree from `untitled.pen` with
> the Pencil MCP (`batch_get` + `get_variables`, `resolveVariables: true`) and
> translate node→element using the same class vocabulary in `hud.css`.
