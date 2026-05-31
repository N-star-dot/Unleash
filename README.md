# 🐕 Unleash — AI Service Dog

> Give anyone the capabilities of a $50,000 service dog. The dog **sees**, the system **remembers**, the voice **keeps you safe**.

A multi-agent AI "service dog" for visually-impaired and anxiety-prone users. A phone/wearable camera sees the world, biometrics sense distress, an LLM decides what matters *right now*, and a calm voice guides the user — escalating to a human caregiver only in a real emergency.

Built for SundAI Hack & Learn / #BosTechWeek · Repo: `N-star-dot/Unleash`

---

## What it does

Specialized agents coordinate over a shared state object (the **conflict bus**), and a single LLM **orchestrator** makes the final safety decision. Two interaction paths share the same agents, voice, and memory:

1. **Hazard pipeline (automatic):** camera + heart-rate → claims → prediction → memory recall → **Gemini orchestrator** decides → action → episode logged.
2. **Voice assistant (on demand):** you speak → intent classified → **Location / Directions / Calendar / general** → spoken answer.

```
            ┌──────────── SYSTEM-INITIATED (hazard) ────────────┐
 Camera ─▶ Perception ─┐                                         │
 HR ─────▶ Biometric ──┼▶ Pattern Detector ▶ Memory ▶ Orchestrator ▶ Action
                       │        (claims)    (recall)   (Gemini)    (speak / call)
                       └────────────── Conflict Bus (shared dict) ┘
            ┌──────────── USER-INITIATED (voice) ───────────────┐
 Mic ▶ Voice(STT) ▶ Assistant.classify ▶ {Location│Directions│Calendar│General} ▶ Voice(TTS)
                                              │
                  Memory: Qdrant + mem0_history.db (SQLite) + W&B Weave
```

## The agents

| Agent | File | Role |
|---|---|---|
| Perception | `perception.py` | YOLOv8n on the camera — people + interaction tracking, **stairs / doors / ladders** (Open Images V7), per-frame JSON (scene, crowd, `risk_level`, detections, hazards). Keeps `vision_generator()`. |
| Biometric | `app.py` | Heart-rate → `TachycardiaAlert` claim when HR > 100. |
| Pattern Detector | `app.py` | Claims → named predictions (e.g. "Panic Attack Precursor", 0.91). |
| Memory | `app.py` | Semantic recall via Mem0 (Gemini embeddings). |
| **Orchestrator** | `app.py` | The LLM decision-maker — Gemini 2.5 Flash picks one directive. |
| Action | `app.py` | Executes it; **emergency places a real Bland AI caregiver call** (simulated if unarmed). |
| Retrospective | `app.py` | Writes the episode outcome back to memory. |
| Voice | `voice_agent.py` | STT (Gemini) + TTS (gTTS → `st.audio`, `say`/pyttsx3 fallback). |
| Assistant / router | `assistant.py` | Classifies → location / directions / calendar / general. |
| Location / Place | `location_agent.py` | "Where am I" (GPS/IP + OpenStreetMap) **and `directions(dest)`** (geocode + OSRM route → distance + walk time). |
| Calendar | `calendar_agent.py` | Google Calendar (OAuth) with `events.json` fallback. |

## UI

- **Command Center** (`ui.py`) — Streamlit operator dashboard in a tactical cyan/teal HUD: live YOLO camera, conflict-bus claims, agent status, biometrics, and **Live Feed / Agents / Biometrics / Place** tabs. Design tokens in `design_export/tokens.css` + `hud.css`.
- **Companion HUD** (`design_export/hud_home.html`, embedded live via `companion_hud.py`) — the cyan-on-black mobile HUD the user carries (HOME / DATA / MAP / UNIT).

## Quick start

```bash
pip install -r requirements.txt          # core (CV + pipeline + UI)
pip install -r requirements-voice.txt     # voice / location / calendar

cp .env.example .env                      # then add GEMINI_API_KEY (required)
#   optional: WANDB_API_KEY, BLAND_API_KEY + CAREGIVER_PHONE
#   WEAVE_PROJECT=<your-wandb-entity>/unleash-service-dog   (use YOUR entity)

streamlit run ui.py          # Command Center dashboard
python  run_simulation.py    # headless hazard pipeline (panic scenario)
python  voice_demo.py --ask "how do I get to Harvard Square?"   # CLI voice agent
```

## Keys & services

| Service | Var | Required? | Used for |
|---|---|---|---|
| Google Gemini | `GEMINI_API_KEY` | **Yes** | orchestrator, Mem0 embeddings, STT, intent routing, chat |
| Weights & Biases | `WANDB_API_KEY` + `WEAVE_PROJECT` | Recommended | Weave decision dashboard |
| Google Calendar | OAuth `credentials.json` | Optional | real calendar (else `events.json`) |
| Bland AI | `BLAND_API_KEY` + `CAREGIVER_PHONE` | Optional | real emergency call (else simulated) |
| YOLOv8 · OpenStreetMap · OSRM · IP-geo | — | none | vision + location run key-less |

## What's real vs simulated (honest framing)

- **Real:** YOLOv8 detection + interaction tracking; the Gemini orchestrator's decision; Mem0 semantic recall; W&B Weave traces; voice STT/TTS; OpenStreetMap geocoding + OSRM directions; the Bland AI emergency call (when armed).
- **Simulated / stubbed:** AirPods audio routing and robot-dog nudge are status strings (hardware hooks); heart-rate is fed via the pipeline (Apple Health is a future hook); cross-session vector memory resets on restart (SQLite history persists).
- **Framing:** *a multi-agent system coordinating over a shared conflict bus, orchestrated by an LLM that makes the final safety decision* — not "six fully autonomous AIs."

## Tech

Python 3.14 · Streamlit · Ultralytics YOLOv8 (PyTorch) · Mem0 + Qdrant · Google GenAI (Gemini 2.5 Flash) · Weights & Biases Weave · gTTS · OpenStreetMap / OSRM · Bland AI.

> ⚠️ **Demo safety:** the emergency path places a *real* phone call when `BLAND_API_KEY` + `CAREGIVER_PHONE` are set and the UI's **ARM REAL CALLS** toggle is on. Point it at your own phone, or leave it off.
