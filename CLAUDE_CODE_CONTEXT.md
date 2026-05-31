# Unleash — Context Brief for Claude Code

> Hand this whole file to Claude Code as the working context. It describes the **Unleash**
> hackathon repo, exactly what runs today, the known bugs, and a prioritized build plan.
> The goal is a multi-agent "AI service dog" for visually impaired / anxiety-prone users
> that senses hazards, reasons with an LLM, takes real-world actions, and logs everything.
>
> Naming note: the **project is Unleash**. The codebase still uses the older internal
> codename "ARGUS" in a few places — the W&B Weave project is `project-argus-service-dog`
> and the Streamlit UI title reads "Project ARGUS". Treat these as Unleash. (Optional
> cleanup task: rename internal "ARGUS" references to "Unleash".)

---

## 1. What this project is (one paragraph)

A real-time, multi-agent assistive system that aims to give a user the capabilities of a
service dog. A phone/webcam camera sees the world (YOLOv8 object detection), a biometric
signal (heart rate) is monitored, a "brain" decides whether the situation warrants action,
and an action layer responds — calming audio, a physical nudge, or a real emergency phone
call to a caregiver. Every decision is persisted to memory (in-RAM vector store + on-disk
SQLite) and traced to a Weights & Biases (Weave) cloud dashboard.

---

## 2. The runtime that ACTUALLY works today

The live demo is **`app.py` driven by `ui.py` (Streamlit)**. It is a **sequential pipeline**:
one shared dict (`ConflictBusState`, a TypedDict) is passed function-to-function. There is
**no LangGraph and no async event bus in the executing path**.

Pipeline order (all functions live in `app.py`, each decorated `@weave.op()`):

1. `process_biometrics(telemetry)` — rule: `heart_rate > 100` → emits a `TachycardiaAlert` claim.
2. `process_vision_queue(telemetry)` — rule: `risk_level == "HIGH"` or `crowd_count > 5` or
   `crowd_density > 0.8` → emits a `VisionAlert` claim. (Vision data comes from `perception.py`.)
3. `pattern_detector(state)` — rule/lookup: turns claims into `active_predictions`
   (e.g. tachycardia → "Panic Attack Precursor" conf 0.91; vision → "Approaching Crowd" conf 0.88).
4. `memory_agent(state)` — **real semantic search** via Mem0 (`memory_client.search(...)`,
   `user_id="user_thtrang_06"`). Returns recalled past episodes.
5. `behavior_orchestrator(state)` — **the only true LLM agent.** Sends predictions + recalled
   memories to **Gemini 2.5 Flash** and asks it to pick exactly ONE of four directives:
   "Play grounding countdown audio through AirPods" / "Initiate a gentle physical nudge" /
   "Call emergency services immediately" / "Maintain passive navigation mode (Safe)".
6. `action_dispatcher(state)` — routing table on the chosen directive:
   - "emergency services" → `_trigger_bland_ai_call()` — **REAL** outbound call via Bland AI
     (`POST https://api.bland.ai/v1/calls`). Falls back to a printed simulation if
     `BLAND_API_KEY` / `CAREGIVER_PHONE` env vars are missing.
   - "airpods" → returns a status STRING only (no real TTS/Bluetooth).
   - "nudge" → returns a status STRING only (no real robot integration).
   - else → passive.
7. `retrospective_agent(state, resolution_success)` — `memory_client.add(...)` writes the
   episode outcome back into Mem0 (vector store + SQLite history).

`ui.py` provides the demo surface: sidebar API-key status (Gemini, W&B), three trigger buttons
(Safe / Approaching Crowd / Panic Attack), and a live YOLO camera toggle that auto-fires the
pipeline on HIGH risk with a 15-second cooldown.

`perception.py` is solid standalone CV: real YOLOv8n on the camera (`yolov8n.pt` is committed),
bounding boxes, scene classification (street/indoor/nature), crowd count, per-object velocity +
collision-risk heuristic, and a structured JSON payload contract (documented in its docstring).

---

## 3. Memory / persistence (the "triple save")

Configured in `app.py` via `Memory.from_config(mem0_config)`:

- **Vector store: Qdrant with `"path": ":memory:"`** → lives in RAM. Provides recall WITHIN a
  running session, but **starts empty on every restart**. ⚠️ This is the key correctness gap.
- **SQLite: `"history_db_path": "mem0_history.db"`** → on-disk raw text history. Persists across
  restarts. (File already contains a real row from a prior run.)
- **W&B Weave: `weave.init("nghiatr38-boston-university/project-argus-service-dog")`** plus
  `@weave.op()` on every function → cloud dashboard with timestamped decision traces. Works.

Mem0 LLM + embedder are both **Gemini** (`gemini-2.5-flash`, `gemini-embedding-001`,
`embedding_model_dims: 768`). Requires `GEMINI_API_KEY`.

---

## 4. Known bugs / gaps (fix list)

**Blocking / high priority**
1. **`requirements.txt` is incomplete** — it lists ONLY `ultralytics`, `opencv-python`,
   `easyocr`, `mediapipe`. It is MISSING everything the app actually imports:
   `streamlit`, `mem0ai`, `weave`, `google-genai`, `requests`, (and `langgraph` for the
   secondary path). A fresh machine cannot run the app. **Fix first.**
2. **Cross-session memory is broken** — Qdrant `:memory:` never rehydrates from
   `mem0_history.db`, so the "it learns over time" claim is false after a restart. Fix:
   either switch Qdrant to a persistent on-disk path, or add a startup step that reloads
   SQLite history into the vector store.

**Secondary path is dead code**
3. There are TWO competing definitions of the system. The `app.py` path (real). And a separate
   LangGraph skeleton: `biometric_agent.py`, `memory_agent.py`, `orchestrator.py`,
   `retrospective_agent.py`, `conflict_bus.py`, `test_agent.py`, `test_architecture.py`.
   This skeleton is **broken**: `biometric_agent.py` does `from conflict_bus import ConflictBusState, Claim`
   but `conflict_bus.py` defines NO `Claim` class. Decide: either delete this path, or repair
   and unify on LangGraph. Do NOT let it distract from the `app.py` demo.

**Pitch-vs-code gaps (not built)**
4. **No Place/GPS agent** — the pitched "geography → memory → updated geography" loop and
   per-location risk scoring do not exist.
5. **No real Voice agent** — no TTS/STT, no tone-by-urgency, no "interrupt itself." AirPods
   output is a printed string.
6. **No Software agent** — no calendar / reminders / contacts (MCP) integration.
7. **Only 1 of ~6 stages is truly LLM-agentic** (the orchestrator). The others are
   deterministic `if`/lookup functions. If the goal is "multiple AI agents," the pattern
   detector is the best candidate to upgrade to real LLM reasoning.

---

## 5. Environment variables required

```
GEMINI_API_KEY=...        # required: orchestrator + Mem0 llm/embedder
WANDB_API_KEY=...          # for Weave cloud tracing
BLAND_API_KEY=...          # optional: enables REAL emergency phone calls
CAREGIVER_PHONE=...        # optional: destination number for the real call
```
Note: there is currently **no `.env.example`** and **no `.env` loading** (no `python-dotenv`
import). Adding both would help reproducibility.

---

## 6. File map

| File | Status | Role |
|------|--------|------|
| `app.py` | ✅ live | The real pipeline + all agent functions + Mem0/Weave/Bland config |
| `ui.py` | ✅ live | Streamlit control panel + live YOLO camera loop |
| `perception.py` | ✅ live | YOLOv8 detection, tracking, JSON payload contract |
| `run_simulation.py` | ✅ works | Headless CLI run of the app.py pipeline (panic scenario) |
| `requirements.txt` | ⚠️ broken | Incomplete — missing most deps |
| `mem0_history.db` | ✅ data | SQLite history (has a real prior episode row) |
| `yolov8n.pt` | ✅ asset | YOLO weights (committed) |
| `conflict_bus.py` | ⚠️ dead | TypedDict for LangGraph path; missing `Claim` class |
| `biometric_agent.py` | ❌ broken | LangGraph node; imports nonexistent `Claim` |
| `memory_agent.py` | ⚠️ dead | LangGraph version (older, Mem0 defaults) |
| `orchestrator.py` | ⚠️ dead | LangGraph version (rule-based, not Gemini) |
| `retrospective_agent.py` | ⚠️ dead | LangGraph version |
| `test_agent.py` | ❌ broken | Exercises broken biometric_agent |
| `test_architecture.py` | ⚠️ dead | Builds the LangGraph graph |
| `README.md` | ⚠️ stub | Just "# multi-agent" |

---

## 7. How to run (once requirements are fixed)

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...   # plus optional WANDB_API_KEY, BLAND_API_KEY, CAREGIVER_PHONE
streamlit run ui.py         # full UI + camera demo
# or headless:
python run_simulation.py
```

---

## 8. Suggested build plan for Claude Code (in priority order)

1. **Fix `requirements.txt`** to include every actual import: pin `streamlit`, `mem0ai`,
   `weave`, `google-genai`, `requests`, `python-dotenv`, plus existing CV deps. Add
   `langgraph` only if keeping the secondary path. Verify a clean install in a fresh venv.
2. **Add `.env.example` + dotenv loading** in `app.py` for reproducibility.
3. **Fix cross-session memory:** make Qdrant persistent OR rehydrate from `mem0_history.db`
   on startup so recall survives restarts. Add a tiny test that proves recall after restart.
4. **Decide the LangGraph path:** either (a) delete the broken/dead files and keep `app.py`
   as the single source of truth, or (b) repair `conflict_bus.py` (add `Claim`) and migrate
   `app.py`'s pipeline into a real LangGraph graph. Recommend (a) for hackathon speed.
5. **Upgrade `pattern_detector` to a real LLM agent** (Gemini) so the "multiple AI agents"
   claim is literally true, while keeping a deterministic fallback for reliability.
6. **Write a real `README.md`**: architecture diagram, setup, env vars, demo script,
   honest "what works / what's simulated" section.
7. (Stretch) Build out one pitched-but-missing agent — **Voice** (real TTS to output) is the
   highest-impact for a live demo; **Place/GPS** is the most novel.

## 9. Important demo cautions
- The Bland AI emergency path places a **real phone call** when keys are set and the
  orchestrator chooses "emergency." Decide intentionally whether to arm it for the demo.
- Don't claim "six autonomous AI agents each reasoning independently" — code shows one true
  LLM agent (orchestrator) + one real retrieval agent (memory) + deterministic functions.
  Safe framing: "a multi-agent system coordinating over a shared conflict bus, orchestrated
  by an LLM that makes the final safety decision."
