"""
Project Unleash — Command Center UI (HUD).

UI layer ported from Phillip's `phillyv2` branch, rewired onto OUR existing
backend. No backend function is modified: this file only *calls* the pipeline
functions we already have (app.py) and the local YOLO vision_generator
(perception.py). Where Phillip's UI depended on his backend (his Perception
class, his Assistant), those paths are dropped or degraded gracefully.

Key safety adaptation: our action_dispatcher always places a REAL Bland AI
emergency call for the "Call emergency services" directive. The sidebar
"ARM REAL CALLS" toggle is honored at the UI level here (run_pipeline), so the
live call only fires when explicitly armed — without changing the backend.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Disable Mem0 telemetry BEFORE app import (app.py imports mem0).
os.environ["MEM0_ENABLE_TELEMETRY"] = "False"
os.environ["MEM0_TELEMETRY"] = "False"

import time
import html

import streamlit as st
import streamlit.components.v1 as components

import hud_theme as hud
from companion_hud import render_companion_hud
import geo_map

# OUR backend pipeline. `pattern_detector` is our `predictive_forecaster`
# (renamed in our app.py) — aliased so the ported UI code reads unchanged.
from app import (
    process_biometrics,
    process_vision_queue,
    multimodal_fusion_agent,
    predictive_forecaster as pattern_detector,
    memory_agent,
    behavior_orchestrator,
    action_dispatcher,
    retrospective_agent,
)

# ---------------------------------------------------------------------------
# Page config + theme — MUST be the first Streamlit calls.
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Unleash — Command Center",
    page_icon="🐕",
)
hud.inject_theme()


# ---------------------------------------------------------------------------
# Session state defaults so every tab renders before any run.
# ---------------------------------------------------------------------------
def _init_state() -> None:
    if "last" not in st.session_state:
        st.session_state.last = {
            "claims": [],
            "predictions": [],
            "memories": [],
            "action": "Maintain passive navigation mode (Safe)",
            "dispatch": "[SYSTEM] Passive observation maintained. No external API triggered.",
            "telemetry": {"heart_rate": 72, "hrv": 65, "risk_level": "LOW", "crowd_count": 0},
            "hr_series": [72, 71, 73, 72, 70, 72],
        }
    if "voice_line" not in st.session_state:
        st.session_state.voice_line = "All clear. I'm here with you."
    if "last_auto_trigger" not in st.session_state:
        st.session_state.last_auto_trigger = 0.0
    if "last_agent_analysis" not in st.session_state:
        st.session_state.last_agent_analysis = 0.0
    if "last_spoken" not in st.session_state:
        st.session_state.last_spoken = ""
    if "voice_name" not in st.session_state:
        st.session_state.voice_name = "Samantha"
    if "voice_rate" not in st.session_state:
        st.session_state.voice_rate = 180


_init_state()

# ---------------------------------------------------------------------------
# LIVE TELEMETRY INJECTION
# Automatically pull real Apple Watch data if the simulator is running it.
# ---------------------------------------------------------------------------
if os.path.exists("live_biometrics.json"):
    import json
    try:
        with open("live_biometrics.json", "r") as f:
            live_bio = json.load(f)
            hr = live_bio.get("heart_rate")
            if hr is not None:
                st.session_state.last["telemetry"]["heart_rate"] = hr
    except Exception:
        pass


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


# ---------------------------------------------------------------------------
# Shared pipeline runner. Mirrors our original execute_scenario order
# (biometrics → vision → fusion → forecaster → memory → orchestrator →
# dispatch → retrospective) and persists the result for cross-tab sharing.
# `arm_real_calls` gates the live Bland AI emergency path AT THE UI LEVEL.
# ---------------------------------------------------------------------------
def run_pipeline(telemetry: dict, arm_real_calls: bool = False) -> dict:
    # 1. Ingest — biometrics + vision claims merged onto the bus.
    try:
        bio = process_biometrics(telemetry).get("active_claims", [])
    except Exception as exc:  # noqa: BLE001 — never crash the dashboard
        bio = []
        st.warning(f"Biometric agent unavailable: {exc}")
    try:
        vis = process_vision_queue(telemetry).get("active_claims", [])
    except Exception as exc:  # noqa: BLE001
        vis = []
        st.warning(f"Vision agent unavailable: {exc}")
    claims = bio + vis

    state = {
        "current_telemetry": telemetry,
        "active_claims": claims,
        "active_predictions": [],
        "retrieved_memories": [],
        "environmental_embedding": [0.0] * 1024,
        "final_action": "Maintain passive navigation mode (Safe)",
    }

    # 1b. Multimodal fusion → environmental embedding used by memory recall.
    try:
        fusion = multimodal_fusion_agent(state)
        state["environmental_embedding"] = fusion.get("environmental_embedding", [0.0] * 1024)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Fusion agent unavailable: {exc}")

    # 2. Pattern detection / forecasting (deterministic, safe).
    predictions = []
    try:
        predictions = pattern_detector(state).get("active_predictions", [])
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Pattern detector unavailable: {exc}")
    state["active_predictions"] = predictions

    # 3. Memory recall.
    memories = []
    try:
        memories = memory_agent(state).get("retrieved_memories", [])
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Memory agent unavailable: {exc}")
    state["retrieved_memories"] = memories

    # 4. Orchestrator decision (degrade to Safe on failure).
    action = "Maintain passive navigation mode (Safe)"
    try:
        action = behavior_orchestrator(state).get("final_action", action)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Orchestrator unavailable: {exc}")
    state["final_action"] = action

    # 5. Action dispatch. SAFETY: our action_dispatcher places a REAL Bland AI
    #    call for the emergency directive regardless of any gate, so we gate the
    #    live path HERE (UI level) instead of modifying the backend function.
    dispatch = "[SYSTEM] Passive observation maintained. No external API triggered."
    try:
        if "emergency services" in action.lower() and not arm_real_calls:
            dispatch = ("[SIMULATED] Emergency directive selected — DISARMED. No live "
                        "Bland AI call placed. Toggle 'ARM REAL CALLS' in the sidebar to enable.")
        else:
            dispatch = action_dispatcher(state).get("execution_status", dispatch)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Action dispatcher unavailable: {exc}")

    # 6. Retrospective write-back (best effort).
    try:
        retrospective_agent(state, resolution_success=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Retrospective agent unavailable: {exc}")

    # Persist for cross-tab + cross-rerun sharing.
    hr = telemetry.get("heart_rate", st.session_state.last["telemetry"].get("heart_rate", 72))
    hr_series = (st.session_state.last.get("hr_series", []) + [hr])[-30:]
    st.session_state.last = {
        "claims": claims,
        "predictions": predictions,
        "memories": memories,
        "action": action,
        "dispatch": dispatch,
        "telemetry": telemetry,
        "hr_series": hr_series,
    }
    st.session_state.voice_line = action
    return st.session_state.last


# ---------------------------------------------------------------------------
# SIDEBAR — brand, nav, source, toggles, API status, arm switch.
# ---------------------------------------------------------------------------
def _api_status_line(label: str, env_var: str) -> str:
    connected = bool(os.environ.get(env_var))
    state = "CONNECTED" if connected else "MISSING"
    variant = "live" if connected else "danger"
    return (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'margin:6px 0;font-family:var(--font-mono);font-size:10px;letter-spacing:1px;">'
        f'<span style="color:var(--text-secondary)">{_esc(label)}</span>'
        f'{hud.pill(state, variant)}</div>'
    )


with st.sidebar:
    st.markdown(
        hud.reactor_header("SERVICE DOG", "UNLEASH // OPERATOR", hud.pill("ONLINE", "live")),
        unsafe_allow_html=True,
    )
    st.markdown(hud.hud_sep("NAVIGATION"), unsafe_allow_html=True)
    nav = st.radio(
        "Console view",
        ["Live Dashboard", "Explore Crowd", "Place Memory", "Audio Config"],
        label_visibility="collapsed",
    )

    st.markdown(hud.hud_sep("PERCEPTION SOURCE"), unsafe_allow_html=True)
    source = st.selectbox(
        "Perception source",
        ["YOLOv8 // CONTINUITY CAM", "YOLOv8 // WEBCAM"],
        label_visibility="collapsed",
    )

    st.markdown(hud.hud_sep("ASSIST MODULES"), unsafe_allow_html=True)
    voice_assist = st.toggle("Voice Assist", value=True, help="Speak the directive aloud (TTS).")
    pinpoint_mapping = st.toggle("Pinpoint Mapping", value=True, help="Resolve live GPS / place.")
    gap_sync_trail = st.toggle("Gap Sync Trail", value=False, help="Sync episode trail to memory.")

    st.markdown(hud.hud_sep("API STATUS"), unsafe_allow_html=True)
    st.markdown(_api_status_line("GEMINI", "GEMINI_API_KEY"), unsafe_allow_html=True)
    st.markdown(_api_status_line("W&B", "WANDB_API_KEY"), unsafe_allow_html=True)
    st.markdown(_api_status_line("BLAND", "BLAND_API_KEY"), unsafe_allow_html=True)

    st.markdown(hud.hud_sep("SAFETY"), unsafe_allow_html=True)
    arm_real_calls = st.toggle(
        "ARM REAL CALLS",
        value=False,
        help="When OFF, the emergency path is always SIMULATED — no live Bland AI call.",
    )
    if arm_real_calls:
        st.markdown(hud.pill("LIVE CALLS ARMED", "danger"), unsafe_allow_html=True)
    else:
        st.markdown(hud.pill("CALLS SAFE / SIMULATED", "cyan"), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HEADER ROW — title + subtitle + LIVE pill.
# ---------------------------------------------------------------------------
head_l, head_r = st.columns([4, 1])
with head_l:
    st.markdown(
        '<h1 style="margin-bottom:2px">COMMAND CENTER</h1>'
        '<p style="font-family:var(--font-mono);font-size:11px;letter-spacing:2px;'
        'color:var(--hud-muted);margin-top:0">UNLEASH MULTI-AGENT SERVICE DOG // REAL-TIME OPS</p>',
        unsafe_allow_html=True,
    )
with head_r:
    st.markdown(
        f'<div style="text-align:right;padding-top:14px">{hud.pill("LIVE // DEPLOY", "live")}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# METRIC CARDS — 4 across.
# ---------------------------------------------------------------------------
last = st.session_state.last
active_claims = last.get("claims", [])
telemetry = last.get("telemetry", {})
hr_now = telemetry.get("heart_rate", 72)
risk_now = telemetry.get("risk_level", "LOW")
_risk_score = {"LOW": 18, "MED": 55, "HIGH": 88}.get(risk_now, 18)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        hud.stat_card(str(len(active_claims)), "ALERTS ACTIVE", "BUS",
                      "danger" if active_claims else "cyan"),
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(hud.stat_card("91", "ALL SYSTEMS", "OK", "success"), unsafe_allow_html=True)
with m3:
    st.markdown(
        hud.stat_card(str(_risk_score), "THREAT INDEX", risk_now,
                      "danger" if risk_now == "HIGH" else "warning"),
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        hud.stat_card(str(hr_now), "HEART RATE", "BPM",
                      "danger" if hr_now > 100 else "success"),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helpers for the claims / directive / voice panels.
# ---------------------------------------------------------------------------
def _claim_severity(ctype: str) -> str:
    return {"TachycardiaAlert": "danger", "VisionAlert": "warning"}.get(ctype, "cyan")


def render_claims_panel(claims: list) -> None:
    if not claims:
        st.markdown(
            hud.panel("CONFLICT BUS", "NO ACTIVE CLAIMS",
                      '<span style="color:var(--hud-success)">● ALL CALM // BUS CLEAR</span>'),
            unsafe_allow_html=True,
        )
        return
    rows = "".join(
        hud.claim_row(c.get("source", "AGENT"), c.get("type", "Claim"),
                      str(c.get("value", "")), _claim_severity(c.get("type", "")))
        for c in claims
    )
    st.markdown(
        hud.panel("CONFLICT BUS", f"{len(claims)} ACTIVE CLAIM(S)", rows,
                  hud.pill("FIRING", "danger")),
        unsafe_allow_html=True,
    )


def render_directive_panel(action: str, dispatch: str) -> None:
    body = (
        '<div role="status" aria-live="assertive" aria-atomic="true">'
        '<span class="vis-hidden">Current directive: </span>'
        f'<div style="font-size:14px;color:var(--text-primary);line-height:1.4">{_esc(action)}</div>'
        f'<div style="font-family:var(--font-mono);font-size:10px;letter-spacing:1px;'
        f'color:var(--hud-label);margin-top:6px">{_esc(dispatch)}</div>'
        '</div>'
    )
    st.markdown(hud.panel("DIRECTIVE", "ORCHESTRATOR DECISION", body,
                          hud.pill("EXECUTED", "cyan")), unsafe_allow_html=True)


def render_voice_output(line: str) -> None:
    body = (
        '<div role="status" aria-live="assertive" aria-atomic="true">'
        '<span class="vis-hidden">Spoken directive: </span>'
        f'<div style="font-size:15px;color:var(--hud-text);font-style:italic">'
        f'&ldquo;{_esc(line)}&rdquo;</div></div>'
    )
    st.markdown(hud.panel("VOICE OUTPUT", "SPOKEN TO HANDLER", body,
                          hud.pill("AUDIO", "live")), unsafe_allow_html=True)


def speak_line(line: str) -> None:
    """Synthesize + play the spoken line when Voice Assist is on. Text always shown."""
    if not voice_assist or not line:
        return
    try:
        from voice_agent import VoiceAgent
        va = VoiceAgent(
            voice=st.session_state.get("voice_name", "Samantha"),
            rate=int(st.session_state.get("voice_rate", 180)),
        )
        data, mime = va.synthesize(line)
        if data:
            st.audio(data, format=mime, autoplay=True)
        else:
            st.caption("Voice synthesis produced no audio (install `gtts`, or run on macOS for `say`).")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Voice synthesis unavailable: {exc}")


# ---------------------------------------------------------------------------
# PERCEPTION SERVER HELPERS (lifecycle + live stream)
# The server is a separate FastAPI process (perception_server.py). We start it
# on demand, restart it when the camera index changes, and pull frames over
# plain HTTP. `source` (Continuity vs Webcam) is a hint shown to the user; the
# authoritative device is the numeric camera index from the sidebar.
# ---------------------------------------------------------------------------
PERCEPTION_PORT = 8000
PERCEPTION_URL = f"http://localhost:{PERCEPTION_PORT}"


def _server_running() -> bool:
    import requests
    try:
        requests.get(f"{PERCEPTION_URL}/payload", timeout=1)
        return True
    except Exception:  # noqa: BLE001
        return False


def ensure_perception_server(cam_index: int) -> bool:
    """Start (or restart) the perception microservice for `cam_index`.

    Returns True once the server answers on /payload. Restarts the process if
    the requested camera index differs from the one it was launched with.
    """
    import subprocess
    import sys
    import time

    proc = st.session_state.get("perception_proc")
    running_cam = st.session_state.get("perception_cam")

    # Camera changed since we launched it → tear the old process down first.
    if proc is not None and running_cam is not None and running_cam != cam_index:
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        st.session_state.perception_proc = None
        st.session_state.perception_cam = None
        time.sleep(1.0)

    # A server we didn't spawn (or already spawned) is already up → reuse it.
    if _server_running():
        st.session_state.perception_cam = cam_index
        return True

    proc = subprocess.Popen(
        [sys.executable, "perception_server.py", "--camera", str(cam_index),
         "--port", str(PERCEPTION_PORT)]
    )
    st.session_state.perception_proc = proc
    st.session_state.perception_cam = cam_index

    with st.status(f"Starting Perception Server (Cam {cam_index})…", expanded=True) as status:
        st.write("Booting background daemon…")
        st.write("Loading YOLOv8 weights (first start can take ~20s)…")
        for _ in range(40):
            if _server_running():
                status.update(label="Perception Server ready.", state="complete")
                return True
            time.sleep(1)
        status.update(
            label="Perception Server did not respond. Check the camera index "
                  "(`python list_cams.py`) and Privacy ▸ Camera permissions.",
            state="error",
        )
    return _server_running()


def render_camera_video() -> None:
    """Embed the perception server's MJPEG stream as a browser-native <img>.

    This renders the live video DIRECTLY in the browser with no Streamlit
    reruns, so the feed tracks smoothly and never blinks. Bounding boxes are
    drawn server-side by YOLOv8.
    """
    components.html(
        f"""
        <div style="width:100%;background:#05080e;border-radius:10px;overflow:hidden;
                    display:flex;align-items:center;justify-content:center;height:100%;">
          <img src="{PERCEPTION_URL}/stream" alt="Live perception feed"
               style="width:100%;height:100%;object-fit:contain;display:block;"
               onerror="this.style.opacity=0.2;" />
        </div>
        """,
        height=440,
    )


# Cadence (seconds) at which the agents sample a frame for full analysis.
ANALYZE_EVERY = 5


def _agents_badge(label: str, variant: str = "cyan") -> str:
    return (
        '<div style="margin:8px 0 4px 0;font-family:var(--font-mono);font-size:10px;'
        'letter-spacing:1px;display:flex;align-items:center;gap:8px;">'
        '<span style="color:var(--text-secondary)">AGENT PIPELINE</span>'
        f'{hud.pill(label, variant)}</div>'
    )


@st.fragment(run_every=2.0)
def perception_overlay_fragment() -> None:
    """Poll the detection payload on its OWN timer and render the live CV
    readout (scene / risk / detection chips). Every ANALYZE_EVERY seconds it
    also samples the current frame and runs the FULL agent pipeline on it,
    showing an "analyzing" indicator while it works.

    Because this is a fragment, only this region reruns — the MJPEG video and
    the rest of the dashboard are never interrupted (no full-page rerun, so the
    live CV stream keeps tracking smoothly the whole time).
    """
    import time
    import requests

    arm_real_calls = bool(st.session_state.get("arm_real_calls_flag", False))
    status_box = st.empty()
    info_box = st.empty()
    chip_box = st.empty()

    try:
        payload_resp = requests.get(f"{PERCEPTION_URL}/payload", timeout=3)
    except requests.exceptions.RequestException:
        info_box.warning("Perception Server not responding — toggle Connect off then on to restart it.")
        return
    if payload_resp.status_code != 200:
        return

    payload = payload_resp.json()
    if "error" in payload:
        info_box.warning(f"Camera: {payload['error']}")
        return
    if payload.get("status") == "initializing" or "scene" not in payload:
        info_box.info("Live — waiting for the first detection…")
        return

    # --- Live CV readout (refreshes every 2s, independent of agent analysis) ---
    info_box.markdown(
        '<div role="status" aria-live="polite" '
        'style="font-family:var(--font-mono);font-size:11px;'
        'letter-spacing:1px;color:var(--hud-cyan)">'
        f'SCENE {_esc(payload.get("scene","?")).upper()} · '
        f'CROWD {_esc(payload.get("crowd_count",0))} · '
        f'RISK {_esc(payload.get("risk_level","LOW"))}</div>',
        unsafe_allow_html=True,
    )
    items = [
        {"text": f'{d.get("label","obj").upper()} {d.get("side","")}',
         "variant": "danger" if d.get("collision_risk", 0) > 0.6 else None}
        for d in payload.get("detections", [])[:6]
    ] + [
        {"text": f'HAZARD {h.get("label","hazard").upper()}', "variant": "warning"}
        for h in payload.get("hazards", [])[:4]
    ]
    if items:
        chip_box.markdown(
            '<div role="status" aria-live="polite">' + hud.chips(items) + '</div>',
            unsafe_allow_html=True,
        )

    # --- Periodic agent analysis: sample this frame every ANALYZE_EVERY s ---
    now = time.time()
    elapsed = now - st.session_state.last_agent_analysis
    has_content = bool(payload.get("detections") or payload.get("hazards") or payload.get("threats"))

    if elapsed >= ANALYZE_EVERY and has_content:
        st.session_state.last_agent_analysis = now
        status_box.markdown(_agents_badge("● ANALYZING FRAME", "warning"), unsafe_allow_html=True)
        # The spinner stays visible for the whole (blocking) pipeline call, so the
        # handler sees the agents are working on the sampled frame.
        with st.spinner("🧠 Agents analyzing sampled frame…"):
            telem = dict(st.session_state.last.get("telemetry", {}))
            frame_telemetry = {
                "heart_rate": telem.get("heart_rate", 72),
                "hrv": telem.get("hrv", 65),
                **payload,
            }
            run_pipeline(frame_telemetry, arm_real_calls)
            # Speak only when the directive actually changed (avoid repeating every 5s).
            line = st.session_state.voice_line
            if line and line != st.session_state.last_spoken:
                speak_line(line)
                st.session_state.last_spoken = line
        status_box.markdown(_agents_badge("ANALYSIS COMPLETE", "live"), unsafe_allow_html=True)
    else:
        remaining = max(0, int(round(ANALYZE_EVERY - elapsed)))
        if has_content:
            status_box.markdown(_agents_badge(f"NEXT SAMPLE IN {remaining}s", "cyan"),
                                unsafe_allow_html=True)
        else:
            status_box.markdown(_agents_badge("IDLE · NO DETECTIONS", "cyan"),
                                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CONFLICT BUS / DIRECTIVE / VITALS panel (shared; live-refreshable).
# ---------------------------------------------------------------------------
def render_bus() -> None:
    last = st.session_state.last
    render_claims_panel(last.get("claims", []))
    render_directive_panel(last.get("action", ""), last.get("dispatch", ""))
    render_voice_output(st.session_state.voice_line)
    telem = last.get("telemetry", {})
    hr_series = last.get("hr_series", [])
    if hr_series:
        st.markdown(hud.hud_sep("HEART RATE // RECENT"), unsafe_allow_html=True)
        st.area_chart({"BPM": hr_series}, height=140, color="#00D4FF")
        st.markdown(hud.hud_sep("HRV // RECENT"), unsafe_allow_html=True)
        hrv_series = [telem.get("hrv", 65)] * max(1, len(hr_series))
        st.area_chart({"HRV": hrv_series}, height=120, color="#34E0CE")


# Same panel, on a 2s timer — used while streaming so agent results appear live.
live_bus_fragment = st.fragment(run_every=2.0)(render_bus)


# ---------------------------------------------------------------------------
# PAGES — each maps to one sidebar NAVIGATION entry.
# ---------------------------------------------------------------------------
def page_live_dashboard() -> None:
    col_cam, col_bus = st.columns([3, 2])

    with col_cam:
        st.markdown(hud.hud_sep("PERCEPTION // LIVE"), unsafe_allow_html=True)

        st.caption("Manual scenario triggers")
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Safe", use_container_width=True):
            run_pipeline({"heart_rate": 72, "hrv": 65, "crowd_density": 0.1,
                          "fall_detected": False, "risk_level": "LOW", "crowd_count": 1},
                         arm_real_calls)
            speak_line(st.session_state.voice_line)
            st.rerun()
        if b2.button("Crowd", use_container_width=True):
            run_pipeline({"heart_rate": 85, "hrv": 45, "crowd_density": 0.9,
                          "fall_detected": False, "risk_level": "MED", "crowd_count": 6},
                         arm_real_calls)
            speak_line(st.session_state.voice_line)
            st.rerun()
        if b3.button("Panic", use_container_width=True):
            run_pipeline({"heart_rate": 115, "hrv": 22, "crowd_density": 0.1,
                          "fall_detected": False, "risk_level": "LOW", "crowd_count": 0},
                         arm_real_calls)
            speak_line(st.session_state.voice_line)
            st.rerun()
        if b4.button("Emergency", use_container_width=True,
                     help="Severe fall + tachycardia. Live call only fires if ARM REAL CALLS is on."):
            run_pipeline({"heart_rate": 180, "hrv": 10, "crowd_density": 0.1,
                          "fall_detected": True, "risk_level": "HIGH", "crowd_count": 0},
                         arm_real_calls)
            speak_line(st.session_state.voice_line)
            st.rerun()

        # ===== LIVE PERCEPTION SERVER (webcam / Continuity Camera) ====
        default_idx = 1 if "CONTINUITY" in source else 0
        cam_index = st.number_input(
            "Camera index", min_value=0, max_value=8, value=default_idx, step=1,
            help=f"Source: {source}. Run `python list_cams.py` to find your "
                 "iPhone/Continuity camera index (usually the highest one).",
        )

        enable_cam = st.checkbox(
            "🔴 Connect & Stream", key="enable_cam",
            help="Starts the perception microservice and streams its live video here.")

        # The overlay fragment reads the arm switch from session state.
        st.session_state.arm_real_calls_flag = arm_real_calls

        if enable_cam:
            if ensure_perception_server(int(cam_index)):
                render_camera_video()
                st.caption("Live perception feed (YOLOv8). Detections update below every 2s.")
                perception_overlay_fragment()
            else:
                st.error("Could not reach the Perception Server. See the status box above.")
        else:
            st.info("Camera disabled. Use the manual scenario buttons above to drive the pipeline.")

    with col_bus:
        st.markdown(hud.hud_sep("CONFLICT BUS // CLAIMS"), unsafe_allow_html=True)
        # While the camera is streaming, render the bus as a fragment so the
        # 5s agent analyses show up here live — without a full-page rerun that
        # would reload the video. Otherwise render it once (button-driven).
        if st.session_state.get("enable_cam"):
            live_bus_fragment()
        else:
            render_bus()


def page_explore_crowd() -> None:
    st.markdown(hud.hud_sep("AGENT MESH // STATUS"), unsafe_allow_html=True)
    agents = ["Perception", "Biometric", "Pattern", "Memory",
              "Orchestrator", "Action", "Retrospective"]
    chips_html = "".join(hud.agent_status(a, "ONLINE") for a in agents)
    st.markdown(f'<div class="chips">{chips_html}</div>', unsafe_allow_html=True)

    last = st.session_state.last
    a_left, a_right = st.columns(2)
    with a_left:
        st.markdown(hud.hud_sep("PREDICTIONS"), unsafe_allow_html=True)
        preds = last.get("predictions", [])
        if preds:
            body = "".join(
                f'<div style="margin:6px 0">'
                f'<span style="color:var(--hud-cyan)">▸ {_esc(p.get("description",""))}</span>'
                f'<span style="font-family:var(--font-mono);font-size:10px;color:var(--hud-muted);'
                f'margin-left:8px">CONF {int(p.get("confidence",0)*100)}%</span></div>'
                for p in preds
            )
        else:
            body = '<span style="color:var(--hud-success)">● NO ANOMALY PREDICTED</span>'
        st.markdown(hud.panel("PATTERN DETECTOR", "ANOMALY FORECAST", body), unsafe_allow_html=True)
    with a_right:
        st.markdown(hud.hud_sep("RECALLED MEMORIES"), unsafe_allow_html=True)
        mems = last.get("memories", [])
        if mems:
            body = "".join(f'<div style="margin:5px 0">◆ {_esc(m)}</div>' for m in mems[:6])
        else:
            body = '<span style="color:var(--hud-muted)">NO EPISODIC RECALL FOR THIS CONTEXT</span>'
        st.markdown(hud.panel("MEMORY AGENT", "MEM0 // EPISODIC", body), unsafe_allow_html=True)

    st.markdown(hud.hud_sep("COMPANION HUD // MIRROR"), unsafe_allow_html=True)
    telem = last.get("telemetry", {})
    preds = last.get("predictions", [])
    confidence = int(preds[-1].get("confidence", 0.91) * 100) if preds else 91
    detections = telem.get("detections", [])
    live_state = {
        "agents_online": 7,
        "confidence": confidence,
        "heart_rate": telem.get("heart_rate", 72),
        "hazards_near": len(detections),
        "heart_variant": "danger" if telem.get("heart_rate", 72) > 100 else "success",
        "hazards_variant": "warning" if detections else "success",
        "spoken": st.session_state.voice_line,
        "tone": "URGENT" if last.get("claims") else "CALM",
    }
    try:
        render_companion_hud(live_state)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Companion HUD unavailable: {exc}")


def page_place_memory() -> None:
    st.markdown(hud.hud_sep("LOCATION // LIVE MAP"), unsafe_allow_html=True)
    # Cache the LocationAgent so it survives the map's live-tracking reruns.
    if "location_agent" not in st.session_state:
        try:
            from location_agent import LocationAgent
            st.session_state["location_agent"] = LocationAgent()
        except Exception as exc:  # noqa: BLE001 — map still works with agent=None
            st.session_state["location_agent"] = None
            st.warning(f"LocationAgent unavailable, map uses browser GPS / default only: {exc}")
    try:
        geo_map.render_location_map(st.session_state["location_agent"])
    except Exception as exc:  # noqa: BLE001 — never let the map crash the console
        st.warning(f"Live location map unavailable: {exc}")

    st.markdown(hud.hud_sep("LOCATION // PLACE"), unsafe_allow_html=True)
    loc = st.session_state.get("location_agent")
    if loc is not None:
        try:
            st.write(loc.where_am_i())
            dest = st.text_input("Directions to", placeholder="e.g. Harvard Square")
            if dest:
                st.write(loc.directions(dest))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Location lookup unavailable: {exc}")

    st.markdown(hud.hud_sep("EPISODIC RECALL"), unsafe_allow_html=True)
    mems = st.session_state.last.get("memories", [])
    if mems:
        body = "".join(f'<div style="margin:5px 0">◆ {_esc(m)}</div>' for m in mems[:8])
    else:
        body = '<span style="color:var(--hud-muted)">NO EPISODIC RECALL FOR THIS CONTEXT</span>'
    st.markdown(hud.panel("PLACE MEMORY", "MEM0 // EPISODIC", body), unsafe_allow_html=True)


def page_audio_config() -> None:
    st.markdown(hud.hud_sep("VOICE // SETTINGS"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.voice_name = st.selectbox(
            "TTS voice (macOS `say`)",
            ["Samantha", "Daniel", "Karen", "Alex", "Tessa", "Moira", "Fred"],
            index=["Samantha", "Daniel", "Karen", "Alex", "Tessa", "Moira", "Fred"].index(
                st.session_state.get("voice_name", "Samantha")
            ),
            help="Used when gTTS is unavailable (offline) and we fall back to macOS speech.",
        )
    with c2:
        st.session_state.voice_rate = st.slider(
            "Speech rate (wpm)", min_value=120, max_value=260,
            value=int(st.session_state.get("voice_rate", 180)), step=10,
        )

    st.caption("Voice Assist is "
               + ("ON" if voice_assist else "OFF — toggle it in the sidebar to hear spoken directives."))

    # ---- TEXT → SPEECH test ----
    st.markdown(hud.hud_sep("SPEAK // TEST"), unsafe_allow_html=True)
    test_text = st.text_input("Say something as the service dog",
                              value=st.session_state.voice_line)
    if st.button("🔊 Speak", use_container_width=True):
        try:
            from voice_agent import VoiceAgent
            va = VoiceAgent(voice=st.session_state.voice_name,
                            rate=int(st.session_state.voice_rate))
            data, mime = va.synthesize(test_text)
            if data:
                st.audio(data, format=mime, autoplay=True)
            else:
                st.warning("No audio engine available. Install `gtts` (pip install gtts) "
                           "or run on macOS so the built-in `say` command is present.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Voice synthesis failed: {exc}")

    # ---- SPEECH → TEXT (talk to your dog) ----
    st.markdown(hud.hud_sep("LISTEN // TALK TO YOUR DOG"), unsafe_allow_html=True)
    if not os.environ.get("GEMINI_API_KEY"):
        st.warning("GEMINI_API_KEY missing — transcription is disabled. "
                   "Add it to .env to enable speech-to-text.")
    audio_val = st.audio_input("Record a command (e.g. 'are we safe to cross?')")
    if audio_val is not None:
        with st.spinner("Transcribing with Gemini…"):
            try:
                from voice_agent import VoiceAgent
                va = VoiceAgent(voice=st.session_state.voice_name,
                                rate=int(st.session_state.voice_rate))
                heard = va.transcribe_bytes(audio_val.getvalue(), suffix=".wav")
            except Exception as exc:  # noqa: BLE001
                heard = ""
                st.error(f"Transcription failed: {exc}")
        if heard:
            st.markdown(
                hud.panel("YOU SAID", "SPEECH → TEXT",
                          f'<div style="font-size:14px;color:var(--hud-text)">{_esc(heard)}</div>'),
                unsafe_allow_html=True,
            )
            reply = f"Understood. {heard}"
            st.session_state.voice_line = reply
            render_voice_output(reply)
            speak_line(reply)
        else:
            st.info("Didn't catch that. Try recording again (and check GEMINI_API_KEY).")


# ---------------------------------------------------------------------------
# NAVIGATION DISPATCH — render the page selected in the sidebar.
# ---------------------------------------------------------------------------
PAGES = {
    "Live Dashboard": page_live_dashboard,
    "Explore Crowd": page_explore_crowd,
    "Place Memory": page_place_memory,
    "Audio Config": page_audio_config,
}
PAGES.get(nav, page_live_dashboard)()
