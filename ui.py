"""Unleash — COMMAND CENTER operator dashboard.

Tactical HUD Streamlit shell wired to the live multi-agent pipeline (app.py),
the YOLOv8 perception loop (perception.py), and the voice / location agents.
Every output is restyled through hud_theme; the companion HUD mirrors live data.

Run:  streamlit run ui.py
"""

from __future__ import annotations

import os
import time
import html

import streamlit as st
import time
import os
import json
import traceback
from app import (
    process_biometrics,
    process_vision_queue,
    multimodal_fusion_agent,
    pattern_detector, 
    memory_agent, 
    behavior_orchestrator, 
    action_dispatcher,
    retrospective_agent,
)

# ---------------------------------------------------------------------------
# Page config + theme — MUST be first Streamlit calls.
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


_init_state()


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


# ---------------------------------------------------------------------------
# Shared pipeline runner. Returns the full result dict and persists it.
# `arm_real_calls` gates the live Bland AI emergency path.
# ---------------------------------------------------------------------------
def run_pipeline(telemetry: dict, arm_real_calls: bool = False) -> dict:
    """Run the full conflict-bus pipeline for one telemetry payload.

    Mirrors execute_scenario from the original ui.py, but returns structured
    data instead of rendering. Real emergency calls only fire when armed.
    """
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
        "final_action": "Maintain passive navigation mode (Safe)",
        # Thread the live-call gate through state (thread-safe, no global env mutation).
        "arm_real_calls": arm_real_calls,
    }
    
    with st.status("Executing Multi-Agent Pipeline...", expanded=True) as status:
        try:
            st.write("📡 **[1/4]** Evaluating raw telemetry streams...")
            bio_output = process_biometrics(telemetry)
            vision_output = process_vision_queue(telemetry)
            
            # Merge claims from both ingest agents
            claims = bio_output.get("active_claims", []) + vision_output.get("active_claims", [])
            state["active_claims"] = claims
            
            st.write("🌌 **[1.5/4]** ImageBind processing multi-modal environmental vectors...")
            fusion_output = multimodal_fusion_agent(state)
            state["environmental_embedding"] = fusion_output.get("environmental_embedding", [0.0]*1024)
            
            st.write("🔍 **[2/4]** Pattern Detector analyzing baseline variations...")
            pattern_output = pattern_detector(state)
            state["active_predictions"] = pattern_output.get("active_predictions", [])
            
            # Explicitly display the predictions on the UI so judges can read the baseline math!
            for pred in state["active_predictions"]:
                st.info(f"**Pattern Detected:** {pred['description']}")
                
            st.write("🧠 **[3/4]** Memory Agent querying Mem0 episodic context...")
            mem_output = memory_agent(state)
            state["retrieved_memories"] = mem_output.get("retrieved_memories", [])
            
            st.write("⚖️ **[4/5]** Behavior Orchestrator resolving action paths...")
            orch_output = behavior_orchestrator(state)
            state["final_action"] = orch_output.get("final_action", "Maintain passive navigation mode (Safe)")
            
            st.write("⚡ **[5/5]** Action Dispatcher routing hardware APIs...")
            dispatch_output = action_dispatcher(state)
            
            status.update(label="Pipeline Execution Complete!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Pipeline execution failed: {type(e).__name__}: {e}")
            st.code(traceback.format_exc(), language="text")
            state["final_action"] = "Error executing action"
            dispatch_output = {"execution_status": "Failed to dispatch action"}
            # Keep the box expanded on failure so the real traceback is visible.
            status.update(label="Pipeline Execution Failed", state="error", expanded=True)
    
    # Display the final execution
    st.success(f"### Final Executed Directive:\n**{state['final_action']}**")
    
    # Simulated physical response
    st.info(f"**[HARDWARE API TRIGGERED:]** {dispatch_output.get('execution_status')}")
    
    # Async memory compaction simulation
    with st.spinner("Retrospective Agent updating long-term episodic memory..."):
        retrospective_agent(state, resolution_success=True)
        st.toast('Episode saved to Mem0 Vector Database!', icon='💾')

    # 2. Pattern detection (deterministic, safe).
    predictions = []
    try:
        from perception import vision_generator
        st.subheader("Live YOLOv8 Inference")
        
        col_cam, col_status = st.columns([2, 1])
        cam_placeholder = col_cam.empty()
        status_placeholder = col_status.empty()
        
        last_trigger_time = 0
        ANALYSIS_INTERVAL = 5  # Capture & analyze one frame for danger every 5 seconds
        analysis_placeholder = col_cam.empty()

        for frame_rgb, payload in vision_generator(camera_index=cam_index):
            if frame_rgb is not None:
                # Use use_container_width to fix deprecation warning
                cam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            if payload:
                if "error" in payload:
                    st.error(payload["error"])
                    break

                live_hr_display = "No recent data"
                if os.path.exists("live_biometrics.json"):
                    try:
                        with open("live_biometrics.json", "r") as f:
                            live_bio = json.load(f)
                            if time.time() - live_bio.get("timestamp", 0) < 60:
                                hr = live_bio.get("heart_rate")
                                if hr is not None:
                                    live_hr_display = f"❤️ {hr} BPM"
                    except Exception:
                        pass

                current_time = time.time()
                time_to_next = int(ANALYSIS_INTERVAL - (current_time - last_trigger_time))
                status_placeholder.info(
                    f"**Scene**: {payload['scene']}\n\n"
                    f"**Crowd Count**: {payload['crowd_count']}\n\n"
                    f"**Risk Level**: {payload['risk_level']}\n\n"
                    f"**Live HR**: {live_hr_display}\n\n"
                    f"**Next scan in**: {max(time_to_next, 0)}s"
                )

                # Capture & analyze the current frame every ANALYSIS_INTERVAL seconds,
                # regardless of risk level. The pipeline acts on whatever danger it finds
                # (and falls back to passive/safe mode when the scene is clear).
                if current_time - last_trigger_time >= ANALYSIS_INTERVAL:
                    with analysis_placeholder.container():
                        st.caption(f"🔎 Frame captured at {time.strftime('%H:%M:%S')} — analyzing for danger...")

                        # Surface what the vision model is currently seeing
                        st.write(f"**Scene:** {payload.get('scene', 'unknown')} | **Risk:** {payload.get('risk_level', 'LOW')}")
                        if payload.get("threats"):
                            st.error(f"⚠️ **THREATS:** {', '.join([t['label'] for t in payload['threats']])}")
                        if payload.get("hazards"):
                            st.warning(f"🚧 **HAZARDS:** {', '.join([h['label'] for h in payload['hazards']])}")
                        if payload.get("text_detected"):
                            st.info(f"📄 **OCR TEXT:** {', '.join(payload['text_detected'])}")

                        # Attempt to load the live Apple HealthKit stream data
                        live_hr = 85  # fallback
                        try:
                            if os.path.exists("live_biometrics.json"):
                                with open("live_biometrics.json", "r") as f:
                                    live_bio = json.load(f)
                                    # Only use data if it's fresh (less than 60 seconds old)
                                    if time.time() - live_bio.get("timestamp", 0) < 60:
                                        if "heart_rate" in live_bio:
                                            live_hr = live_bio["heart_rate"]
                                            st.success(f"❤️ Live Apple Health Sync: {live_hr} BPM")
                        except Exception:
                            pass

                        full_payload = {
                            "heart_rate": live_hr,
                            "hrv": 45,
                            **payload
                        }
                        execute_scenario("Live 5s Vision & Health Scan", full_payload)

                    last_trigger_time = time.time()

    except ImportError:
        st.error("Missing CV dependencies. Please run: pip install opencv-python ultralytics")
