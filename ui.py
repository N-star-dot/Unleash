import os
from dotenv import load_dotenv
load_dotenv()

os.environ["MEM0_ENABLE_TELEMETRY"] = "False"
os.environ["MEM0_TELEMETRY"] = "False"

import streamlit as st
import time
import os
import json
from app import (
    process_biometrics, 
    process_vision_queue,
    pattern_detector, 
    memory_agent, 
    behavior_orchestrator, 
    action_dispatcher,
    retrospective_agent
)

st.set_page_config(page_title="Unleash Control Panel", page_icon="🐕", layout="wide")

st.title("🐕 Project Unleash: Control Panel")
st.markdown("Simulate telemetry and environmental triggers to test the Unleash multi-agent pipeline.")

# Sidebar for configuration status
with st.sidebar:
    st.header("System Status")
    if os.environ.get("GEMINI_API_KEY"):
        st.success("Gemini API: CONNECTED")
    else:
        st.error("Gemini API: MISSING")
        
    if os.environ.get("WANDB_API_KEY"):
        st.success("Weights & Biases: CONNECTED")
    else:
        st.error("Weights & Biases: MISSING")

# Scenario execution function
def execute_scenario(scenario_name: str, telemetry: dict):
    st.subheader(f"Running Scenario: {scenario_name}")
    
    # Initialize the ConflictBusState
    state = {
        "current_telemetry": telemetry,
        "active_claims": [],
        "active_predictions": [],
        "retrieved_memories": [],
        "final_action": ""
    }
    
    with st.status("Executing Multi-Agent Pipeline...", expanded=True) as status:
        try:
            st.write("📡 **[1/4]** Evaluating raw telemetry streams...")
            bio_output = process_biometrics(telemetry)
            vision_output = process_vision_queue(telemetry)
            
            # Merge claims from both ingest agents
            claims = bio_output.get("active_claims", []) + vision_output.get("active_claims", [])
            state["active_claims"] = claims
            
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
            st.error(f"Pipeline execution failed: {str(e)}")
            state["final_action"] = "Error executing action"
            dispatch_output = {"execution_status": "Failed to dispatch action"}
            status.update(label="Pipeline Execution Failed", state="error", expanded=False)
    
    # Display the final execution
    st.success(f"### Final Executed Directive:\n**{state['final_action']}**")
    
    # Simulated physical response
    st.info(f"**[HARDWARE API TRIGGERED:]** {dispatch_output.get('execution_status')}")
    
    # Async memory compaction simulation
    with st.spinner("Retrospective Agent updating long-term episodic memory..."):
        retrospective_agent(state, resolution_success=True)
        st.toast('Episode saved to Mem0 Vector Database!', icon='💾')

# -------------------------------------------------------------
# Trigger Panel
# -------------------------------------------------------------
st.write("---")
st.header("Trigger Scenarios")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🟢 Safe Environment", use_container_width=True):
        payload = {
            "heart_rate": 72,
            "hrv": 65,
            "crowd_density": 0.1,
            "fall_detected": False
        }
        execute_scenario("Safe Environment (Normal HR, No Obstacles)", payload)

with col2:
    if st.button("🟡 Approaching Crowd", use_container_width=True):
        payload = {
            "heart_rate": 85,
            "hrv": 45,
            "crowd_density": 0.9, # Triggers ComputerVisionAgent
            "fall_detected": False
        }
        execute_scenario("Approaching Crowd (High-density objects)", payload)

with col3:
    if st.button("🔴 Panic Precursor", use_container_width=True):
        payload = {
            "heart_rate": 115, # Triggers BiometricAgent
            "hrv": 22,
            "crowd_density": 0.1,
            "fall_detected": False
        }
        execute_scenario("Panic Attack Onset (Elevated HR + Low HRV)", payload)

col4, col5, col6 = st.columns(3)
with col4:
    if st.button("🚨 Extreme Emergency (Call Caregiver)", use_container_width=True):
        payload = {
            "heart_rate": 180, # Extreme Tachycardia
            "hrv": 10,
            "crowd_density": 0.1,
            "fall_detected": True # User fell over
        }
        # We manually inject a severe state into the scenario
        execute_scenario("SEVERE SEIZURE/FALL DETECTED", payload)

# -------------------------------------------------------------
# Live Computer Vision Feed
# -------------------------------------------------------------
st.write("---")
st.header("Live Computer Vision Feed")
st.markdown("Enable this to run the local YOLOv8 object detection model. The feed will automatically trigger the LangGraph pipeline if a high-risk collision or crowd density is detected.")

if st.sidebar.checkbox("🔴 Enable Live YOLO Camera", help="Requires OpenCV and Ultralytics. High CPU usage."):
    cam_index = st.sidebar.number_input("Camera Index (0=Mac, 1=iPhone)", min_value=0, max_value=5, value=0)
    try:
        from perception import vision_generator
        st.subheader("Live YOLOv8 Inference")
        
        col_cam, col_status = st.columns([2, 1])
        cam_placeholder = col_cam.empty()
        status_placeholder = col_status.empty()
        
        last_trigger_time = 0
        COOLDOWN_SECONDS = 2  # Reduced to 2 seconds for smooth rapid analysis
        
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
                        
                status_placeholder.info(
                    f"**Scene**: {payload['scene']}\n\n"
                    f"**Crowd Count**: {payload['crowd_count']}\n\n"
                    f"**Risk Level**: {payload['risk_level']}\n\n"
                    f"**Live HR**: {live_hr_display}"
                )
                
                # Auto-trigger if danger is detected AND cooldown has passed
                if payload['risk_level'] == 'HIGH':
                    current_time = time.time()
                    if current_time - last_trigger_time > COOLDOWN_SECONDS:
                        with status_placeholder.container():
                            st.warning("⚠️ High Risk Detected! Running Agent Analysis...")
                            
                            # Merge CV payload with base biological metrics
                            if payload:
                                st.write(f"**Scene:** {payload.get('scene', 'unknown')} | **Risk:** {payload.get('risk_level', 'LOW')}")
                                
                                if payload.get("threats"):
                                    st.error(f"⚠️ **THREATS:** {', '.join([t['label'] for t in payload['threats']])}")
                                if payload.get("hazards"):
                                    st.warning(f"🚧 **HAZARDS:** {', '.join([h['label'] for h in payload['hazards']])}")
                                if payload.get("text_detected"):
                                    st.info(f"📄 **OCR TEXT:** {', '.join(payload['text_detected'])}")

                            # Attempt to load the live Apple HealthKit stream data
                            live_hr = 85 # fallback
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
                            execute_scenario("Live Vision & Health Hazard Detected", full_payload)
                            
                        last_trigger_time = time.time()
                    else:
                        time_left = int(COOLDOWN_SECONDS - (current_time - last_trigger_time))
                        status_placeholder.warning(f"Hazard tracked. Pipeline in cooldown for {time_left}s...")
                        
    except ImportError:
        st.error("Missing CV dependencies. Please run: pip install opencv-python ultralytics")
