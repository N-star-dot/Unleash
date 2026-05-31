import streamlit as st
import time
import os
from app import (
    process_biometrics, 
    process_vision_queue,
    pattern_detector, 
    memory_agent, 
    behavior_orchestrator, 
    action_dispatcher,
    retrospective_agent
)

st.set_page_config(page_title="ARGUS Control Panel", page_icon="🐕", layout="wide")

st.title("🐕 Project ARGUS: Control Panel")
st.markdown("Simulate telemetry and environmental triggers to test the ARGUS multi-agent pipeline.")

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
        st.write("📡 **[1/4]** Evaluating raw telemetry streams...")
        bio_output = process_biometrics(telemetry)
        vision_output = process_vision_queue(telemetry)
        
        # Merge claims from both ingest agents
        claims = bio_output.get("active_claims", []) + vision_output.get("active_claims", [])
        state["active_claims"] = claims
        time.sleep(0.5)
        
        st.write("🔍 **[2/4]** Pattern Detector analyzing baseline variations...")
        pattern_output = pattern_detector(state)
        state["active_predictions"] = pattern_output.get("active_predictions", [])
        time.sleep(0.5)
        
        st.write("🧠 **[3/4]** Memory Agent querying Mem0 episodic context...")
        mem_output = memory_agent(state)
        state["retrieved_memories"] = mem_output.get("retrieved_memories", [])
        time.sleep(0.5)
        
        st.write("⚖️ **[4/5]** Behavior Orchestrator resolving action paths...")
        orch_output = behavior_orchestrator(state)
        state["final_action"] = orch_output.get("final_action", "Maintain passive navigation mode (Safe)")
        time.sleep(0.5)
        
        st.write("⚡ **[5/5]** Action Dispatcher routing hardware APIs...")
        dispatch_output = action_dispatcher(state)
        
        status.update(label="Pipeline Execution Complete!", state="complete", expanded=False)
    
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
    if st.button("🟢 Trigger: Safe Environment", use_container_width=True):
        payload = {
            "heart_rate": 72,
            "hrv": 65,
            "crowd_density": 0.1,
            "fall_detected": False
        }
        execute_scenario("Safe Environment (Normal HR, No Obstacles)", payload)

with col2:
    if st.button("🟡 Trigger: Approaching Crowd", use_container_width=True):
        payload = {
            "heart_rate": 85,
            "hrv": 45,
            "crowd_density": 0.9, # Triggers ComputerVisionAgent
            "fall_detected": False
        }
        execute_scenario("Approaching Crowd (High-density objects)", payload)

with col3:
    if st.button("🔴 Trigger: Panic Attack Precursor", use_container_width=True):
        payload = {
            "heart_rate": 115, # Triggers BiometricAgent
            "hrv": 22,
            "crowd_density": 0.1,
            "fall_detected": False
        }
        execute_scenario("Panic Attack Onset (Elevated HR + Low HRV)", payload)
