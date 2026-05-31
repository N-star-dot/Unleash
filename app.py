import os
import time
from typing import TypedDict, List, Annotated
import operator
from mem0 import Memory
import weave
from google import genai
from google.genai import types
import requests
from dotenv import load_dotenv

# Load secrets from a local .env file if present (GEMINI_API_KEY, WANDB_API_KEY, etc.)
load_dotenv()

# Disable Mem0's internal telemetry to prevent hidden Qdrant lock crashes in Streamlit
os.environ["MEM0_ENABLE_TELEMETRY"] = "false"
os.environ["MEM0_TELEMETRY"] = "false"

# =====================================================================
# 1. INITIALIZATION & CONFIGURATION
# =====================================================================
# Initialize Weights & Biases Weave tracking.
# Project is overridable via WEAVE_PROJECT. Never let a W&B outage or permission
# issue crash the app — tracing is a nice-to-have, not required to run.
WEAVE_PROJECT = os.environ.get(
    "WEAVE_PROJECT", "nghiatr38-boston-university/project-argus-service-dog"
)
try:
    weave.init(WEAVE_PROJECT)
except Exception as e:
    print(f"[weave] cloud tracing disabled ({type(e).__name__}); continuing without it.")

# Configure Mem0 to use Gemini for its underlying vector extraction
mem0_config = {
    "history_db_path": "mem0_history.db",
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-2.5-flash",
            "api_key": os.environ.get("GEMINI_API_KEY")
        }
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "gemini-embedding-001",
            "api_key": os.environ.get("GEMINI_API_KEY")
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "argus_memory",
            "path": ":memory:",
            "embedding_model_dims": 768
        }
    }
}
# Build the heavy clients fail-soft: importing this module must NEVER raise (ui.py
# imports it at module load, before any of its defensive try/except runs). With no
# GEMINI_API_KEY both constructors raise; in that case we degrade to None and each
# agent below guards on it, so the dashboard still renders and runs in Safe mode.
try:
    memory_client = Memory.from_config(mem0_config)
except Exception as e:
    memory_client = None
    print(f"[mem0] memory disabled ({type(e).__name__}); set GEMINI_API_KEY to enable.")

# Initialize the official Google GenAI client
# It automatically picks up os.environ["GEMINI_API_KEY"]
try:
    ai_client = genai.Client()
except Exception as e:
    ai_client = None
    print(f"[genai] LLM disabled ({type(e).__name__}); set GEMINI_API_KEY to enable.")

# =====================================================================
# 2. THE CONFLICT BUS DEFINITION (LangGraph State)
# =====================================================================
def append_reducer(current: list, new_items: list) -> list:
    return (current or []) + (new_items or [])

class ConflictBusState(TypedDict):
    current_telemetry: dict 
    active_claims: Annotated[List[dict], append_reducer]
    active_predictions: Annotated[List[dict], append_reducer]
    retrieved_memories: Annotated[List[str], append_reducer]
    final_action: str

# =====================================================================
# 3. CORE AGENT LOGIC (Gemini Powered)
# =====================================================================

@weave.op()
def process_biometrics(telemetry: dict) -> dict:
    """Agent 1: Evaluates raw telemetry streams."""
    claims = []
    hr = telemetry.get("heart_rate", 75)
    
    if hr > 100:
        claims.append({
            "source": "BiometricAgent",
            "type": "TachycardiaAlert",
            "value": hr,
            "timestamp": time.time(),
            "ttl": 30
        })
    return {"active_claims": claims}

@weave.op()
def process_vision_queue(telemetry: dict) -> dict:
    """Agent 1b: Evaluates computer vision object density streams."""
    claims = []
    
    # Check for live YOLO schema fields
    risk_level = telemetry.get("risk_level", "LOW")
    crowd_count = telemetry.get("crowd_count", 0)
    
    # Also support the old mock field 'crowd_density' for the Streamlit buttons
    crowd_density = telemetry.get("crowd_density", 0)
    
    if risk_level == "HIGH" or crowd_count > 5 or crowd_density > 0.8:
        claims.append({
            "source": "ComputerVisionAgent",
            "type": "VisionAlert",
            "value": f"Risk: {risk_level}, Crowd: {crowd_count}",
            "timestamp": time.time(),
            "ttl": 30
        })
    return {"active_claims": claims}

@weave.op()
def pattern_detector(state: ConflictBusState) -> dict:
    """Agent 2: Analyzes anomalies against baseline configurations."""
    claims = state.get("active_claims", [])
    predictions = []
    
    if any(c["type"] == "TachycardiaAlert" for c in claims):
        predictions.append({
            "source": "PatternDetector",
            "description": "Panic Attack Precursor Detected (HR spike profile match)",
            "confidence": 0.91,
            "timestamp": time.time()
        })
        
    if any(c["type"] == "VisionAlert" for c in claims):
        predictions.append({
            "source": "PatternDetector",
            "description": "Approaching Crowd Detected (High density objects ahead)",
            "confidence": 0.88,
            "timestamp": time.time()
        })
        
    return {"active_predictions": predictions}

@weave.op()
def memory_agent(state: ConflictBusState) -> dict:
    """Agent 3: Case-based semantic lookup via Gemini-powered Mem0."""
    predictions = state.get("active_predictions", [])
    if not predictions or memory_client is None:
        return {"retrieved_memories": []}

    latest_pred = predictions[-1]["description"]
    
    # Mem0 queries vector database using Gemini embeddings
    memories = memory_client.search(
        query=f"How did we stabilize the user during: {latest_pred}?",
        filters={"user_id": "user_thtrang_06"}
    )
    
    # Mem0 API dictionary response extraction
    results = memories.get("results", []) if isinstance(memories, dict) else memories
    extracted = [m.get("memory", m.get("content", "")) for m in results if isinstance(m, dict)]
    return {"retrieved_memories": extracted}

@weave.op()
def behavior_orchestrator(state: ConflictBusState) -> dict:
    """Agent 4: Resolves action matrices using Gemini Flash for rapid routing."""
    predictions = state.get("active_predictions", [])
    memories = state.get("retrieved_memories", [])

    if not predictions or ai_client is None:
        return {"final_action": "Maintain passive navigation mode (Safe)"}

    # Construct the prompt for Gemini reasoning execution
    prompt = f"""
    Analyze the following multi-agent system state context and select the optimal safety directive.
    
    Active Predictions: {predictions}
    Retrieved Episodic Memories: {memories}
    
    Available Action Directives:
    - "Play grounding countdown audio through AirPods"
    - "Initiate a gentle physical nudge to ground the user"
    - "Call emergency services immediately"
    - "Maintain passive navigation mode (Safe)"
    
    Output strictly the chosen action directive string and nothing else.
    """
    
    # Call Gemini 2.5 Flash for ultra-low latency decision making
    response = ai_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    
    action = response.text.strip()
    return {"final_action": action}

@weave.op()
def retrospective_agent(state: ConflictBusState, resolution_success: bool):
    """Agent 5: Saves outcomes back to Mem0 using Gemini compaction."""
    predictions = state.get("active_predictions", [])
    if not predictions or memory_client is None:
        return

    episode_summary = f"Context: {predictions[-1]['description']} | Action: {state['final_action']} | Success: {resolution_success}"
    
    memory_client.add(
        f"Episode Resolution: {episode_summary}",
        user_id="user_thtrang_06"
    )

def _trigger_bland_ai_call(reason: str, arm_real_calls: bool = False) -> str:
    """Helper to dispatch a live AI voice call via Bland AI.

    When `arm_real_calls` is False the call is ALWAYS simulated — this gate is
    explicit and thread-safe (no global env mutation), so a disarmed run can
    never place a live call even under Streamlit's concurrent reruns.
    """
    if not arm_real_calls:
        return "[SIMULATED BLAND AI] Calls disarmed (ARM REAL CALLS off). Simulated call dispatched."

    bland_api_key = os.environ.get("BLAND_API_KEY")
    caregiver_phone = os.environ.get("CAREGIVER_PHONE")

    if not bland_api_key or not caregiver_phone:
        return "[SIMULATED BLAND AI] Missing BLAND_API_KEY or CAREGIVER_PHONE env vars. Simulated call dispatched."
        
    url = "https://api.bland.ai/v1/calls"
    headers = {"authorization": bland_api_key, "Content-Type": "application/json"}
    
    payload = {
        "phone_number": caregiver_phone,
        "task": f"Hello, this is Project ARGUS. The user is experiencing a severe distress episode labeled as: {reason}. The digital service dog is on-site providing support, but human intervention is requested.",
        "voice": "nat", 
        "reduce_latency": True
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return f"[BLAND AI LIVE] Emergency AI call successfully dispatched to {caregiver_phone}."
    except requests.exceptions.RequestException as e:
        return f"[BLAND AI ERROR] Failed to dispatch call: {str(e)}"

@weave.op()
def action_dispatcher(state: ConflictBusState) -> dict:
    """Agent 5: Physically executes the chosen directives via external APIs."""
    action = state.get("final_action", "")
    predictions = state.get("active_predictions", [])
    reason_context = predictions[-1]["description"] if predictions else "Unknown physiological anomaly"
    # Safety gate threaded explicitly through state (default OFF = always simulated).
    arm_real_calls = bool(state.get("arm_real_calls", False))

    # Mock routing table for physical actions
    if "emergency services" in action.lower():
        status = _trigger_bland_ai_call(reason_context, arm_real_calls)
        return {"execution_status": status}
    elif "airpods" in action.lower():
        # e.g., trigger Apple HealthKit/Bluetooth
        return {"execution_status": "[BLUETOOTH API] Sending audio file to paired AirPods..."}
    elif "nudge" in action.lower():
        # e.g., trigger Robot Dog over WiFi
        return {"execution_status": "[UNITREE UDP] Sending haptic nudge macro to G2 Pro Robot Dog..."}
    else:
        return {"execution_status": "[SYSTEM] Passive observation maintained. No external API triggered."}
