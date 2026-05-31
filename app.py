import os
import time
from typing import TypedDict, List, Annotated
import operator
from mem0 import Memory
import weave
from google import genai
from google.genai import types

# =====================================================================
# 1. INITIALIZATION & CONFIGURATION
# =====================================================================
# Initialize Weights & Biases Weave tracking
weave.init("nghiatr38-boston-university/project-argus-service-dog")

# Configure Mem0 to use Gemini for its underlying vector extraction
mem0_config = {
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
            "embedding_model_dims": 768
        }
    }
}
memory_client = Memory.from_config(mem0_config)

# Initialize the official Google GenAI client
# It automatically picks up os.environ["GEMINI_API_KEY"]
ai_client = genai.Client()

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
    crowd_density = telemetry.get("crowd_density", 0)
    
    if crowd_density > 0.8:
        claims.append({
            "source": "ComputerVisionAgent",
            "type": "VisionAlert",
            "value": crowd_density,
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
    if not predictions:
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
    
    if not predictions:
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
    if not predictions:
        return
        
    episode_summary = f"Context: {predictions[-1]['description']} | Action: {state['final_action']} | Success: {resolution_success}"
    
    memory_client.add(
        f"Episode Resolution: {episode_summary}",
        user_id="user_thtrang_06"
    )

@weave.op()
def action_dispatcher(state: ConflictBusState) -> dict:
    """Agent 5: Physically executes the chosen directives via external APIs."""
    action = state.get("final_action", "")
    
    # Mock routing table for physical actions
    if "emergency services" in action.lower():
        # e.g., trigger Twilio API
        return {"execution_status": "[TWILIO API] Initiating automated phone call to 911 and emergency contacts..."}
    elif "airpods" in action.lower():
        # e.g., trigger Apple HealthKit/Bluetooth
        return {"execution_status": "[BLUETOOTH API] Sending audio file to paired AirPods..."}
    elif "nudge" in action.lower():
        # e.g., trigger Robot Dog over WiFi
        return {"execution_status": "[UNITREE UDP] Sending haptic nudge macro to G2 Pro Robot Dog..."}
    else:
        return {"execution_status": "[SYSTEM] Passive observation maintained. No external API triggered."}
