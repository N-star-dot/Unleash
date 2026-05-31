import weave
from conflict_bus import ConflictBusState

# Initialize the W&B project to start tracking
weave.init("argus-digital-service-dog")

@weave.op()
def behavior_orchestrator(state: ConflictBusState) -> dict:
    telemetry = state.get("current_telemetry", {})
    memories = state.get("retrieved_memories", [])
    
    # Logic to evaluate telemetry against memories goes here
    # Example: strict priority hierarchy checking
    
    if telemetry.get("heart_rate_delta", 0) > 30 and "auditory grounding" in str(memories):
        action = "Initiate auditory countdown at 60BPM."
    else:
        action = "Maintain passive observation."
        
    return {"final_action": action}
