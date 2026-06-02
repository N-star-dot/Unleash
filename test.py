import json
import time
from app import (
    process_biometrics, 
    process_vision_queue,
    multimodal_fusion_agent,
    predictive_forecaster, 
    memory_agent, 
    behavior_orchestrator, 
    action_dispatcher,
    retrospective_agent
)
import traceback

def test():
    telemetry = {
        "heart_rate": 85,
        "hrv": 45,
        "scene": "indoor",
        "crowd_count": 0,
        "risk_level": "LOW",
        "threats": [],
        "hazards": [],
        "detections": []
    }
    state = {
        "current_telemetry": telemetry,
        "active_claims": [],
        "active_predictions": [],
        "retrieved_memories": [],
        "final_action": ""
    }
    
    try:
        bio_output = process_biometrics(telemetry)
        vision_output = process_vision_queue(telemetry)
        claims = bio_output.get("active_claims", []) + vision_output.get("active_claims", [])
        state["active_claims"] = claims
        
        fusion_output = multimodal_fusion_agent(state)
        state["environmental_embedding"] = fusion_output.get("environmental_embedding", [0.0]*1024)
        
        pattern_output = predictive_forecaster(state)
        state["active_predictions"] = pattern_output.get("active_predictions", [])
        
        mem_output = memory_agent(state)
        state["retrieved_memories"] = mem_output.get("retrieved_memories", [])
        
        orch_output = behavior_orchestrator(state)
        state["final_action"] = orch_output.get("final_action", "Maintain passive navigation mode (Safe)")
        
        dispatch_output = action_dispatcher(state)
        print("Success:", dispatch_output)
    except Exception as e:
        print("Error!")
        traceback.print_exc()

test()
