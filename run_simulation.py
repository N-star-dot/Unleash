import time
from app import (
    process_biometrics, 
    pattern_detector, 
    memory_agent, 
    behavior_orchestrator, 
    retrospective_agent
)

def simulate_event(scenario_name: str, telemetry_payload: dict):
    print(f"\n=== Running Scenario: {scenario_name} ===")
    
    # 1. Initialize empty state on the Conflict Bus
    state = {
        "current_telemetry": telemetry_payload,
        "active_claims": [],
        "active_predictions": [],
        "retrieved_memories": [],
        "final_action": ""
    }
    
    # 2. Step-by-step execution through the agent pipeline
    print("[1/4] Biometric Agent processing raw streams...")
    bio_output = process_biometrics(telemetry_payload) # The code had process_biometrics(state["current_telemetry"])
    state["active_claims"] = bio_output.get("active_claims", [])
    
    print("[2/4] Pattern Detector analyzing baseline variations...")
    pattern_output = pattern_detector(state)
    state["active_predictions"] = pattern_output.get("active_predictions", [])
    
    print("[3/4] Memory Agent querying Mem0 episodic context...")
    mem_output = memory_agent(state)
    state["retrieved_memories"] = mem_output.get("retrieved_memories", [])
    
    print("[4/4] Behavior Orchestrator resolving action paths...")
    orch_output = behavior_orchestrator(state)
    state["final_action"] = orch_output.get("final_action", "MAINTAIN_PASSIVE_NAVIGATION_COMPANIONION")
    
    print(f"--> Final Executed Directive: {state['final_action']}")
    
    # 3. Close the loop asynchronously with the Retrospective Agent
    print("[Post-Episode] Retrospective Agent updating episodic memory benchmarks...")
    retrospective_agent(state, resolution_success=True)
    print("=== Scenario Concluded ===\n")

if __name__ == "__main__":
    # Test Scenario: Simulate an oncoming panic attack profile
    panic_profile = {
        "heart_rate": 115,
        "hrv": 22,
        "fall_detected": False
    }
    simulate_event("Panic Attack Onset (Elevated HR + Low HRV)", panic_profile)

# touched 2026-06-03
