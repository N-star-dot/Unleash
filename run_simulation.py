import time
from app import (
    process_biometrics, 
    predictive_forecaster, 
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
    
    print("[2/4] Predictive Forecaster analyzing biometric trajectory...")
    pattern_output = predictive_forecaster(state)
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
    import json
    
    print("Loading raw webhook dump for streaming simulation...")
    try:
        with open("raw_webhook_dump.json", "r") as f:
            payload = json.load(f)
            
        metrics = payload.get("data", {}).get("metrics", [])
        hr_data = []
        for m in metrics:
            if m.get("name") == "heart_rate":
                hr_data = m.get("data", [])
                break
                
        print(f"Found {len(hr_data)} heart rate data points. Streaming first 65 points to fill buffer...")
        
        for i, point in enumerate(hr_data[:65]):
            hr = point.get("Max", 75)
            # Apple Watch HRV is sparse, so we mock it based on HR
            hrv = max(20, 100 - hr/2) 
            
            telemetry = {
                "heart_rate": hr,
                "hrv": hrv,
                "fall_detected": False
            }
            
            # Run silently to avoid console flood until the last few steps
            if i >= 58:
                simulate_event(f"Apple Watch Stream Step {i}", telemetry)
            else:
                # Update the buffer manually or just run the pipeline silently
                from forecast_engine import update_buffer
                update_buffer(hr, hrv)
                if i % 10 == 0:
                    print(f"Filled {i}/60 buffer slots...")
                    
    except Exception as e:
        print(f"Error loading JSON: {e}")
