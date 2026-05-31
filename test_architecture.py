from langgraph.graph import StateGraph, START, END
from conflict_bus import ConflictBusState
from memory_agent import memory_agent
from orchestrator import behavior_orchestrator
from retrospective_agent import retrospective_agent
import time

def setup_graph():
    # Build the graph
    workflow = StateGraph(ConflictBusState)
    
    # Add nodes
    workflow.add_node("memory", memory_agent)
    workflow.add_node("orchestrator", behavior_orchestrator)
    workflow.add_node("retrospective", retrospective_agent)
    
    # Define edges (sequential flow for this episode)
    workflow.add_edge(START, "memory")
    workflow.add_edge("memory", "orchestrator")
    workflow.add_edge("orchestrator", "retrospective")
    workflow.add_edge("retrospective", END)
    
    # Compile the graph
    return workflow.compile()

def simulate_episode():
    app = setup_graph()
    
    print("Starting Architecture Simulation...")
    
    # Initial state simulation: 
    # The biometric agent has pushed telemetry, and pattern detector has pushed a prediction
    initial_state = {
        "current_telemetry": {"heart_rate": 135, "heart_rate_delta": 45},
        "active_predictions": [{"description": "impending panic attack"}],
        "retrieved_memories": [],
        "final_action": ""
    }
    
    print("\n--- Initial State ---")
    print(f"Telemetry: {initial_state['current_telemetry']}")
    print(f"Prediction: {initial_state['active_predictions'][0]['description']}")
    
    print("\n--- Running Graph ---")
    result = app.invoke(initial_state)
    
    print("\n--- Final Output State ---")
    print(f"Retrieved Memories: {result.get('retrieved_memories', [])}")
    print(f"Final Action Taken: {result.get('final_action', 'None')}")
    print("\nThe Retrospective Agent has recorded the episode to Mem0.")

if __name__ == "__main__":
    # Weave tracking is already initialized in orchestrator.py, but we can call it here too.
    import weave
    weave.init("argus-digital-service-dog")
    simulate_episode()
