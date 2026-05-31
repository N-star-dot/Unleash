from langgraph.graph import StateGraph, START, END
from conflict_bus import ConflictBusState
from biometric_agent import biometric_node
import time

def main():
    # Build the graph
    workflow = StateGraph(ConflictBusState)
    
    # Add nodes
    workflow.add_node("biometric_agent", biometric_node)
    
    # Define edges
    workflow.add_edge(START, "biometric_agent")
    workflow.add_edge("biometric_agent", END)
    
    # Compile the graph
    app = workflow.compile()
    
    print("Starting Biometric Agent telemetry simulation...\n")
    
    # Initialize state
    state = {"claims": []}
    
    # Run the graph multiple times to simulate streaming telemetry
    for i in range(10):
        print(f"--- Telemetry Stream {i+1} ---")
        result = app.invoke(state)
        state = result  # Update state with the results for the next iteration (if needed, though claims are appended)
        
        claims = result["claims"]
        print(f"Total claims in bus: {len(claims)}")
        if claims:
            # Print the most recent claims from this iteration
            # We figure out how many new claims were added in this iteration
            # Actually, app.invoke returns the full state after execution
            print("Current claims:")
            for c in claims:
                print(f"  - [{c['source_agent']}] conf: {c['confidence_score']} | ttl: {c['time_to_live']} | time: {c['timestamp']}")
        else:
            print("No critical thresholds crossed.")
            
        print()
        time.sleep(0.5)

if __name__ == "__main__":
    main()
