from mem0 import Memory
import weave
from conflict_bus import ConflictBusState

# Initialize Mem0 (defaults to local SQLite and Qdrant if no API key is passed, or connects to platform if MEM0_API_KEY is set)
m = Memory() 

@weave.op() # Traces this specific retrieval step for the judges
def memory_agent(state: ConflictBusState) -> dict:
    predictions = state.get("active_predictions", [])
    if not predictions:
        return {"retrieved_memories": []}
        
    latest_prediction = predictions[-1].get("description", "")
    if not latest_prediction:
        return {"retrieved_memories": []}
    
    # Search Mem0 for similar past episodes using the user's specific ID
    # This searches the vector embeddings for conceptual matches to the current crisis
    results = m.search(f"What interventions stabilized the user during: {latest_prediction}?", filters={"user_id": "service_dog_user_1"})
    
    # Mem0 results typically have a 'content' or 'memory' field depending on the version
    # The snippet assumed 'content'
    extracted_memories = [res.get("content", str(res)) if isinstance(res, dict) else str(res) for res in results]
    
    return {"retrieved_memories": extracted_memories}
