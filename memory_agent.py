import os
import requests
import weave
from conflict_bus import ConflictBusState
from shaped import ShapedClient

shaped_client = ShapedClient(api_key=os.environ.get("SHAPED_API_KEY"))

@weave.op()
def memory_agent(state: ConflictBusState) -> dict:
    """Agent 3: Case-based reasoning using ShapedQL for ranked retrieval."""
    predictions = state.get("active_predictions", [])
    if not predictions:
        return {"retrieved_memories": []}
        
    latest_pred = predictions[-1].get("description", "")
    if not latest_pred:
        return {"retrieved_memories": []}
    
    # The ShapedQL Query Pipeline
    # 1. Matches semantic similarity of the current anomaly
    # 2. Filters out bad memories (quality_score < 80)
    # 3. Returns only the top 2 most successful interventions
    shaped_query = f"""
    SELECT intervention_chosen
    FROM engine.unleash_memory_engine.retrieve(
        similarity(embedding_ref='precursor_embedding', input_text='{latest_pred}')
    )
    WHERE quality_score >= 80
    LIMIT 2
    """
    
    try:
        response = shaped_client.query(shaped_query)
        
        # Extract the clean list of successful interventions
        top_ranked_memories = [row.get("intervention_chosen") for row in response]
        print(f"[Memory Agent] Shaped retrieved high-confidence context: {top_ranked_memories}")
        
    except Exception as e:
        print(f"[Memory Agent] Shaped API fallback: {e}")
        top_ranked_memories = []
        
    return {"retrieved_memories": top_ranked_memories}
    

