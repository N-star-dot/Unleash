import os
import requests
import weave
import json
from conflict_bus import ConflictBusState
from shaped import Client

shaped_client = Client(api_key=os.environ.get("SHAPED_API_KEY"))

@weave.op()
def memory_agent(state: ConflictBusState) -> dict:
    """Agent 3: Case-based reasoning using ShapedQL for ranked retrieval."""
    predictions = state.get("active_predictions", [])
    if not predictions:
        return {"retrieved_memories": []}
        
    latest_pred = predictions[-1].get("description", "")
    if not latest_pred:
        return {"retrieved_memories": []}
        
    live_vector = state.get("environmental_embedding", [0.0]*1024)
    vector_string = json.dumps(live_vector)
    
    # The ShapedQL Query Pipeline
    # 1. Matches semantic similarity of the current anomaly
    # 2. Filters out bad memories (quality_score < 80)
    # 3. Returns only the top 2 most successful interventions
    shaped_query = f"""
    SELECT intervention_chosen
    FROM engine.unleash_memory_engine.retrieve(
        similarity(embedding_ref='precursor_similarity', input_text='{latest_pred}')
    )
    WHERE quality_score >= 80
    LIMIT 2
    """
    
    try:
        response = shaped_client.execute_query(
            engine_name="unleash_memory_engine",
            query=shaped_query,
            return_metadata=True,
        )

        top_ranked_memories = [
            (r.metadata or {}).get("intervention_chosen")
            for r in response.results
        ]
        top_ranked_memories = [m for m in top_ranked_memories if m]
        print(f"[Memory Agent] Shaped retrieved high-confidence context: {top_ranked_memories}")

    except Exception as e:
        print(f"[Memory Agent] Shaped API fallback: {e}")
        top_ranked_memories = []
        
    return {"retrieved_memories": top_ranked_memories}
    


# touched 2026-06-03
