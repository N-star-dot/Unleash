import os
import weave
import time
from conflict_bus import ConflictBusState
from shaped import ShapedClient

shaped_client = ShapedClient(api_key=os.environ.get("SHAPED_API_KEY"))

@weave.op()
def retrospective_agent(state: ConflictBusState, resolution_success: bool = True) -> dict:
    """Agent 5: Saves structured outcomes to Shaped AI for future ranking."""
    final_action = state.get("final_action")
    predictions = state.get("active_predictions", [])
    
    if not final_action or not predictions:
        return {}
        
    # Calculate a mock quality score based on the outcome
    # In a real scenario, this would be derived from post-intervention HR stabilization
    quality_score = 95 if resolution_success else 20
    
    # Push the structured memory to Shaped
    if final_action != "Maintain passive observation.":
        try:
            shaped_client.tables.insert(
                table_name="Unleash_Data",
                rows=[
                    {
                        "episode_id": f"ep_{int(time.time())}",
                        "precursor_state": predictions[-1].get('description', ''),
                        "intervention_chosen": final_action,
                        "quality_score": quality_score,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                ]
            )
            print(f"[Retrospective Agent] Episode logged to Shaped. Quality: {quality_score}")
        except Exception as e:
            print(f"[Retrospective Agent] Error logging to Shaped: {e}")
        
    return {}

