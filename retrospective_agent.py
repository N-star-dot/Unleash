from mem0 import Memory
from conflict_bus import ConflictBusState
import weave

m = Memory()

@weave.op()
def retrospective_agent(state: ConflictBusState) -> dict:
    final_action = state.get("final_action")
    telemetry = state.get("current_telemetry", {})
    predictions = state.get("active_predictions", [])
    
    if not final_action:
        return {}

    # Format the outcome of the episode to store in Mem0
    episode_summary = (
        f"Episode Outcome: Action '{final_action}' was taken. "
        f"Telemetry at the time: {telemetry}. "
    )
    if predictions:
        episode_summary += f"Crisis predicted: {predictions[-1].get('description')}. "
        
    if final_action != "Maintain passive observation.":
        # We only record significant interventions or episodes
        messages = [
            {"role": "system", "content": "You are recording an intervention episode."},
            {"role": "user", "content": episode_summary}
        ]
        
        # Push back into the database ensuring the system learns for the next iteration
        m.add(messages, user_id="service_dog_user_1")
        
    return {}
