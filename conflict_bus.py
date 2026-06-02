from typing import TypedDict, List, Annotated
import operator

# The strict schema for the shared bus
class ConflictBusState(TypedDict):
    current_telemetry: dict
    active_predictions: List[dict]
    # Annotated with operator.add so memories accumulate during the episode rather than overwriting
    retrieved_memories: Annotated[List[str], operator.add]
    final_action: str
    voice_input: str   # raw text from the user's speech (listen_agent.py)
