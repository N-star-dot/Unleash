import weave
import asyncio
import json
import os
from app import behavior_orchestrator, ConflictBusState

# Initialize Weights & Biases tracing for the evaluation project
weave.init("nghiatr38-boston-university/project-argus-service-dog")

@weave.op()
def evaluate_orchestrator(predictions: list, memories: list) -> str:
    """
    Wrapper that shapes the dataset inputs into the expected LangGraph state format.
    """
    state = {
        "current_telemetry": {},
        "active_claims": [],
        "active_predictions": predictions,
        "retrieved_memories": memories,
        "final_action": ""
    }
    
    # Run the actual orchestrator logic
    result = behavior_orchestrator(state)
    return result["final_action"]

@weave.op()
def action_match_scorer(expected_action: str, output: str) -> dict:
    """
    Scorer that checks if Groq's generated output contains the expected directive.
    """
    # Use substring matching to account for minor LLM formatting variations
    is_match = expected_action.lower().strip() in output.lower().strip()
    return {"is_correct": is_match}

async def run_evaluation():
    print("📦 Loading evaluation dataset...")
    with open("eval_dataset.json", "r") as f:
        dataset = json.load(f)
        
    print(f"🎯 Loaded {len(dataset)} critical emergency scenarios.")
    
    # Define the Evaluation pipeline
    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[action_match_scorer]
    )
    
    print("🚀 Firing scenarios against Groq Llama 3.3 (this will take a few seconds)...")
    # Run the evaluation loop asynchronously
    results = await evaluation.evaluate(evaluate_orchestrator)
    
    print("\n✅ Evaluation Pipeline Complete!")
    print("📊 Check your Weights & Biases Dashboard to view the live scorecard!")

if __name__ == "__main__":
    # Ensure GROQ API Key is set before running
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable is missing.")
        print("Please export it in your terminal before running evaluations.")
        exit(1)
        
    asyncio.run(run_evaluation())
