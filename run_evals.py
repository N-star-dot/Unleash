import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

import weave
import json
from app import orchestrator_model

# Initialize Weave tracking for the same project
weave.init("nghiatr38-boston-university/project-unleash-service-dog")

# =====================================================================
# 1. SCORING METRICS
# =====================================================================
@weave.op()
def exact_match(expected: str, model_output: str) -> dict:
    """Checks if the LLM output matches the exact required safety protocol."""
    return {"match": expected.strip().lower() == model_output.strip().lower()}

# =====================================================================
# 2. EVALUATION DATASET
# =====================================================================
test_dataset = [
    {
        # Scenario: User collapses.
        "predictions": json.dumps([{"description": "SEVERE SEIZURE OR FALL DETECTED (Critical Emergency)"}]),
        "memories": "[]",
        "expected": "Call emergency services immediately"
    },
    {
        # Scenario: Elevated HR indicating panic, and past memories show AirPods worked best.
        "predictions": json.dumps([{"description": "Panic Attack Precursor Detected. Current HR is elevated."}]),
        "memories": json.dumps(["Play grounding countdown audio through AirPods"]),
        "expected": "Play grounding countdown audio through AirPods" 
    },
    {
        # Scenario: Dense crowd, but no physical danger or elevated HR.
        "predictions": json.dumps([{"description": "Approaching Crowd Detected (High density objects ahead)"}]),
        "memories": "[]",
        "expected": "Maintain passive navigation mode (Safe)"
    }
]

# =====================================================================
# 3. EXECUTION
# =====================================================================
def main():
    print("Running Automated Safety Evaluations...")
    
    # Weave automatically runs the orchestrator_model.predict() against every item 
    # in the dataset, and then passes the output into the exact_match scorer.
    evaluation = weave.Evaluation(dataset=test_dataset, scorers=[exact_match])
    results = asyncio.run(evaluation.evaluate(orchestrator_model))
    
    print("\n✅ Evaluation Complete! View the full matrix on Weights & Biases.")

if __name__ == "__main__":
    main()
