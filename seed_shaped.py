import os
import time
from dotenv import load_dotenv
from shaped import Client

load_dotenv()

SHAPED_API_KEY = os.environ.get("SHAPED_API_KEY")
if not SHAPED_API_KEY:
    print("SHAPED_API_KEY is not set.")
    exit(1)

shaped_client = Client(api_key=SHAPED_API_KEY)

print("Seeding Shaped AI database with historical episodes...")

# Some fake precursors and interventions to show case successful and failed outcomes
episodes = [
    {"precursor": "Panic Attack Precursor Detected. Current HR (115) is 30% above user's Apple Health baseline (75).", "intervention": "Play grounding countdown audio through AirPods", "success": True},
    {"precursor": "Panic Attack Precursor Detected. Current HR (115) is 30% above user's Apple Health baseline (75).", "intervention": "Maintain passive observation", "success": False},
    {"precursor": "Panic Attack Precursor Detected. Current HR (120) is 30% above user's Apple Health baseline (80).", "intervention": "Initiate a gentle physical nudge to ground the user", "success": True},
    {"precursor": "Panic Attack Precursor Detected. Current HR (118) is 30% above user's Apple Health baseline (77).", "intervention": "Call emergency services immediately", "success": False},
    {"precursor": "Panic Attack Precursor Detected. Current HR (125) is 30% above user's Apple Health baseline (75).", "intervention": "Play grounding countdown audio through AirPods", "success": True},
    {"precursor": "Approaching Crowd Detected (High density objects ahead)", "intervention": "Initiate a gentle physical nudge to ground the user", "success": True},
    {"precursor": "Approaching Crowd Detected (High density objects ahead)", "intervention": "Maintain passive observation", "success": False},
    {"precursor": "CRITICAL: Deadly Weapon Detected in Field of View!", "intervention": "Call emergency services immediately", "success": True},
    {"precursor": "CRITICAL: Deadly Weapon Detected in Field of View!", "intervention": "Initiate a gentle physical nudge to ground the user", "success": False},
    {"precursor": "Environmental Hazard Detected (Stairs/Doors ahead)", "intervention": "Initiate a gentle physical nudge to ground the user", "success": True},
    {"precursor": "Environmental Hazard Detected (Stairs/Doors ahead)", "intervention": "Maintain passive observation", "success": False},
    {"precursor": "SEVERE SEIZURE OR FALL DETECTED (Critical Emergency)", "intervention": "Call emergency services immediately", "success": True},
    {"precursor": "SEVERE SEIZURE OR FALL DETECTED (Critical Emergency)", "intervention": "Play grounding countdown audio through AirPods", "success": False},
]

headers = {
    "x-api-key": SHAPED_API_KEY,
    "Content-Type": "application/json"
}

records = []
for i, ep in enumerate(episodes):
    quality_score = 95 if ep["success"] else 20
    
    # Calculate a past timestamp as a Unix epoch (the table's timestamp column is Int64)
    timestamp_seconds = int(time.time() - (len(episodes) - i) * 3600)

    records.append({
        "item_id": f"seed_ep_{int(time.time())}_{i}",
        "precursor_state": ep["precursor"],
        "intervention_chosen": ep["intervention"],
        "quality_score": quality_score,
        "timestamp": timestamp_seconds
    })

try:
    shaped_client.insert_table_rows(
        table_name="Unleash_Data_V3",
        rows=records
    )
    print("Seed process completed successfully!")
except Exception as e:
    print(f"Error: {e}")
