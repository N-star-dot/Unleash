import xml.etree.ElementTree as ET
from datetime import datetime
import json
import os

def extract_vital_baselines(xml_path="export.xml"):
    print("[Health Parser] Streaming massive XML file. Extracting HR and HRV...")
    
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found. Please drop your Apple Health export.xml into this directory!")
        return None

    health_data = {
        "heart_rate": [],
        "hrv": []
    }
    
    # iterparse streams the file so you don't run out of RAM
    context = ET.iterparse(xml_path, events=("end",))
    
    for event, elem in context:
        if elem.tag == "Record":
            record_type = elem.get("type")
            
            # Target 1: Heart Rate
            if record_type == "HKQuantityTypeIdentifierHeartRate":
                health_data["heart_rate"].append({
                    "date": elem.get("startDate"),
                    "value": float(elem.get("value"))
                })
                
            # Target 2: Heart Rate Variability (HRV)
            elif record_type == "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":
                health_data["hrv"].append({
                    "date": elem.get("startDate"),
                    "value": float(elem.get("value"))
                })
            
            # Clear the element from memory to keep it fast
            elem.clear()
            
    # Optional: Keep only the most recent 500 records to save LLM context tokens
    health_data["heart_rate"] = health_data["heart_rate"][-500:]
    health_data["hrv"] = health_data["hrv"][-500:]
            
    print(f"[Health Parser] Done! Found {len(health_data['heart_rate'])} HR records and {len(health_data['hrv'])} HRV records.")
    return health_data

if __name__ == "__main__":
    data = extract_vital_baselines()
    if data:
        # Save it to a clean JSON file so the agent can read it instantly
        with open("user_health_baseline.json", "w") as f:
            json.dump(data, f, indent=4)
        print("[Health Parser] Successfully wrote user_health_baseline.json")

# touched 2026-06-03
