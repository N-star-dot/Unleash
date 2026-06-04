import os
import json
import threading
import time
from flask import Flask, request

app = Flask(__name__)

# The file where we will store the most recent live biometric payload
LIVE_DATA_FILE = "live_biometrics.json"

@app.route('/health-webhook', methods=['POST'])
def health_webhook():
    """Receives JSON POST requests from the Health Auto Export app."""
    
    # Authenticate webhook via header or query parameter
    secret = os.environ.get("WEBHOOK_SECRET")
    if secret:
        token = request.headers.get("X-Webhook-Secret") or request.args.get("token")
        if token != secret:
            return {"error": "Unauthorized"}, 401

    # The app automatically sets the Content-Type to application/json
    health_payload = request.json
    
    # Debug: Dump the raw payload to disk so we can see what the app is actually sending
    try:
        with open("raw_webhook_dump.json", "w") as f:
            if request.is_json:
                json.dump(health_payload, f)
            else:
                f.write(request.get_data(as_text=True))
    except Exception as e:
        print(f"Failed to dump raw payload: {e}")
        
    # Extract the latest Heart Rate and HRV
    # Health Auto Export JSON structure varies, but generally looks like this:
    # {"data": {"metrics": [{"name": "heart_rate", "data": [{"qty": 85}]}]}}
    
    try:
        metrics = health_payload.get("data", {}).get("metrics", [])
        
        latest_hr = None
        for m in metrics:
            if m.get("name") in ["heart_rate", "heart_rate_variability"]:
                # Grab the most recent data point
                data_points = m.get("data", [])
                if data_points:
                    # Support raw 'qty' or aggregated 'Avg'
                    data_dict = data_points[-1]
                    qty = data_dict.get("qty") or data_dict.get("Avg")
                    if m.get("name") == "heart_rate":
                        latest_hr = qty
                        
        if latest_hr:
            # Basic validation to ensure heart rate is a realistic positive number
            try:
                latest_hr = float(latest_hr)
                if latest_hr <= 0 or latest_hr > 300:
                    raise ValueError("Unrealistic heart rate value")
            except (ValueError, TypeError):
                print(f"[Webhook Error] Invalid heart rate value: {latest_hr}")
                return {"error": "Invalid heart rate data"}, 400

            print(f"❤️ [APPLE HEALTH] Live HR Update: {latest_hr} BPM")
            
            # Write to a temporary file and atomically replace to prevent read-during-write races
            tmp_file = f"{LIVE_DATA_FILE}.tmp"
            with open(tmp_file, "w") as f:
                json.dump({
                    "heart_rate": latest_hr,
                    "timestamp": time.time()
                }, f)
            os.replace(tmp_file, LIVE_DATA_FILE)
                
    except Exception as e:
        print(f"[Webhook Error] Failed to parse payload: {e}")
    
    return {"status": "success"}, 200

def run_flask_server():
    print("🚀 Starting Apple Health Webhook Server on port 5050...")
    print("To expose this to your iPhone, open a new terminal and run: ngrok http 5050")
    app.run(port=5050, use_reloader=False)

if __name__ == "__main__":
    run_flask_server()

# touched 2026-06-03
