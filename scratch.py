import json

with open("raw_webhook_dump.json", "r") as f:
    payload = json.load(f)

metrics = payload.get("data", {}).get("metrics", [])
for m in metrics:
    if m.get("name") == "heart_rate":
        print(f"Found heart_rate with {len(m.get('data', []))} data points.")
        if m.get("data"):
            print("Last data point:", m["data"][-1])
