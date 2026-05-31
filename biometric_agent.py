import time
import random
from conflict_bus import ConflictBusState, Claim

class BiometricAgent:
    def __init__(self, hr_threshold=100, hrv_threshold=30):
        self.hr_threshold = hr_threshold
        self.hrv_threshold = hrv_threshold

    def simulate_telemetry(self):
        # Simulate Heart Rate and Heart Rate Variability
        hr = random.uniform(60, 130)
        hrv = random.uniform(20, 80)
        return {"hr": hr, "hrv": hrv}

    def process(self) -> list[Claim]:
        telemetry = self.simulate_telemetry()
        claims = []
        
        # Post claim if critical threshold is crossed
        if telemetry["hr"] > self.hr_threshold:
            claims.append(Claim(
                timestamp=time.time(),
                source_agent="BiometricAgent_HR",
                confidence_score=round(random.uniform(0.85, 0.99), 2),
                time_to_live=60
            ))
            
        if telemetry["hrv"] < self.hrv_threshold:
            claims.append(Claim(
                timestamp=time.time(),
                source_agent="BiometricAgent_HRV",
                confidence_score=round(random.uniform(0.80, 0.95), 2),
                time_to_live=60
            ))
            
        return claims

def biometric_node(state: ConflictBusState) -> dict:
    agent = BiometricAgent()
    claims = agent.process()
    
    # Return a dictionary with the key matching the Annotated state field we want to update
    return {"claims": claims}
