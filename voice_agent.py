"""
Voice Agent — converts perception JSON stream to spoken alerts via macOS TTS.

Reads perception.py JSON from stdin, speaks priority-ranked alerts through
whatever audio output is active (AirPods, speakers, etc.).

Priority tiers:
  P3 THREAT   — weapons: interrupt queue, speak immediately, no cooldown
  P2 HAZARD   — stairs / doors: 5s cooldown per label+side
  P1 OBSTACLE — approaching or close nav objects: 3s cooldown per label+side
  P0 AMBIENT  — low light / crowd surge: 10s cooldown

Also emits one JSON line per spoken alert to stdout so it can be piped
into the decision / memory system.

Usage:
    python perception.py --camera 0 | python voice_agent.py
    python perception.py --camera 0 | python voice_agent.py | python orchestrator.py
    VOICE=Karen python voice_agent.py   # change voice
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time


# ─── Config ───────────────────────────────────────────────────────────────────

VOICE = os.environ.get("VOICE", "Samantha")  # macOS voice name

COOLDOWNS = {
    "threat":   0.0,   # always speak
    "hazard":   5.0,   # seconds before repeating same hazard
    "obstacle": 3.0,   # seconds before repeating same obstacle
    "ambient":  10.0,
}

# Label → spoken word for nav detections
NAV_SPEECH = {
    "person":        "person",
    "car":           "car",
    "bicycle":       "bike",
    "motorcycle":    "motorbike",
    "bus":           "bus",
    "truck":         "truck",
    "traffic light": "traffic light",
    "stop sign":     "stop sign",
    "fire hydrant":  "fire hydrant",
    "bench":         "bench",
    "dog":           "dog",
    "cat":           "cat",
    "chair":         "chair",
    "dining table":  "table",
    "couch":         "couch",
}

SIDE_SPEECH = {
    "left":   "on the left",
    "center": "ahead",
    "right":  "on the right",
}


# ─── TTS queue (single speaker thread — no overlapping speech) ────────────────

class Speaker:
    def __init__(self, voice: str):
        self.voice  = voice
        self._q     = queue.PriorityQueue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, text: str, priority: int = 1) -> None:
        """Queue text for speech. Lower priority number = spoken first."""
        self._q.put((priority, time.time(), text))

    def interrupt(self, text: str) -> None:
        """Clear queue and speak immediately (for threats)."""
        while not self._q.empty():
            try: self._q.get_nowait()
            except queue.Empty: break
        self._q.put((0, time.time(), text))

    def _run(self) -> None:
        while True:
            _, _, text = self._q.get()
            subprocess.run(["say", "-v", self.voice, text],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ─── Cooldown tracker ────────────────────────────────────────────────────────

class Cooldown:
    def __init__(self):
        self._last: dict[str, float] = {}

    def ready(self, key: str, seconds: float) -> bool:
        if seconds == 0:
            return True
        now = time.time()
        if now - self._last.get(key, 0) >= seconds:
            self._last[key] = now
            return True
        return False


# ─── Alert builder ────────────────────────────────────────────────────────────

def build_alerts(payload: dict) -> list[dict]:
    """
    Convert one perception frame into a prioritised list of alert dicts.
    Each alert: {priority, tier, text, label, side}
    """
    alerts = []
    ts = payload.get("timestamp", 0)

    # P3 — threats (weapons)
    for t in payload.get("threats", []):
        side = SIDE_SPEECH.get(t["side"], t["side"])
        label = t["label"].capitalize()
        alerts.append({
            "priority": 0,
            "tier":     "threat",
            "text":     f"Warning! {label} {side}",
            "key":      f"threat:{t['label']}:{t['side']}",
            "label":    t["label"],
            "side":     t["side"],
            "confidence": t["confidence"],
        })

    # P2 — hazards (stairs, doors)
    for h in payload.get("hazards", []):
        side  = SIDE_SPEECH.get(h["side"], h["side"])
        label = h["label"].replace("-", " ").capitalize()
        alerts.append({
            "priority": 1,
            "tier":     "hazard",
            "text":     f"{label} {side}",
            "key":      f"hazard:{h['label']}:{h['side']}",
            "label":    h["label"],
            "side":     h["side"],
            "confidence": h["confidence"],
        })

    # P1 — approaching or close nav detections
    for d in payload.get("detections", []):
        vel  = d.get("velocity", "unknown")
        prox = d.get("proximity", "far")
        if vel not in ("approaching", "crossing-L", "crossing-R") and prox != "close":
            continue  # skip far + stationary
        word = NAV_SPEECH.get(d["label"], d["label"])
        side = SIDE_SPEECH.get(d["side"], d["side"])
        if vel == "approaching":
            text = f"{word.capitalize()} approaching {side}"
        elif vel in ("crossing-L", "crossing-R"):
            text = f"{word.capitalize()} crossing {side}"
        else:
            text = f"{word.capitalize()} close {side}"
        alerts.append({
            "priority": 2,
            "tier":     "obstacle",
            "text":     text,
            "key":      f"obstacle:{d['label']}:{d['side']}",
            "label":    d["label"],
            "side":     d["side"],
            "confidence": d["confidence"],
        })

    # P0 — ambient warnings
    if payload.get("ambient_light") in ("dark", "dim"):
        alerts.append({
            "priority": 3,
            "tier":     "ambient",
            "text":     "Low light ahead",
            "key":      "ambient:light",
            "label":    "light",
            "side":     "center",
            "confidence": 1.0,
        })

    crowd = payload.get("crowd_count", 0)
    if crowd >= 5:
        alerts.append({
            "priority": 3,
            "tier":     "ambient",
            "text":     f"Crowded area, {crowd} people",
            "key":      "ambient:crowd",
            "label":    "crowd",
            "side":     "center",
            "confidence": 1.0,
        })

    return alerts


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    speaker  = Speaker(VOICE)
    cooldown = Cooldown()

    print(f"Voice agent ready. Voice: {VOICE}. Listening for perception JSON...",
          file=sys.stderr)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        alerts = build_alerts(payload)
        spoken = []

        for alert in alerts:
            cd = COOLDOWNS[alert["tier"]]
            if not cooldown.ready(alert["key"], cd):
                continue

            if alert["tier"] == "threat":
                speaker.interrupt(alert["text"])
            else:
                speaker.say(alert["text"], priority=alert["priority"])

            spoken.append({
                "tier":       alert["tier"],
                "text":       alert["text"],
                "label":      alert["label"],
                "side":       alert["side"],
                "confidence": alert["confidence"],
            })

            print(f"[{alert['tier'].upper()}] {alert['text']}", file=sys.stderr)

        # Emit to stdout for orchestrator / memory system
        if spoken:
            out = {
                "timestamp": payload.get("timestamp"),
                "scene":     payload.get("scene"),
                "spoken":    spoken,
                "raw_frame": payload,
            }
            print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()


# ─── VoiceAgent class — used by ui.py for in-browser audio playback ──────────

class VoiceAgent:
    """
    Thin wrapper used by the Streamlit UI to synthesize spoken audio bytes.
    Uses macOS `say` to generate an AIFF file, returns raw bytes + mime type.
    Falls back gracefully if `say` is unavailable.
    """

    VOICE = os.environ.get("VOICE", "Samantha")

    def synthesize(self, text: str) -> tuple[bytes | None, str]:
        """Return (audio_bytes, mime_type) or (None, '') on failure."""
        import tempfile, subprocess, os as _os
        if not text:
            return None, ""
        try:
            tmp = tempfile.mktemp(suffix=".aiff")
            subprocess.run(
                ["say", "-v", self.VOICE, "-o", tmp, text],
                check=True, capture_output=True
            )
            with open(tmp, "rb") as f:
                data = f.read()
            _os.unlink(tmp)
            return data, "audio/aiff"
        except Exception:
            return None, ""
