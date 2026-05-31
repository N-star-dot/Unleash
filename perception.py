"""
Perception Agent — Obstacle detection for visually impaired users.
Reads frames from iPhone (Continuity Camera), runs YOLOv8n, emits JSON lines
to stdout. Stderr is used for status/debug so stdout stays pipe-clean.

Output contract (one JSON line per frame, only when detections are non-empty):
{
  "timestamp": <unix_ms>,
  "scene": "street" | "indoor" | "nature" | "unknown",
  "ambient_light": "ok" | "dim" | "dark",
  "crowd_count": <int>,
  "risk_level": "LOW" | "MED" | "HIGH",
  "detections": [
    {
      "label": "person",
      "confidence": 0.87,
      "side": "left" | "center" | "right",
      "proximity": "close" | "far",
      "velocity": "approaching" | "receding" | "crossing-L" | "crossing-R" | "stationary" | "unknown",
      "collision_risk": 0.72
    }
  ]
}
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

DANGER_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "fire hydrant", "bench",
    "dog", "cat", "chair", "dining table", "couch",
}

CONF_THRESHOLD = 0.5

SCENE_MAP = {
    "street": {"car", "truck", "bus", "traffic light", "stop sign", "motorcycle", "bicycle"},
    "indoor": {"chair", "dining table", "couch", "tv", "laptop", "keyboard"},
    "nature": {"bench", "bird", "dog", "cat"},
}


def get_side(cx: float, frame_width: int) -> str:
    third = frame_width / 3
    if cx < third:       return "left"
    if cx < 2 * third:   return "center"
    return "right"


def get_proximity(box_width: float, frame_width: int) -> str:
    return "close" if (box_width / frame_width) > 0.3 else "far"


def classify_scene(seen: set) -> str:
    best, n = "unknown", 0
    for scene, keys in SCENE_MAP.items():
        k = len(keys & seen)
        if k > n:
            best, n = scene, k
    return best


def ambient_level(frame) -> str:
    m = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return "dark" if m < 50 else ("dim" if m < 110 else "ok")


class Tracker:
    """Nearest-neighbour velocity + collision risk estimator (same as viewer.py)."""
    def __init__(self):
        self._hist: dict = {}

    def update(self, label, x1, y1, x2, y2, W, H):
        cx   = (x1 + x2) / 2.0
        cy   = (y1 + y2) / 2.0
        area = (x2 - x1) * (y2 - y1)
        vel, risk = "unknown", 0.20
        prev = self._hist.get(label, [])
        if prev:
            best       = min(prev, key=lambda p: (cx-p[0])**2 + (cy-p[1])**2)
            ratio      = area / best[2] if best[2] > 0 else 1.0
            dx         = cx - best[0]
            if   ratio > 1.10:          vel = "approaching"
            elif ratio < 0.90:          vel = "receding"
            elif abs(dx) > W * 0.015:   vel = "crossing-L" if dx < 0 else "crossing-R"
            else:                       vel = "stationary"
            norm = min(1.0, area / (W * H * 0.20))
            risk = min(1.0, norm * 0.6 + max(0., (ratio-1.)*3.) * 0.4)
            if vel == "receding": risk *= 0.25
        b = self._hist.setdefault(label, [])
        b.append((cx, cy, area))
        self._hist[label] = b[-4:]
        return vel, round(risk, 2)


def vision_generator(camera_index=1):
    print("Loading YOLOv8n model...", file=sys.stderr)
    model = YOLO("yolov8n.pt")
    tracker = Tracker()

    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        # Fallback to default index if macOS AVFOUNDATION fails
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        yield None, {"error": "Camera not found"}
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        results = model(frame, verbose=False)[0]
        detections = []
        seen: set = set()

        for box in results.boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if label not in DANGER_CLASSES or conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw = x2 - x1
            cx = (x1 + x2) / 2
            side = get_side(cx, W)
            prox = get_proximity(bw, W)
            vel, risk = tracker.update(label, x1, y1, x2, y2, W, H)

            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "side": side,
                "proximity": prox,
                "velocity": vel,
                "collision_risk": risk,
            })
            seen.add(label)

            color = (0, 255, 0) if prox == "far" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f} {side} {vel}",
                        (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Convert to RGB for Streamlit rendering
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        payload = None
        if detections:
            crowd = sum(1 for d in detections if d["label"] == "person")
            max_r = max(d["collision_risk"] for d in detections)
            rl = "HIGH" if max_r > 0.60 else ("MED" if max_r > 0.35 else "LOW")
            payload = {
                "timestamp": int(time.time() * 1000),
                "scene": classify_scene(seen),
                "ambient_light": ambient_level(frame),
                "crowd_count": crowd,
                "risk_level": rl,
                "detections": detections,
            }

        yield frame_rgb, payload

    cap.release()
