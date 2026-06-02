"""
Perception Agent — real-time JSON stream for the decision/memory system.

Runs headlessly, emits one JSON line per frame to stdout (only when non-empty).
All status/debug goes to stderr. No voice — that belongs in the decision agent.

Person filter: only emitted when approaching+close OR facing camera (face visible).
This keeps the feed signal-dense rather than noisy.

Output contract:
{
  "timestamp": <unix_ms>,
  "scene": "street" | "indoor" | "nature" | "unknown",
  "ambient_light": "ok" | "dim" | "dark",
  "crowd_count": <int>,          // total persons visible (even if not emitted)
  "detections": [                // non-person obstacles + filtered persons
    {
      "label": "person"|"car"|...,
      "confidence": 0.88,
      "side": "left"|"center"|"right",
      "proximity": "close"|"far",
      "velocity": "approaching"|"receding"|"crossing-L"|"crossing-R"|"stationary"|"unknown",
      "collision_risk": 0.72,
      "facing_camera": true|false   // persons only — face detected in bbox
    }
  ],
  "hazards": [
    {"label": "stairs"|"door-open"|..., "confidence": 0.85, "side": "center"}
  ],
  "threats": [
    {"label": "knife"|"gun", "confidence": 0.72, "side": "right"}
  ],
  "text_detected": ["BUS 42", "STOP"]
}

Usage:
    python perception.py --camera 0
    python perception.py --camera 0 | python decision_agent.py
"""

import argparse
import json
import os
import sys
import time
import threading

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr


# ─── Config ───────────────────────────────────────────────────────────────────

CONF_NAV    = 0.45
CONF_STAIRS = 0.80
CONF_DOORS  = 0.45
CONF_WEAPON = 0.45

NAV_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "fire hydrant", "bench",
    "dog", "cat", "chair", "dining table", "couch",
}

SCENE_MAP = {
    "street": {"car", "truck", "bus", "traffic light", "stop sign", "motorcycle", "bicycle"},
    "indoor": {"chair", "dining table", "couch", "tv", "laptop", "keyboard"},
    "nature": {"bench", "bird", "dog", "cat"},
}

_BASE             = os.path.dirname(os.path.abspath(__file__))
STAIRS_MODEL_PATH = os.path.join(_BASE, "models/stairs/train/weights/best.pt")
DOORS_MODEL_PATH  = os.path.join(_BASE, "models/doors_run/train/weights/best.pt")
WEAPON_MODEL_PATH = os.path.join(_BASE, "models/weapons/train/weights/best.pt")
if not os.path.exists(WEAPON_MODEL_PATH):
    WEAPON_MODEL_PATH = os.path.join(_BASE, "best.pt")

# Face detector for "facing camera" / "talking to me" signal
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ─── Motion tracker ───────────────────────────────────────────────────────────

class Tracker:
    def __init__(self):
        self._hist: dict = {}

    def update(self, label, x1, y1, x2, y2, W, H):
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        area   = (x2-x1)*(y2-y1)
        vel, risk = "unknown", 0.20
        prev = self._hist.get(label, [])
        if prev:
            best  = min(prev, key=lambda p: (cx-p[0])**2+(cy-p[1])**2)
            ratio = area/best[2] if best[2] > 0 else 1.0
            dx    = cx - best[0]
            if   ratio > 1.10:          vel = "approaching"
            elif ratio < 0.90:          vel = "receding"
            elif abs(dx) > W*0.015:     vel = "crossing-L" if dx < 0 else "crossing-R"
            else:                       vel = "stationary"
            norm = min(1.0, area/(W*H*0.20))
            risk = min(1.0, norm*0.6 + max(0., (ratio-1.)*3.)*0.4)
            if vel == "receding": risk *= 0.25
        b = self._hist.setdefault(label, [])
        b.append((cx, cy, area))
        self._hist[label] = b[-4:]
        return vel, round(risk, 2)


# ─── OCR worker ───────────────────────────────────────────────────────────────

class OCRWorker:
    def __init__(self, reader):
        self._reader  = reader
        self._results: list = []
        self._lock    = threading.Lock()
        self._busy    = False

    def submit(self, frame):
        if self._busy: return
        self._busy = True
        threading.Thread(target=self._run, args=(frame.copy(),), daemon=True).start()

    def _run(self, frame):
        small = cv2.resize(frame, (640, 360))
        raw   = self._reader.readtext(small, detail=1, paragraph=False)
        texts = [r[1].strip() for r in raw if r[2] > 0.5 and len(r[1].strip()) > 1]
        with self._lock:
            self._results = texts[:6]
        self._busy = False

    def get(self):
        with self._lock: return list(self._results)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def side_of(cx, W):
    return "left" if cx < W/3 else ("center" if cx < 2*W/3 else "right")

def classify_scene(seen):
    best, n = "unknown", 0
    for scene, keys in SCENE_MAP.items():
        k = len(keys & seen)
        if k > n: best, n = scene, k
    return best

def ambient_level(frame):
    m = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return "dark" if m < 50 else ("dim" if m < 110 else "ok")

def face_in_box(gray, x1, y1, x2, y2):
    cut_y = y1 + int((y2 - y1) * 0.7)
    crop  = gray[y1:cut_y, x1:x2]
    if crop.size == 0:
        return False
    faces = _FACE_CASCADE.detectMultiScale(crop, scaleFactor=1.1, minNeighbors=1, minSize=(15, 15))
    return len(faces) > 0

def person_is_relevant(vel, prox, facing):
    """Emit person if close OR facing camera. Velocity tracked separately for decision agent."""
    return prox == "close" or facing


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "1")))
    args = ap.parse_args()

    print("Loading YOLOv8n...", file=sys.stderr)
    nav_model = YOLO("yolov8n.pt")

    stairs_model = doors_model = weapon_model = None
    for path, name, slot in [
        (STAIRS_MODEL_PATH, "stairs",  "stairs"),
        (DOORS_MODEL_PATH,  "doors",   "doors"),
        (WEAPON_MODEL_PATH, "weapons", "weapon"),
    ]:
        if os.path.exists(path):
            print(f"Loading {name} model...", file=sys.stderr)
            m = YOLO(path)
            if slot == "stairs":  stairs_model = m
            elif slot == "doors": doors_model  = m
            else:                 weapon_model = m
        else:
            print(f"No {name} model (skipping)", file=sys.stderr)

    print("Loading EasyOCR...", file=sys.stderr)
    ocr_worker = OCRWorker(easyocr.Reader(["en"], verbose=False))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}. Run list_cams.py.", file=sys.stderr)
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {args.camera}: {W}x{H}. Ctrl-C to stop.", file=sys.stderr)

    tracker      = Tracker()
    frame_n      = 0
    struct_cache: list = []
    weapon_cache: list = []
    eye_contact_timers: dict = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── YOLOv8n every frame ───────────────────────────────────────────────
        detections  = []
        seen: set   = set()
        total_ppl   = 0

        for box in nav_model.track(frame, persist=True, verbose=False)[0].boxes:
            label = nav_model.names[int(box.cls[0])]
            conf  = float(box.conf[0])
            if label not in NAV_CLASSES or conf < CONF_NAV: continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vel, risk = tracker.update(label, x1, y1, x2, y2, W, H)
            cx   = (x1+x2)/2
            prox = "close" if (x2-x1)/W > 0.30 else "far"
            s    = side_of(cx, W)

            if label == "person":
                total_ppl += 1
                facing = face_in_box(gray, x1, y1, x2, y2)
                if not person_is_relevant(vel, prox, facing):
                    seen.add(label)
                    continue   # skip — not approaching or not facing camera
                
                track_id = int(box.id[0]) if box.id is not None else None
                interacting = False
                
                if facing and track_id is not None:
                    if track_id not in eye_contact_timers:
                        eye_contact_timers[track_id] = time.time()
                    elif time.time() - eye_contact_timers[track_id] >= 5.0:
                        interacting = True
                elif track_id is not None and track_id in eye_contact_timers:
                    del eye_contact_timers[track_id]

                detections.append({
                    "label":          "person",
                    "confidence":     round(conf, 2),
                    "side":           s,
                    "proximity":      prox,
                    "velocity":       vel,
                    "collision_risk": risk,
                    "facing_camera":  bool(facing),
                    "interacting":    interacting,
                })
            else:
                detections.append({
                    "label":          label,
                    "confidence":     round(conf, 2),
                    "side":           s,
                    "proximity":      prox,
                    "velocity":       vel,
                    "collision_risk": risk,
                })
            seen.add(label)

        # ── Specialist models every 3 frames ──────────────────────────────────
        if frame_n % 3 == 0:
            struct_cache = []
            for model, threshold in filter(lambda x: x[0], [
                (stairs_model, CONF_STAIRS),
                (doors_model,  CONF_DOORS),
            ]):
                for box in model(frame, verbose=False)[0].boxes:
                    label = model.names[int(box.cls[0])]
                    conf  = float(box.conf[0])
                    if conf < threshold: continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    struct_cache.append({
                        "label":      "door" if "door" in label else label,
                        "confidence": round(conf, 2),
                        "side":       side_of((x1+x2)/2, W),
                    })

            weapon_cache = []
            if weapon_model:
                for box in weapon_model(frame, verbose=False)[0].boxes:
                    label = weapon_model.names[int(box.cls[0])]
                    conf  = float(box.conf[0])
                    if conf < CONF_WEAPON: continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    weapon_cache.append({
                        "label":      label,
                        "confidence": round(conf, 2),
                        "side":       side_of((x1+x2)/2, W),
                    })

        # ── OCR every 20 frames ───────────────────────────────────────────────
        if frame_n % 20 == 0:
            ocr_worker.submit(frame)

        # ── Emit JSON ─────────────────────────────────────────────────────────
        if detections or struct_cache or weapon_cache:
            print(json.dumps({
                "timestamp":     int(time.time() * 1000),
                "scene":         classify_scene(seen),
                "ambient_light": ambient_level(frame),
                "crowd_count":   total_ppl,
                "detections":    detections,
                "hazards":       struct_cache,
                "threats":       weapon_cache,
                "text_detected": ocr_worker.get(),
            }), flush=True)

        frame_n += 1


if __name__ == "__main__":
    main()
