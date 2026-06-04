"""
Perception Agent — Unleash service dog.

Real-time obstacle + hazard detection feeding the decision/memory brain (app.py).
Two entry points share ONE detection core, so the Streamlit app and the CLI stay
in sync:

    vision_generator(camera_index)  -> generator of (frame_rgb, payload)  [ui.py]
    main()                          -> headless JSON-lines stream to stdout [CLI / piping]

What it detects
  * People & street objects — COCO YOLOv8n, every frame, with interaction tracking
    (approaching / crossing / collision_risk): the "is it coming at me?" signal.
  * Structural hazards — stairs, doors, ladders — every Nth frame. Uses custom
    weights under models/ if present (drop in trained stairs/doors detectors),
    otherwise falls back to the Open Images V7 model, which knows Door/Stairs/Ladder
    out of the box. Swap or tune the weights later without touching anything else.

JSON payload contract (consumed by app.py:process_vision_queue + the orchestrator):
{
  "timestamp": <unix_ms>,
  "scene": "street"|"indoor"|"nature"|"unknown",
  "ambient_light": "ok"|"dim"|"dark",
  "crowd_count": <int>,
  "risk_level": "LOW"|"MED"|"HIGH",        # drives the brain's VisionAlert
  "detections": [ {label, confidence, side, proximity, velocity, collision_risk} ],
  "hazards":    [ {label: "stairs"|"door"|"ladder", confidence, side, proximity} ]
}

Usage:
    python perception.py --camera 0
    python perception.py --camera 0 | python decision_agent.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading

import cv2
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))

# ── detection config ─────────────────────────────────────────────────────────
CONF_NAV = 0.45
CONF_STRUCT = 0.40           # default floor for structural classes (ladder/escalator)
# Per-label structural cutoffs. Stairs/doors false-positive badly in plain rooms,
# so gate them MUCH higher — only report when the model is very sure. (Stairs is
# the most safety-critical hazard for VI users, so don't push so high you miss
# real steps; 0.85 kills the office false positives seen in testing.)
CONF_STRUCT_BY_LABEL = {
    "stairs": 0.95,
    "door":   0.95,
}
STRUCT_EVERY = 3             # run the (heavier) structural model every Nth frame

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
# Open-Images-V7 class names we treat as structural hazards (fallback weights).
OIV7_STRUCT = {"door", "stairs", "ladder", "escalator"}

# Drop trained weights here (e.g. Sofi's). If any exist they REPLACE the oiv7
# fallback, so tuning later is just a matter of dropping in .pt files.
CUSTOM_STRUCT_PATHS = [
    os.path.join(_BASE, "models/stairs/train/weights/best.pt"),
    os.path.join(_BASE, "models/doors_run/train/weights/best.pt"),
]

_BASE             = os.path.dirname(os.path.abspath(__file__))
_LIVE_PERCEPTION  = os.path.join(_BASE, "live_perception.json")
_LIVE_FRAME       = os.path.join(_BASE, "live_frame.jpg")   # annotated frame for the dashboard
STAIRS_MODEL_PATH = os.path.join(_BASE, "models/stairs/train/weights/best.pt")
DOORS_MODEL_PATH  = os.path.join(_BASE, "models/doors_run/train/weights/best.pt")
WEAPON_MODEL_PATH = os.path.join(_BASE, "models/weapons/train/weights/best.pt")
if not os.path.exists(WEAPON_MODEL_PATH):
    WEAPON_MODEL_PATH = os.path.join(_BASE, "best.pt")

def _side(cx: float, W: int) -> str:
    return "left" if cx < W / 3 else ("center" if cx < 2 * W / 3 else "right")


def _ambient(frame) -> str:
    m = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    return "dark" if m < 50 else ("dim" if m < 110 else "ok")


def _scene(seen: set) -> str:
    best, n = "unknown", 0
    for scene, keys in SCENE_MAP.items():
        k = len(keys & seen)
        if k > n:
            best, n = scene, k
    return best


def _norm_struct(label: str) -> str:
    """Normalise a structural class name to stairs/door/ladder/escalator."""
    l = label.lower()
    if "stair" in l:
        return "stairs"
    if "ladder" in l:
        return "ladder"
    if "escalator" in l:
        return "escalator"
    if "door" in l and "handle" not in l:
        return "door"
    return l


class Tracker:
    """Nearest-neighbour velocity + collision-risk per label (the interaction signal)."""

    def __init__(self):
        self._hist: dict = {}

    def update(self, label, x1, y1, x2, y2, W, H):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        area = (x2 - x1) * (y2 - y1)
        vel, risk = "unknown", 0.20
        prev = self._hist.get(label, [])
        if prev:
            best = min(prev, key=lambda p: (cx - p[0]) ** 2 + (cy - p[1]) ** 2)
            ratio = area / best[2] if best[2] > 0 else 1.0
            dx = cx - best[0]
            if ratio > 1.10:
                vel = "approaching"
            elif ratio < 0.90:
                vel = "receding"
            elif abs(dx) > W * 0.015:
                vel = "crossing-L" if dx < 0 else "crossing-R"
            else:
                vel = "stationary"
            norm = min(1.0, area / (W * H * 0.20))
            risk = min(1.0, norm * 0.6 + max(0.0, (ratio - 1.0) * 3.0) * 0.4)
            if vel == "receding":
                risk *= 0.25
        b = self._hist.setdefault(label, [])
        b.append((cx, cy, area))
        self._hist[label] = b[-4:]
        return vel, round(risk, 2)


class Perception:
    """Per-frame detection core shared by the Streamlit generator and the CLI."""

    def __init__(self):
        self.tracker = Tracker()
        self.frame_n = 0
        self._struct_cache: list = []
        self._nav = None
        self._struct: list = []           # list of YOLO models
        self._struct_custom = False

    def _load(self):
        if self._nav is not None:
            return
        from ultralytics import YOLO       # lazy import keeps `import perception` light
        self._nav = YOLO(os.path.join(_BASE, "yolov8n.pt"))
        existing = [p for p in CUSTOM_STRUCT_PATHS if os.path.exists(p)]
        if existing:
            self._struct = [YOLO(p) for p in existing]
            self._struct_custom = True
            print(f"[perception] custom structural weights: {existing}", file=sys.stderr)
        else:
            self._struct = [YOLO("yolov8n-oiv7.pt")]   # auto-downloads once, then cached
            self._struct_custom = False
            print("[perception] Open Images V7 fallback for stairs/doors", file=sys.stderr)

    def analyze(self, frame):
        """frame (BGR ndarray) -> (annotated_rgb, payload|None)."""
        self._load()
        H, W = frame.shape[:2]
        detections, seen = [], set()

        # ── nav model: people + street objects, every frame ──────────────────
        for box in self._nav(frame, verbose=False)[0].boxes:
            label = self._nav.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if label not in NAV_CLASSES or conf < CONF_NAV:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vel, risk = self.tracker.update(label, x1, y1, x2, y2, W, H)
            prox = "close" if (x2 - x1) / W > 0.30 else "far"
            detections.append({
                "label": label, "confidence": round(conf, 2),
                "side": _side((x1 + x2) / 2, W), "proximity": prox,
                "velocity": vel, "collision_risk": risk,
            })
            seen.add(label)
            color = (0, 0, 255) if (prox == "close" or vel == "approaching") else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {vel}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ── structural model: stairs/doors, every Nth frame (cached between) ──
        if self.frame_n % STRUCT_EVERY == 0:
            hazards = []
            for model in self._struct:
                for box in model(frame, verbose=False)[0].boxes:
                    raw = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if not self._struct_custom and raw.lower() not in OIV7_STRUCT:
                        continue
                    label = _norm_struct(raw)
                    # Per-label confidence gate — stairs/doors need to clear a
                    # much higher bar to suppress false positives in empty rooms.
                    if conf < CONF_STRUCT_BY_LABEL.get(label, CONF_STRUCT):
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    hazards.append({
                        "label": label, "confidence": round(conf, 2),
                        "side": _side((x1 + x2) / 2, W),
                        "proximity": "close" if (x2 - x1) / W > 0.30 else "far",
                        "_box": (x1, y1, x2, y2),
                    })
            self._struct_cache = hazards
        for h in self._struct_cache:
            x1, y1, x2, y2 = h["_box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(frame, h["label"], (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

        self.frame_n += 1
        hazards_out = [{k: v for k, v in h.items() if k != "_box"} for h in self._struct_cache]
        crowd = sum(1 for d in detections if d["label"] == "person")
        payload = None
        if detections or hazards_out:
            payload = {
                "timestamp": int(time.time() * 1000),
                "scene": _scene(seen),
                "ambient_light": _ambient(frame),
                "crowd_count": crowd,
                "risk_level": self._risk(detections, hazards_out, crowd),
                "detections": detections,
                "hazards": hazards_out,
            }
        # Write the annotated frame so the dashboard (ui.py) can show the live cam.
        # Every frame (not just on detections) so the view stays smooth. Atomic.
        # Encode explicitly — imwrite infers format from the extension, which the
        # ".tmp" suffix would break, so go through imencode + raw bytes instead.
        try:
            _ok, _buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if _ok:
                _ftmp = _LIVE_FRAME + ".tmp"
                with open(_ftmp, "wb") as _ff:
                    _ff.write(_buf.tobytes())
                os.replace(_ftmp, _LIVE_FRAME)
        except Exception:
            pass
        # Write latest perception state so voice_bridge can read it
        if payload:
            try:
                _tmp = _LIVE_PERCEPTION + ".tmp"
                with open(_tmp, "w") as _f:
                    json.dump(payload, _f)
                os.replace(_tmp, _LIVE_PERCEPTION)
            except Exception:
                pass
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), payload

    @staticmethod
    def _risk(detections, hazards, crowd) -> str:
        """Fuse collision risk, structural hazards and crowd into LOW/MED/HIGH."""
        max_coll = max((d["collision_risk"] for d in detections), default=0.0)
        level = 2 if max_coll > 0.60 else (1 if max_coll > 0.35 else 0)
        for h in hazards:
            if h["label"] == "stairs":               # fall hazard = most critical for VI users
                level = max(level, 2 if h["proximity"] == "close" else 1)
            elif h["proximity"] == "close":          # door/ladder right ahead = obstacle
                level = max(level, 1)
        if crowd > 5:
            level = max(level, 2)
        elif crowd >= 3:
            level = max(level, 1)
        return ("LOW", "MED", "HIGH")[level]


# Shared single-frame analyzer for the PHONE / browser camera (st.camera_input).
_PHOTO_PERCEP = None


def analyze_image_bytes(data):
    """Decode a still image (e.g. st.camera_input bytes) -> (annotated_rgb, payload).

    Lets the PHONE be the camera: open the app on the phone's browser, capture a
    frame, and run the same Perception.analyze() used by the live webcam path —
    so phone snapshots feed the brain with the identical JSON contract."""
    global _PHOTO_PERCEP
    if not data:
        return None, None
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)   # -> BGR
    except Exception:
        return None, None
    if frame is None:
        return None, None
    if _PHOTO_PERCEP is None:
        _PHOTO_PERCEP = Perception()
    return _PHOTO_PERCEP.analyze(frame)


def vision_generator(camera_index=1):
    """Yield (frame_rgb, payload) per frame for Streamlit. (None, {'error'}) on failure."""
    percep = Perception()
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)        # fall back to default index
    if not cap.isOpened():
        yield None, {"error": "Camera not found"}
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            yield percep.analyze(frame)
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser(description="Unleash perception — headless JSON stream")
    ap.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "1")))
    args = ap.parse_args()

    percep = Perception()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(json.dumps({"error": f"cannot open camera {args.camera}; run list_cams.py"}), flush=True)
        sys.exit(1)

    print(f"[perception] camera {args.camera} open; Ctrl-C to stop.", file=sys.stderr)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            _, payload = percep.analyze(frame)
            if payload:
                print(json.dumps(payload), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()
