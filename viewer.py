"""
Enhanced Perception Viewer — service dog system.

Model stack (frame-scheduled to stay real-time):
  YOLOv8n       — COCO obstacle detection          (every frame)
  YOLO-World    — Stairs / curbs / doors / weapons  (every 5 frames, cached)
  MediaPipe Pose — Raised-arm / aggressive posture  (every 3 frames)
  EasyOCR       — Street signs, bus numbers, labels (background thread)

Per-frame signals:
  Motion vectors + collision risk (bbox delta tracking)
  Crowd density + surge detection
  Heuristic scene classification (street / indoor / nature)
  Ambient light warning

Usage:
    python viewer.py               # camera 1 (default)
    python viewer.py --camera 0
Press Q to quit.
"""

import argparse
import os
import sys
import time
import threading

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import easyocr


# ─── Config ───────────────────────────────────────────────────────────────────

CONF_NAV    = 0.45   # YOLOv8n threshold
CONF_WORLD  = 0.28   # YOLO-World — lower because open-vocab is less precise
CONF_WEAPON = 0.32   # weapon sub-threshold (extra sensitive)
CONF_POSE   = 0.50   # MediaPipe min confidence

NAV_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "fire hydrant", "bench",
    "dog", "cat", "chair", "dining table", "couch",
}

HAZARD_PROMPTS = ["stairs", "step", "curb", "pothole", "wet floor sign", "open door", "doorway"]
WEAPON_PROMPTS = ["knife", "gun", "pistol", "baseball bat", "sword"]
WORLD_CLASSES  = HAZARD_PROMPTS + WEAPON_PROMPTS
WEAPON_SET     = set(WEAPON_PROMPTS)

SCENE_MAP = {
    "street": {"car", "truck", "bus", "traffic light", "stop sign", "motorcycle", "bicycle"},
    "indoor": {"chair", "dining table", "couch", "tv", "laptop", "keyboard"},
    "nature": {"bench", "bird", "dog", "cat"},
}

SURGE_DELTA   = 4   # person-count jump that triggers a surge alert
SURGE_WINDOW  = 10  # frames to look back for surge

# BGR colors
C_SAFE   = ( 50, 200,  50)   # green  — far / receding
C_WARN   = ( 30, 150, 255)   # amber  — approaching
C_CLOSE  = ( 20,  50, 220)   # red    — close + approaching
C_THREAT = (  0,   0, 255)   # bright red — weapon
C_HAZARD = (  0, 140, 255)   # orange — structural hazard
C_OCR    = (  0, 215, 255)   # yellow — detected text
C_HUD    = (255, 255, 255)
C_BLACK  = (  0,   0,   0)


# ─── Motion tracker ───────────────────────────────────────────────────────────

class Tracker:
    """
    Nearest-neighbour frame-to-frame tracker per label.
    Estimates velocity and collision risk from bounding-box area change.
    """

    def __init__(self):
        self._hist: dict[str, list] = {}  # label -> [(cx,cy,area), ...]

    def update(self, label: str, x1, y1, x2, y2, W, H) -> tuple[str, float]:
        """Returns (velocity_str, risk 0-1)."""
        cx   = (x1 + x2) / 2.0
        cy   = (y1 + y2) / 2.0
        area = (x2 - x1) * (y2 - y1)

        vel, risk = "unknown", 0.20

        prev = self._hist.get(label, [])
        if prev:
            best        = min(prev, key=lambda p: (cx-p[0])**2 + (cy-p[1])**2)
            pcx, pcy, parea = best
            ratio       = area / parea if parea > 0 else 1.0
            dx          = cx - pcx

            if   ratio > 1.10:           vel = "approaching"
            elif ratio < 0.90:           vel = "receding"
            elif abs(dx) > W * 0.015:    vel = "crossing-L" if dx < 0 else "crossing-R"
            else:                        vel = "stationary"

            norm  = min(1.0, area / (W * H * 0.20))
            appr  = max(0.0, (ratio - 1.0) * 3.0)
            risk  = min(1.0, norm * 0.6 + appr * 0.4)
            if vel == "receding":
                risk *= 0.25

        bucket = self._hist.setdefault(label, [])
        bucket.append((cx, cy, area))
        self._hist[label] = bucket[-4:]

        return vel, round(risk, 2)


# ─── OCR worker ───────────────────────────────────────────────────────────────

class OCRWorker:
    """Runs EasyOCR in a daemon thread; caller polls get() for latest results."""

    def __init__(self, reader: easyocr.Reader):
        self._reader  = reader
        self._results: list[str] = []
        self._lock    = threading.Lock()
        self._busy    = False

    def submit(self, frame: np.ndarray) -> None:
        if self._busy:
            return
        self._busy = True
        threading.Thread(target=self._run, args=(frame.copy(),), daemon=True).start()

    def _run(self, frame: np.ndarray) -> None:
        small = cv2.resize(frame, (640, 360))
        raw   = self._reader.readtext(small, detail=1, paragraph=False)
        texts = [r[1].strip() for r in raw if r[2] > 0.5 and len(r[1].strip()) > 1]
        with self._lock:
            self._results = texts[:6]
        self._busy = False

    def get(self) -> list[str]:
        with self._lock:
            return list(self._results)


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def pill(img, text: str, x: int, y: int, bg: tuple, fs: float = 0.46) -> None:
    """Filled color chip with dark text."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    p = 3
    cv2.rectangle(img, (x, y - th - p * 2), (x + tw + p * 2, y + p), bg, -1)
    cv2.putText(img, text, (x + p, y - p),
                cv2.FONT_HERSHEY_SIMPLEX, fs, C_BLACK, 1, cv2.LINE_AA)


def banner(img, text: str, color: tuple, W: int, y_off: int = 0) -> int:
    """Full-width alert bar. Returns bottom y so other banners can stack."""
    h = 26
    y0, y1 = y_off, y_off + h
    cv2.rectangle(img, (0, y0), (W, y1), color, -1)
    cv2.putText(img, f"  {text}", (4, y0 + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_BLACK, 2, cv2.LINE_AA)
    return y1


# ─── Env helpers ─────────────────────────────────────────────────────────────

def classify_scene(labels: set) -> str:
    best, n = "unknown", 0
    for scene, keys in SCENE_MAP.items():
        k = len(keys & labels)
        if k > n:
            best, n = scene, k
    return best


def ambient(frame: np.ndarray) -> str:
    m = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return "dark" if m < 50 else ("dim" if m < 110 else "ok")


def risk_level(nav: list, world: list, aggressive: bool) -> str:
    if aggressive or any(d["label"] in WEAPON_SET for d in world):
        return "HIGH"
    max_r = max((d["risk"] for d in nav), default=0.0)
    if max_r > 0.60:                       return "HIGH"
    if max_r > 0.35 or world:             return "MED"
    return "LOW"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int,
                    default=int(os.environ.get("CAMERA_INDEX", "1")))
    args = ap.parse_args()

    # ── Model loading ─────────────────────────────────────────────────────────
    print("Loading YOLOv8n...", file=sys.stderr)
    nav_model = YOLO("yolov8n.pt")

    print("Loading YOLO-World (stairs / curbs / weapons)...", file=sys.stderr)
    world_model = YOLO("yolov8s-worldv2.pt")
    world_model.set_classes(WORLD_CLASSES)

    print("Loading MediaPipe Pose...", file=sys.stderr)
    pose_det = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=CONF_POSE,
        min_tracking_confidence=0.5,
    )

    print("Loading EasyOCR (first run ~100 MB download)...", file=sys.stderr)
    ocr_worker = OCRWorker(easyocr.Reader(["en"], verbose=False))

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}. Run list_cams.py.", file=sys.stderr)
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {args.camera}: {W}x{H}. Press Q to quit.", file=sys.stderr)

    # ── State ─────────────────────────────────────────────────────────────────
    tracker     = Tracker()
    frame_n     = 0
    fps         = 0.0
    fps_t       = time.time()
    world_cache: list[dict] = []
    pose_alert  = False
    pcount_hist: list[int]  = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # ── YOLOv8n (every frame) ─────────────────────────────────────────────
        nav_res  = nav_model(frame, verbose=False)[0]
        nav_dets: list[dict] = []
        seen: set[str] = set()

        for box in nav_res.boxes:
            label = nav_model.names[int(box.cls[0])]
            conf  = float(box.conf[0])
            if label not in NAV_CLASSES or conf < CONF_NAV:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vel, risk = tracker.update(label, x1, y1, x2, y2, W, H)
            bw   = x2 - x1
            cx   = (x1 + x2) / 2
            side = "L" if cx < W/3 else ("C" if cx < 2*W/3 else "R")
            prox = "close" if bw / W > 0.30 else "far"
            nav_dets.append(dict(label=label, conf=conf, side=side,
                                 prox=prox, vel=vel, risk=risk,
                                 box=(x1, y1, x2, y2)))
            seen.add(label)

        # ── YOLO-World (every 5 frames) ───────────────────────────────────────
        if frame_n % 5 == 0:
            wr = world_model(frame, verbose=False)[0]
            world_cache = []
            for box in wr.boxes:
                label = world_model.names[int(box.cls[0])]
                conf  = float(box.conf[0])
                thr   = CONF_WEAPON if label in WEAPON_SET else CONF_WORLD
                if conf < thr:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx   = (x1 + x2) / 2
                side = "L" if cx < W/3 else ("C" if cx < 2*W/3 else "R")
                world_cache.append(dict(label=label, conf=conf, side=side,
                                        box=(x1, y1, x2, y2)))

        # ── MediaPipe Pose (every 3 frames) ───────────────────────────────────
        if frame_n % 3 == 0:
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_res = pose_det.process(rgb)
            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                # wrist y < shoulder y means arm is raised (y=0 is top of image)
                L = lm[15].y < lm[11].y - 0.05 and lm[15].visibility > 0.4
                R = lm[16].y < lm[12].y - 0.05 and lm[16].visibility > 0.4
                pose_alert = L or R
            else:
                pose_alert = False

        # ── OCR (background thread, every 20 frames) ──────────────────────────
        if frame_n % 20 == 0:
            ocr_worker.submit(frame)
        ocr_texts = ocr_worker.get()

        # ── Scene / crowd / light ─────────────────────────────────────────────
        scene  = classify_scene(seen)
        light  = ambient(frame)
        n_ppl  = sum(1 for d in nav_dets if d["label"] == "person")
        pcount_hist.append(n_ppl)
        pcount_hist = pcount_hist[-SURGE_WINDOW:]
        surge  = len(pcount_hist) == SURGE_WINDOW and \
                 (max(pcount_hist) - min(pcount_hist)) >= SURGE_DELTA
        rlevel = risk_level(nav_dets, world_cache, pose_alert)

        # ── FPS ───────────────────────────────────────────────────────────────
        now   = time.time()
        fps   = 0.85 * fps + 0.15 / max(now - fps_t, 1e-6)
        fps_t = now

        # ═══ DRAW ════════════════════════════════════════════════════════════

        # Alert banners (stack from top)
        top = 0
        if pose_alert:
            top = banner(frame, "AGGRESSIVE POSTURE DETECTED", C_THREAT, W, top)
        if any(d["label"] in WEAPON_SET for d in world_cache):
            top = banner(frame, "WEAPON DETECTED", C_THREAT, W, top)
        if surge:
            top = banner(frame, f"CROWD SURGE — {n_ppl} people", C_WARN, W, top)
        if light in ("dark", "dim"):
            top = banner(frame, f"LOW LIGHT — {light}", C_WARN, W, top)

        # Nav detections
        for d in nav_dets:
            x1, y1, x2, y2 = d["box"]
            if d["vel"] == "approaching" and d["prox"] == "close":
                color = C_CLOSE
            elif d["vel"] == "approaching" or d["prox"] == "close":
                color = C_WARN
            else:
                color = C_SAFE
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f" {d['label']} {d['conf']:.0%} {d['side']} {d['vel']} r:{d['risk']:.0%} "
            pill(frame, tag, x1, max(y1 - 4, top + 18), color)

        # World detections (hazards + weapons)
        for d in world_cache:
            x1, y1, x2, y2 = d["box"]
            is_weapon = d["label"] in WEAPON_SET
            color = C_THREAT if is_weapon else C_HAZARD
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_weapon else 2)
            tag = f" {d['label']} {d['conf']:.0%} {d['side']} "
            pill(frame, tag, x1, max(y1 - 4, top + 18), color)

        # OCR strip at bottom
        if ocr_texts:
            txt = f"  TEXT: {' | '.join(ocr_texts)}"
            (sw, sh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(frame, (0, H - sh - 12), (W, H), C_BLACK, -1)
            cv2.putText(frame, txt, (4, H - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_OCR, 1, cv2.LINE_AA)

        # HUD — top-left
        rc = C_THREAT if rlevel == "HIGH" else (C_WARN if rlevel == "MED" else C_SAFE)
        for i, line in enumerate([
            f"  cam:{args.camera}  {fps:.1f}fps  scene:{scene}  light:{light}",
            f"  people:{n_ppl}  {'SURGE  ' if surge else ''}risk:{rlevel}",
        ]):
            yh = top + 18 + i * 18
            (lw, lh), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(frame, (0, yh - lh - 2), (lw + 4, yh + 4), C_BLACK, -1)
            cv2.putText(frame, line, (2, yh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        rc if i == 1 else C_HUD, 1, cv2.LINE_AA)

        cv2.imshow("service dog — perception", frame)
        frame_n += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    pose_det.close()
    cv2.destroyAllWindows()
    print("Stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
