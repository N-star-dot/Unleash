"""
Perception Viewer — service dog system.

Models:
  YOLOv8n              — COCO 80-class obstacle detection  (every frame)
  models/stairs/...    — custom stairs detector            (every 3 frames)
  models/doors_run/... — custom door detector              (every 3 frames)
  models/weapons/...   — custom weapon detector            (every 3 frames)
  EasyOCR              — street signs / text               (background thread)

Usage:
    python viewer.py
    python viewer.py --camera 0
Press Q to quit.
"""

import argparse
import os
import sys
import time
import threading

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr


# ─── Config ───────────────────────────────────────────────────────────────────

CONF_NAV      = 0.45
CONF_STRUCT   = 0.65   # stairs + doors — raised to reduce false positives on small objects
CONF_WEAPON   = 0.45   # weapon model threshold

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

SURGE_DELTA  = 4
SURGE_WINDOW = 10

_BASE = os.path.dirname(os.path.abspath(__file__))

STAIRS_MODEL_PATH = os.path.join(_BASE, "models/stairs/train/weights/best.pt")
DOORS_MODEL_PATH  = os.path.join(_BASE, "models/doors_run/train/weights/best.pt")
WEAPON_MODEL_PATH = os.path.join(_BASE, "models/weapons/train/weights/best.pt")
if not os.path.exists(WEAPON_MODEL_PATH):
    WEAPON_MODEL_PATH = os.path.join(_BASE, "best.pt")  # fallback

# BGR colors
C_SAFE   = ( 50, 200,  50)
C_WARN   = ( 30, 150, 255)
C_CLOSE  = ( 20,  50, 220)
C_THREAT = (  0,   0, 255)
C_HAZARD = (  0, 140, 255)
C_OCR    = (  0, 215, 255)
C_HUD    = (255, 255, 255)
C_BLACK  = (  0,   0,   0)


# ─── Motion tracker ───────────────────────────────────────────────────────────

class Tracker:
    def __init__(self):
        self._hist: dict = {}

    def update(self, label, x1, y1, x2, y2, W, H):
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        area   = (x2-x1) * (y2-y1)
        vel, risk = "unknown", 0.20
        prev = self._hist.get(label, [])
        if prev:
            best  = min(prev, key=lambda p: (cx-p[0])**2 + (cy-p[1])**2)
            ratio = area / best[2] if best[2] > 0 else 1.0
            dx    = cx - best[0]
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


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def pill(img, text, x, y, bg, fs=0.46):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    p = 3
    cv2.rectangle(img, (x, y-th-p*2), (x+tw+p*2, y+p), bg, -1)
    cv2.putText(img, text, (x+p, y-p), cv2.FONT_HERSHEY_SIMPLEX, fs, C_BLACK, 1, cv2.LINE_AA)

def banner(img, text, color, W, y_off=0):
    y1 = y_off + 26
    cv2.rectangle(img, (0, y_off), (W, y1), color, -1)
    cv2.putText(img, f"  {text}", (4, y_off+19), cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_BLACK, 2, cv2.LINE_AA)
    return y1

def classify_scene(labels):
    best, n = "unknown", 0
    for scene, keys in SCENE_MAP.items():
        k = len(keys & labels)
        if k > n: best, n = scene, k
    return best

def ambient(frame):
    m = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return "dark" if m < 50 else ("dim" if m < 110 else "ok")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "1")))
    args = ap.parse_args()

    print("Loading YOLOv8n...", file=sys.stderr)
    nav_model = YOLO("yolov8n.pt")

    stairs_model = doors_model = weapon_model = None
    for path, name, attr in [
        (STAIRS_MODEL_PATH, "stairs",  "stairs_model"),
        (DOORS_MODEL_PATH,  "doors",   "doors_model"),
        (WEAPON_MODEL_PATH, "weapons", "weapon_model"),
    ]:
        if os.path.exists(path):
            print(f"Loading {name} model...", file=sys.stderr)
            m = YOLO(path)
            if attr == "stairs_model":  stairs_model = m
            elif attr == "doors_model": doors_model  = m
            else:                       weapon_model = m
        else:
            print(f"No {name} model found (skipping)", file=sys.stderr)

    print("Loading EasyOCR...", file=sys.stderr)
    ocr_worker = OCRWorker(easyocr.Reader(["en"], verbose=False))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}. Run list_cams.py.", file=sys.stderr)
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {args.camera}: {W}x{H}. Press Q to quit.", file=sys.stderr)

    tracker      = Tracker()
    frame_n      = 0
    fps          = 0.0
    fps_t        = time.time()
    struct_cache: list = []
    weapon_cache: list = []
    pcount_hist:  list = []

    while True:
        ret, frame = cap.read()
        if not ret: continue

        # ── YOLOv8n — every frame ─────────────────────────────────────────────
        nav_dets: list = []
        seen: set = set()
        for box in nav_model(frame, verbose=False)[0].boxes:
            label = nav_model.names[int(box.cls[0])]
            conf  = float(box.conf[0])
            if label not in NAV_CLASSES or conf < CONF_NAV: continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vel, risk = tracker.update(label, x1, y1, x2, y2, W, H)
            cx   = (x1+x2)/2
            side = "L" if cx < W/3 else ("C" if cx < 2*W/3 else "R")
            prox = "close" if (x2-x1)/W > 0.30 else "far"
            nav_dets.append(dict(label=label, conf=conf, side=side,
                                 prox=prox, vel=vel, risk=risk, box=(x1,y1,x2,y2)))
            seen.add(label)

        # ── Specialist models — every 3 frames ────────────────────────────────
        if frame_n % 3 == 0:
            struct_cache = []
            for model in filter(None, [stairs_model, doors_model]):
                for box in model(frame, verbose=False)[0].boxes:
                    label = model.names[int(box.cls[0])]
                    conf  = float(box.conf[0])
                    if conf < CONF_STRUCT: continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx   = (x1+x2)/2
                    side = "L" if cx < W/3 else ("C" if cx < 2*W/3 else "R")
                    struct_cache.append(dict(label=label, conf=conf, side=side, box=(x1,y1,x2,y2)))

            weapon_cache = []
            if weapon_model:
                for box in weapon_model(frame, verbose=False)[0].boxes:
                    label = weapon_model.names[int(box.cls[0])]
                    conf  = float(box.conf[0])
                    if conf < CONF_WEAPON: continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx   = (x1+x2)/2
                    side = "L" if cx < W/3 else ("C" if cx < 2*W/3 else "R")
                    weapon_cache.append(dict(label=label, conf=conf, side=side, box=(x1,y1,x2,y2)))

        # ── OCR — background, every 20 frames ────────────────────────────────
        if frame_n % 20 == 0: ocr_worker.submit(frame)
        ocr_texts = ocr_worker.get()

        # ── Scene / crowd / light ─────────────────────────────────────────────
        scene  = classify_scene(seen)
        light  = ambient(frame)
        n_ppl  = sum(1 for d in nav_dets if d["label"] == "person")
        pcount_hist.append(n_ppl)
        pcount_hist = pcount_hist[-SURGE_WINDOW:]
        surge  = len(pcount_hist) == SURGE_WINDOW and \
                 (max(pcount_hist) - min(pcount_hist)) >= SURGE_DELTA

        now = time.time()
        fps = 0.85*fps + 0.15/max(now-fps_t, 1e-6)
        fps_t = now

        # ═══ DRAW ════════════════════════════════════════════════════════════

        top = 0
        if weapon_cache:
            top = banner(frame, f"WEAPON — {weapon_cache[0]['label'].upper()}", C_THREAT, W, top)
        if surge:
            top = banner(frame, f"CROWD SURGE — {n_ppl} people", C_WARN, W, top)
        if light in ("dark", "dim"):
            top = banner(frame, f"LOW LIGHT — {light}", C_WARN, W, top)

        for d in nav_dets:
            x1, y1, x2, y2 = d["box"]
            if d["vel"] == "approaching" and d["prox"] == "close":   color = C_CLOSE
            elif d["vel"] == "approaching" or d["prox"] == "close":  color = C_WARN
            else:                                                      color = C_SAFE
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            pill(frame, f" {d['label']} {d['conf']:.0%} {d['side']} {d['vel']} ",
                 x1, max(y1-4, top+18), color)

        for d in struct_cache:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(frame, (x1,y1), (x2,y2), C_HAZARD, 2)
            pill(frame, f" {d['label']} {d['conf']:.0%} {d['side']} ",
                 x1, max(y1-4, top+18), C_HAZARD)

        for d in weapon_cache:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(frame, (x1,y1), (x2,y2), C_THREAT, 3)
            pill(frame, f" {d['label']} {d['conf']:.0%} {d['side']} ",
                 x1, max(y1-4, top+18), C_THREAT)

        if ocr_texts:
            txt = f"  TEXT: {' | '.join(ocr_texts)}"
            (sw, sh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(frame, (0, H-sh-12), (W, H), C_BLACK, -1)
            cv2.putText(frame, txt, (4, H-6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_OCR, 1, cv2.LINE_AA)

        # HUD
        loaded = " ".join(filter(None, [
            "stairs" if stairs_model else None,
            "doors"  if doors_model  else None,
            "weapon" if weapon_model else None,
        ])) or "nav-only"
        for i, line in enumerate([
            f"  cam:{args.camera}  {fps:.1f}fps  scene:{scene}  light:{light}  [{loaded}]",
            f"  people:{n_ppl}  {'SURGE  ' if surge else ''}struct:{len(struct_cache)}  weapons:{len(weapon_cache)}",
        ]):
            yh = top + 18 + i*18
            (lw, lh), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(frame, (0, yh-lh-2), (lw+4, yh+4), C_BLACK, -1)
            cv2.putText(frame, line, (2, yh), cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_HUD, 1, cv2.LINE_AA)

        cv2.imshow("service dog — perception", frame)
        frame_n += 1
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
