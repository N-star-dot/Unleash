"""
Enumerate all cameras visible to OpenCV on macOS.
Run this first to find which index is your iPhone (Continuity Camera).
"""
import cv2
import sys
import os

MAX_INDEX = 10

print("Scanning cameras (index 0 to {})...".format(MAX_INDEX - 1), file=sys.stderr)


def _open_camera_silent(idx: int):
    """Open camera while suppressing C++-level stderr noise from AVFoundation."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    finally:
        os.dup2(saved_stderr, 2)
        os.close(devnull)
        os.close(saved_stderr)
    return cap


found = []
for idx in range(MAX_INDEX):
    cap = _open_camera_silent(idx)
    if not cap.isOpened():
        cap.release()
        continue

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    found.append({"index": idx, "width": width, "height": height, "fps": fps})
    print(f"  index {idx}: {width}x{height} @ {fps:.1f} fps")

if not found:
    print("No cameras found. Check System Preferences > Privacy > Camera.", file=sys.stderr)
    sys.exit(1)

print(f"\nFound {len(found)} camera(s). iPhone via Continuity Camera is usually the highest index.", file=sys.stderr)
