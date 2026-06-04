import cv2
import sys

for i in range(5):
    print(f"Testing camera index {i}...")
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"  -> Index {i}: Failed to open.")
        continue
        
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"  -> Index {i}: SUCCESS! Read frame shape {frame.shape}")
    else:
        print(f"  -> Index {i}: Opened, but cap.read() returned False (hanging/blocked).")
    
    cap.release()

# touched 2026-06-03
