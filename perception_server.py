import threading
import time
import cv2
import argparse
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

# Import the existing vision generator logic
from perception import vision_generator

app = FastAPI(title="Unleash Perception Service")

latest_frame = None
latest_payload = {"status": "initializing"}

def camera_loop(camera_index):
    global latest_frame, latest_payload
    print(f"Starting camera loop with index {camera_index}...")
    for frame_rgb, payload in vision_generator(camera_index=camera_index):
        if frame_rgb is not None:
            # Convert RGB back to BGR for cv2 encoding
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            latest_frame = frame_bgr
        if payload is not None:
            latest_payload = payload

@app.get("/payload")
def get_payload():
    return latest_payload

def generate_mjpeg():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

@app.get("/stream")
def get_stream():
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()

    # Start the camera in a background thread immediately before the server blocks
    thread = threading.Thread(target=camera_loop, args=(args.camera,), daemon=True)
    thread.start()

    # Wait briefly for YOLO to at least start loading before letting Uvicorn boot up
    time.sleep(1)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
