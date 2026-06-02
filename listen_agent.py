"""
Active Listening Agent — always-on microphone, no push-to-talk.

How it works:
  1. Calibrates silence threshold from ambient noise on startup (~1s)
  2. Continuously monitors mic energy — starts recording when you speak
  3. Stops recording after 0.8s of silence
  4. Transcribes via Google STT (free, no key needed)
  5. Emits one JSON line per utterance to stdout

Output contract:
  {"timestamp": <unix_ms>, "text": "call mom", "source": "user"}

Usage:
    python listen_agent.py
    python listen_agent.py | python decision_agent.py
"""

import json
import os
import sys
import tempfile
import time
import threading
import queue

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import speech_recognition as sr


# ─── Config ──────────────────────────────────────────────────────────────────

SAMPLE_RATE     = 16000
FRAME_MS        = 100       # ms per audio chunk (10 chunks/sec)
FRAME_SAMPLES   = int(SAMPLE_RATE * FRAME_MS / 1000)
SILENCE_FRAMES  = 8         # 0.8s of silence ends an utterance
CALIBRATE_SECS  = 1.5       # ambient noise calibration window
ENERGY_MARGIN   = 1.6       # multiplier above ambient to count as speech
MIN_SPEECH_FRAMES = 3       # ignore utterances shorter than 0.3s (noise bursts)
def _resolve_mic() -> int | None:
    if "MIC_DEVICE" in os.environ:
        return int(os.environ["MIC_DEVICE"])
    import sounddevice as _sd
    for d in _sd.query_devices():
        if "MacBook Air Microphone" in d["name"] and d["max_input_channels"] > 0:
            return d["index"]
    return None  # fall back to system default

MIC_DEVICE = _resolve_mic()


# ─── Audio capture ───────────────────────────────────────────────────────────

audio_q: queue.Queue = queue.Queue()

def _audio_callback(indata, frames, t, status):
    audio_q.put(indata.copy())

def rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


# ─── Transcription ───────────────────────────────────────────────────────────

recognizer = sr.Recognizer()

def transcribe(frames: list[np.ndarray]) -> str:
    audio = np.concatenate(frames, axis=0)
    tmp   = tempfile.mktemp(suffix=".wav")
    wavfile.write(tmp, SAMPLE_RATE, audio)
    with sr.AudioFile(tmp) as src:
        data = recognizer.record(src)
    try:
        return recognizer.recognize_google(data)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"STT error: {e}", file=sys.stderr)
        return ""


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Calibrating ambient noise...", file=sys.stderr)

    # Calibrate: collect CALIBRATE_SECS of ambient audio, compute baseline RMS
    calib_frames = []
    calib_done = threading.Event()

    def _calib_cb(indata, frames, t, status):
        calib_frames.append(indata.copy())
        if len(calib_frames) >= int(CALIBRATE_SECS * 1000 / FRAME_MS):
            calib_done.set()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=FRAME_SAMPLES, callback=_calib_cb,
                        device=MIC_DEVICE):
        calib_done.wait()

    ambient_rms   = float(np.mean([rms(f) for f in calib_frames]))
    speech_thresh = max(ambient_rms * ENERGY_MARGIN, 100)  # lowered floor: 300→100
    print(f"Ambient RMS: {ambient_rms:.0f}  |  Speech threshold: {speech_thresh:.0f}",
          file=sys.stderr)
    print("Listening... speak now, watching energy levels:", file=sys.stderr)

    # Main listen loop
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=FRAME_SAMPLES, callback=_audio_callback,
                        device=MIC_DEVICE):

        speech_frames: list[np.ndarray] = []
        silence_count = 0
        in_speech     = False
        debug_counter = 0

        while True:
            try:
                chunk = audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            energy = rms(chunk)

            debug_counter += 1

            if energy > speech_thresh:
                # Speech detected
                in_speech     = True
                silence_count = 0
                speech_frames.append(chunk)
            elif in_speech:
                # Silence during/after speech
                speech_frames.append(chunk)  # include trailing silence for natural boundary
                silence_count += 1

                if silence_count >= SILENCE_FRAMES:
                    # End of utterance — transcribe
                    in_speech = False
                    if len(speech_frames) >= MIN_SPEECH_FRAMES:
                        text = transcribe(speech_frames)
                        if text:
                            payload = {
                                "timestamp": int(time.time() * 1000),
                                "text":      text,
                                "source":    "user",
                            }
                            print(json.dumps(payload), flush=True)
                            print(f"Heard: {text}", file=sys.stderr)
                    speech_frames = []
                    silence_count = 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
