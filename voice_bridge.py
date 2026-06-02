"""
Voice Bridge — connects listen_agent → brain → spoken response.

Loads the brain in a background thread so the mic starts immediately.
You'll hear "Brain ready" once the pipeline is loaded.

Usage:
    python listen_agent.py | python voice_bridge.py
"""

import json
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MEM0_ENABLE_TELEMETRY", "false")
os.environ.setdefault("MEM0_TELEMETRY", "false")

# Load .env so GROQ_API_KEY etc are always available without manual export
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

VOICE = os.environ.get("VOICE", "Samantha")
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIVE_PERCEPTION = os.path.join(_HERE, "live_perception.json")
_LIVE_VOICE = os.path.join(_HERE, "live_voice.json")


def _write_live_voice(heard: str, thinking: str, response: str, action: str) -> None:
    """Write the latest voice exchange so the dashboard (ui.py) can display it live."""
    try:
        tmp = _LIVE_VOICE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "timestamp": int(time.time() * 1000),
                "heard":     heard,
                "thinking":  thinking,
                "response":  response,
                "action":    action,
            }, f)
        os.replace(tmp, _LIVE_VOICE)  # atomic — UI never reads a half-written file
    except Exception as e:
        print(f"[voice_bridge] could not write live_voice.json: {e}", file=sys.stderr)


def _read_live_perception() -> dict:
    """Read the latest YOLO frame written by perception.py. Returns {} if stale, missing, or dark (unreliable)."""
    try:
        with open(_LIVE_PERCEPTION) as f:
            data = json.load(f)
        age = time.time() - data.get("timestamp", 0) / 1000.0
        if age >= 5 or data.get("ambient_light") == "dark":
            return {}
        return data
    except Exception:
        return {}

# ── Lazy brain loader ─────────────────────────────────────────────────────────

_brain_ready  = threading.Event()
_brain_error  = None
_pipeline_fns = {}

def _load_brain():
    global _brain_error
    try:
        print("Loading brain (in background)...", file=sys.stderr)
        # Suppress weave/gql/wandb noise at the OS level during import
        _devnull = os.open(os.devnull, os.O_WRONLY)
        _saved   = os.dup(2)
        os.dup2(_devnull, 2)
        try:
            from app import (
                process_biometrics,
                process_vision_queue,
                pattern_detector,
                memory_agent,
                behavior_orchestrator,
                action_dispatcher,
            )
        finally:
            os.dup2(_saved, 2)
            os.close(_devnull)
            os.close(_saved)
        _pipeline_fns.update({
            "process_biometrics":    process_biometrics,
            "process_vision_queue":  process_vision_queue,
            "pattern_detector":      pattern_detector,
            "memory_agent":          memory_agent,
            "behavior_orchestrator": behavior_orchestrator,
            "action_dispatcher":     action_dispatcher,
        })
        print("\n✅ Brain ready — you can speak now!\n", file=sys.stderr)
    except Exception as e:
        _brain_error = e
        print(f"Brain load failed: {e}", file=sys.stderr)
    finally:
        _brain_ready.set()

threading.Thread(target=_load_brain, daemon=True).start()


STOP_WORDS = {"ok got it", "ok stop", "stop talking", "stop", "quiet",
              "that's enough", "enough", "ok thanks", "thanks"}

_current_speech = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def speak(text: str) -> None:
    global _current_speech
    if _current_speech and _current_speech.poll() is None:
        _current_speech.kill()
        _current_speech.wait()
    print(f"UNLEASH: {text}", file=sys.stderr)
    _current_speech = subprocess.Popen(["say", "-v", VOICE, text])


def run_pipeline(voice_text: str) -> str:
    state = {
        "current_telemetry":  _read_live_perception(),
        "active_claims":      [],
        "active_predictions": [],
        "retrieved_memories": [],
        "final_action":       "",
        "voice_input":        voice_text,
    }
    state.update(_pipeline_fns["process_biometrics"](state["current_telemetry"]))
    state.update(_pipeline_fns["process_vision_queue"](state["current_telemetry"]))
    state.update(_pipeline_fns["pattern_detector"](state))
    state.update(_pipeline_fns["memory_agent"](state))
    state.update(_pipeline_fns["behavior_orchestrator"](state))
    # Action agent — turns the directive into a (simulated) hardware/API action.
    state.update(_pipeline_fns["action_dispatcher"](state))

    response = state.get("final_action", "")
    _write_live_voice(
        heard=voice_text,
        thinking=state.get("reasoning", ""),
        response=response,
        action=state.get("execution_status", ""),
    )
    return response


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Voice bridge starting (brain loading in background)...", file=sys.stderr)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        text = payload.get("text", "").strip()
        if not text:
            continue

        print(f"\nHeard: '{text}'", file=sys.stderr)

        # Stop words — kill speech and keep listening
        if any(sw in text.lower() for sw in STOP_WORDS):
            if _current_speech and _current_speech.poll() is None:
                _current_speech.kill()
                _current_speech.wait()
            print("(stopped)", file=sys.stderr)
            continue

        # Wait for brain (max 30s)
        if not _brain_ready.wait(timeout=30):
            speak("Still loading, please wait.")
            continue

        if _brain_error:
            speak("Brain failed to load.")
            continue

        print("🧠 Running pipeline...", file=sys.stderr)
        try:
            response = run_pipeline(text)
            print(f"Response: {response}", file=sys.stderr)
            if response and "passive navigation" not in response.lower():
                speak(response)
            else:
                speak("Got it.")
        except Exception as e:
            print(f"Pipeline error: {e}", file=sys.stderr)
            speak("Sorry, something went wrong.")


if __name__ == "__main__":
    main()
