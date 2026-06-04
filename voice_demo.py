"""
Unleash — Voice demo entry point
================================

Run this to try the voice assistant.

    # Text mode (no mic needed) — type questions:
    python voice_demo.py

    # Voice mode — speak into the mic, hear spoken replies (needs GEMINI_API_KEY,
    # a microphone, and: pip install sounddevice soundfile):
    python voice_demo.py --voice

    # One-shot test of a single question:
    python voice_demo.py --ask "what do I have at 4 pm?"
"""

import os
import sys
import argparse

# Optional: load a .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional: start Weave tracing if configured
try:
    import weave
    weave.init("unleash-service-dog")
except Exception:
    pass

from assistant import Assistant


def main():
    ap = argparse.ArgumentParser(description="Unleash voice assistant demo")
    ap.add_argument("--voice", action="store_true", help="use microphone + spoken replies")
    ap.add_argument("--ask", type=str, default=None, help="ask one question and exit")
    ap.add_argument("--seconds", type=float, default=5.0, help="mic listen window")
    args = ap.parse_args()

    assistant = Assistant()

    if args.ask:
        reply = assistant.handle(args.ask)
        print(f"dog > {reply}")
        if args.voice:
            assistant.voice.speak(reply)
        return

    if args.voice:
        assistant.voice.speak("Hi, I'm your Unleash companion. Ask me anything.")
        print("Voice mode. Press Enter to speak, or type 'quit'.")
        while True:
            try:
                cmd = input("(press Enter to talk) > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if cmd in ("quit", "exit"):
                break
            assistant.converse_once(listen_seconds=args.seconds)
        return

    # default: text REPL
    assistant.repl()


if __name__ == "__main__":
    main()

# touched 2026-06-03
