"""
Unleash — Voice Assistant Router
================================

The conversational brain that sits behind the Voice agent. It takes a user
utterance, figures out what they want (intent), routes it to the right agent,
and returns a spoken-ready answer.

    "Where am I right now?"        -> LocationAgent
    "What do I have at 4 pm?"      -> CalendarAgent
    "What's the weather like?" /
    anything else                  -> general chat (Gemini), or a safe canned reply

Intent routing uses Gemini when GEMINI_API_KEY is set, with a fast keyword
fallback so the assistant works (and is testable) with no key at all.

Wiring into the rest of Unleash:
  * This is the *user-initiated Q&A* path — distinct from the hazard pipeline in
    app.py (which is system-initiated). They share the same agents/voice and can
    both speak through VoiceAgent. The UI teammate can call Assistant.handle(text)
    from a text box, or Assistant.converse_once() for a full mic->speak turn.
"""

from __future__ import annotations

import os
import re

try:
    import weave
    _WEAVE = True
except Exception:
    _WEAVE = False


def _op(fn):
    return weave.op()(fn) if _WEAVE else fn


from location_agent import LocationAgent
from calendar_agent import CalendarAgent
from voice_agent import VoiceAgent

INTENTS = ("location", "calendar", "general")


class Assistant:
    def __init__(self, voice: VoiceAgent | None = None):
        self.location = LocationAgent()
        self.calendar = CalendarAgent()
        self.voice = voice or VoiceAgent()

    # ---- intent classification -----------------------------------------------
    @_op
    def classify(self, text: str) -> str:
        """Return one of INTENTS. Gemini if available, else keywords."""
        kw = self._classify_keywords(text)
        if kw:                      # high-confidence keyword hit wins (fast + free)
            return kw
        g = self._classify_gemini(text)
        return g or "general"

    @staticmethod
    def _classify_keywords(text: str) -> str | None:
        t = text.lower()
        if re.search(r"\b(where am i|my location|where are we|what street|"
                     r"where i am|located|address)\b", t):
            return "location"
        if re.search(r"\b(calendar|schedule|appointment|appointments|meeting|"
                     r"event|events|agenda|at \d|do i have|what.?s? at|today|"
                     r"tomorrow|p\.?m\.?|a\.?m\.?)\b", t):
            return "calendar"
        return None

    def _classify_gemini(self, text: str) -> str | None:
        if not os.environ.get("GEMINI_API_KEY"):
            return None
        try:
            from google import genai
            client = genai.Client()
            prompt = (
                "Classify the user's request into exactly one label: "
                "location, calendar, or general.\n"
                "- location: asking where they are, their address, surroundings.\n"
                "- calendar: asking about schedule, events, appointments, a time.\n"
                "- general: anything else.\n"
                f"User: {text}\nLabel:"
            )
            resp = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            label = (resp.text or "").strip().lower()
            return next((i for i in INTENTS if i in label), None)
        except Exception:
            return None

    # ---- general chat ---------------------------------------------------------
    def _general(self, text: str) -> str:
        if not os.environ.get("GEMINI_API_KEY"):
            return ("I can tell you where you are and what's on your schedule. "
                    "Ask me, for example, 'where am I?' or 'what do I have at 4 pm?'")
        try:
            from google import genai
            client = genai.Client()
            prompt = (
                "You are Unleash, a calm, concise AI service dog companion for a "
                "visually impaired user. Answer in one or two short spoken "
                "sentences.\n"
                f"User: {text}"
            )
            resp = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            return (resp.text or "").strip() or "I'm here with you."
        except Exception:
            return "I'm here with you, but I couldn't reach my reasoning service just now."

    # ---- main entry points ----------------------------------------------------
    @_op
    def handle(self, text: str) -> str:
        """Text in -> spoken-ready text out. The core function the UI can call."""
        text = (text or "").strip()
        if not text:
            return "I didn't catch that. Could you say it again?"
        intent = self.classify(text)
        if intent == "location":
            return self.location.answer(text)
        if intent == "calendar":
            return self.calendar.answer(text)
        return self._general(text)

    @_op
    def converse_once(self, text: str | None = None, listen_seconds: float = 5.0) -> str:
        """Full turn: if no text given, listen on the mic; then answer and speak it."""
        if text is None:
            text = self.voice.listen(listen_seconds)
        if not text:
            reply = "I didn't hear anything. I'm still here when you need me."
            self.voice.speak(reply)
            return reply
        reply = self.handle(text)
        self.voice.speak(reply)
        return reply

    def repl(self):
        """Simple text loop for testing without a mic. Ctrl-C to quit."""
        print("Unleash assistant (text mode). Type a question, or 'quit'.\n")
        while True:
            try:
                q = input("you > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q.lower() in ("quit", "exit"):
                break
            print(f"dog > {self.handle(q)}\n")


if __name__ == "__main__":
    Assistant().repl()
