"""
Unleash — Voice Agent (the user's interface to the robodog)
===========================================================

Handles the two ends of a spoken conversation:

    listen()      -> records mic audio, transcribes it to text (speech-to-text)
    speak(text)   -> says text out loud (text-to-speech), e.g. to AirPods

Design choices (easiest + most reliable, no extra vendors):
  * STT: Gemini. We send the recorded audio straight to Gemini for
    transcription, so the only key needed is GEMINI_API_KEY (already used by the
    rest of Unleash). No Whisper install, no extra account.
  * TTS: macOS built-in `say` command (zero setup, routes to the default output
    incl. AirPods). Falls back to pyttsx3, then to printing, on other platforms.

Everything degrades gracefully: with no microphone you can pass text directly,
and with no GEMINI_API_KEY the agent still speaks and accepts typed input — so
it's fully testable without hardware.
"""

from __future__ import annotations

import os
import sys
import shutil
import platform
import subprocess
import tempfile
from typing import Optional

try:
    import weave
    _WEAVE = True
except Exception:
    _WEAVE = False


def _op(fn):
    return weave.op()(fn) if _WEAVE else fn


class VoiceAgent:
    def __init__(self, voice: str = "Samantha", rate: int = 180, sample_rate: int = 16000):
        self.voice = voice          # macOS `say` voice name
        self.rate = rate            # words per minute
        self.sample_rate = sample_rate
        self._has_say = platform.system() == "Darwin" and shutil.which("say") is not None

    # ---- TEXT -> SPEECH (audio bytes, for Streamlit st.audio) -----------------
    @_op
    def synthesize(self, text: str) -> tuple[Optional[bytes], Optional[str]]:
        """Render `text` to audio. Returns (audio_bytes, mime) for st.audio().

        Order: gTTS (mp3, cross-platform, works in any browser) -> macOS `say`
        (aiff) -> (None, None) if nothing is available. Use this in Streamlit:
            data, mime = va.synthesize(reply)
            if data: st.audio(data, format=mime, autoplay=True)
        """
        text = (text or "").strip()
        if not text:
            return None, None
        # 1) gTTS -> mp3 (best for Streamlit; plays in every browser)
        try:
            from gtts import gTTS
            buf = tempfile.mktemp(suffix=".mp3")
            gTTS(text=text).save(buf)
            with open(buf, "rb") as f:
                return f.read(), "audio/mp3"
        except Exception:
            pass
        # 2) macOS `say` -> aiff
        if self._has_say:
            try:
                out = tempfile.mktemp(suffix=".aiff")
                subprocess.run(
                    ["say", "-v", self.voice, "-r", str(self.rate), "-o", out, text],
                    check=False,
                )
                with open(out, "rb") as f:
                    return f.read(), "audio/aiff"
            except Exception:
                pass
        return None, None

    @_op
    def speak(self, text: str) -> None:
        """Say `text` aloud on the local machine (CLI/desktop use, not Streamlit).
        macOS `say` -> pyttsx3 -> print fallback."""
        text = (text or "").strip()
        if not text:
            return
        if self._has_say:
            try:
                subprocess.run(
                    ["say", "-v", self.voice, "-r", str(self.rate), text],
                    check=False,
                )
                return
            except Exception:
                pass
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.say(text)
            engine.runAndWait()
            return
        except Exception:
            pass
        print(f"🔊 (TTS unavailable) Unleash says: {text}")

    # ---- MIC -> AUDIO ---------------------------------------------------------
    def record(self, seconds: float = 5.0) -> Optional[str]:
        """Record `seconds` of mic audio to a temp WAV. Returns path or None."""
        try:
            import sounddevice as sd
            import soundfile as sf
        except Exception:
            print("[voice] sounddevice/soundfile not installed — "
                  "use transcribe(path) or pass text directly.", file=sys.stderr)
            return None
        try:
            print(f"🎙️  Listening for {seconds:.0f}s…", file=sys.stderr)
            audio = sd.rec(int(seconds * self.sample_rate),
                           samplerate=self.sample_rate, channels=1)
            sd.wait()
            path = tempfile.mktemp(suffix=".wav")
            sf.write(path, audio, self.sample_rate)
            return path
        except Exception as e:
            print(f"[voice] recording failed: {e}", file=sys.stderr)
            return None

    # ---- AUDIO -> TEXT (Gemini STT) ------------------------------------------
    @_op
    def transcribe(self, wav_path: str) -> str:
        """Transcribe an audio FILE to text using Gemini. '' on failure."""
        if not wav_path or not os.path.exists(wav_path):
            return ""
        if not os.environ.get("GEMINI_API_KEY"):
            print("[voice] GEMINI_API_KEY missing — cannot transcribe.", file=sys.stderr)
            return ""
        try:
            from google import genai
            client = genai.Client()
            uploaded = client.files.upload(file=wav_path)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    "Transcribe this audio to plain text. Return only the words spoken.",
                    uploaded,
                ],
            )
            return (resp.text or "").strip()
        except Exception as e:
            print(f"[voice] transcription failed: {e}", file=sys.stderr)
            return ""

    @_op
    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        """Transcribe raw audio bytes (e.g. from Streamlit's mic widget). '' on failure."""
        if not audio_bytes:
            return ""
        try:
            path = tempfile.mktemp(suffix=suffix)
            with open(path, "wb") as f:
                f.write(audio_bytes)
            return self.transcribe(path)
        except Exception as e:
            print(f"[voice] byte transcription failed: {e}", file=sys.stderr)
            return ""

    # ---- convenience ----------------------------------------------------------
    @_op
    def listen(self, seconds: float = 5.0) -> str:
        """Record from mic and return the transcribed text ('' if unavailable)."""
        wav = self.record(seconds)
        return self.transcribe(wav) if wav else ""


if __name__ == "__main__":
    va = VoiceAgent()
    va.speak("Hello, I'm your Unleash service dog. I'm online and listening.")
    if os.environ.get("GEMINI_API_KEY"):
        text = va.listen(4.0)
        print("You said:", text)
