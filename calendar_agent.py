"""
Unleash — Calendar / Software Agent
===================================

Answers questions like "What do I have at 4 pm?" or "What's on my schedule
today?" by reading a local events store (events.json).

Why local JSON: it's bulletproof on stage (no OAuth, no network) and easy for
teammates to edit live. The interface is written so you can later swap the
backend for the real Google Calendar API without touching the voice/assistant
layer — just reimplement `_load_events()` to call the Calendar API and map each
event into the same dict shape:

    {"id","title","date"(YYYY-MM-DD or "TODAY"),"start"("HH:MM"),"end","location","notes"}

Time understanding: a query time like "4 pm", "16:00", "four thirty" is parsed
with a regex first; if GEMINI_API_KEY is set we fall back to Gemini to extract a
time, so loose phrasing still works.
"""

from __future__ import annotations

import os
import re
import json
import datetime as dt
from pathlib import Path
from typing import Optional

try:
    import weave
    _WEAVE = True
except Exception:
    _WEAVE = False


def _op(fn):
    return weave.op()(fn) if _WEAVE else fn


EVENTS_PATH = Path(__file__).with_name("events.json")


class CalendarAgent:
    def __init__(self, events_path: Path = EVENTS_PATH):
        self.events_path = Path(events_path)

    # ---- backend --------------------------------------------------------------
    def _load_events(self) -> list[dict]:
        """Load + normalize today's events.

        Tries the real Google Calendar API first (if credentials are present),
        and falls back to the local events.json so the demo never breaks. Both
        paths return the SAME dict shape:
            {"id","title","date","start","end","location","notes"}
        """
        gcal = self._load_events_google()
        if gcal is not None:
            return gcal
        return self._load_events_local()

    def _load_events_local(self) -> list[dict]:
        try:
            raw = json.loads(self.events_path.read_text())
        except Exception:
            return []
        today = dt.date.today().isoformat()
        events = []
        for e in raw.get("events", []):
            date = e.get("date", "TODAY")
            date = today if date in (None, "", "TODAY") else date
            events.append({**e, "date": date})
        return events

    def _load_events_google(self) -> Optional[list[dict]]:
        """Read today's events from Google Calendar.

        Returns None (so the caller falls back to local JSON) if the Google
        libraries aren't installed or no credentials exist. Auth uses an OAuth
        flow: place `credentials.json` (OAuth client) next to this file; the
        first run opens a browser and caches `token.json`.
        """
        creds_file = self.events_path.with_name("credentials.json")
        token_file = self.events_path.with_name("token.json")
        if not creds_file.exists() and not token_file.exists():
            return None
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
        except Exception:
            return None  # libs not installed -> fall back

        SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
        creds = None
        try:
            if token_file.exists():
                creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                elif creds_file.exists():
                    flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES)
                    creds = flow.run_local_server(port=0)
                else:
                    return None
                token_file.write_text(creds.to_json())

            service = build("calendar", "v3", credentials=creds)
            tz = dt.timezone.utc
            today = dt.date.today()
            start = dt.datetime.combine(today, dt.time.min).astimezone()
            end = dt.datetime.combine(today, dt.time.max).astimezone()
            result = service.events().list(
                calendarId="primary",
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            ).execute()

            events = []
            for item in result.get("items", []):
                s = item.get("start", {})
                e = item.get("end", {})
                start_dt = s.get("dateTime") or s.get("date")
                end_dt = e.get("dateTime") or e.get("date")
                events.append({
                    "id": item.get("id", ""),
                    "title": item.get("summary", "(no title)"),
                    "date": today.isoformat(),
                    "start": self._iso_to_hhmm(start_dt),
                    "end": self._iso_to_hhmm(end_dt),
                    "location": item.get("location", ""),
                    "notes": item.get("description", ""),
                })
            return events
        except Exception:
            return None  # any failure -> safe fallback to local

    @staticmethod
    def _iso_to_hhmm(value: Optional[str]) -> str:
        """'2026-05-31T16:00:00-04:00' or '2026-05-31' -> 'HH:MM'."""
        if not value:
            return "00:00"
        if "T" not in value:        # all-day event
            return "00:00"
        try:
            d = dt.datetime.fromisoformat(value)
            return f"{d.hour:02d}:{d.minute:02d}"
        except Exception:
            return "00:00"

    # ---- time parsing ---------------------------------------------------------
    @staticmethod
    def _parse_time_regex(text: str) -> Optional[str]:
        """Pull an 'HH:MM' 24h time out of free text. Returns None if not found."""
        t = text.lower()
        # 4 pm / 4:30pm / 4 p.m.
        m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", t)
        if m:
            hour = int(m.group(1)) % 12
            minute = int(m.group(2) or 0)
            if m.group(3).startswith("p"):
                hour += 12
            return f"{hour:02d}:{minute:02d}"
        # 24h "16:00" or "at 16"
        m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", t)
        if m:
            return f"{int(m.group(1)):02d}:{m.group(2)}"
        return None

    def _parse_time_gemini(self, text: str) -> Optional[str]:
        """Use Gemini to extract a 24h time if the regex missed. Best-effort."""
        if not os.environ.get("GEMINI_API_KEY"):
            return None
        try:
            from google import genai
            client = genai.Client()
            prompt = (
                "Extract the clock time the user is asking about and return ONLY "
                "24-hour HH:MM, or NONE if there's no specific time.\n"
                f"User: {text}\nTime:"
            )
            resp = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            out = (resp.text or "").strip()
            m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", out)
            return f"{int(m.group(1)):02d}:{m.group(2)}" if m else None
        except Exception:
            return None

    def extract_time(self, text: str) -> Optional[str]:
        return self._parse_time_regex(text) or self._parse_time_gemini(text)

    # ---- queries --------------------------------------------------------------
    @_op
    def events_at(self, hhmm: str, date: Optional[str] = None, window_min: int = 60) -> list[dict]:
        """Events overlapping a target time (within +/- window for fuzzy matches)."""
        date = date or dt.date.today().isoformat()
        target = self._to_min(hhmm)
        hits = []
        for e in self._load_events():
            if e["date"] != date:
                continue
            start = self._to_min(e.get("start", "00:00"))
            end = self._to_min(e.get("end", e.get("start", "00:00")))
            # match if target falls inside the event, or within the fuzzy window of its start
            if start <= target <= end or abs(start - target) <= window_min:
                hits.append(e)
        return sorted(hits, key=lambda e: self._to_min(e.get("start", "00:00")))

    @_op
    def events_today(self) -> list[dict]:
        today = dt.date.today().isoformat()
        evs = [e for e in self._load_events() if e["date"] == today]
        return sorted(evs, key=lambda e: self._to_min(e.get("start", "00:00")))

    # ---- spoken answers -------------------------------------------------------
    @_op
    def answer(self, query: str) -> str:
        """Route a natural calendar question to a spoken-ready answer."""
        hhmm = self.extract_time(query)
        if hhmm:
            evs = self.events_at(hhmm)
            spoken = self._fmt_time(hhmm)
            if not evs:
                return f"You have nothing scheduled around {spoken}."
            if len(evs) == 1:
                return f"At {spoken} you have: {self._describe(evs[0])}."
            joined = "; ".join(self._describe(e) for e in evs)
            return f"Around {spoken} you have: {joined}."
        # No specific time -> summarize the day
        evs = self.events_today()
        if not evs:
            return "You have nothing on your calendar today."
        joined = "; ".join(f"{self._fmt_time(e['start'])}, {e['title']}" for e in evs)
        return f"Today you have: {joined}."

    # ---- helpers --------------------------------------------------------------
    @staticmethod
    def _to_min(hhmm: str) -> int:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    @staticmethod
    def _fmt_time(hhmm: str) -> str:
        h, m = (int(x) for x in hhmm.split(":"))
        suffix = "am" if h < 12 else "pm"
        h12 = h % 12 or 12
        return f"{h12}:{m:02d} {suffix}" if m else f"{h12} {suffix}"

    @staticmethod
    def _describe(e: dict) -> str:
        s = e.get("title", "an event")
        if e.get("location"):
            s += f" at {e['location']}"
        return s


if __name__ == "__main__":
    cal = CalendarAgent()
    for q in ["What do I have at 4 pm?", "what's at 12:30?", "anything at 3pm?",
              "what's on my schedule today?"]:
        print(f"Q: {q}\nA: {cal.answer(q)}\n")

# touched 2026-06-03
