"""
Unleash — Location / Place Agent
================================

Answers "Where am I?" and basic place questions for the user.

The robodog has a camera (Perception agent) but NO GPS chip, so this agent
resolves location through a fallback chain that is reliable on a laptop indoors:

    1. Manual override .......... agent.set_location(lat, lon)  (e.g. piped from a phone)
    2. Named preset ............. agent.use_preset("5th_and_main")
    3. IP-based geolocation ..... free, no API key  (ip-api.com)
    4. Hard default ............. Boston, MA

Whatever coordinate wins is reverse-geocoded to a human address via
OpenStreetMap Nominatim (free, no API key). Everything is wrapped so a network
failure never crashes the demo — it degrades to coordinates or the last good
answer.

To use REAL phone GPS later: feed the phone's lat/lon into set_location() (e.g.
over the same channel the Continuity Camera uses) and the rest works unchanged.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

import ssl
try:                                     # use certifi's CA bundle for HTTPS (stdlib
    import certifi                       # urllib otherwise fails cert checks on some
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())  # Python installs)
except Exception:
    _SSL_CTX = ssl.create_default_context()

try:
    import weave
    _WEAVE = True
except Exception:  # weave optional for standalone use
    _WEAVE = False

    def _noop_op(fn=None, **_):
        return fn if fn else (lambda f: f)


def _op(fn):
    """Apply @weave.op() if weave is importable, else no-op."""
    return weave.op()(fn) if _WEAVE else fn


# A few demo-friendly presets. Add your own.
PRESETS = {
    "home":          {"lat": 42.3601, "lon": -71.0589, "label": "Home"},
    "5th_and_main":  {"lat": 42.3554, "lon": -71.0640, "label": "5th & Main"},
    "harvard_sq":    {"lat": 42.3736, "lon": -71.1190, "label": "Harvard Square"},
    "mass_general":  {"lat": 42.3631, "lon": -71.0686, "label": "Mass General Hospital"},
}

DEFAULT_COORD = {"lat": 42.3601, "lon": -71.0589}  # Boston, MA
USER_AGENT = "UnleashServiceDog/1.0 (hackathon demo)"
HTTP_TIMEOUT = 6


@dataclass
class LocationAgent:
    override: Optional[dict] = None          # {"lat":..,"lon":..} manual / phone GPS
    preset: Optional[str] = None             # key into PRESETS
    last_known: Optional[dict] = field(default=None)  # cache of last good fix
    _geo_cache: dict = field(default_factory=dict)    # reverse-geocode cache

    # ---- ways to set position -------------------------------------------------
    def set_location(self, lat: float, lon: float) -> None:
        """Manually set coordinates (e.g. piped from a real phone GPS)."""
        self.override = {"lat": float(lat), "lon": float(lon)}

    def use_preset(self, name: str) -> None:
        if name not in PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Options: {list(PRESETS)}")
        self.preset = name

    def clear(self) -> None:
        self.override = None
        self.preset = None

    # ---- resolve current coordinate ------------------------------------------
    @_op
    def get_current_coord(self) -> dict:
        """Return {'lat','lon','source'} using the fallback chain."""
        if self.override:
            return {**self.override, "source": "manual/phone-gps"}
        if self.preset:
            p = PRESETS[self.preset]
            return {"lat": p["lat"], "lon": p["lon"], "source": f"preset:{self.preset}"}
        ip = self._ip_geolocate()
        if ip:
            self.last_known = ip
            return {**ip, "source": "ip-geolocation"}
        if self.last_known:
            return {**self.last_known, "source": "last-known"}
        return {**DEFAULT_COORD, "source": "default"}

    def _ip_geolocate(self) -> Optional[dict]:
        """Free IP-based location, no API key. Returns coord dict or None."""
        try:
            url = "http://ip-api.com/json/?fields=status,lat,lon,city,regionName,country"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT, context=_SSL_CTX) as r:
                data = json.loads(r.read().decode())
            if data.get("status") == "success":
                return {"lat": data["lat"], "lon": data["lon"]}
        except Exception:
            return None
        return None

    # ---- reverse geocode ------------------------------------------------------
    @_op
    def reverse_geocode(self, lat: float, lon: float) -> dict:
        """Coordinate -> human address via OpenStreetMap Nominatim (no key).

        Results are cached per ~11m grid cell (4 decimal places) so repeated
        "where am I?" questions are instant and never trip Nominatim's
        1-request/second rate limit during a demo."""
        key = (round(lat, 4), round(lon, 4))
        cached = self._geo_cache.get(key)
        if cached is not None:
            return cached
        try:
            params = urllib.parse.urlencode({
                "lat": lat, "lon": lon, "format": "jsonv2", "zoom": 18,
                "addressdetails": 1,
            })
            url = f"https://nominatim.openstreetmap.org/reverse?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT, context=_SSL_CTX) as r:
                data = json.loads(r.read().decode())
            addr = data.get("address", {})
            road = addr.get("road") or addr.get("pedestrian") or addr.get("footway")
            city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("suburb")
            result = {
                "display_name": data.get("display_name", ""),
                "road": road,
                "city": city,
                "state": addr.get("state"),
                "country": addr.get("country"),
            }
            self._geo_cache[key] = result
            return result
        except Exception:
            return {"display_name": "", "road": None, "city": None}

    # ---- the user-facing answer ----------------------------------------------
    @_op
    def where_am_i(self) -> str:
        """Spoken-ready answer to 'where am I right now?'"""
        coord = self.get_current_coord()
        geo = self.reverse_geocode(coord["lat"], coord["lon"])

        # Build a natural sentence from whatever detail we got.
        if geo.get("road") and geo.get("city"):
            place = f"{geo['road']} in {geo['city']}"
        elif geo.get("city"):
            place = geo["city"]
        elif geo.get("display_name"):
            place = geo["display_name"].split(",")[0]
        else:
            place = f"latitude {coord['lat']:.4f}, longitude {coord['lon']:.4f}"

        # If we used a labelled preset, prefer its friendly name.
        if coord["source"].startswith("preset:"):
            label = PRESETS[self.preset]["label"]
            return f"You're at {label} — near {place}."
        return f"You're near {place}."

    # ---- directions / mapping -------------------------------------------------
    def _geocode_place(self, query: str) -> Optional[dict]:
        """Free-text place -> {lat, lon, name} via Nominatim search (no key)."""
        try:
            params = urllib.parse.urlencode({"q": query, "format": "jsonv2", "limit": 1})
            url = f"https://nominatim.openstreetmap.org/search?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT, context=_SSL_CTX) as r:
                hits = json.loads(r.read().decode())
            if hits:
                h = hits[0]
                return {"lat": float(h["lat"]), "lon": float(h["lon"]),
                        "name": h.get("display_name", query).split(",")[0]}
        except Exception:
            return None
        return None

    def _route(self, a: dict, b: dict) -> Optional[dict]:
        """Street route a->b via the free OSRM demo server. {meters, seconds} or None.

        The public OSRM server only exposes the driving profile, so we use its
        distance (a good proxy for the on-street walking path) and estimate the
        walk time ourselves."""
        try:
            coords = f"{a['lon']},{a['lat']};{b['lon']},{b['lat']}"
            url = f"https://router.project-osrm.org/route/v1/driving/{coords}?overview=false"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT, context=_SSL_CTX) as r:
                data = json.loads(r.read().decode())
            routes = data.get("routes") or []
            if routes:
                return {"meters": routes[0]["distance"], "seconds": routes[0]["duration"]}
        except Exception:
            return None
        return None

    @_op
    def directions(self, destination: str) -> str:
        """Spoken-ready walking directions from the current location to a place."""
        dest = (destination or "").strip()
        if not dest:
            return "Where would you like to go?"
        target = self._geocode_place(dest)
        if not target:
            return f"I couldn't find {dest} on the map."
        here = self.get_current_coord()
        route = self._route(here, target)
        if not route:
            return (f"I found {target['name']}, but I couldn't reach the routing "
                    "service for a path right now.")
        km = route["meters"] / 1000.0
        walk_min = max(1, round(km / 5.0 * 60))      # ~5 km/h walking pace
        dist = f"{km:.1f} kilometers" if km >= 1 else f"{int(route['meters'])} meters"
        return f"{target['name']} is about {dist} away — roughly a {walk_min} minute walk."

    @_op
    def answer(self, query: str) -> str:
        """Handle any location-type question. Currently maps everything to where_am_i."""
        return self.where_am_i()


if __name__ == "__main__":
    agent = LocationAgent()
    print("[default / IP]:", agent.where_am_i())
    agent.use_preset("5th_and_main")
    print("[preset]:", agent.where_am_i())
    agent.set_location(42.3736, -71.1190)
    print("[manual]:", agent.where_am_i())
