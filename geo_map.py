"""geo_map.py — GREEN CORE real-GPS current-location map for the Unleash HUD.

Renders the user's REAL browser GPS position (or the LocationAgent IP/preset
fallback) as a cyan #00D4FF dot on a Carto dark-matter pydeck basemap, dropped
into the ASSISTANT tab beneath the WHERE AM I panel.

Design contract (mirrors feat_assistant.py conventions):
  * NEVER call st.* at import time — everything lives inside functions.
  * NEVER raise — every backend / network / component call is wrapped; on
    failure we degrade to st.warning + a hud panel, and a dot ALWAYS shows
    (worst case the LocationAgent "default" Boston coord).
  * Accessibility is the product: the locate control is a labelled native
    Streamlit button; the GPS-vs-IP status is conveyed as TEXT (a hud.pill
    'GPS LOCK' / 'IP ESTIMATE'), never color alone; map status text lives in an
    aria-live region.
  * inject_theme() is assumed to have already run in feat_ui.py.

Architecture note — LIVE MOVING DOT upgrade path: get_browser_location() is a
pure one-shot read and render_location_map() always rebuilds the deck from the
CURRENT coord in session_state. Nothing about the GPS layer is one-shot-specific,
so the live upgrade is purely additive (mount an st_autorefresh + append fixes to
a track list) — see the clearly-marked block at the bottom of render_location_map.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

import hud_theme as hud


# Cyan #00D4FF as an RGBA list for pydeck (the HUD primary accent).
_CYAN_RGBA = [0, 212, 255, 220]          # dot fill (opaque)
_CYAN_RING_RGBA = [0, 212, 255, 40]      # accuracy ring (translucent)
_CYAN_RING_LINE = [0, 212, 255, 140]     # accuracy ring outline
_CYAN_TRAIL_RGBA = [0, 212, 255, 180]    # live breadcrumb path (under the dot)

# LIVE MOVING DOT — hard cap on the breadcrumb so session memory stays BOUNDED.
# Beyond this we drop the OLDEST points (FIFO); the list is never unbounded.
_MAX_TRACK_POINTS = 500


def get_browser_location() -> Optional[dict]:
    """One-shot REAL GPS fix via the browser Geolocation API.

    Uses streamlit_js_eval.get_geolocation() (one-shot getCurrentPosition under
    the hood). Returns a normalized dict or None — NEVER raises.

    Returns:
        {"lat": float, "lon": float, "accuracy": float|None, "source": "gps"}
        on a successful fix, or None when the component is missing, the user
        denied permission, the page is not HTTPS/localhost, or the fix is empty.
    """
    try:
        from streamlit_js_eval import get_geolocation
    except Exception:  # noqa: BLE001 — component not installed → no GPS, just fall back
        return None

    try:
        fix = get_geolocation()
    except Exception:  # noqa: BLE001 — JS bridge hiccup → fall back, never crash
        return None

    # The component returns None until the browser answers (and stays None on
    # denial / non-secure-context), or a dict shaped like the JS GeolocationPosition.
    if not fix or not isinstance(fix, dict):
        return None
    coords = fix.get("coords") or {}
    lat = coords.get("latitude")
    lon = coords.get("longitude")
    if lat is None or lon is None:
        return None
    try:
        return {
            "lat": float(lat),
            "lon": float(lon),
            "accuracy": float(coords["accuracy"]) if coords.get("accuracy") is not None else None,
            "source": "gps",
        }
    except (TypeError, ValueError):
        return None


def _resolve_coord(location_agent) -> dict:
    """Pick the coord to plot: prefer a stored GPS fix, else the agent fallback.

    Always returns {"lat","lon","accuracy","source"} with usable floats — the
    LocationAgent.get_current_coord() worst case is the "default" Boston coord,
    so a dot ALWAYS shows. Never raises.
    """
    fix = st.session_state.get("feat_gps_fix")
    if fix:
        return fix

    # Fall back to LocationAgent's unified accessor (override → preset → IP →
    # last-known → default). Its `source` is one of: manual/phone-gps,
    # preset:{name}, ip-geolocation, last-known, default.
    if location_agent is not None:
        try:
            c = location_agent.get_current_coord()
            return {
                "lat": float(c["lat"]),
                "lon": float(c["lon"]),
                "accuracy": None,
                "source": c.get("source", "default"),
            }
        except Exception:  # noqa: BLE001 — agent misbehaved → hard default below
            pass

    # Absolute last resort if even the agent is None/broken (Boston, == preset:home).
    return {"lat": 42.3601, "lon": -71.0589, "accuracy": None, "source": "default"}


def _is_gps_lock(source: str) -> bool:
    """True only for a REAL device fix; everything else is an honest estimate."""
    return source in ("gps", "manual/phone-gps")


def _status_pill_html(source: str) -> str:
    """HONEST status pill: 'GPS LOCK' (live) for a real fix, else 'IP ESTIMATE'."""
    if _is_gps_lock(source):
        return hud.pill("GPS LOCK", "live")
    if source.startswith("preset:"):
        return hud.pill("PRESET", "cyan")
    # ip-geolocation / last-known / default → not a precise device fix; say so.
    return hud.pill("IP ESTIMATE", "warning")


def _status_caption(coord: dict) -> str:
    """Plain-text, aria-live status line (text-not-color-only) under the map."""
    src = coord.get("source", "default")
    acc = coord.get("accuracy")
    if _is_gps_lock(src):
        precision = f" · ±{int(acc)} M" if acc else ""
        msg = f"GPS LOCK · REAL DEVICE LOCATION{precision}"
    elif src.startswith("preset:"):
        msg = f"PRESET LOCATION · {src.split(':', 1)[1].upper()}"
    elif src == "ip-geolocation":
        msg = "IP ESTIMATE · APPROXIMATE — NOT A GPS FIX"
    elif src == "last-known":
        msg = "LAST-KNOWN LOCATION · IP LOOKUP UNAVAILABLE"
    else:
        msg = "DEFAULT LOCATION · NO GPS / IP FIX YET"
    return (
        '<div role="status" aria-live="polite" aria-atomic="true" '
        'style="font-family:var(--font-mono);font-size:10px;letter-spacing:1px;'
        f'color:var(--hud-label);margin-top:8px">{msg}</div>'
    )


def _build_deck(coord: dict, track: Optional[list] = None):
    """Build the Carto-dark pydeck Deck with a cyan dot + accuracy ring.

    Token-free: map_provider='carto' + map_style='dark' resolves to Carto's
    dark-matter basemap (no Mapbox/Carto credentials). Returns a pdk.Deck.

    When `track` (a list of [lon, lat] points) has >=2 points, a cyan PathLayer
    breadcrumb is drawn UNDER the dot/ring. `track` defaults to None so the
    one-shot path (_build_deck(coord)) is unchanged.
    """
    import pydeck as pdk

    lat, lon = coord["lat"], coord["lon"]
    acc = coord.get("accuracy")

    layers = []

    # Live breadcrumb FIRST so it sits UNDER the ring + dot. Needs >=2 points to
    # be a line. Points are already [lon, lat] (pydeck order) — kept as-is.
    if track and len(track) >= 2:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{"path": list(track)}],
                get_path="path",
                get_color=_CYAN_TRAIL_RGBA,
                width_min_pixels=2,
                pickable=False,
            )
        )

    # Translucent accuracy-radius ring (only when we have a real radius, in m).
    if acc and acc > 0:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=[{"position": [lon, lat]}],     # pydeck wants [lon, lat]
                get_position="position",
                get_radius=float(acc),               # meters
                radius_units="meters",
                get_fill_color=_CYAN_RING_RGBA,
                get_line_color=_CYAN_RING_LINE,
                stroked=True,
                line_width_min_pixels=1,
                pickable=False,
            )
        )

    # The cyan current-location dot (fixed pixel size so it reads at any zoom).
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=[{"position": [lon, lat], "label": "You are here"}],
            get_position="position",
            get_radius=7,
            radius_units="pixels",
            get_fill_color=_CYAN_RGBA,
            get_line_color=[255, 255, 255, 230],
            stroked=True,
            line_width_min_pixels=2,
            pickable=True,
        )
    )

    return pdk.Deck(
        map_provider="carto",            # explicit → guarantees token-free path
        map_style="dark",                # Carto dark-matter-gl
        initial_view_state=pdk.ViewState(
            latitude=lat, longitude=lon, zoom=15, pitch=0, bearing=0
        ),
        layers=layers,
        tooltip={"text": "{label}"},
    )


def _inject_map_panel_css(panel_key: str) -> None:
    """Frame the map container as a HUD .panel (matches hud_theme's .panel).

    st.container(key=...) emits a `st-key-{key}` DOM class we target so the live
    pydeck chart sits inside the same bordered cyan card as every other panel:
    --hud-card bg, --hud-border, --r-md radius. Re-emitted every run on purpose
    (Streamlit rebuilds the DOM each rerun, so we must NOT gate this behind a
    session flag or the frame would vanish on the next interaction).
    """
    st.markdown(
        f"""<style>
        .st-key-{panel_key} {{
            background: var(--hud-card);
            border: 1px solid var(--hud-border);
            border-radius: var(--r-md);
            overflow: hidden;
            margin: .4rem 0;
            padding: 0 !important;
        }}
        .st-key-{panel_key} > div {{ gap: 0 !important; }}
        .st-key-{panel_key} .panel-head {{ border-bottom: 1px solid var(--hud-border); }}
        .st-key-{panel_key} [data-testid="stDeckGlJsonChart"] {{ margin: 0; }}
        </style>""",
        unsafe_allow_html=True,
    )


def render_location_map(location_agent, *, key_prefix: str = "feat") -> None:
    """Render the GREEN CORE locate-me button + live current-location map.

    Flow:
      1. A labelled native "LOCATE ME (GPS)" button fires get_browser_location().
      2. On a real fix → store it in session_state AND pipe it into the
         LocationAgent via set_location(lat, lon) so the rest of the app (place
         text, directions origin) honors the real device position.
      3. Always resolve a coord to plot (GPS fix → agent IP/preset/default) so a
         dot ALWAYS shows, then draw the Carto-dark pydeck chart.
      4. Show an HONEST status pill ('GPS LOCK' vs 'IP ESTIMATE') + aria-live text.

    NEVER calls st.* at import; NEVER raises (degrades to st.warning / a panel).
    """
    fix_key = f"{key_prefix}_gps_fix"
    track_key = f"{key_prefix}_gps_track"
    st.session_state.setdefault(fix_key, None)
    st.session_state.setdefault(track_key, [])   # bounded breadcrumb (FIFO, capped)
    # Keep a stable alias the helpers read regardless of key_prefix.
    if fix_key != "feat_gps_fix":
        st.session_state["feat_gps_fix"] = st.session_state.get(fix_key)

    # --- 1. Locate control + Live tracking toggle (labelled native controls) ----
    ctrl_locate, ctrl_live = st.columns([3, 2])
    with ctrl_live:
        live = st.toggle(
            "Live tracking",
            value=False,
            key=f"{key_prefix}_gps_live",
            help="Poll your browser GPS every ~3s and trace a breadcrumb trail. "
                 "Leave OFF for a single one-shot fix.",
        )

    # --- 1a. LIVE MODE (additive) — mount poller + fix EVERY run, no click needed.
    #     When OFF this whole block is skipped: no autorefresh, no polling. -------
    if live:
        autorefresh_ok = True
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=3000, key=f"{key_prefix}_gps_poll")
        except Exception:  # noqa: BLE001 — component missing → degrade to one-shot
            autorefresh_ok = False
            st.warning(
                "Live tracking needs streamlit-autorefresh "
                "(`pip install streamlit-autorefresh`). Falling back to one-shot "
                "— press 'Locate me (GPS)' to update."
            )
        if autorefresh_ok:
            # Unconditional poll: each 3s rerun re-fires the one-shot read; on a
            # real fix store it, push to the agent, and append to the BOUNDED track.
            fix = get_browser_location()
            if fix:
                st.session_state[fix_key] = fix
                st.session_state["feat_gps_fix"] = fix
                if location_agent is not None:
                    try:
                        location_agent.set_location(fix["lat"], fix["lon"])
                    except Exception:  # noqa: BLE001 — never crash the live loop
                        pass
                track = st.session_state[track_key]
                point = [fix["lon"], fix["lat"]]   # pydeck order [lon, lat]
                # Dedupe identical consecutive samples (a stationary device repeats).
                if not track or track[-1] != point:
                    track.append(point)
                    # BOUNDED: drop oldest beyond the cap — never unbounded growth.
                    if len(track) > _MAX_TRACK_POINTS:
                        del track[: len(track) - _MAX_TRACK_POINTS]

    # --- 1b. One-shot locate button (UNCHANGED behavior; works with live OFF) ---
    with ctrl_locate:
        locate_clicked = st.button(
            "Locate me (GPS)",
            key=f"{key_prefix}_locate_gps",
            help="Use your browser's real GPS to drop a pin at your exact location "
                 "(needs location permission + HTTPS).",
            use_container_width=True,
        )
    if locate_clicked:
        fix = get_browser_location()
        if fix:
            st.session_state[fix_key] = fix
            st.session_state["feat_gps_fix"] = fix
            # Pipe the real fix into the agent → unified app honors device GPS.
            if location_agent is not None:
                try:
                    location_agent.set_location(fix["lat"], fix["lon"])
                except Exception as exc:  # noqa: BLE001 — never crash the tab
                    st.warning(f"Couldn't apply GPS fix to location agent: {exc}")
        else:
            st.warning(
                "No GPS fix — allow location access and ensure the page is served "
                "over HTTPS (browser geolocation is blocked on plain HTTP). "
                "Showing best available estimate instead."
            )

    # --- 1c. Clear trail (labelled) — only shown when a breadcrumb exists --------
    if st.session_state[track_key]:
        if st.button(
            "Clear trail",
            key=f"{key_prefix}_clear_trail",
            help="Erase the recorded GPS breadcrumb trail.",
            use_container_width=True,
        ):
            st.session_state[track_key] = []

    # --- 2. Resolve the coord to plot (GPS fix preferred, else agent fallback) --
    coord = _resolve_coord(location_agent)
    track = st.session_state[track_key]   # bounded [lon,lat] breadcrumb (may be [])

    # --- 3. Draw the map INSIDE a HUD .panel frame (matches WHERE AM I etc.) ----
    #     The chart is a live widget, so we can't pass it to hud.panel(); instead
    #     we style a keyed st.container as .panel and reuse the exact .panel-head
    #     markup (cyan dot + title + subtitle + status pills) inside it.
    live_pill = hud.pill("LIVE TRACKING", "live") if live else ""
    src = coord.get("source", "default")
    panel_key = f"{key_prefix}_map_panel"
    _inject_map_panel_css(panel_key)
    with st.container(key=panel_key):
        st.markdown(
            '<div class="panel-head"><span class="dot"></span>'
            '<div class="pht"><h3 class="t">CURRENT LOCATION</h3>'
            '<div class="s">LIVE MAP // CARTO DARK</div></div>'
            '<div style="margin-left:auto;display:flex;gap:6px">'
            f'{_status_pill_html(src)}{live_pill}</div></div>',
            unsafe_allow_html=True,
        )
        try:
            deck = _build_deck(coord, track)
            st.pydeck_chart(deck, use_container_width=True)
        except Exception as exc:  # noqa: BLE001 — pydeck/basemap failure ≠ tab crash
            st.warning(f"Map unavailable: {exc}")
            st.markdown(
                '<div style="padding:12px;color:var(--hud-muted)">Could not render '
                f'the live map. Coordinates: {coord["lat"]:.5f}, {coord["lon"]:.5f}.'
                '</div>',
                unsafe_allow_html=True,
            )

    # --- 4. Honest status caption (aria-live), beneath the framed panel --------
    st.markdown(_status_caption(coord), unsafe_allow_html=True)

    # =======================================================================
    # LIVE MOVING DOT — IMPLEMENTED ABOVE (steps 1a/1b/1c + _build_deck track).
    # -----------------------------------------------------------------------
    # The "Live tracking" toggle mounts st_autorefresh(interval=3000) (guarded
    # import → falls back to one-shot if the component is missing), re-fires
    # get_browser_location() every ~3s, and appends each [lon,lat] fix to a
    # BOUNDED (cap _MAX_TRACK_POINTS, FIFO) track that _build_deck draws as a
    # cyan PathLayer UNDER the dot. One-shot path is unchanged when live is OFF.
    # Original recipe retained below for reference:
    #
    #   1. pip install streamlit-autorefresh  (now uncommented in requirements.txt)
    #   2. At the TOP of this function, mount the poller (guard the import):
    #
    #          try:
    #              from streamlit_autorefresh import st_autorefresh
    #              st_autorefresh(interval=3000, key=f"{key_prefix}_gps_poll")
    #          except Exception:
    #              pass  # live mode optional — one-shot still works
    #
    #      Each 3s tick reruns the script, so get_browser_location() RE-FIRES and
    #      the new fix flows through the exact same code path above (button press
    #      no longer required once polling is mounted — call it unconditionally:
    #          fix = get_browser_location()
    #          if fix: st.session_state[fix_key] = fix; ... set_location(...)).
    #
    #   3. (Optional trail) Keep a track list and add a PathLayer:
    #          track = st.session_state.setdefault(f"{key_prefix}_gps_track", [])
    #          if fix: track.append([fix["lon"], fix["lat"]])   # lon,lat for pydeck
    #          # then in _build_deck, prepend:
    #          #   pdk.Layer("PathLayer", data=[{"path": track}], get_path="path",
    #          #             get_color=_CYAN_RING_LINE, width_min_pixels=2)
    #      (A real OSRM route trail would need location_agent._route to request
    #       overview=full&geometries=geojson — see recon; not required for the dot.)
    #
    # GUARD HEAVY SINGLETONS against the rerun storm: the autorefresh reruns the
    # WHOLE script, so wrap mem0/Qdrant init and weave.init behind @st.cache_resource
    # or a session_state flag (the Qdrant single-instance lock + Weave must NOT
    # re-init on every 3s tick). The cached agents in feat_assistant.py already
    # live in session_state, so they are rerun-safe as-is.
    # =======================================================================
