"""Unleash — COMMAND CENTER operator dashboard.

Tactical HUD Streamlit shell wired to the live multi-agent pipeline (app.py),
the YOLOv8 perception loop (perception.py), and the voice / location agents.
Every output is restyled through hud_theme; the companion HUD mirrors live data.

Run:  streamlit run ui.py
"""

from __future__ import annotations

import os
import time
import html

import streamlit as st

import hud_theme as hud
from companion_hud import render_companion_hud

# Pipeline agents (app.py is the source of truth for the hazard pipeline).
from app import (
    process_biometrics,
    process_vision_queue,
    pattern_detector,
    memory_agent,
    behavior_orchestrator,
    action_dispatcher,
    retrospective_agent,
)

# ---------------------------------------------------------------------------
# Page config + theme — MUST be first Streamlit calls.
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Unleash — Command Center",
    page_icon="🐕",
)
hud.inject_theme()


# ---------------------------------------------------------------------------
# Session state defaults so every tab renders before any run.
# ---------------------------------------------------------------------------
def _init_state() -> None:
    if "last" not in st.session_state:
        st.session_state.last = {
            "claims": [],
            "predictions": [],
            "memories": [],
            "action": "Maintain passive navigation mode (Safe)",
            "dispatch": "[SYSTEM] Passive observation maintained. No external API triggered.",
            "telemetry": {"heart_rate": 72, "hrv": 65, "risk_level": "LOW", "crowd_count": 0},
            "hr_series": [72, 71, 73, 72, 70, 72],
        }
    if "voice_line" not in st.session_state:
        st.session_state.voice_line = "All clear. I'm here with you."
    if "last_auto_trigger" not in st.session_state:
        st.session_state.last_auto_trigger = 0.0


_init_state()


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


# ---------------------------------------------------------------------------
# Shared pipeline runner. Returns the full result dict and persists it.
# `arm_real_calls` gates the live Bland AI emergency path.
# ---------------------------------------------------------------------------
def run_pipeline(telemetry: dict, arm_real_calls: bool = False) -> dict:
    """Run the full conflict-bus pipeline for one telemetry payload.

    Mirrors execute_scenario from the original ui.py, but returns structured
    data instead of rendering. Real emergency calls only fire when armed.
    """
    # 1. Ingest — biometrics + vision claims merged onto the bus.
    try:
        bio = process_biometrics(telemetry).get("active_claims", [])
    except Exception as exc:  # noqa: BLE001 — never crash the dashboard
        bio = []
        st.warning(f"Biometric agent unavailable: {exc}")
    try:
        vis = process_vision_queue(telemetry).get("active_claims", [])
    except Exception as exc:  # noqa: BLE001
        vis = []
        st.warning(f"Vision agent unavailable: {exc}")
    claims = bio + vis

    state = {
        "current_telemetry": telemetry,
        "active_claims": claims,
        "active_predictions": [],
        "retrieved_memories": [],
        "final_action": "Maintain passive navigation mode (Safe)",
        # Thread the live-call gate through state (thread-safe, no global env mutation).
        "arm_real_calls": arm_real_calls,
    }

    # 2. Pattern detection (deterministic, safe).
    predictions = []
    try:
        predictions = pattern_detector(state).get("active_predictions", [])
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Pattern detector unavailable: {exc}")
    state["active_predictions"] = predictions

    # 3. Memory recall (Mem0 / Gemini — may need a key).
    memories = []
    try:
        memories = memory_agent(state).get("retrieved_memories", [])
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Memory agent unavailable (check GEMINI_API_KEY): {exc}")
    state["retrieved_memories"] = memories

    # 4. Orchestrator decision (Gemini LLM — degrade to Safe on failure).
    action = "Maintain passive navigation mode (Safe)"
    try:
        action = behavior_orchestrator(state).get("final_action", action)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Orchestrator unavailable (check GEMINI_API_KEY): {exc}")
    state["final_action"] = action

    # 5. Action dispatch. The LIVE emergency branch is gated by state["arm_real_calls"]
    #    threaded into action_dispatcher -> _trigger_bland_ai_call: when disarmed the
    #    call is ALWAYS simulated. No global os.environ mutation, so this is safe under
    #    Streamlit's concurrent reruns.
    dispatch = "[SYSTEM] Passive observation maintained. No external API triggered."
    try:
        dispatch = action_dispatcher(state).get("execution_status", dispatch)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Action dispatcher unavailable: {exc}")

    # 6. Retrospective write-back (best effort).
    try:
        retrospective_agent(state, resolution_success=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Retrospective agent unavailable: {exc}")

    # Persist for cross-tab + cross-rerun sharing.
    hr = telemetry.get("heart_rate", st.session_state.last["telemetry"].get("heart_rate", 72))
    hr_series = (st.session_state.last.get("hr_series", []) + [hr])[-30:]
    st.session_state.last = {
        "claims": claims,
        "predictions": predictions,
        "memories": memories,
        "action": action,
        "dispatch": dispatch,
        "telemetry": telemetry,
        "hr_series": hr_series,
    }
    # Spoken line shown as text everywhere (accessibility: always visible).
    st.session_state.voice_line = action
    return st.session_state.last


# ---------------------------------------------------------------------------
# SIDEBAR — brand, nav, source, toggles, API status, arm switch.
# ---------------------------------------------------------------------------
def _api_status_line(label: str, env_var: str) -> str:
    connected = bool(os.environ.get(env_var))
    state = "CONNECTED" if connected else "MISSING"
    variant = "live" if connected else "danger"
    return (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'margin:6px 0;font-family:var(--font-mono);font-size:10px;letter-spacing:1px;">'
        f'<span style="color:var(--text-secondary)">{_esc(label)}</span>'
        f'{hud.pill(state, variant)}</div>'
    )


with st.sidebar:
    st.markdown(
        hud.reactor_header("SERVICE DOG", "UNLEASH // OPERATOR", hud.pill("ONLINE", "live")),
        unsafe_allow_html=True,
    )
    st.markdown(hud.hud_sep("NAVIGATION"), unsafe_allow_html=True)
    nav = st.radio(
        "Console view",
        ["Live Dashboard", "Explore Crowd", "Place Memory", "Audio Config"],
        label_visibility="collapsed",
    )

    st.markdown(hud.hud_sep("PERCEPTION SOURCE"), unsafe_allow_html=True)
    source = st.selectbox(
        "Perception source",
        ["YOLOv8 // CONTINUITY CAM", "YOLOv8 // WEBCAM"],
        label_visibility="collapsed",
    )

    st.markdown(hud.hud_sep("ASSIST MODULES"), unsafe_allow_html=True)
    voice_assist = st.toggle("Voice Assist", value=True, help="Speak the directive aloud (TTS).")
    pinpoint_mapping = st.toggle("Pinpoint Mapping", value=True, help="Resolve live GPS / place.")
    gap_sync_trail = st.toggle("Gap Sync Trail", value=False, help="Sync episode trail to memory.")

    st.markdown(hud.hud_sep("API STATUS"), unsafe_allow_html=True)
    st.markdown(_api_status_line("GEMINI", "GEMINI_API_KEY"), unsafe_allow_html=True)
    st.markdown(_api_status_line("W&B", "WANDB_API_KEY"), unsafe_allow_html=True)
    st.markdown(_api_status_line("BLAND", "BLAND_API_KEY"), unsafe_allow_html=True)

    st.markdown(hud.hud_sep("SAFETY"), unsafe_allow_html=True)
    arm_real_calls = st.toggle(
        "ARM REAL CALLS",
        value=False,
        help="When OFF, the emergency path is always SIMULATED — no live Bland AI call.",
    )
    if arm_real_calls:
        st.markdown(hud.pill("LIVE CALLS ARMED", "danger"), unsafe_allow_html=True)
    else:
        st.markdown(hud.pill("CALLS SAFE / SIMULATED", "cyan"), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HEADER ROW — title + subtitle + LIVE pill.
# ---------------------------------------------------------------------------
head_l, head_r = st.columns([4, 1])
with head_l:
    st.markdown(
        '<h1 style="margin-bottom:2px">COMMAND CENTER</h1>'
        '<p style="font-family:var(--font-mono);font-size:11px;letter-spacing:2px;'
        'color:var(--hud-muted);margin-top:0">UNLEASH MULTI-AGENT SERVICE DOG // REAL-TIME OPS</p>',
        unsafe_allow_html=True,
    )
with head_r:
    st.markdown(
        f'<div style="text-align:right;padding-top:14px">{hud.pill("LIVE // DEPLOY", "live")}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# METRIC CARDS — 4 across.
# ---------------------------------------------------------------------------
last = st.session_state.last
active_claims = last.get("claims", [])
telemetry = last.get("telemetry", {})
hr_now = telemetry.get("heart_rate", 72)
risk_now = telemetry.get("risk_level", "LOW")
_risk_score = {"LOW": 18, "MED": 55, "HIGH": 88}.get(risk_now, 18)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        hud.stat_card(str(len(active_claims)), "ALERTS ACTIVE", "BUS",
                      "danger" if active_claims else "cyan"),
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(hud.stat_card("91", "ALL SYSTEMS", "OK", "success"), unsafe_allow_html=True)
with m3:
    st.markdown(
        hud.stat_card(str(_risk_score), "THREAT INDEX", risk_now,
                      "danger" if risk_now == "HIGH" else "warning"),
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        hud.stat_card(str(hr_now), "HEART RATE", "BPM",
                      "danger" if hr_now > 100 else "success"),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helpers for the claims / directive / voice panels.
# ---------------------------------------------------------------------------
def _claim_severity(ctype: str) -> str:
    return {"TachycardiaAlert": "danger", "VisionAlert": "warning"}.get(ctype, "cyan")


def render_claims_panel(claims: list[dict]) -> None:
    if not claims:
        st.markdown(
            hud.panel("CONFLICT BUS", "NO ACTIVE CLAIMS",
                      '<span style="color:var(--hud-success)">● ALL CALM // BUS CLEAR</span>'),
            unsafe_allow_html=True,
        )
        return
    rows = "".join(
        hud.claim_row(c.get("source", "AGENT"), c.get("type", "Claim"),
                      str(c.get("value", "")), _claim_severity(c.get("type", "")))
        for c in claims
    )
    st.markdown(
        hud.panel("CONFLICT BUS", f"{len(claims)} ACTIVE CLAIM(S)", rows,
                  hud.pill("FIRING", "danger")),
        unsafe_allow_html=True,
    )


def render_directive_panel(action: str, dispatch: str) -> None:
    # Safety-critical directive: announce changes to assistive tech (WCAG 4.1.3).
    body = (
        '<div role="status" aria-live="assertive" aria-atomic="true">'
        '<span class="vis-hidden">Current directive: </span>'
        f'<div style="font-size:14px;color:var(--text-primary);line-height:1.4">{_esc(action)}</div>'
        f'<div style="font-family:var(--font-mono);font-size:10px;letter-spacing:1px;'
        f'color:var(--hud-label);margin-top:6px">{_esc(dispatch)}</div>'
        '</div>'
    )
    st.markdown(hud.panel("DIRECTIVE", "ORCHESTRATOR DECISION", body,
                          hud.pill("EXECUTED", "cyan")), unsafe_allow_html=True)


def render_voice_output(line: str) -> None:
    # Spoken line mirrored as text in a live region so it is announced when it changes.
    body = (
        '<div role="status" aria-live="assertive" aria-atomic="true">'
        '<span class="vis-hidden">Spoken directive: </span>'
        f'<div style="font-size:15px;color:var(--hud-text);font-style:italic">'
        f'&ldquo;{_esc(line)}&rdquo;</div></div>'
    )
    st.markdown(hud.panel("VOICE OUTPUT", "SPOKEN TO HANDLER", body,
                          hud.pill("AUDIO", "live")), unsafe_allow_html=True)


def speak_line(line: str) -> None:
    """Synthesize + play the spoken line when Voice Assist is on. Text always shown."""
    if not voice_assist or not line:
        return
    try:
        from voice_agent import VoiceAgent
        data, mime = VoiceAgent().synthesize(line)
        if data:
            st.audio(data, format=mime, autoplay=True)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Voice synthesis unavailable: {exc}")


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_feed, tab_agents, tab_bio, tab_place = st.tabs(
    ["Live Feed", "Agents", "Biometrics", "Place"]
)

# ===== LIVE FEED =============================================================
with tab_feed:
    col_cam, col_bus = st.columns([3, 2])

    with col_cam:
        st.markdown(hud.hud_sep("PERCEPTION // LIVE"), unsafe_allow_html=True)

        # Pick where frames come from. Phone = st.camera_input (works on a phone
        # browser); Local = the on-device webcam / iPhone Continuity YOLO loop.
        cam_source = st.radio(
            "Camera source",
            ["Phone / browser camera", "Local webcam / Continuity"],
            help="Phone uses your device browser camera; Local runs YOLOv8 on this Mac.",
        )

        # If we just switched AWAY from the local source, release any cached camera
        # generator so we never hold the webcam open while on the phone path.
        if cam_source != "Local webcam / Continuity" and "cam_gen" in st.session_state:
            try:
                st.session_state.cam_gen.close()
            except Exception:  # noqa: BLE001
                pass
            st.session_state.pop("cam_gen", None)
            st.session_state.pop("cam_index_active", None)

        st.caption("Manual scenario triggers")
        b1, b2, b3 = st.columns(3)
        if b1.button("Safe", use_container_width=True):
            run_pipeline({"heart_rate": 72, "hrv": 65, "crowd_density": 0.1,
                          "fall_detected": False, "risk_level": "LOW", "crowd_count": 1},
                         arm_real_calls)
            speak_line(st.session_state.voice_line)
            st.rerun()
        if b2.button("Approaching Crowd", use_container_width=True):
            run_pipeline({"heart_rate": 85, "hrv": 45, "crowd_density": 0.9,
                          "fall_detected": False, "risk_level": "MED", "crowd_count": 6},
                         arm_real_calls)
            speak_line(st.session_state.voice_line)
            st.rerun()
        if b3.button("Panic Attack", use_container_width=True):
            run_pipeline({"heart_rate": 115, "hrv": 22, "crowd_density": 0.1,
                          "fall_detected": False, "risk_level": "LOW", "crowd_count": 0},
                         arm_real_calls)
            speak_line(st.session_state.voice_line)
            st.rerun()

        # ===== PHONE / BROWSER CAMERA ====================================
        if cam_source == "Phone / browser camera":
            st.caption(
                "To use your PHONE's camera, open this app on the phone's browser "
                "via the Network URL (same Wi-Fi), e.g. "
                "http://<your-mac-ip>:<port> — then tap below."
            )
            photo = st.camera_input("Tap to use your phone's camera")
            if photo is not None:
                try:
                    import perception

                    rgb, payload = perception.analyze_image_bytes(photo.getvalue())
                    if rgb is not None:
                        st.image(
                            rgb, channels="RGB", use_container_width=True,
                            caption="Captured frame; detections listed below.",
                        )
                    else:
                        st.warning("Could not decode that photo — try capturing again.")

                    if payload and "error" not in payload:
                        st.markdown(
                            '<div role="status" aria-live="polite" '
                            'style="font-family:var(--font-mono);font-size:11px;'
                            'letter-spacing:1px;color:var(--hud-cyan)">'
                            f'SCENE {_esc(payload.get("scene","?")).upper()} · '
                            f'CROWD {_esc(payload.get("crowd_count",0))} · '
                            f'RISK {_esc(payload.get("risk_level","LOW"))}</div>',
                            unsafe_allow_html=True,
                        )
                        items = [
                            {"text": f'{d.get("label","obj").upper()} {d.get("side","")}',
                             "variant": "danger" if d.get("collision_risk", 0) > 0.6 else None}
                            for d in payload.get("detections", [])[:6]
                        ] + [
                            {"text": f'HAZARD {h.get("label","hazard").upper()}', "variant": "warning"}
                            for h in payload.get("hazards", [])[:4]
                        ]
                        if items:
                            st.markdown(
                                '<div role="status" aria-live="polite">'
                                + hud.chips(items) + '</div>',
                                unsafe_allow_html=True,
                            )

                        # Feed the hazard pipeline: auto on HIGH risk, else on button.
                        # Cooldown-guard the HIGH auto-run so the still photo persisting
                        # in session_state can't re-fire the pipeline (and possible live
                        # calls) on every rerun. The manual button is never gated.
                        risk = payload.get("risk_level", "LOW")
                        run_now = False
                        if risk == "HIGH":
                            now = time.time()
                            if now - st.session_state.last_auto_trigger > 15:
                                st.session_state.last_auto_trigger = now
                                run_now = True
                                st.markdown(
                                    '<div role="alert" aria-live="assertive" '
                                    'style="font-family:var(--font-mono);font-size:11px;'
                                    'letter-spacing:1px;color:var(--hud-danger)">'
                                    'HIGH RISK — running agent pipeline…</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.info("HIGH risk already handled — cooldown active. Recapture to re-run.")
                        else:
                            run_now = st.button("Analyze hazard", use_container_width=True)
                        if run_now:
                            run_pipeline({"heart_rate": 85, "hrv": 45, **payload}, arm_real_calls)
                            speak_line(st.session_state.voice_line)
                            st.success("Photo analyzed and fed to the pipeline.")
                    elif payload and "error" in payload:
                        st.warning(f"Camera: {payload['error']}")
                    else:
                        st.info("No detections in that frame. Capture again to retry.")
                except ImportError:
                    st.warning("CV deps missing — run: pip install opencv-python ultralytics")
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Photo analysis error: {exc}")

        # ===== LOCAL WEBCAM / CONTINUITY ================================
        else:
            cam_index = st.number_input("Camera index", min_value=0, max_value=8, value=1, step=1)
            st.caption("Run `python list_cams.py` to find your iPhone/Continuity camera index.")
            enable_cam = st.checkbox(
                "Enable Live YOLO Camera", key="enable_cam",
                help="Runs local YOLOv8. High CPU. Requires opencv + ultralytics.")
            # st.tabs renders EVERY tab body each rerun and cannot report which tab
            # is visually active server-side, so the self-rerun streaming loop below
            # would keep firing even when the operator is on another tab (pinning the
            # app in a continuous high-CPU rerun cycle). Gate the loop on an explicit
            # Streaming guard the operator toggles only while watching this tab.
            streaming = st.checkbox(
                "Streaming", key="cam_streaming",
                help="Pull live frames + self-rerun. Turn OFF when viewing another tab "
                     "to stop the rerun loop and idle the camera.")

            # If the camera index changed since we cached a generator, drop the old
            # one so we reopen on the new device (avoids reading the wrong camera).
            if st.session_state.get("cam_index_active") != int(cam_index) and "cam_gen" in st.session_state:
                try:
                    st.session_state.cam_gen.close()
                except Exception:  # noqa: BLE001
                    pass
                st.session_state.pop("cam_gen", None)
                st.session_state.pop("cam_index_active", None)

            if enable_cam and streaming:
                try:
                    from perception import vision_generator

                    cam_box = st.empty()
                    info_box = st.empty()
                    chip_box = st.empty()
                    COOLDOWN = 15           # seconds between auto-pipeline runs
                    MAX_FRAMES_PER_RUN = 30  # bounded batch keeps Streamlit responsive

                    # FLICKER FIX: build the generator ONCE and reuse it across reruns.
                    # The old code opened + released the camera every rerun, so the
                    # camera light cycled on/off. Now we cache the generator in
                    # session_state and only close it on OFF / source-switch / index
                    # change / error — keeping the camera continuously ON.
                    gen = st.session_state.get("cam_gen")
                    if gen is None:
                        gen = vision_generator(camera_index=int(cam_index))
                        st.session_state.cam_gen = gen
                        st.session_state.cam_index_active = int(cam_index)

                    triggered = False
                    stop = False
                    try:
                        for i in range(MAX_FRAMES_PER_RUN):
                            if not st.session_state.get("enable_cam", True) \
                                    or not st.session_state.get("cam_streaming", False):
                                break
                            frame_rgb, payload = next(gen)
                            if frame_rgb is not None:
                                cam_box.image(
                                    frame_rgb, channels="RGB", use_container_width=True,
                                    caption="Live perception feed; detections listed below.",
                                )
                            if not payload:
                                continue
                            if "error" in payload:
                                info_box.warning(f"Camera: {payload['error']}")
                                stop = True
                                break

                            info_box.markdown(
                                '<div role="status" aria-live="polite" '
                                'style="font-family:var(--font-mono);font-size:11px;'
                                'letter-spacing:1px;color:var(--hud-cyan)">'
                                f'SCENE {_esc(payload.get("scene","?")).upper()} · '
                                f'CROWD {_esc(payload.get("crowd_count",0))} · '
                                f'RISK {_esc(payload.get("risk_level","LOW"))}</div>',
                                unsafe_allow_html=True,
                            )
                            items = [
                                {"text": f'{d.get("label","obj").upper()} {d.get("side","")}',
                                 "variant": "danger" if d.get("collision_risk", 0) > 0.6 else None}
                                for d in payload.get("detections", [])[:6]
                            ] + [
                                {"text": f'HAZARD {h.get("label","hazard").upper()}', "variant": "warning"}
                                for h in payload.get("hazards", [])[:4]
                            ]
                            if items:
                                chip_box.markdown(
                                    '<div role="status" aria-live="polite">'
                                    + hud.chips(items) + '</div>',
                                    unsafe_allow_html=True,
                                )

                            if payload.get("risk_level") == "HIGH":
                                now = time.time()
                                if now - st.session_state.last_auto_trigger > COOLDOWN:
                                    st.session_state.last_auto_trigger = now
                                    info_box.markdown(
                                        '<div role="alert" aria-live="assertive" '
                                        'style="font-family:var(--font-mono);font-size:11px;'
                                        'letter-spacing:1px;color:var(--hud-danger)">'
                                        'HIGH RISK — running agent pipeline…</div>',
                                        unsafe_allow_html=True,
                                    )
                                    run_pipeline({"heart_rate": 85, "hrv": 45, **payload}, arm_real_calls)
                                    speak_line(st.session_state.voice_line)
                                    triggered = True
                                    break
                    except StopIteration:
                        stop = True
                    except Exception as exc:  # noqa: BLE001
                        info_box.warning(f"Camera/vision error: {exc}")
                        stop = True

                    # Only release the camera on a real stop condition. A normal batch
                    # finishing must NOT close the generator (that caused the flicker).
                    if stop:
                        try:
                            gen.close()
                        except Exception:  # noqa: BLE001
                            pass
                        st.session_state.pop("cam_gen", None)
                        st.session_state.pop("cam_index_active", None)

                    if triggered:
                        st.success("Episode analyzed and saved. Resuming feed.")
                    # Keep streaming the next batch only while the camera is enabled AND
                    # the Streaming guard is on — so the self-rerun loop stops firing when
                    # the operator leaves this tab (st.tabs can't tell us server-side).
                    if st.session_state.get("enable_cam", True) \
                            and st.session_state.get("cam_streaming", False) and not stop:
                        st.rerun()
                except ImportError:
                    st.warning("CV deps missing — run: pip install opencv-python ultralytics")
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Camera/vision error: {exc}")
            else:
                # Camera OFF or Streaming paused: release the cached generator so the
                # webcam light actually goes out and the self-rerun loop stops (no
                # lingering open capture across reruns).
                if "cam_gen" in st.session_state:
                    try:
                        st.session_state.cam_gen.close()
                    except Exception:  # noqa: BLE001
                        pass
                    st.session_state.pop("cam_gen", None)
                    st.session_state.pop("cam_index_active", None)
                if enable_cam and not streaming:
                    st.info("Streaming paused — camera idle. Check Streaming to resume the live feed.")
                else:
                    st.info("Camera disabled. Use the manual scenario buttons above to drive the pipeline.")

    with col_bus:
        st.markdown(hud.hud_sep("CONFLICT BUS // CLAIMS"), unsafe_allow_html=True)
        last = st.session_state.last
        render_claims_panel(last.get("claims", []))
        render_directive_panel(last.get("action", ""), last.get("dispatch", ""))
        render_voice_output(st.session_state.voice_line)
        hr_series = last.get("hr_series", [])
        if hr_series:
            st.markdown(hud.hud_sep("HEART RATE // RECENT"), unsafe_allow_html=True)
            st.area_chart({"BPM": hr_series}, height=140, color="#00D4FF")

# ===== AGENTS ================================================================
with tab_agents:
    st.markdown(hud.hud_sep("AGENT MESH // STATUS"), unsafe_allow_html=True)
    agents = ["Perception", "Biometric", "Pattern", "Memory",
              "Orchestrator", "Action", "Retrospective"]
    chips_html = "".join(hud.agent_status(a, "ONLINE") for a in agents)
    st.markdown(f'<div class="chips">{chips_html}</div>', unsafe_allow_html=True)

    last = st.session_state.last
    a_left, a_right = st.columns(2)
    with a_left:
        st.markdown(hud.hud_sep("PREDICTIONS"), unsafe_allow_html=True)
        preds = last.get("predictions", [])
        if preds:
            body = "".join(
                f'<div style="margin:6px 0">'
                f'<span style="color:var(--hud-cyan)">▸ {_esc(p.get("description",""))}</span>'
                f'<span style="font-family:var(--font-mono);font-size:10px;color:var(--hud-muted);'
                f'margin-left:8px">CONF {int(p.get("confidence",0)*100)}%</span></div>'
                for p in preds
            )
        else:
            body = '<span style="color:var(--hud-success)">● NO ANOMALY PREDICTED</span>'
        st.markdown(hud.panel("PATTERN DETECTOR", "ANOMALY FORECAST", body), unsafe_allow_html=True)
    with a_right:
        st.markdown(hud.hud_sep("RECALLED MEMORIES"), unsafe_allow_html=True)
        mems = last.get("memories", [])
        if mems:
            body = "".join(f'<div style="margin:5px 0">◆ {_esc(m)}</div>' for m in mems[:6])
        else:
            body = '<span style="color:var(--hud-muted)">NO EPISODIC RECALL FOR THIS CONTEXT</span>'
        st.markdown(hud.panel("MEMORY AGENT", "MEM0 // EPISODIC", body), unsafe_allow_html=True)

# ===== BIOMETRICS ============================================================
with tab_bio:
    st.markdown(hud.hud_sep("VITALS // TELEMETRY"), unsafe_allow_html=True)
    last = st.session_state.last
    telem = last.get("telemetry", {})
    hr_series = last.get("hr_series", [])
    bm1, bm2 = st.columns(2)
    with bm1:
        st.markdown(hud.stat_card(str(telem.get("heart_rate", 72)), "HEART RATE", "BPM",
                                  "danger" if telem.get("heart_rate", 72) > 100 else "success"),
                    unsafe_allow_html=True)
        if hr_series:
            st.area_chart({"BPM": hr_series}, height=160, color="#00D4FF")
    with bm2:
        st.markdown(hud.stat_card(str(telem.get("hrv", 65)), "HRV", "ms",
                                  "warning" if telem.get("hrv", 65) < 40 else "cyan"),
                    unsafe_allow_html=True)
        hrv_series = [telem.get("hrv", 65)] * max(1, len(hr_series))
        st.area_chart({"HRV": hrv_series}, height=160, color="#34E0CE")

    st.markdown(hud.hud_sep("COMPANION HUD // MIRROR"), unsafe_allow_html=True)
    preds = last.get("predictions", [])
    confidence = int(preds[-1].get("confidence", 0.91) * 100) if preds else 91
    detections = telem.get("detections", [])
    live_state = {
        "agents_online": 7,
        "confidence": confidence,
        "heart_rate": telem.get("heart_rate", 72),
        "hazards_near": len(detections),
        "heart_variant": "danger" if telem.get("heart_rate", 72) > 100 else "success",
        "hazards_variant": "warning" if detections else "success",
        "spoken": st.session_state.voice_line,
        "tone": "URGENT" if last.get("claims") else "CALM",
    }
    try:
        render_companion_hud(live_state)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Companion HUD unavailable: {exc}")

# ===== PLACE =================================================================
with tab_place:
    st.markdown(hud.hud_sep("VOICE // ASSISTANT"), unsafe_allow_html=True)
    try:
        from voice_ui import render_voice_panel
        render_voice_panel()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Voice panel unavailable: {exc}")

    st.markdown(hud.hud_sep("LOCATION // PLACE"), unsafe_allow_html=True)
    try:
        from location_agent import LocationAgent
        loc = LocationAgent()
        st.write(loc.where_am_i())
        dest = st.text_input("Directions to", placeholder="e.g. Harvard Square")
        if dest:
            st.write(loc.directions(dest))
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Location agent unavailable: {exc}")
