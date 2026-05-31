"""feat_safety.py — SAFETY tab ("FEAT // Safety Pipeline"), the hazard-detection story.

Renders the SAFETY feature body into the current Streamlit container. It visualizes
the live multi-agent hazard pipeline from app.py + perception.py:

    Perception -> Biometric -> Pattern -> Memory -> Orchestrator -> Action

Layout (top -> bottom):
  * TitleBlock        — hud_sep "SAFETY PIPELINE"
  * Metrics row       — 4 stat_card (RISK LEVEL / SCENE / HEART RATE bpm / ACTION)
  * Pipeline panel    — the 6-stage agent chain with per-stage status chips
  * Camera + claims   — left: live YOLO camera + manual scenario buttons;
                        right: CONFLICT BUS claims + DIRECTIVE + SPOKEN line

Design: tactical cyan/teal-on-black HUD via hud_theme only. Accessibility is the
product — every control is labeled, the spoken line is always shown on-screen (never
audio-only), severity is paired with text/icon (never color alone), and live agent
output is wrapped in aria-live regions.

Public surface:
    render() -> None      # assumes hud_theme.inject_theme() already ran in feat_ui.py
"""

from __future__ import annotations

import html
import time

import streamlit as st

import hud_theme as hud

# Pipeline agents — app.py is the source of truth for the hazard pipeline.
# Import defensively: a missing dependency must NOT crash the whole tab.
try:
    from app import (
        process_biometrics,
        process_vision_queue,
        pattern_detector,
        memory_agent,
        behavior_orchestrator,
        action_dispatcher,
        retrospective_agent,
    )
    _PIPELINE_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 — degrade gracefully, never crash the tab
    process_biometrics = process_vision_queue = pattern_detector = None  # type: ignore
    memory_agent = behavior_orchestrator = action_dispatcher = None  # type: ignore
    retrospective_agent = None  # type: ignore
    _PIPELINE_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SAFE_ACTION = "Maintain passive navigation mode (Safe)"
_SAFE_DISPATCH = "[SYSTEM] Passive observation maintained. No external API triggered."
_SAFE_VOICE = "All clear. I'm here with you."

_COOLDOWN = 15            # seconds between auto-pipeline runs from the live feed
_MAX_FRAMES = 30          # bound the camera loop so Streamlit stays responsive

# Scenario presets for the three manual buttons.
_SCENARIOS = {
    "Safe": {
        "heart_rate": 72, "hrv": 65, "crowd_density": 0.1,
        "fall_detected": False, "risk_level": "LOW", "crowd_count": 1,
        "scene": "indoor",
    },
    "Approaching Crowd": {
        "heart_rate": 85, "hrv": 45, "crowd_density": 0.9,
        "fall_detected": False, "risk_level": "MED", "crowd_count": 6,
        "scene": "street",
    },
    "Panic Attack": {
        "heart_rate": 115, "hrv": 22, "crowd_density": 0.1,
        "fall_detected": False, "risk_level": "LOW", "crowd_count": 0,
        "scene": "indoor",
    },
}

# Pipeline stages, in execution order, mapped to the result key proving each ran.
_STAGES = [
    ("Perception", "Vision / YOLO", "scene"),
    ("Biometric", "Heart-rate claims", "bio_claims"),
    ("Pattern", "Anomaly forecast", "predictions"),
    ("Memory", "Episodic recall", "memories"),
    ("Orchestrator", "Gemini directive", "action"),
    ("Action", "Dispatch / arm gate", "dispatch"),
]


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


# ---------------------------------------------------------------------------
# Session state — namespaced "feat_*" keys so the tab renders before any run.
# ---------------------------------------------------------------------------
def _init_state() -> None:
    if "feat_safety_last" not in st.session_state:
        st.session_state.feat_safety_last = {
            "claims": [],
            "bio_claims": [],
            "vis_claims": [],
            "predictions": [],
            "memories": [],
            "action": _SAFE_ACTION,
            "dispatch": _SAFE_DISPATCH,
            "telemetry": {
                "heart_rate": 72, "hrv": 65, "risk_level": "LOW",
                "crowd_count": 0, "scene": "indoor",
            },
            "ran": False,        # has any run completed this session?
        }
    if "feat_safety_voice" not in st.session_state:
        st.session_state.feat_safety_voice = _SAFE_VOICE
    if "feat_safety_last_auto" not in st.session_state:
        st.session_state.feat_safety_last_auto = 0.0


# ---------------------------------------------------------------------------
# Shared run helper — builds telemetry, merges biometric + vision claims, then
# runs pattern -> memory -> orchestrator -> action (arm-gated) -> retrospective.
# Stores everything in st.session_state so reruns + the rest of the tab persist.
# ---------------------------------------------------------------------------
def _run_pipeline(telemetry: dict, arm_real_calls: bool = False) -> dict:
    """Run the full conflict-bus pipeline for one telemetry payload.

    Every backend call is wrapped so a missing GEMINI_API_KEY / network / camera
    failure degrades to Safe instead of crashing. The live emergency call only
    fires when ``arm_real_calls`` is True (default OFF -> always simulated).
    """
    if _PIPELINE_IMPORT_ERROR is not None or process_biometrics is None:
        st.warning(
            "Safety pipeline backend unavailable "
            f"({type(_PIPELINE_IMPORT_ERROR).__name__ if _PIPELINE_IMPORT_ERROR else 'import'}). "
            "Showing last known state."
        )
        return st.session_state.feat_safety_last

    # 1. Ingest — biometric + vision claims merged onto the conflict bus.
    bio_claims: list = []
    try:
        bio_claims = process_biometrics(telemetry).get("active_claims", [])
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Biometric agent unavailable: {exc}")
    vis_claims: list = []
    try:
        vis_claims = process_vision_queue(telemetry).get("active_claims", [])
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Vision agent unavailable: {exc}")
    claims = bio_claims + vis_claims

    state: dict = {
        "current_telemetry": telemetry,
        "active_claims": claims,
        "active_predictions": [],
        "retrieved_memories": [],
        "final_action": _SAFE_ACTION,
        # Thread the live-call gate through state (thread-safe, no env mutation).
        "arm_real_calls": bool(arm_real_calls),
    }

    # 2. Pattern detection (deterministic, safe).
    predictions: list = []
    try:
        predictions = pattern_detector(state).get("active_predictions", [])
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Pattern detector unavailable: {exc}")
    state["active_predictions"] = predictions

    # 3. Memory recall (Mem0 / Gemini — may need a key).
    memories: list = []
    try:
        memories = memory_agent(state).get("retrieved_memories", [])
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Memory agent unavailable (check GEMINI_API_KEY): {exc}")
    state["retrieved_memories"] = memories

    # 4. Orchestrator decision (Gemini LLM — degrade to Safe on failure).
    action = _SAFE_ACTION
    try:
        action = behavior_orchestrator(state).get("final_action", action)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Orchestrator unavailable (check GEMINI_API_KEY): {exc}")
    state["final_action"] = action

    # 5. Action dispatch — the LIVE emergency branch is gated by arm_real_calls.
    dispatch = _SAFE_DISPATCH
    try:
        dispatch = action_dispatcher(state).get("execution_status", dispatch)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Action dispatcher unavailable: {exc}")

    # 6. Retrospective write-back (best effort).
    try:
        retrospective_agent(state, resolution_success=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Retrospective agent unavailable: {exc}")

    # Persist for cross-rerun sharing.
    st.session_state.feat_safety_last = {
        "claims": claims,
        "bio_claims": bio_claims,
        "vis_claims": vis_claims,
        "predictions": predictions,
        "memories": memories,
        "action": action,
        "dispatch": dispatch,
        "telemetry": telemetry,
        "ran": True,
    }
    # Spoken line shown as text everywhere (accessibility: always visible).
    st.session_state.feat_safety_voice = action
    return st.session_state.feat_safety_last


# ---------------------------------------------------------------------------
# Section: TitleBlock + metrics row
# ---------------------------------------------------------------------------
def _claim_severity(ctype: str) -> str:
    """Conflict-bus claim severity: Tachycardia=danger, Vision=warning, else cyan."""
    return {"TachycardiaAlert": "danger", "VisionAlert": "warning"}.get(ctype, "cyan")


def _short_action(action: str) -> str:
    """Condense a directive string into a 1-2 word ACTION metric label."""
    a = (action or "").lower()
    if "emergency" in a:
        return "EMERGENCY"
    if "grounding" in a or "airpods" in a or "countdown" in a:
        return "GROUNDING"
    if "nudge" in a:
        return "NUDGE"
    return "PASSIVE"


def _render_title_and_metrics() -> None:
    last = st.session_state.feat_safety_last
    telem = last.get("telemetry", {})
    risk = str(telem.get("risk_level", "LOW")).upper()
    scene = str(telem.get("scene", "—")).upper()
    hr = telem.get("heart_rate", 72)
    action = last.get("action", _SAFE_ACTION)
    action_short = _short_action(action)

    # TitleBlock — hud_sep emits a real <h2> for screen-reader navigation.
    st.markdown(hud.hud_sep("SAFETY PIPELINE"), unsafe_allow_html=True)
    st.markdown(
        '<p style="font-family:var(--font-mono);font-size:11px;letter-spacing:2px;'
        'color:var(--hud-muted);margin:0 0 .6rem">'
        'HAZARD DETECTION // PERCEPTION → BIOMETRIC → PATTERN → MEMORY → '
        'ORCHESTRATOR → ACTION</p>',
        unsafe_allow_html=True,
    )

    risk_variant = {"HIGH": "danger", "MED": "warning"}.get(risk, "success")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            hud.stat_card(risk, "RISK LEVEL", "FUSED", risk_variant),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            hud.stat_card(scene, "SCENE", "VISION", "cyan"),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            hud.stat_card(str(hr), "HEART RATE", "BPM",
                          "danger" if hr > 100 else "success"),
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            hud.stat_card(action_short, "ACTION", "DIRECTIVE",
                          "danger" if action_short == "EMERGENCY"
                          else ("warning" if action_short != "PASSIVE" else "cyan")),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section: PIPELINE panel — agent chain with per-stage status.
# ---------------------------------------------------------------------------
def _stage_state(key: str, last: dict) -> tuple[str, str]:
    """Return (display_state, variant) for a pipeline stage from the last run.

    Stages are ONLINE once a run completed; an empty result reads STANDBY so we
    never imply work happened when it didn't (honest status, WCAG 1.4.1: text-paired).
    """
    if not last.get("ran"):
        return "STANDBY", "warning"
    val = last.get(key)
    has = bool(val) if isinstance(val, (list, str)) else val is not None
    # Perception/Memory legitimately produce nothing in a calm scene -> READY.
    if key in ("scene", "memories", "bio_claims") and not has:
        return "READY", "cyan"
    return ("ONLINE", "success") if has else ("READY", "cyan")


def _render_pipeline_panel() -> None:
    last = st.session_state.feat_safety_last
    st.markdown(hud.hud_sep("PIPELINE // AGENT CHAIN"), unsafe_allow_html=True)

    # Build the chain as agent_status chips joined by arrow connectors, wrapped in
    # an aria-live region so assistive tech hears stage transitions when they change.
    parts: list[str] = []
    for idx, (name, _sub, key) in enumerate(_STAGES):
        disp, _variant = _stage_state(key, last)
        parts.append(hud.agent_status(name, disp))
        if idx < len(_STAGES) - 1:
            parts.append(
                '<span aria-hidden="true" style="color:var(--hud-cyan-40);'
                'font-family:var(--font-mono);font-size:11px;margin:0 1px">→</span>'
            )

    chain_html = (
        '<div role="group" aria-label="Agent pipeline chain status" '
        'style="display:flex;flex-wrap:wrap;align-items:center;gap:2px">'
        + "".join(parts) + '</div>'
    )

    # Per-stage detail rows so the substages + their plain-text state are explicit.
    rows: list[str] = []
    for name, sub, key in _STAGES:
        disp, variant = _stage_state(key, last)
        accent = {"success": "--hud-success", "warning": "--hud-warning",
                  "danger": "--hud-danger"}.get(variant, "--hud-cyan")
        rows.append(
            '<div style="display:flex;align-items:center;gap:10px;'
            'padding:6px 0;border-bottom:1px solid var(--border-subtle)">'
            f'<span style="min-width:108px;font-family:var(--font-mono);font-size:11px;'
            f'letter-spacing:1px;color:var(--hud-cyan)">{_esc(name).upper()}</span>'
            f'<span style="flex:1;font-size:12px;color:var(--text-secondary)">{_esc(sub)}</span>'
            f'<span style="font-family:var(--font-mono);font-size:9px;letter-spacing:1px;'
            f'font-weight:700;color:var({accent})">● {_esc(disp)}</span></div>'
        )

    pill_html = hud.pill("RAN", "live") if last.get("ran") else hud.pill("STANDBY", "warning")
    body = (
        '<div role="status" aria-live="polite" aria-atomic="true">'
        + chain_html
        + '<div style="margin-top:10px">' + "".join(rows) + '</div></div>'
    )
    st.markdown(
        hud.panel("AGENT PIPELINE", "PERCEPTION → ACTION", body, pill_html),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section: claims / directive / voice column.
# ---------------------------------------------------------------------------
def _render_claims_column() -> None:
    last = st.session_state.feat_safety_last
    claims = last.get("claims", [])

    st.markdown(hud.hud_sep("CONFLICT BUS // CLAIMS"), unsafe_allow_html=True)
    if not claims:
        st.markdown(
            hud.panel(
                "CONFLICT BUS", "NO ACTIVE CLAIMS",
                '<div role="status" aria-live="polite">'
                '<span style="color:var(--hud-success)">● ALL CALM // BUS CLEAR</span></div>',
            ),
            unsafe_allow_html=True,
        )
    else:
        rows = "".join(
            hud.claim_row(
                c.get("source", "AGENT"),
                c.get("type", "Claim"),
                str(c.get("value", "")),
                _claim_severity(c.get("type", "")),
            )
            for c in claims
        )
        st.markdown(
            hud.panel(
                "CONFLICT BUS", f"{len(claims)} ACTIVE CLAIM(S)",
                f'<div role="status" aria-live="polite">{rows}</div>',
                hud.pill("FIRING", "danger"),
            ),
            unsafe_allow_html=True,
        )

    # DIRECTIVE — the chosen action + dispatch status, announced to assistive tech.
    action = last.get("action", _SAFE_ACTION)
    dispatch = last.get("dispatch", _SAFE_DISPATCH)
    directive_body = (
        '<div role="status" aria-live="assertive" aria-atomic="true">'
        '<span class="vis-hidden">Current directive: </span>'
        f'<div style="font-size:14px;color:var(--text-primary);line-height:1.4">{_esc(action)}</div>'
        f'<div style="font-family:var(--font-mono);font-size:10px;letter-spacing:1px;'
        f'color:var(--hud-label);margin-top:6px">{_esc(dispatch)}</div></div>'
    )
    st.markdown(
        hud.panel("DIRECTIVE", "ORCHESTRATOR DECISION", directive_body,
                  hud.pill("EXECUTED", "cyan")),
        unsafe_allow_html=True,
    )

    # SPOKEN line — always shown as text (never audio-only), in a live region.
    line = st.session_state.feat_safety_voice
    voice_body = (
        '<div role="status" aria-live="assertive" aria-atomic="true">'
        '<span class="vis-hidden">Spoken to handler: </span>'
        f'<div style="font-size:15px;color:var(--hud-text);font-style:italic">'
        f'&ldquo;{_esc(line)}&rdquo;</div></div>'
    )
    st.markdown(
        hud.panel("SPOKEN", "VOICE TO HANDLER", voice_body, hud.pill("ON-SCREEN", "live")),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section: camera + manual controls column.
# ---------------------------------------------------------------------------
def _detection_chips(payload: dict) -> list[dict]:
    items = [
        {"text": f'{str(d.get("label", "obj")).upper()} {d.get("side", "")}'.strip(),
         "variant": "danger" if d.get("collision_risk", 0) > 0.6 else None}
        for d in payload.get("detections", [])[:6]
    ]
    items += [
        {"text": f'HAZARD {str(h.get("label", "hazard")).upper()}', "variant": "warning"}
        for h in payload.get("hazards", [])[:4]
    ]
    return items


def _run_live_camera(cam_index: int, arm_real_calls: bool) -> None:
    """Loop perception.vision_generator for a bounded number of frames.

    Stops on frame budget, on the checkbox being unticked, or on a camera error.
    Auto-runs the pipeline on the first HIGH-risk frame (cooldown-guarded). The
    generator is always closed in a finally so the camera is released.
    """
    try:
        from perception import vision_generator
    except ImportError:
        st.warning("CV deps missing — run: pip install opencv-python ultralytics")
        return
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Perception import error: {exc}")
        return

    cam_box = st.empty()
    info_box = st.empty()
    chip_box = st.empty()
    triggered = False

    try:
        gen = vision_generator(camera_index=int(cam_index))
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Camera/vision error: {exc}")
        return

    try:
        for i, (frame_rgb, payload) in enumerate(gen):
            # Stop: frame budget reached or the checkbox was unticked.
            if i >= _MAX_FRAMES or not st.session_state.get("feat_safety_cam", True):
                break
            if frame_rgb is not None:
                # Dynamic caption so the image's visible/accessible text reflects
                # what is actually in frame (a11y); chips below remain the source
                # of truth in their aria-live region.
                if payload:
                    cap = (
                        f"Live feed — scene "
                        f'{str(payload.get("scene", "?")).upper()}, '
                        f'{payload.get("crowd_count", 0)} people, '
                        f'risk {payload.get("risk_level", "LOW")}; '
                        "detections + hazards listed below."
                    )
                else:
                    cap = "Live perception feed; detections + hazards listed below."
                cam_box.image(
                    frame_rgb, channels="RGB", use_container_width=True,
                    caption=cap,
                )
            if not payload:
                continue
            if "error" in payload:
                info_box.warning(f"Camera: {payload['error']}")
                break

            # Live scene summary (aria-live: announced to assistive tech).
            info_box.markdown(
                '<div role="status" aria-live="polite" '
                'style="font-family:var(--font-mono);font-size:11px;letter-spacing:1px;'
                'color:var(--hud-cyan)">'
                f'SCENE {_esc(str(payload.get("scene", "?")).upper())} · '
                f'CROWD {_esc(payload.get("crowd_count", 0))} · '
                f'RISK {_esc(payload.get("risk_level", "LOW"))}</div>',
                unsafe_allow_html=True,
            )
            items = _detection_chips(payload)
            if items:
                chip_box.markdown(
                    '<div role="status" aria-live="polite" '
                    'aria-label="Detections and hazards">' + hud.chips(items) + '</div>',
                    unsafe_allow_html=True,
                )

            # Auto-run the pipeline on HIGH risk, cooldown-guarded.
            if payload.get("risk_level") == "HIGH":
                now = time.time()
                if now - st.session_state.feat_safety_last_auto > _COOLDOWN:
                    st.session_state.feat_safety_last_auto = now
                    info_box.markdown(
                        '<div role="alert" aria-live="assertive" '
                        'style="font-family:var(--font-mono);font-size:11px;letter-spacing:1px;'
                        'color:var(--hud-danger)">HIGH RISK — running agent pipeline…</div>',
                        unsafe_allow_html=True,
                    )
                    _run_pipeline({"heart_rate": 85, "hrv": 45, **payload}, arm_real_calls)
                    triggered = True
                    break
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Camera/vision error: {exc}")
    finally:
        # Always close the generator so its finally-block releases the camera.
        try:
            gen.close()
        except Exception:  # noqa: BLE001
            pass

    if triggered:
        st.success("Episode analyzed and saved. Resuming feed.")
    # Re-trigger so the checkbox can stop the feed on the next rerun.
    if st.session_state.get("feat_safety_cam", True):
        st.rerun()


def _render_camera_column(arm_real_calls: bool) -> None:
    st.markdown(hud.hud_sep("PERCEPTION // CAMERA"), unsafe_allow_html=True)

    cam_index = st.number_input(
        "Camera index", min_value=0, max_value=8, value=1, step=1,
        key="feat_safety_cam_index",
        help="Which camera the YOLO feed reads (0 = default webcam).",
    )
    enable_cam = st.checkbox(
        "Enable Live YOLO Camera", key="feat_safety_cam",
        help="Runs local YOLOv8 on the selected camera. High CPU. "
             "Requires opencv-python + ultralytics.",
    )

    st.caption("Manual scenario triggers — drive the pipeline without a camera")
    b1, b2, b3 = st.columns(3)
    btns = (
        (b1, "Safe"),
        (b2, "Approaching Crowd"),
        (b3, "Panic Attack"),
    )
    for col, label in btns:
        if col.button(label, use_container_width=True, key=f"feat_safety_btn_{label}"):
            _run_pipeline(_SCENARIOS[label], arm_real_calls)
            st.rerun()

    if enable_cam:
        _run_live_camera(int(cam_index), arm_real_calls)
    else:
        st.info(
            "Camera disabled. Use the manual scenario buttons above to drive the "
            "pipeline, or enable the live YOLO camera."
        )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def render() -> None:
    """Render the SAFETY tab body. Assumes inject_theme() already ran in feat_ui.py."""
    _init_state()

    # ARM REAL CALLS gate — read from the shared session_state key written by the
    # SYSTEM tab (feat_system._K_ARM = "arm_real_calls"). Default OFF.
    arm_real_calls = bool(st.session_state.get("arm_real_calls", False))

    _render_title_and_metrics()
    _render_pipeline_panel()

    # Camera (left, wider) + claims / decision (right).
    col_cam, col_bus = st.columns([3, 2])
    with col_cam:
        _render_camera_column(arm_real_calls)
    with col_bus:
        _render_claims_column()

    # Honest status footer: surface the live-call arm state in plain text + pill.
    if arm_real_calls:
        st.markdown(
            '<div style="margin-top:.6rem">' + hud.pill("LIVE CALLS ARMED", "danger") + '</div>',
            unsafe_allow_html=True,
        )
        st.caption("Live emergency calls are ARMED. The emergency directive may place a real call.")
    else:
        st.markdown(
            '<div style="margin-top:.6rem">' + hud.pill("CALLS SAFE / SIMULATED", "cyan") + '</div>',
            unsafe_allow_html=True,
        )
        st.caption("Live calls are OFF. The emergency path is always simulated (no real call).")
