"""feat_memory.py — FEAT // Memory & Persistence tab body.

The "it remembers / it learns" story. Renders the persistence layers behind the
conflict bus — plus a live recall demo, all in the tactical cyan/teal HUD. Honest
by design (PRD §2, 🟡 caveat): Shaped is the primary ranked episodic recall
engine (cross-session), episodic *history* persists in SQLite across restarts,
and a secondary local mem0/Qdrant vector layer starts EMPTY on each launch
(`path=":memory:"`) and is NOT rehydrated from SQLite yet. The panel labels this
plainly.

Backend it maps to (app.py):
  - shaped_client       → Shaped ranked recall (primary; needs SHAPED_API_KEY)
  - mem0_config         → local mem0/Qdrant `:memory:`, dim 384, history_db `mem0_history.db`
  - memory_client       → mem0 client (None when SHAPED_API_KEY is missing)
  - retrospective_agent → writes episodes to Shaped
  - memory_agent        → ranked recall via Shaped
  - WEAVE_PROJECT       → W&B Weave cloud decision trace

Exposes exactly:  def render() -> None
inject_theme() is assumed to have already run (in feat_ui.py).
"""

from __future__ import annotations

import html
import os
import sqlite3

import streamlit as st

import hud_theme as hud

# Pull the live backend handles fail-soft. Importing app.py must already be safe
# (it degrades memory_client to None with no key), but guard anyway so a missing
# dependency in app.py can never blank out this tab.
try:
    from app import memory_client, mem0_config, WEAVE_PROJECT
except Exception as exc:  # noqa: BLE001 — never crash the tab on import
    memory_client = None
    mem0_config = {}
    WEAVE_PROJECT = os.environ.get(
        "WEAVE_PROJECT", "phillipsle997-boston-university/unleash-service-dog"
    )
    _IMPORT_ERR = exc
else:
    _IMPORT_ERR = None

# Memory namespace used by the pipeline (mirrors app.py memory_agent filters).
_USER_ID = "user_thtrang_06"


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


# ---------------------------------------------------------------------------
# Session defaults — so the tab renders fully before any interaction.
# ---------------------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("feat_mem_query", "")
    st.session_state.setdefault("feat_mem_results", [])      # list[str] of recalled episodes
    st.session_state.setdefault("feat_mem_searched", False)  # has a search run this session?
    st.session_state.setdefault("feat_mem_recall_hits", 0)   # cumulative successful recalls


# ---------------------------------------------------------------------------
# SQLite read — the REAL, honest row count from mem0_history.db.
# Wrapped in try/except: a missing or locked file must NOT crash the tab.
# ---------------------------------------------------------------------------
def _sqlite_stats() -> dict:
    """Return {'rows': int|None, 'path': str, 'exists': bool, 'error': str|None}.

    Reads the `history` audit table that Mem0 writes — this is what actually
    persists across restarts, so it is the honest source for the row count.
    """
    path = "mem0_history.db"
    if isinstance(mem0_config, dict):
        path = mem0_config.get("history_db_path", path)

    info = {"rows": None, "path": path, "exists": os.path.exists(path), "error": None}
    if not info["exists"]:
        info["error"] = "file not created yet"
        return info

    conn = None
    try:
        # read-only connection so we never lock the db Mem0 may be using
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
        row = conn.execute("SELECT COUNT(*) FROM history").fetchone()
        info["rows"] = int(row[0]) if row else 0
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        info["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    return info


def _vector_dim() -> int:
    """Embedding dim from mem0_config; sensible default 768."""
    try:
        return int(
            mem0_config.get("vector_store", {}).get("config", {}).get(
                "embedding_model_dims", 768
            )
        )
    except Exception:  # noqa: BLE001
        return 768


def _qdrant_path() -> str:
    try:
        return str(
            mem0_config.get("vector_store", {}).get("config", {}).get("path", ":memory:")
        )
    except Exception:  # noqa: BLE001
        return ":memory:"


# ---------------------------------------------------------------------------
# Recall — guarded call into memory_client.search(...).
# ---------------------------------------------------------------------------
def _recall(query: str) -> tuple[list[str], str | None]:
    """Return (episodes, warning). Honest about the in-session-only caveat.

    memory_client may be None (no GROQ_API_KEY for mem0's LLM) — guard and degrade
    to a warning instead of crashing.
    """
    if memory_client is None:
        return [], (
            "Local mem0/Qdrant layer offline — set GROQ_API_KEY to enable it. "
            "(The pipeline's primary recall is Shaped; SQLite history below still persists.)"
        )
    try:
        res = memory_client.search(query=query, filters={"user_id": _USER_ID})
    except Exception as exc:  # noqa: BLE001 — network / model / key failure
        return [], f"Recall failed: {type(exc).__name__}: {exc}"

    # Mem0 returns either {"results": [...]} or a bare list.
    rows = res.get("results", []) if isinstance(res, dict) else (res or [])
    episodes = [
        (r.get("memory") or r.get("content") or "").strip()
        for r in rows
        if isinstance(r, dict)
    ]
    return [e for e in episodes if e], None


# ---------------------------------------------------------------------------
# Small HTML builders (token colors only).
# ---------------------------------------------------------------------------
def _kv(label: str, value: str, accent_var: str = "--hud-cyan") -> str:
    """A mono label / value row used inside the TripleSave panel bodies."""
    return (
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'gap:10px;padding:3px 0;font-family:var(--font-mono);font-size:11px;'
        'letter-spacing:1px">'
        f'<span style="color:var(--hud-label)">{_esc(label)}</span>'
        f'<span style="color:var({accent_var})">{_esc(value)}</span></div>'
    )


def _note(text: str, accent_var: str) -> str:
    """An honest-status note line — text + a leading ▸ marker (never color-alone)."""
    return (
        '<div style="display:flex;gap:7px;margin-top:8px;font-family:var(--font-mono);'
        'font-size:10px;letter-spacing:.5px;line-height:1.45;'
        f'color:var(--hud-label)"><span aria-hidden="true" '
        f'style="color:var({accent_var})">&#9656;</span>'
        f'<span>{_esc(text)}</span></div>'
    )


def _link(url: str, text: str) -> str:
    return (
        f'<a href="{_esc(url)}" target="_blank" rel="noopener noreferrer" '
        f'style="color:var(--accent);text-decoration:underline">{_esc(text)}</a>'
    )


# ---------------------------------------------------------------------------
# RENDER
# ---------------------------------------------------------------------------
def render() -> None:
    """Render the Memory & Persistence tab body into the current container."""
    _init_state()

    if _IMPORT_ERR is not None:
        st.warning(
            "Memory backend import degraded "
            f"({type(_IMPORT_ERR).__name__}); showing offline view."
        )

    sql = _sqlite_stats()
    dim = _vector_dim()
    qdrant_path = _qdrant_path()
    weave_link = f"https://wandb.ai/{WEAVE_PROJECT}/weave"

    # ---- TitleBlock ----
    st.markdown(
        hud.reactor_header(
            "MEMORY &amp; PERSISTENCE",
            "FEAT // IT REMEMBERS — IT LEARNS",
            hud.pill("EPISODIC", "cyan"),
        ),
        unsafe_allow_html=True,
    )
    st.markdown(hud.hud_sep("MEMORY & PERSISTENCE"), unsafe_allow_html=True)

    # ---- Metrics row ----
    # EPISODES TODAY is read honestly from SQLite (None -> "—"); RECALL HITS is
    # the in-session counter; VECTOR DIM comes from mem0_config.
    episodes_val = "—" if sql["rows"] is None else str(sql["rows"])
    episodes_badge = "SQLITE" if sql["rows"] is not None else "OFFLINE"
    mem_online = memory_client is not None              # local mem0/Qdrant vector layer
    shaped_on = bool(os.environ.get("SHAPED_API_KEY"))  # pipeline's Shaped recall engine
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            hud.stat_card(episodes_val, "EPISODES STORED", episodes_badge, "memory"),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            hud.stat_card(
                str(st.session_state.feat_mem_recall_hits),
                "RECALL HITS",
                "SESSION",
                "success" if st.session_state.feat_mem_recall_hits else "cyan",
            ),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            hud.stat_card(str(dim), "VECTOR DIM", "MINILM", "cyan"),
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            hud.stat_card(
                "ON" if shaped_on else "OFF",
                "RECALL ENGINE",
                "SHAPED" if shaped_on else "NO KEY",
                "success" if shaped_on else "warning",
            ),
            unsafe_allow_html=True,
        )

    # ---- TRIPLE-SAVE row ----
    st.markdown(hud.hud_sep("RECALL // PERSISTENCE LAYERS"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    # 1) SHAPED RANKED RECALL — primary cross-session episodic recall (memory accent).
    with c1:
        body = (
            _kv("BACKEND", "Shaped (ranked recall)", "--hud-memory")
            + _kv("LOCAL LAYER", "mem0 / Qdrant", "--hud-memory")
            + _kv("PATH", qdrant_path, "--hud-memory")
            + _kv("COLLECTION", "unleash_memory", "--hud-memory")
            + _kv("DIM", str(dim), "--hud-memory")
            + _kv("EMBEDDER", "all-MiniLM-L6-v2 (HF)", "--hud-memory")
            + _note(
                "Shaped ranks episodes across sessions. The local mem0/Qdrant "
                "layer rebuilds on restart (not rehydrated from SQLite yet).",
                "--hud-memory",
            )
        )
        st.markdown(
            hud.panel(
                "SHAPED RANKED RECALL",
                "RANKED // CROSS-SESSION",
                body,
                hud.pill("CROSS-SESSION", "cyan"),
            ),
            unsafe_allow_html=True,
        )

    # 2) SQLITE mem0_history.db — persists across restarts; REAL row count (success accent).
    with c2:
        if sql["error"]:
            rows_line = _kv("ROWS", "unavailable", "--hud-warning")
            status_note = _note(f"Read note: {sql['error']}", "--hud-warning")
            pill_html = hud.pill("DEGRADED", "warning")
        else:
            rows_line = _kv("ROWS", str(sql["rows"]), "--hud-success")
            status_note = _note(
                "Persists across restarts. This is the real history row count.",
                "--hud-success",
            )
            pill_html = hud.pill("PERSISTED", "live")
        body = (
            _kv("FILE", sql["path"], "--hud-success")
            + _kv("TABLE", "history", "--hud-success")
            + rows_line
            + status_note
        )
        st.markdown(
            hud.panel("SQLITE mem0_history.db", "DURABLE // ON-DISK", body, pill_html),
            unsafe_allow_html=True,
        )

    # 3) W&B WEAVE — cloud decision trace (cyan accent).
    with c3:
        weave_on = bool(os.environ.get("WANDB_API_KEY"))
        body = (
            _kv("PROJECT", WEAVE_PROJECT, "--hud-cyan")
            + '<div style="padding:3px 0;font-family:var(--font-mono);font-size:11px;'
            'letter-spacing:1px;display:flex;justify-content:space-between;gap:10px">'
            '<span style="color:var(--hud-label)">TRACE</span>'
            f'<span>{_link(weave_link, "open in W&amp;B")}</span></div>'
            + _note(
                "Cloud decision trace of every @weave.op agent call. "
                + ("Streaming live." if weave_on else "Local-only (no WANDB_API_KEY)."),
                "--hud-cyan",
            )
        )
        st.markdown(
            hud.panel(
                "W&B WEAVE",
                "CLOUD // DECISION TRACE",
                body,
                hud.pill("CLOUD" if weave_on else "LOCAL", "cyan"),
            ),
            unsafe_allow_html=True,
        )

    # ---- RECALL DEMO ----
    st.markdown(hud.hud_sep("RECALL // EPISODIC LOOKUP"), unsafe_allow_html=True)

    with st.form("feat_mem_recall_form", clear_on_submit=False):
        query = st.text_input(
            "Recall a situation",  # labeled control (a11y)
            key="feat_mem_query",
            placeholder="e.g. approaching crowd, panic attack precursor…",
            help="In-session probe of the local mem0/Qdrant vector layer. "
                 "(The pipeline's primary recall engine is Shaped.)",
        )
        submitted = st.form_submit_button("Recall episodes", use_container_width=False)

    if submitted and query.strip():
        episodes, warn = _recall(query.strip())
        st.session_state.feat_mem_results = episodes
        st.session_state.feat_mem_searched = True
        if episodes:
            st.session_state.feat_mem_recall_hits += 1
        if warn:
            st.warning(warn)

    _render_recall_results(query)


def _render_recall_results(query: str) -> None:
    """Render recalled episodes in an aria-live region (announced as they change)."""
    results = st.session_state.get("feat_mem_results", [])
    searched = st.session_state.get("feat_mem_searched", False)

    if results:
        rows = "".join(
            hud.claim_row("MEM0", "Recalled episode", _esc(ep), "memory")
            for ep in results[:8]
        )
        inner = (
            '<span class="vis-hidden">'
            f'{len(results)} episode(s) recalled for query {_esc(query)}: </span>'
            + rows
        )
        st.markdown(
            '<div role="status" aria-live="polite" aria-atomic="true">'
            + hud.panel(
                "RETRIEVED EPISODES",
                f"{len(results)} MATCH(ES) // IN-SESSION",
                inner,
                hud.pill("RECALLED", "live"),
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    elif searched:
        body = (
            '<div role="status" aria-live="polite" aria-atomic="true">'
            '<span style="color:var(--hud-label);font-family:var(--font-mono);'
            'font-size:12px;letter-spacing:1px">NO EPISODES RECALLED THIS SESSION.</span>'
            '<div style="color:var(--hud-muted);font-family:var(--font-mono);'
            'font-size:10px;letter-spacing:.5px;margin-top:6px;line-height:1.45">'
            'The local mem0/Qdrant layer starts empty on launch and isn&#39;t rehydrated '
            'from SQLite yet &mdash; run the safety pipeline to write episodes, then recall.'
            '</div></div>'
        )
        st.markdown(
            hud.panel("RETRIEVED EPISODES", "0 MATCHES // IN-SESSION", body),
            unsafe_allow_html=True,
        )
    else:
        body = (
            '<span style="color:var(--hud-muted);font-family:var(--font-mono);'
            'font-size:11px;letter-spacing:1px">'
            'Enter a situation above and recall to search episodic memory.</span>'
        )
        st.markdown(
            hud.panel("RETRIEVED EPISODES", "AWAITING QUERY", body),
            unsafe_allow_html=True,
        )

    # Honest, always-visible footer caveat (PRD §2 🟡).
    st.markdown(
        '<p style="font-family:var(--font-mono);font-size:10px;letter-spacing:.5px;'
        'color:var(--hud-muted);margin-top:10px;line-height:1.5">'
        'HONEST STATUS &mdash; episodic <b style="color:var(--hud-success)">history '
        'persists</b> in SQLite across restarts; '
        '<b style="color:var(--hud-memory)">the local vector layer rebuilds</b> each session '
        '(mem0/Qdrant <code>:memory:</code>, not rehydrated from SQLite yet).</p>',
        unsafe_allow_html=True,
    )
