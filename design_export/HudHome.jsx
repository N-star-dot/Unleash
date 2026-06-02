/**
 * Unleash — HUD HOME (React)
 * Generated from untitled.pen frame `QyOvA`.
 *
 * Usage:
 *   import "./tokens.css"; import "./hud.css";
 *   import HudHome from "./HudHome";
 *   <HudHome state={liveState} />
 *
 * All copy/values are props so the UI can be wired to the live backend
 * (see UI_DESIGN_PRD.md §10 Backend hooks). Defaults reproduce the mockup.
 *
 * Icons: lucide-react.  npm i lucide-react
 */
import {
  BatteryFull, Radio, Cpu, Target, HeartPulse, TriangleAlert,
  ScanEye, Volume2, House, Activity, Map as MapIcon, User,
} from "lucide-react";

const Corners = () => (
  <>
    <span className="corner tl-h" /><span className="corner tl-v" />
    <span className="corner tr-h" /><span className="corner tr-v" />
    <span className="corner bl-h" /><span className="corner bl-v" />
    <span className="corner br-h" /><span className="corner br-v" />
  </>
);

function StatCard({ icon: Icon, badge, value, label, variant }) {
  return (
    <div className={`stat-card ${variant ? "is-" + variant : ""}`}>
      <div className="top">
        <span className="ib"><Icon size={14} /></span>
        <span className="cb">{badge}</span>
      </div>
      <div className="num">{value}</div>
      <div className="lbl">{label}</div>
      <Corners />
    </div>
  );
}

function Chip({ text, variant }) {
  return (
    <span className={`chip ${variant ? "is-" + variant : ""}`}>
      <span className="d" />{text}
    </span>
  );
}

const TABS = [
  { id: "home", label: "HOME", icon: House },
  { id: "data", label: "DATA", icon: Activity },
  { id: "map",  label: "MAP",  icon: MapIcon },
  { id: "unit", label: "UNIT", icon: User },
];

export default function HudHome({
  active = "home",
  onTab = () => {},
  stats = [
    { icon: Cpu,           badge: "OK",   value: "6",  label: "AGENTS ONLINE" },
    { icon: Target,        badge: "HI",   value: "91", label: "CONFIDENCE" },
    { icon: HeartPulse,    badge: "BPM",  value: "72", label: "HEART RATE",   variant: "success" },
    { icon: TriangleAlert, badge: "NEAR", value: "2",  label: "HAZARDS NEAR", variant: "warning" },
  ],
  detections = [
    { text: "RED LIGHT", variant: "danger" },
    { text: "CROSSWALK" },
    { text: "CAR x2", variant: "warning" },
    { text: "CURB", variant: "success" },
  ],
  spoken = "RED LIGHT. TWO CARS ON YOUR RIGHT. WAIT.",
  tone = "URGENT",
  frameInfo = "FRAME 4210 // 30FPS",
}) {
  return (
    <div className="hud-screen">
      {/* StatusBar */}
      <div className="hud-statusbar">
        <span className="time">09:41</span>
        <span className="sbr"><span className="net">5G</span><BatteryFull size={16} color="var(--hud-cyan)" /></span>
      </div>

      {/* Header */}
      <div className="hud-header">
        <div className="hud-reactor">
          <div className="ring1" /><div className="ring2" />
          <div className="core"><Radio size={8} /></div>
        </div>
        <div className="hud-htxt">
          <div className="hud-title">COMMAND</div>
          <div className="hud-sub">SERVICE DOG // ONLINE</div>
        </div>
        <span className="pill pill--live"><span className="dot" />LIVE</span>
      </div>

      {/* Body */}
      <div className="hud-body">
        <div className="hud-sep"><span>SENSORS // ACTIVE</span></div>

        <div className="stat-grid">
          <div className="stat-row">{stats.slice(0, 2).map((s, i) => <StatCard key={i} {...s} />)}</div>
          <div className="stat-row">{stats.slice(2, 4).map((s, i) => <StatCard key={i} {...s} />)}</div>
        </div>

        {/* Perception */}
        <div className="panel">
          <div className="panel-head">
            <span className="dot" />
            <div className="pht"><span className="t">PERCEPTION</span><span className="s">YOLOV8 // CONTINUITY CAM</span></div>
            <span className="pill pill--danger"><span className="dot" />REC</span>
          </div>
          <div className="panel-body">
            <div className="cam">
              <span className="det-tag" style={{ left: 118, top: 6 }}>RED LIGHT 0.94</span>
              <span className="frame-tag"><ScanEye size={10} color="var(--hud-cyan)" />{frameInfo}</span>
            </div>
            <div className="chips">{detections.map((d, i) => <Chip key={i} {...d} />)}</div>
          </div>
        </div>

        {/* Voice */}
        <div className="voice">
          <div className="voice-head">
            <span className="vhl"><Volume2 size={14} />VOICE OUTPUT</span>
            <span className="pill pill--danger">{tone}</span>
          </div>
          <div className="voice-body">
            <p className="voice-spoken">&ldquo;{spoken}&rdquo;</p>
          </div>
        </div>
      </div>

      {/* Tab bar */}
      <nav className="hud-tabbar">
        {TABS.map(({ id, label, icon: Icon }) => (
          <a key={id} className={`hud-tab ${active === id ? "is-active" : ""}`} onClick={() => onTab(id)}>
            <span className="ul" /><Icon size={21} /><span className="lbl">{label}</span>
          </a>
        ))}
      </nav>
    </div>
  );
}
