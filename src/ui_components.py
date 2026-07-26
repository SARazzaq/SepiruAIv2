"""
APEX UI Components — Premium HTML component library.
Renders high-fidelity UI elements into Streamlit via st.markdown.
"""
import streamlit as st
from typing import Optional, Literal


# ── Metric Card ───────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, icon: str = "",
                delta: str = "", delta_dir: Literal["up","down",""] = "") -> str:
    delta_html = ""
    if delta:
        arrow = "↑" if delta_dir == "up" else ("↓" if delta_dir == "down" else "")
        delta_html = f'<div class="card-delta {delta_dir}">{arrow} {delta}</div>'
    icon_html = f'<span class="card-icon">{icon}</span>' if icon else ""
    return f"""
    <div class="metric-card">
        {icon_html}
        <h3>{label}</h3>
        <h2>{value}</h2>
        {delta_html}
    </div>"""


def render_metric_cards(items: list[tuple], cols: int = 4):
    """items = list of (label, value) or (label, value, icon) tuples"""
    columns = st.columns(cols)
    for i, item in enumerate(items):
        label = item[0]; value = item[1]
        icon  = item[2] if len(item) > 2 else ""
        with columns[i % cols]:
            st.markdown(metric_card(label, value, icon), unsafe_allow_html=True)


# ── Glass Card ────────────────────────────────────────────────────────────────
def glass_card(content: str, padding: str = "1.8rem") -> str:
    return f'<div class="glass-card" style="padding:{padding}">{content}</div>'


# ── Section Label ─────────────────────────────────────────────────────────────
def section_label(text: str) -> str:
    return f'<div class="section-label"><span>{text}</span></div>'


def render_section(text: str):
    st.markdown(section_label(text), unsafe_allow_html=True)


# ── Badge ─────────────────────────────────────────────────────────────────────
def badge(text: str, variant: Literal["gold","green","indigo","rose"] = "gold",
          dot: bool = False) -> str:
    dot_html = '<span class="badge-dot"></span>' if dot else ""
    return f'<span class="badge badge-{variant}">{dot_html}{text}</span>'


# ── Status Pill ───────────────────────────────────────────────────────────────
def status_pill(text: str, online: bool = True) -> str:
    dot_cls = "live-dot" if online else "offline-dot"
    return f'<div class="status-pill"><span class="{dot_cls}"></span>{text}</div>'


# ── Insight Box ───────────────────────────────────────────────────────────────
def insight_box(content: str) -> str:
    return f'<div class="insight-box">{content}</div>'


def render_insight(content: str):
    st.markdown(insight_box(content), unsafe_allow_html=True)


# ── Stat Row ──────────────────────────────────────────────────────────────────
def stat_row(label: str, value: str) -> str:
    return f"""
    <div class="stat-row">
        <span class="label">{label}</span>
        <span class="value">{value}</span>
    </div>"""


def stat_table(rows: list[tuple], title: str = "") -> str:
    title_html = f'<div style="font-family:Syne,sans-serif;font-size:.65rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--text-dim);margin-bottom:.8rem;">{title}</div>' if title else ""
    rows_html = "".join(stat_row(k, v) for k, v in rows)
    return f'<div class="glass-card">{title_html}{rows_html}</div>'


def render_stat_table(rows: list[tuple], title: str = ""):
    st.markdown(stat_table(rows, title), unsafe_allow_html=True)


# ── Progress Ring ─────────────────────────────────────────────────────────────
def progress_ring(value: float, max_val: float = 100,
                  label: str = "", size: int = 80) -> str:
    pct = min(value / max_val, 1.0)
    circumference = 283
    offset = circumference * (1 - pct)
    display = f"{value:.0f}%" if max_val == 100 else f"{value:.1f}"
    r = size // 2 - 6
    cx = cy = size // 2
    return f"""
    <div class="progress-ring-wrap">
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
            <defs>
                <linearGradient id="goldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#c9a84c"/>
                    <stop offset="100%" style="stop-color:#e8c96a"/>
                </linearGradient>
            </defs>
            <circle class="ring-bg" cx="{cx}" cy="{cy}" r="{r}"/>
            <circle class="ring-fill" cx="{cx}" cy="{cy}" r="{r}"
                    stroke-dashoffset="{offset:.1f}"
                    style="stroke-dasharray:{circumference};stroke-dashoffset:{offset:.1f}"/>
            <text x="{cx}" y="{cy+1}" text-anchor="middle" dominant-baseline="middle"
                  style="fill:var(--gold);font-family:Syne,sans-serif;font-size:{size//6}px;font-weight:700;transform:rotate(90deg);transform-origin:{cx}px {cy}px;">
                {display}
            </text>
        </svg>
        <span class="ring-label">{label}</span>
    </div>"""


# ── Horizontal Rule ───────────────────────────────────────────────────────────
def apex_hr() -> str:
    return '<hr class="apex-hr">'


def render_hr():
    st.markdown(apex_hr(), unsafe_allow_html=True)


# ── Page Header (for sub-pages) ───────────────────────────────────────────────
def page_header(title: str, accent: str, subtitle: str) -> str:
    return f"""
    <div class="app-header">
        <h1>{title} <span>{accent}</span></h1>
        <p>{subtitle}</p>
    </div>"""


def render_page_header(title: str, accent: str, subtitle: str):
    st.markdown(page_header(title, accent, subtitle), unsafe_allow_html=True)


# ── Upload CTA ────────────────────────────────────────────────────────────────
def upload_cta(title: str = "Drop your data here",
               subtitle: str = "Supports .csv and .xlsx",
               features: list[str] | None = None) -> str:
    chips = ""
    if features:
        chip_html = "".join(f'<span class="feature-chip">{f}</span>' for f in features)
        chips = f'<div class="feature-list">{chip_html}</div>'
    return f"""
    <div class="upload-cta">
        <h2>{title}</h2>
        <p>{subtitle}</p>
        {chips}
    </div>"""


# ── Typing Indicator ──────────────────────────────────────────────────────────
def typing_indicator() -> str:
    return """
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0 12px;">
        <span style="color:var(--text-dim);font-size:.65rem;font-family:'DM Sans',sans-serif;
                     letter-spacing:2.5px;text-transform:uppercase;font-weight:400;">
            Processing
        </span>
        <div style="display:flex;gap:4px;align-items:center;">
            <div style="width:5px;height:5px;border-radius:50%;background:var(--gold);
                        animation:_td 1.2s ease-in-out infinite 0s;"></div>
            <div style="width:5px;height:5px;border-radius:50%;background:var(--gold);
                        animation:_td 1.2s ease-in-out infinite 0.2s;"></div>
            <div style="width:5px;height:5px;border-radius:50%;background:var(--gold);
                        animation:_td 1.2s ease-in-out infinite 0.4s;"></div>
        </div>
    </div>
    <style>
    @keyframes _td{0%,100%{opacity:1;transform:scale(1) translateY(0)}
                   50%{opacity:.12;transform:scale(.5) translateY(-3px)}}
    </style>"""


# ── Toast Notification ────────────────────────────────────────────────────────
def toast(title: str, message: str, icon: str = "✦") -> str:
    return f"""
    <div class="apex-toast">
        <span class="toast-icon">{icon}</span>
        <div>
            <div class="toast-title">{title}</div>
            <div class="toast-msg">{message}</div>
        </div>
    </div>"""


# ── CSS + Motion loader ───────────────────────────────────────────────────────
def load_all_styles(base_path: str = "assets"):
    """Load all CSS + inject cursor and animations on every page."""
    import streamlit.components.v1 as components
    from pathlib import Path

    # Resolve absolute path — works from any working directory
    # Try relative first, then look relative to this file's location
    base = Path(base_path)
    if not base.is_absolute():
        # Try from cwd
        if not base.exists():
            # Fall back to path relative to this src/ file → ../assets
            base = Path(__file__).parent.parent / base_path

    files = ["style.css", "components.css", "animations.css"]
    combined = ""
    for f in files:
        fpath = base / f
        try:
            combined += fpath.read_text(encoding="utf-8") + "\n"
        except FileNotFoundError:
            pass
    st.markdown(f"<style>{combined}</style>", unsafe_allow_html=True)

    from src.animations import apex_motion_engine
    components.html(apex_motion_engine(), height=0, scrolling=False)
