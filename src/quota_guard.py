"""
Quota Guard — tracks daily API usage in session state.
Uses st.session_state (no file I/O — works on Streamlit Cloud).
Groq free tier: 14,400 req/day.
"""

from datetime import datetime, timezone, timedelta


DAILY_LIMIT = 14000


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _get_state() -> dict:
    try:
        import streamlit as st
        if "_quota" not in st.session_state or st.session_state["_quota"].get("date") != _today_utc():
            st.session_state["_quota"] = {"date": _today_utc(), "count": 0}
        return st.session_state["_quota"]
    except Exception:
        return {"date": _today_utc(), "count": 0}


def get_usage() -> dict:
    d = _get_state()
    remaining = max(0, DAILY_LIMIT - d["count"])
    return {
        "date":      d["date"],
        "count":     d["count"],
        "remaining": remaining,
        "limit":     DAILY_LIMIT,
        "pct_used":  round(d["count"] / DAILY_LIMIT * 100, 1),
        "exhausted": remaining == 0,
    }


def increment(n: int = 1):
    try:
        import streamlit as st
        d = _get_state()
        d["count"] = d.get("count", 0) + n
        st.session_state["_quota"] = d
    except Exception:
        pass


def can_proceed() -> bool:
    return get_usage()["remaining"] > 0


def reset_time_utc() -> str:
    now = datetime.now(timezone.utc)
    next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    delta = next_midnight - now
    h, rem = divmod(int(delta.total_seconds()), 3600)
    m = rem // 60
    return f"{h}h {m}m"
