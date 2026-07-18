import pages.theme_customizer
"""Theme manager for Spamlyser Pro — persistent dark/light mode with session state."""

import json
import os
from pathlib import Path
from typing import Literal

import streamlit as st

Theme = Literal["light", "dark", "auto"]

_THEME_FILE = Path(__file__).resolve().parent.parent / "data" / "theme_pref.json"


def _persist(theme: Theme) -> None:
    _THEME_FILE.parent.mkdir(parents=True, exist_ok=True)
    _THEME_FILE.write_text(json.dumps({"theme": theme}), encoding="utf-8")


def _load_persisted() -> Theme:
    try:
        if _THEME_FILE.exists():
            data = json.loads(_THEME_FILE.read_text(encoding="utf-8"))
            return data.get("theme", "auto")
    except (json.JSONDecodeError, OSError):
        pass
    return "auto"


def init_theme() -> None:
    """Ensure ``st.session_state.theme`` is populated from persistence."""
    if "theme" not in st.session_state:
        st.session_state.theme = _load_persisted()


def get_active_theme() -> Theme:
    """Return the effective theme (resolving ``auto`` to light/dark)."""
    pref = st.session_state.get("theme", "auto")
    if pref != "auto":
        return pref
    return "dark"


def set_theme(theme: Theme) -> None:
    """Update session state and persist the choice."""
    st.session_state.theme = theme
    _persist(theme)


def theme_css_variables() -> str:
    """Return a ``<style>`` block with theme-aware CSS custom properties."""
    theme = get_active_theme()
    if theme == "dark":
        return """
<style>
:root {
  --bg-primary: #1a1a1a;
  --bg-secondary: #2d2d2d;
  --text-primary: #e0e0e0;
  --text-secondary: #a0a0a0;
  --card-bg: #2d2d2d;
  --card-border: #404040;
  --input-bg: #333333;
  --input-border: #555555;
  --focus-ring: #00d4aa;
  --accent: #00d4aa;
  --error: #ff6b6b;
  --success: #51cf66;
  --warning: #fcc419;
}
</style>"""
    return """
<style>
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f5f7fa;
  --text-primary: #333333;
  --text-secondary: #666666;
  --card-bg: #ffffff;
  --card-border: #e0e0e0;
  --input-bg: #ffffff;
  --input-border: #cccccc;
  --focus-ring: #1e40af;
  --accent: #1e40af;
  --error: #e03131;
  --success: #2f9e44;
  --warning: #e67700;
}
</style>"""


def inject_theme_toggle() -> None:
    """Render a small theme toggle in the sidebar (call once per page)."""
    init_theme()
    current = st.session_state.theme
    label = {"light": "☀️ Light", "dark": "🌙 Dark", "auto": "🔄 Auto"}
    chosen = st.sidebar.selectbox(
        "Theme",
        options=["light", "dark", "auto"],
        format_func=lambda x: label[x],
        index=["light", "dark", "auto"].index(current),
        key="theme_selector",
    )
    if chosen != current:
        set_theme(chosen)
        st.rerun()
