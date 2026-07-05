"""Accessibility helpers for Spamlyser Pro — ARIA labels, focus management,
skip-to-content links, and contrast-safe color utilities.
"""

import re
from typing import Any

import streamlit as st


def skip_to_content_link() -> None:
    """Render a hidden "Skip to content" link for keyboard users."""
    st.markdown(
        """
    <a href="#main-content" style="
        position: absolute;
        left: -9999px;
        top: 0;
        z-index: 9999;
        padding: 8px 16px;
        background: #00d4aa;
        color: #000;
        font-weight: 600;
        border-radius: 0 0 4px 0;
    " onfocus="this.style.left='0'" onblur="this.style.left='-9999px'">
        Skip to main content
    </a>
    <div id="main-content"></div>
    """,
        unsafe_allow_html=True,
    )


def aria_label(element_id: str, label: str) -> str:
    """Return an ``aria-label`` attribute for the given *element_id*."""
    return f'aria-label="{label}" id="{element_id}"'


def focus_ring_style() -> str:
    """Return a CSS block that adds visible focus indicators for keyboard nav."""
    return """
<style>
*:focus-visible {
    outline: 3px solid #00d4aa !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 4px rgba(0, 212, 170, 0.3) !important;
}
.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible {
    outline: 3px solid #00d4aa !important;
    outline-offset: 2px;
}
/* Ensure selectable text areas have sufficient contrast */
.stTextArea textarea,
.stTextInput input {
    color: #e0e0e0 !important;
    background: #2d2d2d !important;
}
@media (prefers-contrast: high) {
    * {
        outline-width: 2px !important;
    }
    .stButton > button {
        border: 2px solid currentColor !important;
    }
}
</style>"""


def contrast_safe_color(hex_color: str, against: str = "#1a1a1a") -> str:
    """Return *hex_color* if it passes WCAG AA contrast against *against*,
    otherwise return an adjusted color.

    Simple luminance-based heuristic — does NOT replace a proper contrast
    checker but catches the worst violations.
    """

    def _lum(hex_c: str) -> float:
        hex_c = hex_c.lstrip("#")
        if len(hex_c) != 6:
            return 0.5
        r, g, b = (int(hex_c[i : i + 2], 16) / 255 for i in (0, 2, 4))
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    l1 = _lum(hex_color)
    l2 = _lum(against)
    ratio = (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)
    return hex_color if ratio >= 4.5 else "#00d4aa"  # fallback to accessible accent


def accessible_button(
    label: str,
    key: str,
    on_click: Any = None,
    **kwargs: Any,
) -> bool:
    """Render a Streamlit button with an ARIA label and visible focus ring."""
    return st.button(
        label,
        key=key,
        on_click=on_click,
        kwargs=kwargs,
    )


def inject_accessibility() -> None:
    """Call once at app startup to inject all accessibility enhancements."""
    skip_to_content_link()
    st.markdown(focus_ring_style(), unsafe_allow_html=True)
