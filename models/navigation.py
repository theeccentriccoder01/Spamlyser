"""Top navigation bar for page routing."""

import streamlit as st

PAGES = {
    "home": "🏠 Home",
    "analyzer": "🔍 SMS Analyzer",
    "about": "ℹ️ About",
    "features": "⚡ Features",
    "analytics": "📊 Analytics",
    "trends": "📈 Trend Analytics",
    "models": "🤖 Models",
    "feedback": "💬 Feedback",
    "help": "❓ Help",
    "contact": "📞 Contact",
    "docs": "📚 Docs",
    "api": "🔌 API",
    "what_if": "🧪 What-If",
    "settings": "⚙️ Settings",
}


def top_navigation_bar(navigate_to):
    """Render a horizontal navigation bar at the top of the page."""
    current = st.session_state.get("current_page", "home")
    cols = st.columns(len(PAGES))
    for col, (page_key, page_label) in zip(cols, PAGES.items(), strict=False):
        with col:
            is_active = page_key == current
            if is_active:
                st.markdown(
                    f'<div style="text-align:center;padding:6px 0;'
                    f"background:#00d4aa20;border-radius:8px;"
                    f'border:1px solid #00d4aa;">'
                    f'<span style="color:#00d4aa;font-weight:600;font-size:0.85rem;">'
                    f"{page_label}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                if st.button(
                    page_label,
                    key=f"nav_top_{page_key}",
                    use_container_width=True,
                    help=f"Go to {page_label}",
                ):
                    navigate_to(page_key)
