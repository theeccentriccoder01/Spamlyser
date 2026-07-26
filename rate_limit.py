import time

import streamlit as st


def check_rate_limit(action_key: str, max_requests: int, window_seconds: int) -> bool:
    """
    Checks if the user has exceeded the rate limit for a specific action.
    Returns True if allowed, False if rate limited.
    """
    if "rate_limits" not in st.session_state:
        st.session_state.rate_limits = {}
        
    now = time.time()
    history = st.session_state.rate_limits.get(action_key, [])
    
    # Remove timestamps older than the window
    history = [t for t in history if now - t < window_seconds]
    
    if len(history) >= max_requests:
        st.session_state.rate_limits[action_key] = history
        return False
        
    history.append(now)
    st.session_state.rate_limits[action_key] = history
    return True

def check_debounce(action_key: str, wait_seconds: int = 2) -> bool:
    """
    Checks if enough time has passed since the last action to avoid double-clicks.
    Returns True if allowed, False if debounced.
    """
    debounce_key = f"debounce_{action_key}"
    now = time.time()
    last_click = st.session_state.get(debounce_key, 0)
    
    if now - last_click < wait_seconds:
        return False
        
    st.session_state[debounce_key] = now
    return True
