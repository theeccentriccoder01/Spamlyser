import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="What-If Scenario Simulator", layout="wide")
st.title("🎛️ What-If Analysis Scenario Simulator")
st.markdown("Interactively tweak message features to see how the classifier decision boundary shifts in real-time.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Feature Tweaks")
    st.slider("Keyword Urgency Score", 0.0, 1.0, 0.5)
    st.slider("URL Suspiciousness", 0.0, 1.0, 0.2)
    st.slider("Sender Reputation", 0.0, 100.0, 85.0)
    st.number_input("Special Character Ratio", min_value=0.0, max_value=1.0, value=0.05)
    st.toggle("Contains Actionable Link", value=True)
    st.button("Run Simulation", type="primary", use_container_width=True)

with col2:
    st.subheader("Decision Boundary Visualization")
    st.info("Simulation Result: **HAM (Not Spam)**")
    st.progress(0.25, text="Spam Probability: 25%")
    
    # Mock chart
    chart_data = pd.DataFrame(
        np.random.randn(20, 3) * [0.5, 0.5, 0.5] + [0.25, 0.3, 0.4],
        columns=['Urgency', 'URL Score', 'Risk']
    )
    st.area_chart(chart_data)
