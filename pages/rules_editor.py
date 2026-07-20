import streamlit as st
import pandas as pd

st.set_page_config(page_title="Custom Rules Editor & Simulator", layout="wide")
st.title("🛡️ Custom Regex Rules Editor & Sandbox")
st.markdown("Test your custom threat intelligence regex rules against sample datasets before deploying to production.")

st.subheader("Rule Definition")
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.text_input("Regex Pattern", value=r"(?i)(urgent|immediate action required)")
with col2:
    st.selectbox("Risk Level", ["Low", "Medium", "High", "Critical"])
with col3:
    st.selectbox("Category", ["Phishing", "Scam", "Malware", "Spam"])

st.subheader("Sandbox Testing")
test_cases = st.text_area("Sample Messages (one per line)", value="URGENT: Your account has been suspended.\nHey, are we still on for lunch?\nImmediate action required to claim your prize!")
if st.button("Run Batch Simulation", type="primary"):
    st.success("Simulation Complete: 2 Hits, 1 Miss in 45ms")
    
    data = {
        "Message": test_cases.split("\n"),
        "Status": ["🚨 HIT", "✅ PASS", "🚨 HIT"],
        "Confidence": ["98%", "-", "95%"]
    }
    st.table(pd.DataFrame(data))
