import streamlit as st

st.set_page_config(page_title="Dynamic Theme Customizer", layout="centered")
st.title("🎨 Dynamic Theme & Accessibility")
st.markdown("Customize your workspace appearance and enable accessibility presets.")

st.subheader("Theme Presets")
theme = st.select_slider(
    "Select Global Theme",
    options=["Light", "Dark", "High Contrast (Vision Impaired)", "Color-Blind Safe"]
)

st.subheader("Accessibility Controls")
st.toggle("Enable Screen Reader Optimizations", value=True)
st.toggle("Reduce Motion / Animations", value=False)
st.slider("UI Scaling (Text & Elements)", 75, 150, 100, format="%d%%")

st.divider()
st.subheader("Preview")
if theme == "Dark":
    st.info("Currently previewing: Dark Theme")
elif theme == "High Contrast (Vision Impaired)":
    st.warning("Currently previewing: High Contrast")
else:
    st.success(f"Currently previewing: {theme}")

st.button("Apply Changes Globally", type="primary", use_container_width=True)
