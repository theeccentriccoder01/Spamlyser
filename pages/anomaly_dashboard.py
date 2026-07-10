"""Anomaly detection dashboard — surfaces outlier messages and patterns."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_anomaly_dashboard() -> None:
    """Render the anomaly detection dashboard."""
    st.markdown("## 🔍 Anomaly Detection & Outlier Analysis")
    st.markdown("Identifies unusual messages, model disagreements, and confidence outliers.")

    history = st.session_state.get("classification_history", [])
    if not history:
        st.info("No analysis history available. Run some SMS analyses first.")
        return

    from models.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector()
    df = detector.score_messages(history)

    # Overview metrics
    total = len(df)
    anomalies = df["is_anomaly"].sum()
    cols = st.columns(5)
    cols[0].metric("Total Messages", total)
    cols[1].metric("Anomalies Detected", int(anomalies), delta=f"{anomalies/total*100:.1f}%" if total else "0%")
    cols[2].metric("Avg Content Score", f"{df['content_score'].mean():.2f}")
    cols[3].metric("Avg Disagreement", f"{df['disagreement_score'].mean():.1%}")
    cols[4].metric("Confidence Outliers", int(df["confidence_anomaly"].sum()))

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Anomaly Scatter", "📋 Top Outliers", "🧩 Breakdown", "📈 Trends",
    ])

    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = px.scatter(
                df,
                x="confidence_zscore" if "confidence_zscore" in df.columns else df.index,
                y="composite_anomaly",
                color="is_anomaly",
                hover_data=["message", "disagreement_score"],
                title="Anomaly Score Distribution",
                color_continuous_scale=["#44bb44", "#ffa657", "#ff4444"],
                labels={"composite_anomaly": "Anomaly Score", "index": "Message #"},
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
            fig.update_layout(
                height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"),
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### Filter")
            min_score = st.slider("Min Anomaly Score", 0.0, float(df["composite_anomaly"].max()), 0.0)
            show_only = st.checkbox("Show only anomalies")
            df_display = df[df["composite_anomaly"] >= min_score]
            if show_only:
                df_display = df_display[df_display["is_anomaly"] == 1]
            st.metric("Showing", len(df_display))

    with tab2:
        top_n = st.slider("Number of top outliers", 5, 30, 10)
        top_df = detector.top_anomalies(df, top_n)
        for i, (_, row) in enumerate(top_df.iterrows()):
            label = row.get("ensemble_predictions", {}).get("majority_voting", {}).get("label", "?")
            conf = row.get("ensemble_predictions", {}).get("majority_voting", {}).get("confidence", 0.0)
            msg = str(row.get("message", ""))[:100]
            anomaly_type = "critical" if row.get("is_anomaly") else "info"
            explanations = detector.get_outlier_explanations(row)
            color = "#ff4444" if row.get("is_anomaly") else "#58a6ff"
            st.markdown(f"""<div class="anomaly-card {anomaly_type}">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-weight:700;color:white">#{i+1} Anomaly</span>
                    <span style="color:{color};font-weight:600">{label} ({conf:.1%})</span>
                </div>
                <div style="color:#c9d1d9;margin:6px 0;font-size:0.85rem">{msg}</div>
                <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:4px">
                    {''.join(f'<span style="background:{color}20;color:{color};padding:1px 8px;border-radius:10px;font-size:0.7rem">{e}</span>' for e in explanations)}
                </div>
                <div style="font-size:0.7rem;color:#585858;margin-top:4px">
                    Score: {row.get('composite_anomaly', 0):.2f} | Content: {row.get('content_score', 0):.2f} | Disagreement: {row.get('disagreement_score', 0):.1%}
                </div>
            </div>""", unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            anomaly_counts = df["is_anomaly"].value_counts().rename({0: "Normal", 1: "Anomaly"})
            fig1 = px.pie(
                values=anomaly_counts.values, names=anomaly_counts.index,
                title="Anomaly Ratio", color_discrete_sequence=["#44bb44", "#ff4444"],
            )
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9"))
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            # Score distribution
            fig2 = px.histogram(
                df, x="composite_anomaly", nbins=30,
                title="Anomaly Score Distribution",
                color_discrete_sequence=["#00d4aa"],
            )
            fig2.add_vline(x=np.percentile(df["composite_anomaly"], 90), line_dash="dash", line_color="#ff4444")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9"))
            st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        if "timestamp" in df.columns:
            df["ts"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("ts")
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=df["ts"], y=df["composite_anomaly"],
                mode="lines+markers", name="Anomaly Score",
                line=dict(color="#00d4aa", width=2),
                marker=dict(size=6, color=df["is_anomaly"].map({0: "#44bb44", 1: "#ff4444"})),
            ))
            fig3.update_layout(
                title="Anomaly Score Over Time",
                height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"), hovermode="x",
            )
            st.plotly_chart(fig3, use_container_width=True)
