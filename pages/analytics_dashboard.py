"""Advanced analytics dashboard for Spamlyser Pro.

Provides interactive visualizations, real-time performance metrics,
threat trend analysis, and model comparison charts using Plotly.
"""

import collections
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _load_analysis_history() -> list[dict[str, Any]]:
    """Load analysis history from all available session state sources."""
    history: list[dict[str, Any]] = []
    history.extend(st.session_state.get("classification_history", []))
    history.extend(st.session_state.get("ensemble_history", []))
    return history


def _compute_kpis(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute key performance indicators from analysis history."""
    if not history:
        return {
            "total_analyses": 0,
            "spam_count": 0,
            "ham_count": 0,
            "avg_confidence": 0.0,
            "spam_rate": 0.0,
            "unique_models": [],
        }
    df = pd.DataFrame(history)
    total = len(df)
    spam_count = int((df["prediction"] == "SPAM").sum())
    ham_count = total - spam_count
    avg_confidence = float(df["confidence"].mean())
    unique_models = list(df["model"].unique()) if "model" in df.columns else []
    return {
        "total_analyses": total,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "avg_confidence": avg_confidence,
        "spam_rate": spam_count / total * 100 if total > 0 else 0.0,
        "unique_models": unique_models,
    }


def _filter_history_by_date(history: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    """Filter history to keep only records from the last `days` days."""
    if days <= 0:
        return history
    cutoff = datetime.now() - timedelta(days=days)
    filtered = []
    for h in history:
        ts = h.get("timestamp")
        # Ensure ts is a datetime object
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = pd.to_datetime(ts)
        if isinstance(ts, datetime) or isinstance(ts, pd.Timestamp):
            if ts >= cutoff:
                filtered.append(h)
        else:
            # If timestamp is missing or malformed, we keep it just in case
            filtered.append(h)
    return filtered


def _extract_keywords(history: list[dict[str, Any]]) -> dict[str, int]:
    """Extract and count word frequencies from SPAM messages."""
    text = ""
    for h in history:
        if h.get("prediction") == "SPAM":
            msg = h.get("preprocessed") or h.get("message", "")
            text += " " + str(msg).lower()
    
    words = re.findall(r'\b[a-z]{3,}\b', text)
    stopwords = {
        "the", "and", "you", "for", "that", "this", "with", "from", "your", "have", 
        "are", "not", "will", "all", "can", "out", "our", "has", "was", "just", 
        "get", "how", "what", "who", "when", "why", "but", "they", "them"
    }
    words = [w for w in words if w not in stopwords]
    return dict(collections.Counter(words).most_common(50))


def _build_top_keywords_chart(keyword_counts: dict[str, int]) -> go.Figure:
    """Build a horizontal bar chart of the top 10 SPAM keywords."""
    if not keyword_counts:
        return go.Figure()
    top_10 = dict(list(keyword_counts.items())[:10])
    df = pd.DataFrame(list(top_10.items()), columns=["Keyword", "Frequency"])
    df = df.sort_values("Frequency", ascending=True)
    fig = px.bar(
        df,
        x="Frequency",
        y="Keyword",
        orientation="h",
        title="Top 10 SPAM Keywords",
        color="Frequency",
        color_continuous_scale=px.colors.sequential.Reds,
    )
    fig.update_layout(height=400)
    return fig


def _build_keyword_treemap(keyword_counts: dict[str, int]) -> go.Figure:
    """Build a treemap visualization of SPAM keywords."""
    if not keyword_counts:
        return go.Figure()
    df = pd.DataFrame(list(keyword_counts.items()), columns=["Keyword", "Frequency"])
    df["Root"] = "SPAM Keywords"
    fig = px.treemap(
        df,
        path=["Root", "Keyword"],
        values="Frequency",
        title="SPAM Keyword Frequencies",
        color="Frequency",
        color_continuous_scale=px.colors.sequential.Teal,
    )
    fig.update_layout(height=400)
    return fig


def _build_timeline_chart(history: list[dict[str, Any]]) -> go.Figure:
    """Build a time-series chart showing analysis volume and spam rate."""
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["rolling_spam_rate"] = (df["prediction"] == "SPAM").rolling(
        20, min_periods=1
    ).mean() * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["rolling_spam_rate"],
            mode="lines",
            name="Spam Rate (20-msg rolling)",
            line=dict(color="#ff6b6b", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["confidence"],
            mode="markers",
            name="Confidence",
            marker=dict(color="#4ecdc4", size=4, opacity=0.5),
        )
    )
    fig.update_layout(
        title="Analysis Timeline",
        xaxis_title="Time",
        yaxis_title="Percentage / Confidence",
        height=400,
        hovermode="x unified",
    )
    return fig


def _build_threat_distribution(history: list[dict[str, Any]]) -> go.Figure:
    """Build a sunburst chart of threat type distribution."""
    rows = []
    for h in history:
        tt = h.get("threat_type") or h.get("threat_type", "Unknown")
        if tt and tt != "None":
            rows.append({"threat_type": tt, "count": 1})
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    summary = df.groupby("threat_type").sum().reset_index()
    fig = px.pie(
        summary,
        values="count",
        names="threat_type",
        title="Threat Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def _build_model_comparison(history: list[dict[str, Any]]) -> go.Figure:
    """Build a grouped bar chart comparing model accuracy and confidence."""
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    if "model" not in df.columns:
        return go.Figure()
    stats = (
        df.groupby("model")
        .agg(avg_confidence=("confidence", "mean"), count=("confidence", "count"))
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=stats["model"],
            y=stats["avg_confidence"],
            name="Avg Confidence",
            marker_color="#4ecdc4",
        )
    )
    fig.add_trace(
        go.Bar(
            x=stats["model"],
            y=stats["count"] / stats["count"].max() * 100,
            name="Usage Rate (%)",
            marker_color="#667eea",
        )
    )
    fig.update_layout(
        title="Model Performance Comparison",
        barmode="group",
        height=400,
        yaxis_title="Score / Percentage",
    )
    return fig


def render_dashboard():
    """Main render function for the analytics dashboard page."""
    st.markdown(
        """
    <div style="text-align:center;padding:20px 0;background:linear-gradient(90deg,#1a1a1a,#2d2d2d);
               border-radius:15px;margin-bottom:30px;border:1px solid #404040;">
        <h1 style="color:#00d4aa;font-size:3rem;margin:0;text-shadow:0 0 20px rgba(0,212,170,0.3);">
            📊 Analytics Dashboard
        </h1>
        <p style="color:#d1d1d1;margin:10px 0 0;font-size:1.2rem;">
            Real-time performance metrics and threat intelligence
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Filter by date range
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Dashboard Metrics")
    with col2:
        time_filter = st.selectbox(
            "Time Range",
            ["All Time", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
            index=0
        )
    
    days_map = {"All Time": 0, "Last 24 Hours": 1, "Last 7 Days": 7, "Last 30 Days": 30}
    days = days_map[time_filter]

    raw_history = _load_analysis_history()
    history = _filter_history_by_date(raw_history, days)
    kpis = _compute_kpis(history)

    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        st.metric("Total Analyses", f"{kpis['total_analyses']:,}")
    with kpi_cols[1]:
        st.metric("Spam Detected", f"{kpis['spam_count']:,}")
    with kpi_cols[2]:
        st.metric("Spam Rate", f"{kpis['spam_rate']:.1f}%")
    with kpi_cols[3]:
        st.metric("Avg Confidence", f"{kpis['avg_confidence']:.1%}")
    with kpi_cols[4]:
        st.metric("Models Used", f"{len(kpis['unique_models'])}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Timeline", "🎯 Threat Distribution", "☁️ Keywords", "🤖 Model Comparison", "⚙️ Export"]
    )

    with tab1:
        fig = _build_timeline_chart(history)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No analysis history available yet. Run some analyses first!")

    with tab2:
        fig = _build_threat_distribution(history)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No threat data available yet.")

    with tab3:
        kw_counts = _extract_keywords(history)
        if kw_counts:
            col_a, col_b = st.columns(2)
            with col_a:
                fig_bar = _build_top_keywords_chart(kw_counts)
                st.plotly_chart(fig_bar, use_container_width=True)
            with col_b:
                fig_tree = _build_keyword_treemap(kw_counts)
                st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info("No SPAM keyword data available for the selected timeframe.")

    with tab4:
        fig = _build_model_comparison(history)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model data available yet.")

    with tab5:
        st.markdown("### 📥 Export Analytics Data")
        export_json = json.dumps(
            {
                "kpis": kpis,
                "history": [
                    {k: str(v) if isinstance(v, datetime) else v for k, v in h.items()}
                    for h in history
                ],
            },
            indent=2,
            default=str,
        )
        st.download_button(
            "📥 Download JSON Report",
            data=export_json,
            file_name=f"spamlyser_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

if __name__ == "__main__":
    render_dashboard()
