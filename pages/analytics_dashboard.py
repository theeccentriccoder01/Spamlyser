"""Advanced analytics dashboard for Spamlyser Pro.

Provides interactive visualizations, real-time performance metrics,
threat trend analysis, and model comparison charts using Plotly.
"""

import json
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


def _build_timeline_chart(history: list[dict[str, Any]]) -> go.Figure:
    """Build a time-series chart showing analysis volume and spam rate."""
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["rolling_spam_rate"] = (
        (df["prediction"] == "SPAM").rolling(20, min_periods=1).mean() * 100
    )
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

    history = _load_analysis_history()
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

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Timeline", "🎯 Threat Distribution", "🤖 Model Comparison", "⚙️ Export"]
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
        fig = _build_model_comparison(history)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model data available yet.")

    with tab4:
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
