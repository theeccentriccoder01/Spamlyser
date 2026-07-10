"""Real-time streaming visualizer for batch SMS classification."""

import time
from collections import deque
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


class StreamVisualizer:
    """Renders animated classification cards and a live timeline chart."""

    def __init__(self, max_cards: int = 20, window_seconds: int = 60):
        self.max_cards = max_cards
        self.window_seconds = window_seconds
        self.timeline: deque[dict] = deque(maxlen=100)
        self.card_container = st.empty()
        self.chart_container = st.empty()
        self.stats_container = st.empty()

    def render_card(self, result: dict, index: int, total: int) -> None:
        label = result.get("label", "?")
        is_spam = label == "SPAM"
        color = "#ff4444" if is_spam else "#44bb44"
        confidence = result.get("confidence", 0.0)
        threat = result.get("threat", "N/A")
        message = result.get("message", "")[:80]
        model = result.get("method", "Ensemble")

        card_html = f"""<div class="stream-card { 'spam' if is_spam else 'ham' }" style="
            animation: slideIn 0.3s ease-out;
            background: {'linear-gradient(135deg, #ff444410, #1a1a1a)' if is_spam else 'linear-gradient(135deg, #44bb4410, #1a1a1a)'};
            border-left: 4px solid {color};
            border-radius: 8px; padding: 10px 16px; margin: 4px 0;
            display: flex; justify-content: space-between; align-items: center;">
            <div style="flex:1">
                <span style="color:{color};font-weight:700;font-size:0.9rem">{label}</span>
                <span style="color:#8b949e;font-size:0.75rem;margin-left:8px">{model}</span>
                <div style="color:#c9d1d9;font-size:0.8rem;margin-top:2px">{message}</div>
            </div>
            <div style="text-align:right;min-width:120px">
                <div style="font-size:0.85rem;font-weight:600;color:{color}">{confidence:.1%}</div>
                <div style="font-size:0.7rem;color:#8b949e">{threat}</div>
                <div style="font-size:0.65rem;color:#585858">{index}/{total}</div>
            </div>
        </div>"""

        existing = self.card_container.markdown("", unsafe_allow_html=True)
        combined = card_html + (existing if hasattr(self, '_card_html') else "")
        self.card_container.markdown(combined, unsafe_allow_html=True)
        self._card_html = combined

    def render_chart(self, history: list[dict]) -> None:
        if len(history) < 2:
            return
        df = pd.DataFrame(history)
        df["ts"] = pd.to_datetime(df.get("timestamp", datetime.now()))
        df = df.set_index("ts").last(f"{self.window_seconds}s").reset_index()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("Spam Probability (Rolling)", "Throughput"),
            row_heights=[0.6, 0.4],
        )
        fig.add_trace(go.Scatter(
            x=df["ts"], y=df.get("confidence", [0.5]*len(df)),
            mode="lines+markers", name="Confidence",
            line=dict(color="#00d4aa", width=2),
            marker=dict(size=5, color=df.get("label", ["HAM"]*len(df)).map(
                lambda l: "#ff4444" if l == "SPAM" else "#44bb44"
            )),
        ), row=1, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.3, row=1, col=1)

        throughput = df.resample("5s", on="ts").size().reset_index() if len(df) > 1 else df[["ts"]]
        if len(throughput) > 1:
            fig.add_trace(go.Bar(
                x=throughput["ts"], y=throughput.iloc[:,1] if throughput.shape[1] > 1 else [],
                name="Msg/5s", marker_color="#00d4aa80",
            ), row=2, col=1)

        fig.update_layout(
            height=300, margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9", size=10),
            hovermode="x unified",
            showlegend=False,
        )
        fig.update_xaxes(gridcolor="#30363d", row=2, col=1)
        fig.update_yaxes(gridcolor="#30363d", row=1, col=1)
        self.chart_container.plotly_chart(fig, use_container_width=True)

    def render_stats(self, stats: dict) -> None:
        cols = self.stats_container.columns(5)
        cols[0].metric("Processed", stats.get("processed", 0))
        cols[1].metric("Spam", stats.get("spam", 0))
        cols[2].metric("Ham", stats.get("ham", 0))
        cols[3].metric("Avg Conf", f"{stats.get('avg_confidence', 0):.1%}")
        cols[4].metric("Speed", f"{stats.get('speed', 0):.1f}/s")

    def reset(self) -> None:
        self.card_container = st.empty()
        self.chart_container = st.empty()
        self.stats_container = st.empty()
        self.timeline.clear()
        self._card_html = ""
