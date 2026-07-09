"""Cross-session trend analytics dashboard — Gantt timeline + KPI comparison."""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from datetime import datetime


def render_trend_analytics() -> None:
    """Render the cross-session trend analytics page."""
    st.markdown("## 📈 Cross-Session Trend Analytics")
    st.markdown("Compare analysis patterns across sessions, detect drift, and track KPIs over time.")

    history = st.session_state.get("classification_history", [])
    if not history:
        st.info("No analysis history available. Run some SMS analyses first.")
        return

    from models.session_analytics import SessionAnalytics
    analytics = SessionAnalytics(gap_minutes=30)
    sessions = analytics.segment_sessions(history)

    if not sessions:
        st.warning("Could not segment sessions from the available history.")
        return

    comparison_df = analytics.compute_comparison(sessions)
    drift_data = analytics.compute_drift(sessions)

    # Summary metrics
    total_sessions = len(sessions)
    total_msgs = sum(s["total"] for s in sessions)
    avg_spam_rate = np.mean([s["spam_rate"] for s in sessions])
    latest_trend = drift_data["deltas"][-1]["direction"] if drift_data["deltas"] else "stable"

    cols = st.columns(5)
    cols[0].metric("Sessions", total_sessions)
    cols[1].metric("Total Messages", total_msgs)
    cols[2].metric("Avg Spam Rate", f"{avg_spam_rate:.1%}")
    cols[3].metric("Drift Events", drift_data["total_drift_sessions"])
    cols[4].metric("Latest Trend", latest_trend.upper())

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Gantt Timeline", "📋 Session Comparison", "🔄 Drift Analysis", "📥 Export",
    ])

    with tab1:
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, s in enumerate(sessions):
            spam_pct = s["spam_rate"]
            bar_color = f"rgb({int(255 * spam_pct)}, {int(255 * (1 - spam_pct))}, 100)"
            fig.add_trace(go.Bar(
                name=f"Session {s['id']}",
                x=[s["duration_seconds"] / 60],
                y=[f"Session {s['id']}"],
                orientation="h",
                marker=dict(color=bar_color),
                text=f"{s['total']} msgs | {spam_pct:.0%} spam",
                textposition="inside",
                hovertemplate=(
                    f"Session {s['id']}<br>"
                    f"Duration: {s['duration_seconds']/60:.1f} min<br>"
                    f"Messages: {s['total']}<br>"
                    f"Spam Rate: {spam_pct:.1%}<br>"
                    f"Avg Conf: {s['avg_confidence']:.1%}<br>"
                    f"Models: {s['model_count']}<br>"
                    f"<extra></extra>"
                ),
                width=0.6,
            ))

        fig.update_layout(
            title="Session Timeline (Width = Duration, Color = Spam Rate)",
            xaxis_title="Duration (minutes)", yaxis_title="",
            height=200 + 40 * total_sessions,
            barmode="stack", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9"),
            showlegend=False, margin=dict(l=100, r=20, t=40, b=20),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📅 Session Details"):
            for s in sessions:
                st.markdown(f"""<div style="background:#161b22;border-radius:8px;padding:12px;margin:4px 0;border-left:3px solid {'#ff4444' if s['spam_rate'] > 0.5 else '#44bb44'}">
                    <b>Session {s['id']}</b> — {s['start'].strftime('%b %d, %I:%M %p') if hasattr(s['start'], 'strftime') else s['start']}
                    <span style="float:right">{s['total']} msgs | {s['spam_rate']:.0%} spam | {s['throughput_per_hour']:.0f}/hr</span>
                    <div style="font-size:0.8rem;color:#8b949e;margin-top:4px">Models: {', '.join(s['models_used'].keys())}</div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        if len(sessions) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                s1_idx = st.selectbox("Session A", range(len(sessions)),
                    format_func=lambda i: f"Session {i} ({sessions[i]['start'].strftime('%b %d %I:%M %p') if hasattr(sessions[i]['start'], 'strftime') else sessions[i]['start']})",
                    key="s1")
            with col2:
                s2_idx = st.selectbox("Session B", range(len(sessions)), index=min(1, len(sessions)-1),
                    format_func=lambda i: f"Session {i} ({sessions[i]['start'].strftime('%b %d %I:%M %p') if hasattr(sessions[i]['start'], 'strftime') else sessions[i]['start']})",
                    key="s2")

            s1, s2 = sessions[s1_idx], sessions[s2_idx]
            metrics = [
                ("Messages", s1["total"], s2["total"], f"{s2['total'] - s1['total']:+d}"),
                ("Spam Rate", f"{s1['spam_rate']:.1%}", f"{s2['spam_rate']:.1%}", f"{s2['spam_rate'] - s1['spam_rate']:+.1%}"),
                ("Avg Confidence", f"{s1['avg_confidence']:.1%}", f"{s2['avg_confidence']:.1%}", f"{s2['avg_confidence'] - s1['avg_confidence']:+.1%}"),
                ("Duration", f"{s1['duration_seconds']/60:.1f}m", f"{s2['duration_seconds']/60:.1f}m", f"{s2['duration_seconds'] - s1['duration_seconds']:+.0f}s"),
                ("Throughput", f"{s1['throughput_per_hour']:.0f}/hr", f"{s2['throughput_per_hour']:.0f}/hr", f"{s2['throughput_per_hour'] - s1['throughput_per_hour']:+.0f}/hr"),
                ("Models Used", s1["model_count"], s2["model_count"], f"{s2['model_count'] - s1['model_count']:+d}"),
            ]
            st.markdown("### 📊 Side-by-Side Comparison")
            st.markdown(f"""<div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:8px;font-size:0.85rem">
                <div style="font-weight:600;color:#8b949e">Metric</div>
                <div style="font-weight:600;color:#58a6ff;text-align:right">Session {s1['id']}</div>
                <div style="font-weight:600;color:#58a6ff;text-align:right">Session {s2['id']}</div>
                <div style="font-weight:600;color:#8b949e;text-align:right">Δ</div>
                {"".join(f'<div style="color:#c9d1d9">{m[0]}</div><div style="text-align:right">{m[1]}</div><div style="text-align:right">{m[2]}</div><div style="text-align:right;font-weight:600;color:{"#44bb44" if "+" in m[3][:1] else "#ff4444" if "-" in m[3][:1] else "#ffa657"}">{m[3]}</div>' for m in metrics)}
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Need at least 2 sessions for comparison.")

    with tab3:
        if drift_data["deltas"]:
            drift_df = pd.DataFrame(drift_data["deltas"])
            fig_drift = go.Figure()
            fig_drift.add_trace(go.Scatter(
                x=list(range(len(drift_df))),
                y=drift_df["spam_rate_delta"],
                mode="lines+markers+text",
                name="Spam Rate Δ",
                line=dict(color="#ffa657", width=2),
                marker=dict(size=10, color=drift_df["direction"].map(
                    {"up": "#ff4444", "down": "#44bb44", "stable": "#ffa657"})),
                text=[f"{d:+.1%}" for d in drift_df["spam_rate_delta"]],
                textposition="top center",
            ))
            fig_drift.add_trace(go.Scatter(
                x=list(range(len(drift_df))),
                y=drift_df["confidence_delta"],
                mode="lines+markers",
                name="Confidence Δ",
                line=dict(color="#00d4aa", width=2, dash="dot"),
                marker=dict(size=8, color="#00d4aa"),
            ))
            if "kl_divergence" in drift_df.columns:
                fig_drift.add_trace(go.Bar(
                    x=list(range(len(drift_df))),
                    y=drift_df["kl_divergence"],
                    name="KL Divergence",
                    marker_color="#58a6ff80",
                    yaxis="y2",
                ))

            fig_drift.update_layout(
                title="Session-to-Session Drift",
                height=400, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9"),
                hovermode="x unified",
                xaxis=dict(title="Session Transition", tickmode="linear"),
                yaxis=dict(title="Delta", tickformat=".0%"),
                yaxis2=dict(title="KL Divergence", overlaying="y", side="right", tickformat=".2f"),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_drift, use_container_width=True)

            st.markdown("### Drift Details")
            for d in drift_data["deltas"]:
                emoji = {"up": "🔴", "down": "🟢", "stable": "🟡"}.get(d["direction"], "⚪")
                sev = d.get("drift_severity", "unknown")
                sev_color = {"high": "#ff4444", "medium": "#ffa657", "low": "#44bb44"}.get(sev, "#8b949e")
                st.markdown(f"""<div style="background:#161b22;border-radius:8px;padding:10px;margin:3px 0">
                    <span style="font-weight:600">{emoji} Session {d['from_session']} → Session {d['to_session']}</span>
                    <span style="margin-left:12px;font-size:0.85rem">
                        Spam Δ: <span style="color:{'#ff4444' if d['spam_rate_delta'] > 0 else '#44bb44'};font-weight:600">{d['spam_rate_delta']:+.1%}</span>
                        | Conf Δ: <span style="font-weight:600">{d['confidence_delta']:+.1%}</span>
                        | Vol Change: <span style="font-weight:600">{d['volume_change']:+d}</span>
                        | KL: <span style="color:{sev_color};font-weight:600">{d.get('kl_divergence', 0):.3f}</span>
                    </span>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No drift detected between sessions (need at least 2).")

    with tab4:
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            "📥 Download Session Data (CSV)",
            csv, f"session_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv", use_container_width=True,
        )
        if drift_data["deltas"]:
            drift_json = pd.DataFrame(drift_data["deltas"]).to_json(orient="records", indent=2)
            st.download_button(
                "📥 Download Drift Data (JSON)",
                drift_json, f"drift_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json", use_container_width=True,
            )
