"""Benchmark automation dashboard for Spamlyser Pro.

Provides interactive visualizations of model performance benchmarks,
historical trends, and regression detection.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path


def render_benchmark_dashboard():
    st.markdown("# ⏱️ Model Benchmark Dashboard")
    st.markdown("Track and compare model performance over time.")

    try:
        from models.benchmark_automation import (
            BenchmarkHistory,
            run_automated_benchmark,
        )
    except ImportError:
        st.warning("Benchmark automation module not available.")
        return

    from config import BENCHMARK_AUTO_RUNS, BENCHMARK_HISTORY_PATH

    history = BenchmarkHistory(BENCHMARK_HISTORY_PATH)

    tab1, tab2, tab3 = st.tabs(["Run Benchmark", "History", "Trends"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            sample_size = st.number_input(
                "Sample size", min_value=5, max_value=50, value=10
            )
            warmup = st.number_input(
                "Warmup runs", min_value=0, max_value=5, value=1
            )
            if st.button("▶️ Run Benchmark", type="primary", use_container_width=True):
                with st.spinner("Running benchmark... This may take a moment."):
                    result = run_automated_benchmark(
                        sample_size=sample_size, warmup_runs=warmup
                    )
                if result:
                    st.success("Benchmark completed!")
                    cols = st.columns(4)
                    cols[0].metric(
                        "Avg Latency", f"{result.avg_latency_ms:.1f}ms"
                    )
                    cols[1].metric("Accuracy", f"{result.accuracy:.1%}")
                    cols[2].metric("Precision", f"{result.precision:.1%}")
                    cols[3].metric("Recall", f"{result.recall:.1%}")
                else:
                    st.error("Benchmark failed. Check logs for details.")

        with col2:
            st.info(
                "**Note:** Benchmarks run the ensemble classifier against a sample"
                " of test data and measure latency, accuracy, precision, and recall.\n\n"
                "Results are automatically saved to the benchmark history for trend analysis."
            )

    with tab2:
        records = history.get_all()
        if records:
            df = pd.DataFrame(records)
            st.dataframe(
                df[["timestamp", "avg_latency_ms", "accuracy", "precision", "recall"]]
                .sort_values("timestamp", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "📥 Export History as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"benchmark_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

            if st.button("🗑️ Clear History", type="secondary"):
                history.clear()
                st.rerun()
        else:
            st.info("No benchmark records yet. Run a benchmark to generate data.")

    with tab3:
        records = history.get_all()
        if len(records) >= 2:
            df = pd.DataFrame(records)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            fig_latency = px.line(
                df,
                x="timestamp",
                y="avg_latency_ms",
                title="Latency Trend Over Time",
                markers=True,
                labels={
                    "timestamp": "Date",
                    "avg_latency_ms": "Avg Latency (ms)",
                },
            )
            st.plotly_chart(fig_latency, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig_accuracy = px.line(
                    df,
                    x="timestamp",
                    y="accuracy",
                    title="Accuracy Trend",
                    markers=True,
                )
                st.plotly_chart(fig_accuracy, use_container_width=True)

            with col2:
                recent = df.tail(5)
                recent["model"] = "Ensemble"
                fig_bar = px.bar(
                    recent,
                    x="timestamp",
                    y=["precision", "recall"],
                    title="Recent Precision & Recall",
                    barmode="group",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) >= 2 else None
            if previous is not None:
                latency_change = (
                    (latest["avg_latency_ms"] - previous["avg_latency_ms"])
                    / previous["avg_latency_ms"]
                    * 100
                )
                accuracy_change = latest["accuracy"] - previous["accuracy"]
                delta_color = "inverse" if latency_change > 5 else "normal"
                st.metric(
                    "Latency Change",
                    f"{latency_change:+.1f}%",
                    delta=f"{'⚠️' if abs(latency_change) > 10 else '✅'} vs previous run",
                    delta_color=delta_color,
                )
                st.metric(
                    "Accuracy Change",
                    f"{accuracy_change:+.2%}",
                    delta=f"{'⬆️' if accuracy_change > 0 else '⬇️'} vs previous",
                )
        else:
            st.info(
                "Need at least 2 benchmark records to show trends. "
                "Run the benchmark at least twice."
            )

    st.markdown("---")
    st.caption("Benchmarks run on CPU by default. Performance may vary based on system resources.")


if __name__ == "__main__":
    render_benchmark_dashboard()
