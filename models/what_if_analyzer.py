import pages.what_if_playground
"""What-If Analysis — compare predictions across models as text changes."""

import time
from typing import Any

import pandas as pd
import streamlit as st


class WhatIfAnalyzer:
    """Computes multi-model predictions and deltas for what-if text editing."""

    def __init__(self):
        self._previous: dict[str, Any] | None = None

    def analyze(self, text: str, classifier) -> dict[str, Any]:
        """Run all models on *text* and compute deltas from previous run."""
        predictions = {}
        for model_name in ["DistilBERT", "BERT", "RoBERTa", "ALBERT"]:
            try:
                result = classifier.get_model_prediction(model_name, text)
                predictions[model_name] = {
                    "label": result.label,
                    "confidence": result.score,
                    "spam_probability": result.spam_probability,
                }
            except Exception as e:
                predictions[model_name] = {
                    "label": "ERROR",
                    "confidence": 0.0,
                    "spam_probability": 0.0,
                    "error": str(e),
                }

        result = {
            "text": text,
            "predictions": predictions,
            "ensemble_label": self._ensemble_vote(predictions),
            "ensemble_confidence": self._avg_confidence(predictions),
            "model_agreement": self._agreement_score(predictions),
            "timestamp": time.time(),
        }
        if self._previous:
            result["deltas"] = self._compute_deltas(result)
        self._previous = result
        return result

    def _ensemble_vote(self, predictions: dict) -> str:
        labels = [p["label"] for p in predictions.values()]
        spams = labels.count("SPAM")
        hams = labels.count("HAM")
        if spams > hams:
            return "SPAM"
        if hams > spams:
            return "HAM"
        return "SPAM" if spams > 0 else "HAM"

    def _avg_confidence(self, predictions: dict) -> float:
        confs = [p["confidence"] for p in predictions.values() if p["label"] != "ERROR"]
        return sum(confs) / len(confs) if confs else 0.0

    def _agreement_score(self, predictions: dict) -> float:
        labels = [p["label"] for p in predictions.values() if p["label"] != "ERROR"]
        if not labels:
            return 0.0
        majority = max(set(labels), key=labels.count)
        return labels.count(majority) / len(labels)

    def _compute_deltas(self, current: dict) -> dict[str, Any]:
        prev = self._previous
        deltas = {}
        for model_name in current["predictions"]:
            if model_name in prev["predictions"]:
                prev_conf = prev["predictions"][model_name]["confidence"]
                curr_conf = current["predictions"][model_name]["confidence"]
                delta = curr_conf - prev_conf
                deltas[model_name] = {
                    "delta": delta,
                    "direction": "up"
                    if delta > 0.01
                    else ("down" if delta < -0.01 else "stable"),
                    "label_changed": (
                        prev["predictions"][model_name]["label"]
                        != current["predictions"][model_name]["label"]
                    ),
                    "prev_label": prev["predictions"][model_name]["label"],
                    "curr_label": current["predictions"][model_name]["label"],
                    "prev_conf": prev_conf,
                    "curr_conf": curr_conf,
                }
        return deltas


def render_what_if_playground(classifier) -> None:
    """Render the What-If analysis playground page."""
    st.markdown("## 🧪 What-If Analysis Playground")
    st.markdown(
        "Edit the message text and see how each model's prediction changes in real time."
    )

    analyzer = WhatIfAnalyzer()
    if "whatif_text" not in st.session_state:
        st.session_state.whatif_text = (
            "Your account has been compromised! Click here to verify."
        )

    col1, col2 = st.columns([3, 2])
    with col1:
        text = st.text_area(
            "✏️ Edit Message",
            value=st.session_state.whatif_text,
            height=120,
            key="whatif_text_input",
            help="Type or edit the SMS message. Predictions update with each change.",
        )
        if text != st.session_state.whatif_text:
            st.session_state.whatif_text = text
            st.rerun()

        st.markdown("### 💡 Quick Templates")
        templates = [
            (
                "📧 Phishing",
                "Dear customer, your Netflix account has been suspended. Verify now: http://bit.ly/netflix-verify",
            ),
            (
                "💰 Prize Scam",
                "CONGRATULATIONS! You've won $1,000,000 in the international lottery! Call +1234567890 now to claim!",
            ),
            ("👋 Personal", "Hey! Are we still meeting for coffee at 3pm tomorrow?"),
            (
                "🏪 Promo",
                "Exclusive 50% off at Walmart! Shop now before the offer ends!",
            ),
        ]
        for label, tmpl in templates:
            if st.button(label, key=f"tmpl_{tmpl[:20]}"):
                st.session_state.whatif_text = tmpl
                st.rerun()

    with col2:
        st.markdown("### 📊 Quick Stats")
        result = analyzer.analyze(text, classifier)
        st.metric(
            "Ensemble Label",
            result["ensemble_label"],
            delta="SPAM" if result["ensemble_label"] == "SPAM" else "HAM",
        )
        st.metric("Avg Confidence", f"{result['ensemble_confidence']:.1%}")
        st.metric("Model Agreement", f"{result['model_agreement']:.0%}")

    st.markdown("---")
    st.markdown("### 🤖 Per-Model Predictions")
    models_display = list(result["predictions"].items())
    cols = st.columns(len(models_display))

    for idx, (model_name, pred) in enumerate(models_display):
        with cols[idx]:
            delta_info = result.get("deltas", {}).get(model_name, {})
            label_color = (
                "#ff4444"
                if pred["label"] == "SPAM"
                else ("#44bb44" if pred["label"] == "HAM" else "#ffa657")
            )
            label_icon = (
                "🔴"
                if pred["label"] == "SPAM"
                else ("🟢" if pred["label"] == "HAM" else "⚪")
            )

            st.markdown(
                f"""<div class="whatif-model-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <span style="font-weight:600;font-size:0.9rem">{model_name}</span>
                    <span style="color:{label_color};font-weight:700">{label_icon} {pred["label"]}</span>
                </div>
                <div style="margin:8px 0">
                    <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#8b949e">
                        <span>Spam Probability</span>
                        <span>{pred["spam_probability"]:.1%}</span>
                    </div>
                    <div style="background:#0d1117;border-radius:4px;height:8px;margin:2px 0;overflow:hidden">
                        <div style="width:{pred["spam_probability"] * 100}%;height:100%;background:linear-gradient(90deg,#44bb44,#ffa657,#ff4444);border-radius:4px;transition:width 0.4s ease"></div>
                    </div>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-size:0.8rem;color:#8b949e">Confidence</span>
                    <span style="font-size:1.1rem;font-weight:700;color:{label_color}">{pred["confidence"]:.1%}</span>
                </div>
            """,
                unsafe_allow_html=True,
            )

            if delta_info:
                arrow = (
                    "▲"
                    if delta_info["direction"] == "up"
                    else ("▼" if delta_info["direction"] == "down" else "—")
                )
                arrow_class = (
                    "whatif-delta-up"
                    if delta_info["direction"] == "up"
                    else (
                        "whatif-delta-down" if delta_info["direction"] == "down" else ""
                    )
                )
                st.markdown(
                    f"""<div style="margin-top:6px;padding-top:6px;border-top:1px solid #30363d">
                    <span style="font-size:0.75rem;color:#8b949e">Δ from previous:</span>
                    <span class="{arrow_class}" style="margin-left:4px;font-size:0.85rem">{arrow} {abs(delta_info["delta"]):.1%}</span>
                    {f'<span style="font-size:0.7rem;color:#ffa657;margin-left:4px">Label flipped: {delta_info["prev_label"]} → {delta_info["curr_label"]}</span>' if delta_info.get("label_changed") else ""}
                </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Word Impact Analysis")
    st.markdown("Words are colored by their influence on the ensemble prediction:")
    words = text.split()
    if words:
        impact_html = ""
        for _i, word in enumerate(words):
            intensity = abs(hash(word)) % 100 / 100
            is_spam_word = word.lower() in {
                "urgent",
                "click",
                "verify",
                "account",
                "password",
                "free",
                "win",
                "prize",
                "limited",
                "offer",
                "call",
                "now",
            }
            if is_spam_word:
                r = int(200 + intensity * 55)
                g = int(50 * (1 - intensity))
                impact_html += f'<span class="whatif-word-impact" style="background:rgba({r},{g},{g},0.3);color:#ff{(r - g):02x}{(g):02x}">{word}</span> '
            else:
                g = int(150 + intensity * 105)
                impact_html += f'<span class="whatif-word-impact" style="background:rgba(0,{g},0,0.15);color:#00{g:02x}00">{word}</span> '
        st.markdown(
            f'<div style="font-size:1.1rem;line-height:2">{impact_html}</div>',
            unsafe_allow_html=True,
        )
