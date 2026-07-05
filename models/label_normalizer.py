from models.unicode_handler import clean_unicode_text

"""
Label normalisation for SMS spam classification (issue #15).

HuggingFace text-classification pipelines return whatever labels are in the
model's ``id2label`` config. Fine-tuned models that did not set human-readable
labels return ``LABEL_0`` / ``LABEL_1`` instead of ``HAM`` / ``SPAM``.

The single-message prediction path in ``app.py`` used the raw label directly as
a dictionary key for stats tracking::

    st.session_state.model_stats[selected_model_name][label.lower()] += 1

where ``model_stats`` is initialised with only ``'spam'`` / ``'ham'`` /
``'total'`` keys. A ``LABEL_0`` / ``LABEL_1`` (or any unexpected) label produced
a key such as ``'label_1'`` that does not exist, raising ``KeyError`` and
crashing the prediction. The same raw label was also compared with
``== "SPAM"`` / ``== "HAM"``, so an unmapped label would be silently treated as
HAM (misclassification).

``normalize_label`` maps any raw label to the canonical ``"SPAM"`` / ``"HAM"``,
fixing both the crash and the silent-misclassification risk in one place.
"""

from typing import Optional

# Raw labels that mean "spam" / "ham" across the models this app loads.
_SPAM_LABELS = {"SPAM", "LABEL_1"}
_HAM_LABELS = {"HAM", "LABEL_0"}


def normalize_label(raw_label, spam_probability: float | None = None) -> str:
    """Map a model's raw label to canonical ``"SPAM"`` or ``"HAM"``.

    Args:
        raw_label: The label emitted by the model pipeline (e.g. ``"SPAM"``,
            ``"HAM"``, ``"LABEL_0"``, ``"LABEL_1"``, or any other value).
        spam_probability: Optional probability that the message is spam, used
            only as a fallback when the label itself is unrecognised.

    Returns:
        ``"SPAM"`` or ``"HAM"``. Recognised spam labels (``"SPAM"``,
        ``"LABEL_1"``) map to ``"SPAM"``; recognised ham labels (``"HAM"``,
        ``"LABEL_0"``) map to ``"HAM"``. For an unrecognised label, falls back
        to ``spam_probability`` (``>= 0.5`` -> ``"SPAM"``) when a usable numeric
        value is provided, otherwise defaults to ``"HAM"``.
    """
    label = str(raw_label).strip().upper()
    if label in _SPAM_LABELS:
        return "SPAM"
    if label in _HAM_LABELS:
        return "HAM"
    if spam_probability is not None:
        try:
            return "SPAM" if float(spam_probability) >= 0.5 else "HAM"
        except (TypeError, ValueError):
            return "HAM"
    return "HAM"
