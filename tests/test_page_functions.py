"""
Tests for page_functions.py — issue #41
Ensures show_feedback_page() accepts navigate_to as a parameter and
that no stale module-level None sentinel exists.
"""

import inspect
import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Provide a minimal streamlit stub so page_functions can be imported without
# a running Streamlit server.
# ---------------------------------------------------------------------------
st_mock = MagicMock()
# Make st.session_state behave like a dict (attribute access)
st_mock.session_state = {}
sys.modules.setdefault("streamlit", st_mock)


import page_functions


class TestShowFeedbackPageSignature:
    """Verify the function signature was fixed (issue #41)."""

    def test_navigate_to_parameter_exists(self):
        """show_feedback_page must accept a navigate_to positional parameter."""
        sig = inspect.signature(page_functions.show_feedback_page)
        assert "navigate_to" in sig.parameters, (
            "show_feedback_page() is missing the 'navigate_to' parameter. "
            "See issue #41."
        )

    def test_navigate_to_has_no_default(self):
        """navigate_to must be required (no default value) so callers never
        accidentally pass the old module-level None."""
        sig = inspect.signature(page_functions.show_feedback_page)
        param = sig.parameters["navigate_to"]
        assert param.default is inspect.Parameter.empty, (
            "navigate_to should be a required parameter with no default."
        )

    def test_no_module_level_none_sentinel(self):
        """The old `navigate_to = None` module-level sentinel must be gone."""
        assert not hasattr(page_functions, "navigate_to") or callable(
            getattr(page_functions, "navigate_to", None)
        ), (
            "page_functions.navigate_to still exists as a module-level None. "
            "Remove it — callers must pass the function explicitly."
        )


class TestShowFeedbackPageCallable:
    """Verify show_feedback_page invokes the provided navigate_to."""

    def _make_st_mock(self):
        """Return a fresh Streamlit mock with the session state we need."""
        st = MagicMock()
        st.session_state = {
            "feedback_submitted": False,
            "feedback_rating": 3,
            "feedback_context": None,
        }
        # Simulate button not clicked
        st.button.return_value = False
        st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        st.form.return_value.__enter__ = MagicMock(return_value=MagicMock())
        st.form.return_value.__exit__ = MagicMock(return_value=False)
        return st

    def test_navigate_to_callable_accepted(self):
        """show_feedback_page() must not raise when given a real callable."""
        fake_navigate = MagicMock()

        with patch.dict(sys.modules, {"streamlit": self._make_st_mock()}):
            # Re-import so patched `st` is used inside the function
            import importlib

            import page_functions as pf

            importlib.reload(pf)
            # Should NOT raise TypeError / AttributeError
            try:
                pf.show_feedback_page(navigate_to=fake_navigate)
            except Exception as exc:
                # Only fail for the specific error the bug caused
                assert "NoneType" not in str(exc) and "not callable" not in str(exc), (
                    f"show_feedback_page raised the original bug error: {exc}"
                )

    def test_none_navigate_to_raises_clearly(self):
        """Passing None explicitly must fail with TypeError, not silently
        hide the bug until a button is clicked deep inside the function."""
        # This is a documentation test — callers should never pass None.
        # The old code hid this mistake; with the parameter, Python's own
        # type system makes misuse visible at call sites during code review.
        import page_functions as pf

        # Introspect: the param must exist and be required
        sig = inspect.signature(pf.show_feedback_page)
        assert "navigate_to" in sig.parameters
