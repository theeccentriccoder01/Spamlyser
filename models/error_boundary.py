"""Error boundary framework — wraps page functions with graceful error handling."""

import functools
import logging
import traceback
from typing import Any, Callable, TypeVar

import streamlit as st

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class PageError(Exception):
    """Base exception for Spamlyser page-level errors."""

    def __init__(self, message: str, page: str = "", recoverable: bool = True):
        super().__init__(message)
        self.page = page
        self.recoverable = recoverable


class ModelLoadError(PageError):
    """Raised when a model fails to initialise."""


class DataAccessError(PageError):
    """Raised when storage reads/writes fail."""


class ConfigurationError(PageError):
    """Raised when the app config is invalid."""


def error_boundary(page_func: F, fallback_message: str | None = None) -> F:
    """Decorator that wraps a Streamlit page function in a try/except block.

    On failure it logs the traceback and renders a user-friendly error panel
    instead of crashing the whole app.
    """

    @functools.wraps(page_func)
    def wrapper(*args, **kwargs):
        try:
            return page_func(*args, **kwargs)
        except PageError as exc:
            logger.error(
                "PageError in %s (recoverable=%s): %s",
                exc.page or page_func.__name__,
                exc.recoverable,
                exc,
            )
            render_error_panel(
                title=f"⚠️ {exc.page or page_func.__name__} Error",
                message=str(exc),
                detail=traceback.format_exc() if exc.recoverable else None,
                recoverable=exc.recoverable,
            )
        except Exception as exc:
            logger.error(
                "Unhandled error in %s: %s", page_func.__name__, exc, exc_info=True
            )
            render_error_panel(
                title="⚠️ Unexpected Error",
                message=fallback_message or "Something went wrong. Please try again.",
                detail=traceback.format_exc(),
                recoverable=True,
            )

    return wrapper  # type: ignore[return-value]


def render_error_panel(
    title: str = "⚠️ Error",
    message: str = "An error occurred.",
    detail: str | None = None,
    recoverable: bool = True,
) -> None:
    """Display a styled error panel in the Streamlit UI."""
    st.markdown(f"### {title}")
    st.error(message)
    if detail and st.checkbox(
        "Show technical details", key=f"err_detail_{hash(title)}"
    ):
        st.code(detail, language="traceback")
    if recoverable:
        st.button("🔄 Retry", on_click=st.rerun, type="primary")
    else:
        st.warning("This error is not recoverable. Please restart the app.")


def safe_execute(
    fn: Callable[..., Any],
    default: Any = None,
    error_message: str = "Operation failed",
    logger_name: str | None = None,
    **kwargs,
) -> Any:
    """Execute *fn* with sensible error handling.

    Returns *default* on failure rather than raising.
    """
    log = logging.getLogger(logger_name or __name__)
    try:
        return fn(**kwargs)
    except Exception as exc:
        log.warning("%s: %s", error_message, exc)
        return default
