import re
import html
import signal
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

MAX_SMS_LENGTH = 1000
REGEX_TIMEOUT = 0.5

_TIMER_TRIGGERED = False


def _timeout_handler(signum, frame):
    global _TIMER_TRIGGERED
    _TIMER_TRIGGERED = True
    raise TimeoutError("Regex execution timed out")


def safe_regex(
    pattern: str,
    text: str,
    method: Callable = re.search,
    timeout: float = REGEX_TIMEOUT,
    flags: int = 0,
    default=None,
):
    """Execute a regex operation with a timeout guard.

    Falls back to ``default`` (or ``None``) if the regex times out or raises
    any exception.  On Unix this uses ``signal.setitimer``; on platforms
    without ``signal.SIGALRM`` the regex runs without a watchdog timer and
    only the standard try/except protection applies.
    """
    global _TIMER_TRIGGERED
    _TIMER_TRIGGERED = False

    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        try:
            compiled = re.compile(pattern, flags)
            result = method(compiled, text)
            if _TIMER_TRIGGERED:
                logger.warning("Regex timed out: %s", pattern[:60])
                return default
            return result
        except (TimeoutError, re.error, RecursionError, ValueError) as exc:
            logger.warning(
                "Regex failed (%s): %s", exc.__class__.__name__, pattern[:60]
            )
            return default
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


def safe_regex_search(
    pattern: str,
    text: str,
    timeout: float = REGEX_TIMEOUT,
    flags: int = 0,
    default=None,
):
    return safe_regex(
        pattern, text, method=re.search, timeout=timeout, flags=flags, default=default
    )


def safe_regex_match(
    pattern: str,
    text: str,
    timeout: float = REGEX_TIMEOUT,
    flags: int = 0,
    default=None,
):
    return safe_regex(
        pattern, text, method=re.match, timeout=timeout, flags=flags, default=default
    )


def safe_regex_findall(
    pattern: str,
    text: str,
    timeout: float = REGEX_TIMEOUT,
    flags: int = 0,
    default=None,
):
    return safe_regex(
        pattern, text, method=re.findall, timeout=timeout, flags=flags, default=default
    )


def safe_regex_sub(
    pattern: str,
    repl: str,
    text: str,
    timeout: float = REGEX_TIMEOUT,
    flags: int = 0,
    default=None,
):
    """Execute ``re.sub`` with a timeout. Returns ``default`` on failure."""
    global _TIMER_TRIGGERED
    _TIMER_TRIGGERED = False

    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        try:
            compiled = re.compile(pattern, flags)
            result = compiled.sub(repl, text)
            if _TIMER_TRIGGERED:
                logger.warning("Regex sub timed out: %s", pattern[:60])
                return default if default is not None else text
            return result
        except (TimeoutError, re.error, RecursionError, ValueError) as exc:
            logger.warning(
                "Regex sub failed (%s): %s", exc.__class__.__name__, pattern[:60]
            )
            return default if default is not None else text
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


def sanitize_text(text: Optional[str], max_length: int = MAX_SMS_LENGTH) -> str:
    """Truncate to ``max_length`` and HTML-escape the result."""
    if not text:
        return ""
    raw = str(text)[:max_length]
    return html.escape(raw)


def strip_html_unsafe(text: str) -> str:
    """Remove known HTML/script tags before display-only rendering."""
    text = re.sub(r"(?i)<script[^>]*>.*?</script>", "", text)
    text = re.sub(r"(?i)<[^>]*>", "", text)
    return text


def validate_sms_message(message: str) -> tuple:
    """Validate an SMS message.

    Returns ``(is_valid: bool, error_msg: str, sanitized: str)`` where the
    sanitized string has HTML tags stripped (safe for model inference).
    """
    if not message or not message.strip():
        return False, "Message cannot be empty.", ""
    if len(message) > MAX_SMS_LENGTH:
        return (
            False,
            f"Message exceeds {MAX_SMS_LENGTH} characters ({len(message)} given).",
            strip_html_unsafe(message[:MAX_SMS_LENGTH]),
        )
    return True, "", strip_html_unsafe(message)
