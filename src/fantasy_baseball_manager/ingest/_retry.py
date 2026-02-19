import logging
from collections.abc import Callable
from typing import Any

import httpx
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger(__name__)


def default_http_retry(label: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a tenacity retry decorator configured for HTTP calls.

    *label* is interpolated into the warning message emitted before each
    retry attempt, e.g. ``"Retrying <label> (attempt 2): <error>"``.
    """

    def _log_retry(retry_state: RetryCallState) -> None:
        logger.warning("Retrying %s (attempt %d): %s", label, retry_state.attempt_number, retry_state.outcome)

    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        before_sleep=_log_retry,
        reraise=True,
    )
