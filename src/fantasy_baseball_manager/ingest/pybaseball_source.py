import logging
import time
from collections.abc import Callable
from typing import Any

import requests
from pybaseball import (
    batting_stats_bref,
    pitching_stats_bref,
)
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

_NETWORK_ERRORS = (requests.RequestException, ConnectionError, TimeoutError)


def _log_retry(retry_state: RetryCallState) -> None:
    logger.warning("Retrying pybaseball call (attempt %d): %s", retry_state.attempt_number, retry_state.outcome)


_network_retry = retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(_NETWORK_ERRORS),
    before_sleep=_log_retry,
    reraise=True,
)


class BrefBattingSource:
    def __init__(
        self,
        retry: Callable[[Callable[..., Any]], Callable[..., Any]] = _network_retry,
    ) -> None:
        self._retrying_fetch = retry(self._do_fetch)

    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "batting_stats_bref"

    def _do_fetch(self, **params: Any) -> list[dict[str, Any]]:
        logger.debug("Calling %s(%s)", "batting_stats_bref", params)
        t0 = time.perf_counter()
        try:
            df = batting_stats_bref(**params)
        except _NETWORK_ERRORS:
            raise
        except Exception as exc:
            raise RuntimeError(f"pybaseball fetch failed: {exc}") from exc
        logger.debug("%s returned %d rows in %.1fs", "batting_stats_bref", len(df), time.perf_counter() - t0)
        return df.to_dict("records")

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        return self._retrying_fetch(**params)


class BrefPitchingSource:
    def __init__(
        self,
        retry: Callable[[Callable[..., Any]], Callable[..., Any]] = _network_retry,
    ) -> None:
        self._retrying_fetch = retry(self._do_fetch)

    @property
    def source_type(self) -> str:
        return "pybaseball"

    @property
    def source_detail(self) -> str:
        return "pitching_stats_bref"

    def _do_fetch(self, **params: Any) -> list[dict[str, Any]]:
        logger.debug("Calling %s(%s)", "pitching_stats_bref", params)
        t0 = time.perf_counter()
        try:
            df = pitching_stats_bref(**params)
        except _NETWORK_ERRORS:
            raise
        except Exception as exc:
            raise RuntimeError(f"pybaseball fetch failed: {exc}") from exc
        logger.debug("%s returned %d rows in %.1fs", "pitching_stats_bref", len(df), time.perf_counter() - t0)
        return df.to_dict("records")

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        return self._retrying_fetch(**params)
