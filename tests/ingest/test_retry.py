import logging

import httpx
import pytest
from tenacity import wait_none

from fantasy_baseball_manager.ingest._retry import default_http_retry


class TestDefaultHttpRetry:
    def test_returns_decorator(self) -> None:
        decorator = default_http_retry("Test")
        assert callable(decorator)

    def test_retries_on_transport_error(self) -> None:
        calls: list[int] = []

        decorator = default_http_retry("Test")
        decorator.wait = wait_none()  # type: ignore[attr-defined]

        @decorator
        def flaky() -> str:
            calls.append(1)
            if len(calls) < 3:
                raise httpx.TransportError("connection failed")
            return "ok"

        assert flaky() == "ok"
        assert len(calls) == 3

    def test_retries_on_http_status_error(self) -> None:
        calls: list[int] = []

        decorator = default_http_retry("Test")
        decorator.wait = wait_none()  # type: ignore[attr-defined]

        @decorator
        def flaky() -> str:
            calls.append(1)
            if len(calls) < 2:
                raise httpx.HTTPStatusError(
                    "500", request=httpx.Request("GET", "http://x"), response=httpx.Response(500)
                )
            return "ok"

        assert flaky() == "ok"
        assert len(calls) == 2

    def test_exhausts_after_3_attempts_and_reraises(self) -> None:
        decorator = default_http_retry("Test")
        decorator.wait = wait_none()  # type: ignore[attr-defined]

        @decorator
        def always_fails() -> str:
            raise httpx.TransportError("down")

        with pytest.raises(httpx.TransportError, match="down"):
            always_fails()

    def test_log_message_includes_label(self, caplog: pytest.LogCaptureFixture) -> None:
        decorator = default_http_retry("MyLabel")
        decorator.wait = wait_none()  # type: ignore[attr-defined]
        calls: list[int] = []

        @decorator
        def flaky() -> str:
            calls.append(1)
            if len(calls) < 2:
                raise httpx.TransportError("oops")
            return "ok"

        with caplog.at_level(logging.WARNING):
            flaky()

        assert any("Retrying MyLabel" in msg for msg in caplog.messages)
