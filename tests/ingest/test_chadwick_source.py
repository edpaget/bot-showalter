import csv
import io
import zipfile
from typing import Any

import httpx
import pytest

from fantasy_baseball_manager.ingest.chadwick_source import ChadwickRegisterSource
from fantasy_baseball_manager.ingest.protocols import DataSource


class FakeTransport(httpx.BaseTransport):
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._response


class FailNTransport(httpx.BaseTransport):
    """Returns error responses for the first N requests, then succeeds."""

    def __init__(self, fail_count: int, success_response: httpx.Response) -> None:
        self._fail_count = fail_count
        self._success_response = success_response
        self._call_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            return httpx.Response(503, content=b"Service Unavailable")
        return self._success_response

    @property
    def call_count(self) -> int:
        return self._call_count


_FIELDS = ["key_mlbam", "name_first", "name_last", "key_fangraphs", "key_bbref", "key_retro"]


def _build_zip(rows: list[dict[str, Any]], filename: str = "register-master/people-0.csv") -> bytes:
    """Build a ZIP archive containing a single CSV file with the given rows."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr(filename, buf.getvalue())
    return zip_buf.getvalue()


def _zip_response(zip_bytes: bytes) -> httpx.Response:
    return httpx.Response(200, content=zip_bytes, headers={"content-type": "application/zip"})


class TestChadwickRegisterSource:
    def test_satisfies_datasource_protocol(self) -> None:
        source = ChadwickRegisterSource()
        assert isinstance(source, DataSource)

    def test_source_type_and_detail(self) -> None:
        source = ChadwickRegisterSource()
        assert source.source_type == "chadwick_bureau"
        assert source.source_detail == "chadwick_register"

    def test_valid_zip_returns_list_of_dicts(self) -> None:
        rows = [
            {
                "key_mlbam": "545361",
                "name_first": "Mike",
                "name_last": "Trout",
                "key_fangraphs": "10155",
                "key_bbref": "troutmi01",
                "key_retro": "troum001",
            },
            {
                "key_mlbam": "660271",
                "name_first": "Shohei",
                "name_last": "Ohtani",
                "key_fangraphs": "19755",
                "key_bbref": "ohtansh01",
                "key_retro": "ohtas001",
            },
        ]
        zip_bytes = _build_zip(rows)
        client = httpx.Client(transport=FakeTransport(_zip_response(zip_bytes)))
        source = ChadwickRegisterSource(client=client)

        result = source.fetch()

        assert len(result) == 2
        assert result[0]["key_mlbam"] == "545361"
        assert result[0]["name_first"] == "Mike"
        assert result[1]["key_mlbam"] == "660271"

    def test_rows_without_key_mlbam_are_excluded(self) -> None:
        rows = [
            {
                "key_mlbam": "",
                "name_first": "Unknown",
                "name_last": "Player",
                "key_fangraphs": "",
                "key_bbref": "",
                "key_retro": "",
            },
            {
                "key_mlbam": "545361",
                "name_first": "Mike",
                "name_last": "Trout",
                "key_fangraphs": "10155",
                "key_bbref": "troutmi01",
                "key_retro": "troum001",
            },
        ]
        zip_bytes = _build_zip(rows)
        client = httpx.Client(transport=FakeTransport(_zip_response(zip_bytes)))
        source = ChadwickRegisterSource(client=client)

        result = source.fetch()

        assert len(result) == 1
        assert result[0]["key_mlbam"] == "545361"

    def test_empty_csv_returns_empty_list(self) -> None:
        zip_bytes = _build_zip([])
        client = httpx.Client(transport=FakeTransport(_zip_response(zip_bytes)))
        source = ChadwickRegisterSource(client=client)

        result = source.fetch()

        assert result == []

    def test_retry_on_503_then_success(self) -> None:
        rows = [
            {
                "key_mlbam": "545361",
                "name_first": "Mike",
                "name_last": "Trout",
                "key_fangraphs": "10155",
                "key_bbref": "troutmi01",
                "key_retro": "troum001",
            },
        ]
        zip_bytes = _build_zip(rows)
        transport = FailNTransport(fail_count=2, success_response=_zip_response(zip_bytes))
        client = httpx.Client(transport=transport)
        source = ChadwickRegisterSource(client=client)

        result = source.fetch()

        assert len(result) == 1
        assert transport.call_count == 3

    def test_exhausted_retries_raises(self) -> None:
        rows = [
            {
                "key_mlbam": "545361",
                "name_first": "Mike",
                "name_last": "Trout",
                "key_fangraphs": "10155",
                "key_bbref": "troutmi01",
                "key_retro": "troum001",
            },
        ]
        zip_bytes = _build_zip(rows)
        transport = FailNTransport(fail_count=5, success_response=_zip_response(zip_bytes))
        client = httpx.Client(transport=transport)
        source = ChadwickRegisterSource(client=client)

        with pytest.raises(httpx.HTTPStatusError):
            source.fetch()

        assert transport.call_count == 3
