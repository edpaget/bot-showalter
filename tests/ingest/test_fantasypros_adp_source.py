from pathlib import Path
from typing import Any

import pytest

from fantasy_baseball_manager.ingest.adp_mapper import _discover_provider_columns
from fantasy_baseball_manager.ingest.fantasypros_adp_source import (
    FantasyProsADPSource,
    _parse_table_html,
)
from fantasy_baseball_manager.ingest.protocols import DataSource

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fantasypros_adp_table.html"
_FIXTURE_HTML = _FIXTURE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Mock Playwright objects
# ---------------------------------------------------------------------------


class _MockLocator:
    def __init__(self, html: str) -> None:
        self._html = html

    def wait_for(self, **kwargs: Any) -> None:
        pass

    def inner_html(self) -> str:
        return self._html


class _MockPage:
    def __init__(self, html: str) -> None:
        self._html = html

    def goto(self, url: str, **kwargs: Any) -> None:
        pass

    def locator(self, selector: str) -> _MockLocator:
        return _MockLocator(self._html)


class _MockBrowser:
    def __init__(self, html: str) -> None:
        self._html = html

    def new_page(self) -> _MockPage:
        return _MockPage(self._html)

    def close(self) -> None:
        pass


class _MockBrowserType:
    def __init__(self, html: str) -> None:
        self._html = html

    def launch(self, **kwargs: Any) -> _MockBrowser:
        return _MockBrowser(self._html)


class _MockPlaywright:
    def __init__(self, html: str) -> None:
        self.chromium = _MockBrowserType(html)

    def __enter__(self) -> "_MockPlaywright":
        return self

    def __exit__(self, *args: object) -> None:
        pass


class _TimeoutLocator:
    def wait_for(self, **kwargs: Any) -> None:
        msg = "Timeout 30000ms exceeded."
        raise TimeoutError(msg)

    def inner_html(self) -> str:
        return ""


class _TimeoutPage:
    def goto(self, url: str, **kwargs: Any) -> None:
        pass

    def locator(self, selector: str) -> _TimeoutLocator:
        return _TimeoutLocator()


class _TimeoutBrowser:
    def new_page(self) -> _TimeoutPage:
        return _TimeoutPage()

    def close(self) -> None:
        pass


class _TimeoutBrowserType:
    def launch(self, **kwargs: Any) -> _TimeoutBrowser:
        return _TimeoutBrowser()


class _TimeoutPlaywright:
    def __init__(self) -> None:
        self.chromium = _TimeoutBrowserType()

    def __enter__(self) -> "_TimeoutPlaywright":
        return self

    def __exit__(self, *args: object) -> None:
        pass


# ---------------------------------------------------------------------------
# _parse_table_html unit tests
# ---------------------------------------------------------------------------


def _strip_table_wrapper(html: str) -> str:
    """Extract inner HTML from the fixture (strip outermost <table> tag)."""
    # The fixture is a full <table>; _parse_table_html expects a full <table>
    return html


class TestParseTableHtml:
    def test_parses_headers_from_thead(self) -> None:
        rows = _parse_table_html(_FIXTURE_HTML)
        assert rows  # non-empty
        expected_keys = {"Rank", "Player", "Team", "Positions", "ESPN", "Yahoo", "CBS", "NFBC", "AVG"}
        assert set(rows[0].keys()) == expected_keys

    def test_parses_rows_from_tbody(self) -> None:
        rows = _parse_table_html(_FIXTURE_HTML)
        assert len(rows) == 20
        assert rows[0]["Player"] == "Shohei Ohtani"
        assert rows[0]["Rank"] == "1"
        assert rows[19]["Player"] == "Vladimir Guerrero Jr."

    def test_rank_player_team_positions_avg_present(self) -> None:
        rows = _parse_table_html(_FIXTURE_HTML)
        required_keys = {"Rank", "Player", "Team", "Positions", "AVG"}
        for row in rows:
            assert required_keys <= set(row.keys())

    def test_provider_columns_detected(self) -> None:
        rows = _parse_table_html(_FIXTURE_HTML)
        header = list(rows[0].keys())
        providers = _discover_provider_columns(header)
        slugs = [slug for _, slug in providers]
        assert "espn" in slugs
        assert "yahoo" in slugs
        assert "cbs" in slugs
        assert "nfbc" in slugs

    def test_empty_table_returns_empty_list(self) -> None:
        html = "<table><thead><tr><th>Rank</th><th>Player</th></tr></thead><tbody></tbody></table>"
        rows = _parse_table_html(html)
        assert rows == []

    def test_no_header_raises_error(self) -> None:
        html = "<table><thead></thead><tbody><tr><td>1</td></tr></tbody></table>"
        with pytest.raises(RuntimeError, match="No header row"):
            _parse_table_html(html)


# ---------------------------------------------------------------------------
# FantasyProsADPSource tests
# ---------------------------------------------------------------------------


class TestFantasyProsADPSource:
    def test_satisfies_datasource_protocol(self) -> None:
        source = FantasyProsADPSource()
        assert isinstance(source, DataSource)

    def test_source_type_and_detail(self) -> None:
        source = FantasyProsADPSource()
        assert source.source_type == "fantasypros_web"
        assert "fantasypros.com" in source.source_detail

    def test_fetch_returns_row_dicts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Extract inner table content (strip outer <table> tags) for mock locator
        inner = _FIXTURE_HTML.split(">", 1)[1].rsplit("</table>", 1)[0]
        mock_pw = _MockPlaywright(inner)
        monkeypatch.setattr(
            "fantasy_baseball_manager.ingest.fantasypros_adp_source.sync_playwright",
            lambda: mock_pw,
        )

        source = FantasyProsADPSource()
        rows = source.fetch()
        assert len(rows) == 20
        assert rows[0]["Player"] == "Shohei Ohtani"
        assert rows[0]["Rank"] == "1"
        assert "AVG" in rows[0]

    def test_fetch_rows_compatible_with_mapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inner = _FIXTURE_HTML.split(">", 1)[1].rsplit("</table>", 1)[0]
        mock_pw = _MockPlaywright(inner)
        monkeypatch.setattr(
            "fantasy_baseball_manager.ingest.fantasypros_adp_source.sync_playwright",
            lambda: mock_pw,
        )

        source = FantasyProsADPSource()
        rows = source.fetch()
        header = list(rows[0].keys())
        providers = _discover_provider_columns(header)
        assert len(providers) >= 2
        slugs = {slug for _, slug in providers}
        assert "espn" in slugs
        assert "yahoo" in slugs

    def test_table_not_found_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_pw = _TimeoutPlaywright()
        monkeypatch.setattr(
            "fantasy_baseball_manager.ingest.fantasypros_adp_source.sync_playwright",
            lambda: mock_pw,
        )

        source = FantasyProsADPSource()
        with pytest.raises(TimeoutError):
            source.fetch()
