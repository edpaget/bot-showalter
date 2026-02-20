from pathlib import Path
from typing import Any

import pytest

from fantasy_baseball_manager.ingest.adp_mapper import _discover_provider_columns
from fantasy_baseball_manager.ingest.fantasypros_adp_source import (
    FantasyProsADPSource,
    _normalize_rows,
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


class TestParseTableHtml:
    def test_parses_headers_from_thead(self) -> None:
        rows = _parse_table_html(_FIXTURE_HTML)
        assert rows
        expected_keys = {"Rank", "Player (Team)", "Yahoo", "CBS", "RTS", "NFBC", "FT", "ESPN", "AVG"}
        assert set(rows[0].keys()) == expected_keys

    def test_parses_rows_from_tbody(self) -> None:
        rows = _parse_table_html(_FIXTURE_HTML)
        assert len(rows) == 20
        assert rows[0]["Player (Team)"] == "Shohei Ohtani (LAD - SP,DH)"
        assert rows[0]["Rank"] == "1"

    def test_rank_and_avg_present(self) -> None:
        rows = _parse_table_html(_FIXTURE_HTML)
        for row in rows:
            assert "Rank" in row
            assert "AVG" in row

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
# _normalize_rows unit tests
# ---------------------------------------------------------------------------


class TestNormalizeRows:
    def test_splits_player_team_column(self) -> None:
        rows = _parse_table_html(_FIXTURE_HTML)
        normalized = _normalize_rows(rows)
        assert normalized[0]["Player"] == "Shohei Ohtani"
        assert normalized[0]["Team"] == "LAD"
        assert normalized[0]["Positions"] == "SP,DH"

    def test_preserves_other_columns(self) -> None:
        rows = _parse_table_html(_FIXTURE_HTML)
        normalized = _normalize_rows(rows)
        assert normalized[0]["Rank"] == "1"
        assert normalized[0]["AVG"] == "1.0"
        assert "ESPN" in normalized[0]

    def test_all_rows_have_player_team_positions(self) -> None:
        rows = _parse_table_html(_FIXTURE_HTML)
        normalized = _normalize_rows(rows)
        for row in normalized:
            assert "Player" in row
            assert "Team" in row
            assert "Positions" in row

    def test_passthrough_when_no_combined_column(self) -> None:
        rows = [{"Rank": "1", "Player": "Test Player", "Team": "NYY", "AVG": "1.0"}]
        result = _normalize_rows(rows)
        assert result == rows

    def test_empty_rows_passthrough(self) -> None:
        assert _normalize_rows([]) == []

    def test_handles_suffix_in_name(self) -> None:
        rows = _parse_table_html(_FIXTURE_HTML)
        normalized = _normalize_rows(rows)
        vlad = normalized[19]
        assert vlad["Player"] == "Vladimir Guerrero Jr."
        assert vlad["Team"] == "TOR"
        assert vlad["Positions"] == "1B"


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
        assert rows[0]["Team"] == "LAD"
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
        assert "Player" in rows[0]
        assert "Team" in rows[0]
        assert "Positions" in rows[0]
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
