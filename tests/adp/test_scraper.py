"""Tests for Yahoo ADP scraper."""

from datetime import datetime

import pytest

from fantasy_baseball_manager.adp.models import ADPData
from fantasy_baseball_manager.adp.scraper import YahooADPScraper

# Fixture mimicking Yahoo's actual HTML structure
YAHOO_ADP_HTML_FIXTURE = """
<html>
<body>
<table data-tst="table">
  <thead>
    <tr><th>Player</th><th>Rank</th><th>Pos Rank</th><th>CER</th><th>%Drafted</th><th>Preseason</th><th>All Drafts</th></tr>
  </thead>
  <tbody>
    <tr data-tst="table-row-0">
      <td><div data-tst="player"><div data-tst="player-name">Mike Trout</div><span data-tst="player-position">OF</span></div></td>
      <td><div class="Ta(c)">1</div></td>
      <td><div class="Ta(c)">1</div></td>
      <td><div class="Ta(c)">1</div></td>
      <td><div class="Ta(c)">99.8%</div></td>
      <td><div class="Ta(c)">1.5</div></td>
      <td><div class="Ta(c)">1.5</div></td>
    </tr>
    <tr data-tst="table-row-1">
      <td><div data-tst="player"><div data-tst="player-name">Shohei Ohtani</div><span data-tst="player-position">DH,SP</span></div></td>
      <td><div class="Ta(c)">2</div></td>
      <td><div class="Ta(c)">1</div></td>
      <td><div class="Ta(c)">2</div></td>
      <td><div class="Ta(c)">99.5%</div></td>
      <td><div class="Ta(c)">2.3</div></td>
      <td><div class="Ta(c)">2.3</div></td>
    </tr>
    <tr data-tst="table-row-2">
      <td><div data-tst="player"><div data-tst="player-name">Ronald Acuña Jr.</div><span data-tst="player-position">OF</span></div></td>
      <td><div class="Ta(c)">3</div></td>
      <td><div class="Ta(c)">2</div></td>
      <td><div class="Ta(c)">3</div></td>
      <td><div class="Ta(c)">98.2%</div></td>
      <td><div class="Ta(c)">3.1</div></td>
      <td><div class="Ta(c)">3.1</div></td>
    </tr>
  </tbody>
</table>
</body>
</html>
"""

YAHOO_ADP_HTML_MALFORMED = """
<html>
<body>
<table data-tst="table">
  <tbody>
    <tr data-tst="table-row-0">
      <td><div data-tst="player"><div data-tst="player-name">Valid Player</div><span data-tst="player-position">OF</span></div></td>
      <td><div class="Ta(c)">1</div></td>
      <td><div class="Ta(c)">1</div></td>
      <td><div class="Ta(c)">1</div></td>
      <td><div class="Ta(c)">90%</div></td>
      <td><div class="Ta(c)">5.0</div></td>
      <td><div class="Ta(c)">5.0</div></td>
    </tr>
    <tr data-tst="table-row-1">
      <td><div data-tst="player"><span data-tst="player-position">1B</span></div></td>
      <td><div class="Ta(c)">2</div></td>
      <td><div class="Ta(c)">1</div></td>
      <td><div class="Ta(c)">2</div></td>
      <td><div class="Ta(c)">80%</div></td>
      <td><div class="Ta(c)">invalid</div></td>
      <td><div class="Ta(c)">invalid</div></td>
    </tr>
    <tr data-tst="table-row-2">
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</body>
</html>
"""


class TestYahooADPScraper:
    """Tests for YahooADPScraper."""

    def test_parse_adp_table_extracts_players(self) -> None:
        """Test that _parse_adp_table extracts player data correctly."""
        scraper = YahooADPScraper()
        entries = scraper._parse_adp_table(YAHOO_ADP_HTML_FIXTURE)

        assert len(entries) == 3
        assert entries[0].name == "Mike Trout"
        assert entries[0].adp == 1.5
        assert entries[0].positions == ("OF",)
        assert entries[0].percent_drafted == 99.8

    def test_parse_adp_table_multiple_positions(self) -> None:
        """Test parsing player with multiple positions."""
        scraper = YahooADPScraper()
        entries = scraper._parse_adp_table(YAHOO_ADP_HTML_FIXTURE)

        ohtani = entries[1]
        assert ohtani.name == "Shohei Ohtani"
        assert ohtani.positions == ("DH", "SP")

    def test_parse_adp_table_special_characters(self) -> None:
        """Test parsing player names with special characters."""
        scraper = YahooADPScraper()
        entries = scraper._parse_adp_table(YAHOO_ADP_HTML_FIXTURE)

        acuna = entries[2]
        assert acuna.name == "Ronald Acuña Jr."
        assert acuna.adp == 3.1

    def test_parse_adp_table_handles_malformed_rows(self) -> None:
        """Test that malformed rows are skipped gracefully."""
        scraper = YahooADPScraper()
        entries = scraper._parse_adp_table(YAHOO_ADP_HTML_MALFORMED)

        assert len(entries) == 1
        assert entries[0].name == "Valid Player"

    def test_normalize_name(self) -> None:
        """Test player name normalization."""
        scraper = YahooADPScraper()

        assert scraper._normalize_name("Mike Trout") == "mike trout"
        assert scraper._normalize_name("Ronald Acuña Jr.") == "ronald acuna jr"
        assert scraper._normalize_name("Vladimir Guerrero Jr.") == "vladimir guerrero jr"
        assert scraper._normalize_name("Bobby Witt Jr.") == "bobby witt jr"
        assert scraper._normalize_name("José Ramírez") == "jose ramirez"

    def test_fetch_adp_returns_adp_data(self) -> None:
        """Test that fetch_adp returns properly structured ADPData."""

        class MockScraper(YahooADPScraper):
            """Mock scraper that returns fixture HTML."""

            def _fetch_page_html(self) -> str:
                return YAHOO_ADP_HTML_FIXTURE

        scraper = MockScraper()
        result = scraper.fetch_adp()

        assert isinstance(result, ADPData)
        assert result.source == "yahoo"
        assert len(result.entries) == 3
        assert isinstance(result.fetched_at, datetime)

    def test_parse_percent_drafted(self) -> None:
        """Test parsing percent drafted values."""
        scraper = YahooADPScraper()
        entries = scraper._parse_adp_table(YAHOO_ADP_HTML_FIXTURE)

        assert entries[0].percent_drafted == 99.8
        assert entries[1].percent_drafted == 99.5
        assert entries[2].percent_drafted == 98.2


@pytest.mark.integration
class TestYahooADPScraperIntegration:
    """Integration tests that hit the live Yahoo site."""

    def test_fetch_adp_live(self) -> None:
        """Test fetching ADP from live Yahoo site."""
        scraper = YahooADPScraper()
        result = scraper.fetch_adp()

        assert isinstance(result, ADPData)
        assert result.source == "yahoo"
        assert len(result.entries) > 0
        assert result.entries[0].adp > 0
