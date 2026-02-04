"""Tests for Yahoo ADP scraper."""

from datetime import datetime

import pytest

from fantasy_baseball_manager.adp.models import ADPData
from fantasy_baseball_manager.adp.scraper import YahooADPScraper

YAHOO_ADP_HTML_FIXTURE = """
<html>
<body>
<table class="Table">
  <thead>
    <tr>
      <th>Rank</th>
      <th>Player</th>
      <th>Avg Pick</th>
      <th>% Drafted</th>
    </tr>
  </thead>
  <tbody>
    <tr class="player-row" data-row="0">
      <td>1</td>
      <td>
        <div class="player-name">
          <a href="/players/123">Mike Trout</a>
          <span class="position">OF - LAA</span>
        </div>
      </td>
      <td>1.5</td>
      <td>99.8%</td>
    </tr>
    <tr class="player-row" data-row="1">
      <td>2</td>
      <td>
        <div class="player-name">
          <a href="/players/456">Shohei Ohtani</a>
          <span class="position">DH,SP - LAD</span>
        </div>
      </td>
      <td>2.3</td>
      <td>99.5%</td>
    </tr>
    <tr class="player-row" data-row="2">
      <td>3</td>
      <td>
        <div class="player-name">
          <a href="/players/789">Ronald Acuña Jr.</a>
          <span class="position">OF - ATL</span>
        </div>
      </td>
      <td>3.1</td>
      <td>98.2%</td>
    </tr>
  </tbody>
</table>
</body>
</html>
"""

YAHOO_ADP_HTML_MALFORMED = """
<html>
<body>
<table class="Table">
  <tbody>
    <tr class="player-row" data-row="0">
      <td>1</td>
      <td>
        <div class="player-name">
          <a href="/players/123">Valid Player</a>
          <span class="position">OF - NYY</span>
        </div>
      </td>
      <td>5.0</td>
      <td>90%</td>
    </tr>
    <tr class="player-row" data-row="1">
      <td>2</td>
      <td>
        <div class="player-name">
          <span class="position">1B - BOS</span>
        </div>
      </td>
      <td>invalid</td>
      <td>80%</td>
    </tr>
    <tr class="player-row" data-row="2">
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
