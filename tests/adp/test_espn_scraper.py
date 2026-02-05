"""Tests for ESPN ADP scraper."""

from datetime import datetime

import pytest

from fantasy_baseball_manager.adp.espn_scraper import ESPNADPScraper
from fantasy_baseball_manager.adp.models import ADPData

# ESPN uses a table structure with data-idx attributes for rows
# Player info is in the first cell, ADP in a later column
ESPN_ADP_HTML_FIXTURE = """
<html>
<body>
<div class="ResponsiveTable">
  <table class="Table">
    <thead>
      <tr>
        <th>RK</th>
        <th>PLAYER</th>
        <th>POS</th>
        <th>ADP</th>
        <th>AVG $</th>
        <th>%ROST</th>
      </tr>
    </thead>
    <tbody class="Table__TBODY">
      <tr data-idx="0" class="Table__TR">
        <td>1</td>
        <td>
          <div class="player-column">
            <span class="player-name">Mike Trout</span>
          </div>
        </td>
        <td>OF</td>
        <td>1.5</td>
        <td>$52</td>
        <td>99.8%</td>
      </tr>
      <tr data-idx="1" class="Table__TR">
        <td>2</td>
        <td>
          <div class="player-column">
            <span class="player-name">Shohei Ohtani</span>
          </div>
        </td>
        <td>DH, SP</td>
        <td>2.3</td>
        <td>$50</td>
        <td>99.5%</td>
      </tr>
      <tr data-idx="2" class="Table__TR">
        <td>3</td>
        <td>
          <div class="player-column">
            <span class="player-name">Ronald Acuña Jr.</span>
          </div>
        </td>
        <td>OF</td>
        <td>3.1</td>
        <td>$48</td>
        <td>98.2%</td>
      </tr>
    </tbody>
  </table>
</div>
</body>
</html>
"""

ESPN_ADP_HTML_MALFORMED = """
<html>
<body>
<div class="ResponsiveTable">
  <table class="Table">
    <tbody class="Table__TBODY">
      <tr data-idx="0" class="Table__TR">
        <td>1</td>
        <td>
          <div class="player-column">
            <span class="player-name">Valid Player</span>
          </div>
        </td>
        <td>OF</td>
        <td>5.0</td>
        <td>$40</td>
        <td>90%</td>
      </tr>
      <tr data-idx="1" class="Table__TR">
        <td>2</td>
        <td>
          <div class="player-column">
            <span class="player-name"></span>
          </div>
        </td>
        <td>1B</td>
        <td>invalid</td>
        <td>$30</td>
        <td>80%</td>
      </tr>
      <tr data-idx="2" class="Table__TR">
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </tbody>
  </table>
</div>
</body>
</html>
"""


class TestESPNADPScraper:
    """Tests for ESPNADPScraper."""

    def test_parse_adp_table_extracts_players(self) -> None:
        """Test that _parse_adp_table extracts player data correctly."""
        scraper = ESPNADPScraper()
        entries = scraper._parse_adp_table(ESPN_ADP_HTML_FIXTURE)

        assert len(entries) == 3
        assert entries[0].name == "Mike Trout"
        assert entries[0].adp == 1.5
        assert entries[0].positions == ("OF",)

    def test_parse_adp_table_multiple_positions(self) -> None:
        """Test parsing player with multiple positions."""
        scraper = ESPNADPScraper()
        entries = scraper._parse_adp_table(ESPN_ADP_HTML_FIXTURE)

        ohtani = entries[1]
        assert ohtani.name == "Shohei Ohtani"
        assert ohtani.positions == ("DH", "SP")
        assert ohtani.adp == 2.3

    def test_parse_adp_table_special_characters(self) -> None:
        """Test parsing player names with special characters."""
        scraper = ESPNADPScraper()
        entries = scraper._parse_adp_table(ESPN_ADP_HTML_FIXTURE)

        acuna = entries[2]
        assert acuna.name == "Ronald Acuña Jr."
        assert acuna.adp == 3.1

    def test_parse_adp_table_handles_malformed_rows(self) -> None:
        """Test that malformed rows are skipped gracefully."""
        scraper = ESPNADPScraper()
        entries = scraper._parse_adp_table(ESPN_ADP_HTML_MALFORMED)

        assert len(entries) == 1
        assert entries[0].name == "Valid Player"

    def test_fetch_adp_returns_adp_data(self) -> None:
        """Test that fetch_adp returns properly structured ADPData."""

        class MockScraper(ESPNADPScraper):
            """Mock scraper that returns fixture HTML."""

            def _fetch_page_html(self) -> str:
                return ESPN_ADP_HTML_FIXTURE

        scraper = MockScraper()
        result = scraper.fetch_adp()

        assert isinstance(result, ADPData)
        assert result.source == "espn"
        assert len(result.entries) == 3
        assert isinstance(result.fetched_at, datetime)

    def test_percent_drafted_not_parsed(self) -> None:
        """Test that percent_drafted is None (ESPN doesn't provide useful %drafted)."""
        scraper = ESPNADPScraper()
        entries = scraper._parse_adp_table(ESPN_ADP_HTML_FIXTURE)

        # ESPN %ROST is roster % not draft %, so we don't parse it
        assert entries[0].percent_drafted is None


@pytest.mark.integration
class TestESPNADPScraperIntegration:
    """Integration tests that hit the live ESPN site."""

    def test_fetch_adp_live(self) -> None:
        """Test fetching ADP from live ESPN site."""
        scraper = ESPNADPScraper()
        result = scraper.fetch_adp()

        assert isinstance(result, ADPData)
        assert result.source == "espn"
        assert len(result.entries) > 0
        assert result.entries[0].adp > 0
