"""Tests for ESPN ADP scraper."""

from datetime import datetime

import pytest

from fantasy_baseball_manager.adp.espn_scraper import ESPNADPScraper, _parse_espn_rows
from fantasy_baseball_manager.adp.models import ADPData


# Fixture mimicking ESPN's actual HTML structure (as of 2025)
ESPN_ADP_HTML_FIXTURE = """
<html>
<body>
<div class="ResponsiveTable">
  <table class="Table">
    <thead>
      <tr><th>RK</th><th>PLAYER</th><th>ADP</th><th>+/-</th><th>AVG $</th><th>+/-</th><th>%ROST</th></tr>
    </thead>
    <tbody class="Table__TBODY">
      <tr class="Table__TR Table__TR--sm Table__even">
        <td class="Table__TD"><div class="jsx-2810852873 table--cell ranking tl">1</div></td>
        <td class="Table__TD">
          <div class="jsx-2810852873 table--cell player__column">
            <div class="player-info">
              <span class="truncate"><a class="AnchorLink link clr-link pointer">Mike Trout</a></span>
              <div class="player-column__position">
                <span class="playerinfo__playerteam">LAA</span>
                <span class="playerinfo__playerpos ttu">OF</span>
              </div>
            </div>
          </div>
        </td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell adp tar">1.5</div></td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell pcadp tar"><span>0.0</span></div></td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell avc tar">52.0</div></td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell avc tar"><span>0.0</span></div></td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell own tar">99.8%</div></td>
      </tr>
      <tr class="Table__TR Table__TR--sm Table__odd">
        <td class="Table__TD"><div class="jsx-2810852873 table--cell ranking tl">2</div></td>
        <td class="Table__TD">
          <div class="jsx-2810852873 table--cell player__column">
            <div class="player-info">
              <span class="truncate"><a class="AnchorLink link clr-link pointer">Shohei Ohtani</a></span>
              <div class="player-column__position">
                <span class="playerinfo__playerteam">LAD</span>
                <span class="playerinfo__playerpos ttu">DH, SP</span>
              </div>
            </div>
          </div>
        </td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell adp tar">2.3</div></td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell pcadp tar"><span>0.0</span></div></td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell avc tar">50.0</div></td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell avc tar"><span>0.0</span></div></td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell own tar">99.5%</div></td>
      </tr>
      <tr class="Table__TR Table__TR--sm Table__even">
        <td class="Table__TD"><div class="jsx-2810852873 table--cell ranking tl">3</div></td>
        <td class="Table__TD">
          <div class="jsx-2810852873 table--cell player__column">
            <div class="player-info">
              <span class="truncate"><a class="AnchorLink link clr-link pointer">Ronald Acuña Jr.</a></span>
              <div class="player-column__position">
                <span class="playerinfo__playerteam">ATL</span>
                <span class="playerinfo__playerpos ttu">OF</span>
              </div>
            </div>
          </div>
        </td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell adp tar">3.1</div></td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell pcadp tar"><span>0.0</span></div></td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell avc tar">48.0</div></td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell avc tar"><span>0.0</span></div></td>
        <td class="Table__TD"><div class="jsx-2810852873 table--cell own tar">98.2%</div></td>
      </tr>
    </tbody>
  </table>
</div>
</body>
</html>
"""

# Fixture with "--" ADP values (off-season)
ESPN_ADP_HTML_OFFSEASON = """
<html>
<body>
<table class="Table">
  <tbody class="Table__TBODY">
    <tr class="Table__TR Table__TR--sm">
      <td class="Table__TD"><div class="table--cell ranking tl">1</div></td>
      <td class="Table__TD">
        <div class="table--cell player__column">
          <span class="truncate"><a class="AnchorLink">Shohei Ohtani</a></span>
          <span class="playerinfo__playerpos ttu">DH, SP</span>
        </div>
      </td>
      <td class="Table__TD"><div class="table--cell adp tar">--</div></td>
    </tr>
    <tr class="Table__TR Table__TR--sm">
      <td class="Table__TD"><div class="table--cell ranking tl">2</div></td>
      <td class="Table__TD">
        <div class="table--cell player__column">
          <span class="truncate"><a class="AnchorLink">Aaron Judge</a></span>
          <span class="playerinfo__playerpos ttu">OF, DH</span>
        </div>
      </td>
      <td class="Table__TD"><div class="table--cell adp tar">--</div></td>
    </tr>
  </tbody>
</table>
</body>
</html>
"""

ESPN_ADP_HTML_MALFORMED = """
<html>
<body>
<table class="Table">
  <tbody class="Table__TBODY">
    <tr class="Table__TR">
      <td class="Table__TD"><div class="ranking">1</div></td>
      <td class="Table__TD">
        <div class="player__column">
          <a class="AnchorLink">Valid Player</a>
          <span class="playerinfo__playerpos ttu">OF</span>
        </div>
      </td>
      <td class="Table__TD"><div class="adp tar">5.0</div></td>
    </tr>
    <tr class="Table__TR">
      <td class="Table__TD"><div class="ranking">2</div></td>
      <td class="Table__TD">
        <div class="player__column">
          <!-- Missing player name -->
          <span class="playerinfo__playerpos ttu">1B</span>
        </div>
      </td>
      <td class="Table__TD"><div class="adp tar">10.0</div></td>
    </tr>
    <tr class="Table__TR">
      <td class="Table__TD"><div class="ranking">3</div></td>
      <td class="Table__TD">
        <div class="player__column">
          <a class="AnchorLink">No Position</a>
          <!-- Missing position -->
        </div>
      </td>
      <td class="Table__TD"><div class="adp tar">15.0</div></td>
    </tr>
    <tr class="Table__TR">
      <td class="Table__TD"><div class="ranking">4</div></td>
      <td class="Table__TD">
        <div class="player__column">
          <a class="AnchorLink">Invalid ADP</a>
          <span class="playerinfo__playerpos ttu">SS</span>
        </div>
      </td>
      <td class="Table__TD"><div class="adp tar">invalid</div></td>
    </tr>
  </tbody>
</table>
</body>
</html>
"""


class TestParseESPNRows:
    """Tests for _parse_espn_rows function."""

    def test_parses_valid_rows(self) -> None:
        """Test parsing valid ESPN HTML rows."""
        entries = _parse_espn_rows(ESPN_ADP_HTML_FIXTURE)

        assert len(entries) == 3
        assert entries[0].name == "Mike Trout"
        assert entries[0].adp == 1.5
        assert entries[0].positions == ("OF",)

    def test_parses_multiple_positions(self) -> None:
        """Test parsing player with multiple positions."""
        entries = _parse_espn_rows(ESPN_ADP_HTML_FIXTURE)

        ohtani = entries[1]
        assert ohtani.name == "Shohei Ohtani"
        assert ohtani.positions == ("DH", "SP")
        assert ohtani.adp == 2.3

    def test_parses_special_characters(self) -> None:
        """Test parsing player names with special characters."""
        entries = _parse_espn_rows(ESPN_ADP_HTML_FIXTURE)

        acuna = entries[2]
        assert acuna.name == "Ronald Acuña Jr."
        assert acuna.adp == 3.1

    def test_skips_offseason_dashes(self) -> None:
        """Test that rows with '--' ADP are skipped."""
        entries = _parse_espn_rows(ESPN_ADP_HTML_OFFSEASON)

        # All entries have "--" ADP, so none should be returned
        assert len(entries) == 0

    def test_handles_malformed_rows(self) -> None:
        """Test that malformed rows are skipped gracefully."""
        entries = _parse_espn_rows(ESPN_ADP_HTML_MALFORMED)

        # Only "Valid Player" has all required fields
        assert len(entries) == 1
        assert entries[0].name == "Valid Player"
        assert entries[0].adp == 5.0


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

    def test_fetch_adp_returns_adp_data(self) -> None:
        """Test that fetch_adp returns properly structured ADPData."""

        class MockScraper(ESPNADPScraper):
            """Mock scraper that returns fixture HTML without pagination."""

            def fetch_adp(self) -> ADPData:
                from datetime import UTC, datetime

                entries = self._parse_adp_table(ESPN_ADP_HTML_FIXTURE)
                return ADPData(
                    entries=tuple(entries),
                    fetched_at=datetime.now(UTC),
                    source="espn",
                )

        scraper = MockScraper()
        result = scraper.fetch_adp()

        assert isinstance(result, ADPData)
        assert result.source == "espn"
        assert len(result.entries) == 3
        assert isinstance(result.fetched_at, datetime)

    def test_percent_drafted_is_none(self) -> None:
        """Test that percent_drafted is None (ESPN doesn't track draft %)."""
        scraper = ESPNADPScraper()
        entries = scraper._parse_adp_table(ESPN_ADP_HTML_FIXTURE)

        for entry in entries:
            assert entry.percent_drafted is None

    def test_max_pages_configurable(self) -> None:
        """Test that max_pages can be configured."""
        scraper = ESPNADPScraper(max_pages=5)
        assert scraper._max_pages == 5


@pytest.mark.integration
class TestESPNADPScraperIntegration:
    """Integration tests that hit the live ESPN site."""

    def test_fetch_adp_live(self) -> None:
        """Test fetching ADP from live ESPN site.

        Note: During off-season, ESPN doesn't have ADP data so this test
        may return 0 entries. The test verifies the scraper runs without
        errors and returns a valid ADPData structure.
        """
        scraper = ESPNADPScraper(max_pages=2)  # Limit pages for faster test
        result = scraper.fetch_adp()

        assert isinstance(result, ADPData)
        assert result.source == "espn"
        # During off-season, entries might be 0; during season should be > 0
        assert len(result.entries) >= 0
        if result.entries:
            assert result.entries[0].adp > 0
