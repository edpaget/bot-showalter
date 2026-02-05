"""Yahoo ADP scraper using Playwright."""

import contextlib
import re
import unicodedata
from datetime import UTC, datetime
from html.parser import HTMLParser

from playwright.sync_api import sync_playwright

from fantasy_baseball_manager.adp.models import ADPData, ADPEntry

YAHOO_ADP_URL = "https://baseball.fantasysports.yahoo.com/b1/draftanalysis?count=300"


class _ADPTableParser(HTMLParser):
    """HTML parser for extracting ADP data from Yahoo's table.

    Yahoo's table structure:
    - Rows: <tr data-tst="table-row-N">
    - Player name: <div data-tst="player-name">Name</div>
    - Position: <span data-tst="player-position">POS</span>
    - Columns: Player, Rank, PosRank, CER, %Drafted, Preseason ADP, All Drafts ADP, ...
    """

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[ADPEntry] = []
        self._in_player_row = False
        self._in_player_name = False
        self._in_player_position = False
        self._current_name: str | None = None
        self._current_position: str | None = None
        self._cell_index = 0
        self._cell_values: list[tuple[int, str]] = []
        self._in_cell_div = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        data_tst = attrs_dict.get("data-tst") or ""

        # Start of a player row
        if tag == "tr" and data_tst.startswith("table-row-"):
            self._in_player_row = True
            self._current_name = None
            self._current_position = None
            self._cell_index = 0
            self._cell_values = []

        elif self._in_player_row:
            # Track cell boundaries
            if tag == "td":
                self._cell_index += 1
                self._in_cell_div = False

            # Player name div
            elif tag == "div" and data_tst == "player-name":
                self._in_player_name = True

            # Position span
            elif tag == "span" and data_tst == "player-position":
                self._in_player_position = True

            # Cell content div (for numeric values like %Drafted, ADP)
            elif tag == "div" and self._cell_index >= 5:
                self._in_cell_div = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._in_player_row:
            self._finalize_row()
            self._in_player_row = False

        elif tag == "div" and self._in_player_name:
            self._in_player_name = False

        elif tag == "span" and self._in_player_position:
            self._in_player_position = False

        elif tag == "div":
            self._in_cell_div = False

    def handle_data(self, data: str) -> None:
        data = data.strip()
        if not data:
            return

        if self._in_player_name:
            self._current_name = data

        elif self._in_player_position:
            self._current_position = data

        elif self._in_cell_div and self._in_player_row and self._cell_index >= 5:
            # Capture cell values for %Drafted (col 5), Preseason ADP (col 6), All Drafts ADP (col 7)
            self._cell_values.append((self._cell_index, data))

    def _finalize_row(self) -> None:
        if not self._current_name or not self._current_position:
            return

        # Extract values by column index
        values_by_col: dict[int, str] = {col: val for col, val in self._cell_values}

        # Get ADP - prefer "All Drafts" (col 7), fall back to "Preseason" (col 6)
        adp_str = values_by_col.get(7) or values_by_col.get(6)
        if not adp_str:
            return

        try:
            adp = float(adp_str)
        except ValueError:
            return

        # Get %Drafted (col 5)
        percent_drafted: float | None = None
        pct_str = values_by_col.get(5)
        if pct_str:
            with contextlib.suppress(ValueError):
                percent_drafted = float(pct_str.rstrip("%"))

        # Parse position
        positions = tuple(p.strip() for p in self._current_position.split(","))

        entry = ADPEntry(
            name=self._current_name,
            adp=adp,
            positions=positions,
            percent_drafted=percent_drafted,
        )
        self.entries.append(entry)


class YahooADPScraper:
    """Scraper for Yahoo Fantasy Baseball ADP data.

    Uses Playwright to render the React-based Yahoo ADP page and extracts
    player ADP information.
    """

    def __init__(self, url: str = YAHOO_ADP_URL) -> None:
        """Initialize the scraper.

        Args:
            url: The Yahoo ADP page URL.
        """
        self._url = url

    def fetch_adp(self) -> ADPData:
        """Fetch ADP data from Yahoo.

        Returns:
            ADPData containing all player ADP entries.
        """
        html = self._fetch_page_html()
        entries = self._parse_adp_table(html)

        return ADPData(
            entries=tuple(entries),
            fetched_at=datetime.now(UTC),
            source="yahoo",
        )

    def _fetch_page_html(self) -> str:
        """Fetch the page HTML using Playwright.

        Returns:
            The rendered HTML content.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(
                    user_agent=(
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                )
                # Use domcontentloaded instead of networkidle - Yahoo has persistent
                # analytics requests that prevent networkidle from ever triggering
                page.goto(self._url, wait_until="domcontentloaded", timeout=60000)
                # Wait for actual player data to load (not just skeleton/loading state)
                # The table initially has data-tst="loading" and skeleton rows
                # Wait for player links to appear in the table body
                page.wait_for_selector("table tbody a", timeout=30000)
                return page.content()
            finally:
                browser.close()

    def _parse_adp_table(self, html: str) -> list[ADPEntry]:
        """Parse ADP entries from HTML.

        Args:
            html: The page HTML content.

        Returns:
            List of ADPEntry objects.
        """
        parser = _ADPTableParser()
        parser.feed(html)
        return parser.entries

    def _normalize_name(self, name: str) -> str:
        """Normalize a player name for matching.

        - Converts to lowercase
        - Removes accents/diacritics
        - Removes periods from Jr./Sr./III

        Args:
            name: The player name.

        Returns:
            Normalized name string.
        """
        normalized = unicodedata.normalize("NFD", name)
        normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
        normalized = normalized.lower()
        normalized = re.sub(r"\.", "", normalized)
        return normalized
