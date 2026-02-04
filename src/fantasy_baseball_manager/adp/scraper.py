"""Yahoo ADP scraper using Playwright."""

import re
import unicodedata
from datetime import UTC, datetime
from html.parser import HTMLParser

from playwright.sync_api import sync_playwright

from fantasy_baseball_manager.adp.models import ADPData, ADPEntry

YAHOO_ADP_URL = "https://baseball.fantasysports.yahoo.com/b1/draftanalysis"


class _ADPTableParser(HTMLParser):
    """HTML parser for extracting ADP data from Yahoo's table."""

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[ADPEntry] = []
        self._in_player_row = False
        self._in_player_name = False
        self._in_position_span = False
        self._in_link = False
        self._current_cells: list[str] = []
        self._current_name: str | None = None
        self._current_position: str | None = None
        self._cell_index = 0
        self._capture_text = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)

        class_attr = attrs_dict.get("class") or ""

        if tag == "tr" and "player-row" in class_attr:
            self._in_player_row = True
            self._current_cells = []
            self._current_name = None
            self._current_position = None
            self._cell_index = 0

        elif self._in_player_row and tag == "td":
            self._cell_index += 1
            self._capture_text = True

        elif self._in_player_row and tag == "a":
            self._in_link = True

        elif self._in_player_row and tag == "span" and "position" in class_attr:
            self._in_position_span = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._in_player_row:
            self._finalize_row()
            self._in_player_row = False

        elif tag == "td":
            self._capture_text = False

        elif tag == "a":
            self._in_link = False

        elif tag == "span" and self._in_position_span:
            self._in_position_span = False

    def handle_data(self, data: str) -> None:
        data = data.strip()
        if not data:
            return

        if self._in_link and self._in_player_row:
            self._current_name = data

        elif self._in_position_span:
            self._current_position = data

        elif self._capture_text and self._cell_index in (3, 4):
            self._current_cells.append(data)

    def _finalize_row(self) -> None:
        if not self._current_name or not self._current_position:
            return

        if len(self._current_cells) < 2:
            return

        try:
            adp = float(self._current_cells[0])
        except (ValueError, IndexError):
            return

        percent_drafted: float | None = None
        try:
            pct_str = self._current_cells[1].rstrip("%")
            percent_drafted = float(pct_str)
        except (ValueError, IndexError):
            pass

        position_str = self._current_position.split(" - ")[0]
        positions = tuple(p.strip() for p in position_str.split(","))

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
                page = browser.new_page()
                page.goto(self._url, wait_until="networkidle")
                page.wait_for_selector("table", timeout=10000)
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
