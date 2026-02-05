"""ESPN ADP scraper using Playwright."""

import contextlib
from datetime import UTC, datetime
from html.parser import HTMLParser

from playwright.sync_api import sync_playwright

from fantasy_baseball_manager.adp.models import ADPData, ADPEntry

ESPN_ADP_URL = "https://fantasy.espn.com/baseball/livedraftresults"


class _ESPNTableParser(HTMLParser):
    """HTML parser for extracting ADP data from ESPN's table.

    ESPN's table structure:
    - Table body: <tbody class="Table__TBODY">
    - Rows: <tr data-idx="N" class="Table__TR">
    - Player name: <span class="player-name">Name</span> or <a class="AnchorLink">Name</a>
    - Columns: RK, PLAYER, POS, ADP, AVG $, %ROST
    """

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[ADPEntry] = []
        self._in_table_body = False
        self._in_player_row = False
        self._in_player_name = False
        self._current_name: str | None = None
        self._current_position: str | None = None
        self._current_adp: float | None = None
        self._cell_index = 0
        self._in_cell = False
        self._cell_text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        class_attr = attrs_dict.get("class") or ""

        # Start of table body
        if tag == "tbody" and "Table__TBODY" in class_attr:
            self._in_table_body = True

        elif self._in_table_body:
            # Start of a player row
            if tag == "tr" and "Table__TR" in class_attr:
                self._in_player_row = True
                self._current_name = None
                self._current_position = None
                self._current_adp = None
                self._cell_index = 0

            elif self._in_player_row:
                # Track cell boundaries
                if tag == "td":
                    self._cell_index += 1
                    self._in_cell = True
                    self._cell_text = ""

                # Player name - look for common ESPN patterns
                elif tag == "span" and "player-name" in class_attr:
                    self._in_player_name = True
                elif tag == "a" and "AnchorLink" in class_attr and self._cell_index == 2:
                    # ESPN sometimes uses anchor links for player names
                    self._in_player_name = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "tbody":
            self._in_table_body = False

        elif tag == "tr" and self._in_player_row:
            self._finalize_row()
            self._in_player_row = False

        elif (tag == "span" or tag == "a") and self._in_player_name:
            self._in_player_name = False

        elif tag == "td" and self._in_cell:
            self._process_cell()
            self._in_cell = False

    def handle_data(self, data: str) -> None:
        data = data.strip()
        if not data:
            return

        if self._in_player_name:
            self._current_name = data

        elif self._in_cell and self._in_player_row:
            # Accumulate cell text
            if self._cell_text:
                self._cell_text += " "
            self._cell_text += data

    def _process_cell(self) -> None:
        """Process accumulated cell text based on column index."""
        text = self._cell_text.strip()
        if not text:
            return

        # Cell 3: Position
        if self._cell_index == 3:
            self._current_position = text

        # Cell 4: ADP
        elif self._cell_index == 4:
            with contextlib.suppress(ValueError):
                self._current_adp = float(text)

    def _finalize_row(self) -> None:
        if not self._current_name or not self._current_position or self._current_adp is None:
            return

        # Parse positions (ESPN uses ", " as separator)
        positions = tuple(p.strip() for p in self._current_position.split(","))

        entry = ADPEntry(
            name=self._current_name,
            adp=self._current_adp,
            positions=positions,
            percent_drafted=None,  # ESPN %ROST is roster % not draft %
        )
        self.entries.append(entry)


class ESPNADPScraper:
    """Scraper for ESPN Fantasy Baseball ADP data.

    Uses Playwright to render the React-based ESPN ADP page and extracts
    player ADP information.
    """

    def __init__(self, url: str = ESPN_ADP_URL) -> None:
        """Initialize the scraper.

        Args:
            url: The ESPN ADP page URL.
        """
        self._url = url

    def fetch_adp(self) -> ADPData:
        """Fetch ADP data from ESPN.

        Returns:
            ADPData containing all player ADP entries.
        """
        html = self._fetch_page_html()
        entries = self._parse_adp_table(html)

        return ADPData(
            entries=tuple(entries),
            fetched_at=datetime.now(UTC),
            source="espn",
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
                page.goto(self._url, wait_until="domcontentloaded", timeout=60000)
                # Wait for table data to load
                page.wait_for_selector("table tbody tr", timeout=30000)
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
        parser = _ESPNTableParser()
        parser.feed(html)
        return parser.entries
