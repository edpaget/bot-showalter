"""ESPN ADP scraper using Playwright."""

import logging
import re
from datetime import UTC, datetime

from playwright.sync_api import Page, sync_playwright

from fantasy_baseball_manager.adp.models import ADPData, ADPEntry

ESPN_ADP_URL = "https://fantasy.espn.com/baseball/livedraftresults"

logger = logging.getLogger(__name__)


def _parse_espn_rows(html: str) -> list[ADPEntry]:
    """Parse ADP entries from ESPN HTML.

    ESPN's table structure (as of 2025):
    - Table body: <tbody class="Table__TBODY">
    - Each row has cells with specific classes:
      - "ranking": Rank number
      - "player__column": Contains player name and position
      - "adp": Average Draft Position
      - "pcadp": ADP percent change
      - "avc": Auction value
      - "own": Ownership %

    Player name is in: <a class="AnchorLink ...">Name</a>
    Position is in: <span class="playerinfo__playerpos ...">POS</span>
    ADP is in: <div class="...adp...">value</div>

    Args:
        html: The page HTML content.

    Returns:
        List of ADPEntry objects.
    """
    entries: list[ADPEntry] = []

    # Find all table rows
    row_pattern = re.compile(r'<tr[^>]*class="[^"]*Table__TR[^"]*"[^>]*>(.*?)</tr>', re.DOTALL)

    for row_match in row_pattern.finditer(html):
        row_html = row_match.group(1)

        # Extract player name from AnchorLink in player column
        # Pattern: <a class="AnchorLink link clr-link pointer" ...>Player Name</a>
        name_match = re.search(
            r'<a[^>]*class="[^"]*AnchorLink[^"]*"[^>]*>([^<]+)</a>',
            row_html,
        )
        if not name_match:
            continue
        name = name_match.group(1).strip()

        # Extract position from playerinfo__playerpos span
        # Pattern: <span class="playerinfo__playerpos ttu">DH, SP</span>
        pos_match = re.search(
            r'<span[^>]*class="[^"]*playerinfo__playerpos[^"]*"[^>]*>([^<]+)</span>',
            row_html,
        )
        if not pos_match:
            continue
        position_str = pos_match.group(1).strip()
        positions = tuple(p.strip() for p in position_str.split(","))

        # Extract ADP from the cell with class "adp"
        # Pattern: <div ... class="...adp tar...">....</div>
        # The ADP value might be a number or "--"
        adp_match = re.search(
            r'<div[^>]*class="[^"]*\badp\b[^"]*"[^>]*>([^<]*)</div>',
            row_html,
        )
        if not adp_match:
            continue

        adp_text = adp_match.group(1).strip()
        if adp_text == "--" or not adp_text:
            # No ADP data available (off-season)
            continue

        try:
            adp = float(adp_text)
        except ValueError:
            continue

        entries.append(
            ADPEntry(
                name=name,
                adp=adp,
                positions=positions,
                percent_drafted=None,
            )
        )

    return entries


class ESPNADPScraper:
    """Scraper for ESPN Fantasy Baseball ADP data.

    Uses Playwright to render the React-based ESPN ADP page and extracts
    player ADP information. Handles pagination to fetch all players.

    Note: ESPN's "Live Draft Results" page only has ADP data when drafts
    are actively happening (typically closer to and during the season).
    During the off-season, ADP values will be "--" and no entries will
    be returned.
    """

    def __init__(self, url: str = ESPN_ADP_URL, max_pages: int = 10) -> None:
        """Initialize the scraper.

        Args:
            url: The ESPN ADP page URL.
            max_pages: Maximum number of pages to scrape.
        """
        self._url = url
        self._max_pages = max_pages

    def fetch_adp(self) -> ADPData:
        """Fetch ADP data from ESPN.

        Returns:
            ADPData containing all player ADP entries.
        """
        all_entries: list[ADPEntry] = []

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
                page.wait_for_selector("table tbody tr", timeout=30000)

                # Parse first page
                html = page.content()
                entries = self._parse_adp_table(html)
                all_entries.extend(entries)
                logger.debug("Page 1: found %d entries", len(entries))

                # Handle pagination
                for page_num in range(2, self._max_pages + 1):
                    if not self._go_to_next_page(page, page_num):
                        break
                    html = page.content()
                    entries = self._parse_adp_table(html)
                    if not entries:
                        break
                    all_entries.extend(entries)
                    logger.debug("Page %d: found %d entries", page_num, len(entries))

            finally:
                browser.close()

        logger.info("ESPN scraper: fetched %d total entries", len(all_entries))

        return ADPData(
            entries=tuple(all_entries),
            fetched_at=datetime.now(UTC),
            source="espn",
        )

    def _go_to_next_page(self, page: Page, page_num: int) -> bool:
        """Navigate to the next page of results.

        Args:
            page: Playwright page object.
            page_num: The page number to navigate to.

        Returns:
            True if navigation succeeded, False if no more pages.
        """
        try:
            # ESPN uses pagination with numbered links
            # Look for the pagination item with the target page number
            selector = f'.Pagination__list__item:has-text("{page_num}")'
            pagination_item = page.query_selector(selector)

            if not pagination_item:
                logger.debug("No pagination item found for page %d", page_num)
                return False

            # Check if it's already active (current page)
            class_attr = pagination_item.get_attribute("class") or ""
            if "active" in class_attr:
                logger.debug("Page %d is already active", page_num)
                return False

            # Click the pagination item
            pagination_item.click()

            # Wait for table to update
            page.wait_for_timeout(1000)  # Brief wait for React to update
            page.wait_for_selector("table tbody tr", timeout=10000)

            return True

        except Exception as e:
            logger.debug("Failed to navigate to page %d: %s", page_num, e)
            return False

    def _parse_adp_table(self, html: str) -> list[ADPEntry]:
        """Parse ADP entries from HTML.

        Args:
            html: The page HTML content.

        Returns:
            List of ADPEntry objects.
        """
        return _parse_espn_rows(html)
