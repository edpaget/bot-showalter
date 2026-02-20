import logging
import re
from html.parser import HTMLParser
from typing import Any

from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)

_URL = "https://www.fantasypros.com/mlb/adp/overall.php"
_TABLE_SELECTOR = "#data"
_NAV_TIMEOUT_MS = 60_000
_TABLE_TIMEOUT_MS = 30_000


class _TableParser(HTMLParser):
    """Parse an HTML table into a list of header strings and a list of row-lists."""

    def __init__(self) -> None:
        super().__init__()
        self.headers: list[str] = []
        self.rows: list[list[str]] = []
        self._in_thead = False
        self._in_tbody = False
        self._in_th = False
        self._in_td = False
        self._current_row: list[str] = []
        self._current_cell: str = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "thead":
            self._in_thead = True
        elif tag == "tbody":
            self._in_tbody = True
        elif tag == "th" and self._in_thead:
            self._in_th = True
            self._current_cell = ""
        elif tag == "td" and self._in_tbody:
            self._in_td = True
            self._current_cell = ""
        elif tag == "tr" and self._in_tbody:
            self._current_row = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "thead":
            self._in_thead = False
        elif tag == "tbody":
            self._in_tbody = False
        elif tag == "th" and self._in_th:
            self._in_th = False
            self.headers.append(self._current_cell.strip())
        elif tag == "td" and self._in_td:
            self._in_td = False
            self._current_row.append(self._current_cell.strip())
        elif tag == "tr" and self._in_tbody and self._current_row:
            self.rows.append(self._current_row)
            self._current_row = []

    def handle_data(self, data: str) -> None:
        if self._in_th:
            self._current_cell += data
        elif self._in_td:
            self._current_cell += data


def _parse_table_html(html: str) -> list[dict[str, Any]]:
    """Parse an HTML table string into row dicts matching the FantasyPros CSV format."""
    parser = _TableParser()
    parser.feed(html)

    if not parser.headers:
        msg = "No header row found in table HTML"
        raise RuntimeError(msg)

    result: list[dict[str, Any]] = []
    for row_values in parser.rows:
        row_dict: dict[str, Any] = {}
        for i, header in enumerate(parser.headers):
            row_dict[header] = row_values[i] if i < len(row_values) else ""
        result.append(row_dict)

    return result


_PLAYER_TEAM_RE = re.compile(r"^(.+?)\s*\((\w+)\s*-\s*(.+)\)$")


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split 'Player (Team)' column into separate Player, Team, Positions columns."""
    if not rows or "Player (Team)" not in rows[0]:
        return rows

    result: list[dict[str, Any]] = []
    for row in rows:
        new_row: dict[str, Any] = {}
        for key, value in row.items():
            if key == "Player (Team)":
                m = _PLAYER_TEAM_RE.match(value)
                if m:
                    new_row["Player"] = m.group(1).strip()
                    new_row["Team"] = m.group(2).strip()
                    new_row["Positions"] = m.group(3).strip()
                else:
                    new_row["Player"] = value
                    new_row["Team"] = ""
                    new_row["Positions"] = ""
            else:
                new_row[key] = value
        result.append(new_row)
    return result


class FantasyProsADPSource:
    """Fetch live ADP data from FantasyPros via headless Chromium."""

    @property
    def source_type(self) -> str:
        return "fantasypros_web"

    @property
    def source_detail(self) -> str:
        return _URL

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        logger.debug("Fetching ADP data from %s", _URL)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(_URL, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
                locator = page.locator(_TABLE_SELECTOR)
                locator.wait_for(timeout=_TABLE_TIMEOUT_MS)
                html = locator.inner_html()
            finally:
                browser.close()

        rows = _parse_table_html(f"<table>{html}</table>")
        rows = _normalize_rows(rows)
        logger.debug("Parsed %d rows from FantasyPros ADP page", len(rows))
        return rows
