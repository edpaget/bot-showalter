"""FanGraphs projection source."""

import logging
from datetime import UTC, datetime
from urllib.request import Request, urlopen

import json

from fantasy_baseball_manager.projections.models import (
    BattingProjection,
    PitchingProjection,
    ProjectionData,
    ProjectionSystem,
)

logger = logging.getLogger(__name__)

FANGRAPHS_API_BASE = "https://www.fangraphs.com/api/projections"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class FanGraphsProjectionSource:
    """Fetches projections from FanGraphs API.

    Supports Steamer, ZiPS, and other projection systems available via
    the FanGraphs projections API.
    """

    def __init__(self, system: ProjectionSystem = ProjectionSystem.STEAMER) -> None:
        """Initialize the projection source.

        Args:
            system: The projection system to fetch (default: Steamer).
        """
        self._system = system

    def fetch_projections(self) -> ProjectionData:
        """Fetch projections from FanGraphs.

        Returns:
            ProjectionData containing batting and pitching projections.
        """
        logger.info("Fetching %s projections from FanGraphs", self._system.value)

        batting_json = self._fetch_json(self._batting_url())
        pitching_json = self._fetch_json(self._pitching_url())

        batting = tuple(self._parse_batting(p) for p in batting_json)
        pitching = tuple(self._parse_pitching(p) for p in pitching_json)

        logger.info(
            "Fetched %d batting and %d pitching projections",
            len(batting),
            len(pitching),
        )

        return ProjectionData(
            batting=batting,
            pitching=pitching,
            system=self._system,
            fetched_at=datetime.now(UTC),
        )

    def _batting_url(self) -> str:
        """Construct the batting projections API URL."""
        return (
            f"{FANGRAPHS_API_BASE}"
            f"?type={self._system.value}"
            f"&stats=bat"
            f"&pos=all"
            f"&team=0"
            f"&players=0"
            f"&lg=all"
        )

    def _pitching_url(self) -> str:
        """Construct the pitching projections API URL."""
        return (
            f"{FANGRAPHS_API_BASE}"
            f"?type={self._system.value}"
            f"&stats=pit"
            f"&pos=all"
            f"&team=0"
            f"&players=0"
            f"&lg=all"
        )

    def _fetch_json(self, url: str) -> list[dict]:
        """Fetch JSON data from a URL.

        Args:
            url: The URL to fetch.

        Returns:
            Parsed JSON as a list of dictionaries.
        """
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_batting(self, data: dict) -> BattingProjection:
        """Parse a batting projection from JSON.

        Args:
            data: Raw JSON dictionary for a single player.

        Returns:
            BattingProjection instance.
        """
        mlbam_id = data.get("xMLBAMID")
        return BattingProjection(
            player_id=str(data["playerid"]),
            mlbam_id=str(mlbam_id) if mlbam_id is not None else None,
            name=data["PlayerName"],
            team=data["Team"],
            position=data.get("minpos", ""),
            g=int(data.get("G", 0)),
            pa=int(data.get("PA", 0)),
            ab=int(data.get("AB", 0)),
            h=int(data.get("H", 0)),
            singles=int(data.get("1B", 0)),
            doubles=int(data.get("2B", 0)),
            triples=int(data.get("3B", 0)),
            hr=int(data.get("HR", 0)),
            r=int(data.get("R", 0)),
            rbi=int(data.get("RBI", 0)),
            sb=int(data.get("SB", 0)),
            cs=int(data.get("CS", 0)),
            bb=int(data.get("BB", 0)),
            so=int(data.get("SO", 0)),
            hbp=int(data.get("HBP", 0)),
            sf=int(data.get("SF", 0)),
            sh=int(data.get("SH", 0)),
            obp=float(data.get("OBP", 0.0)),
            slg=float(data.get("SLG", 0.0)),
            ops=float(data.get("OPS", 0.0)),
            woba=float(data.get("wOBA", 0.0)),
            war=float(data.get("WAR", 0.0)),
        )

    def _parse_pitching(self, data: dict) -> PitchingProjection:
        """Parse a pitching projection from JSON.

        Args:
            data: Raw JSON dictionary for a single player.

        Returns:
            PitchingProjection instance.
        """
        mlbam_id = data.get("xMLBAMID")
        return PitchingProjection(
            player_id=str(data["playerid"]),
            mlbam_id=str(mlbam_id) if mlbam_id is not None else None,
            name=data["PlayerName"],
            team=data["Team"],
            g=int(data.get("G", 0)),
            gs=int(data.get("GS", 0)),
            ip=float(data.get("IP", 0.0)),
            w=int(data.get("W", 0)),
            l=int(data.get("L", 0)),
            sv=int(data.get("SV", 0)),
            hld=int(data.get("HLD", 0)),
            so=int(data.get("SO", 0)),
            bb=int(data.get("BB", 0)),
            hbp=int(data.get("HBP", 0)),
            h=int(data.get("H", 0)),
            er=int(data.get("ER", 0)),
            hr=int(data.get("HR", 0)),
            era=float(data.get("ERA", 0.0)),
            whip=float(data.get("WHIP", 0.0)),
            fip=float(data.get("FIP", 0.0)),
            war=float(data.get("WAR", 0.0)),
        )
