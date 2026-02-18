import logging
from typing import Any

import httpx
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger(__name__)


def _log_retry(retry_state: RetryCallState) -> None:
    logger.warning("Retrying MLB API request (attempt %d): %s", retry_state.attempt_number, retry_state.outcome)


_BASE_URL = "https://statsapi.mlb.com/api/v1/stats"

_SPORT_IDS: dict[str, int] = {
    "AAA": 11,
    "AA": 12,
    "A+": 13,
    "A": 14,
    "ROK": 16,
}


class MLBMinorLeagueBattingSource:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0))

    @property
    def source_type(self) -> str:
        return "mlb_api"

    @property
    def source_detail(self) -> str:
        return "milb_batting"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _fetch_with_retry(self, params: dict[str, Any]) -> httpx.Response:
        response = self._client.get(_BASE_URL, params=params)
        response.raise_for_status()
        return response

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        season: int = params["season"]
        level: str = params["level"]
        sport_id = _SPORT_IDS[level]

        logger.debug("GET %s season=%d level=%s", _BASE_URL, season, level)
        response = self._fetch_with_retry(
            {
                "group": "hitting",
                "stats": "season",
                "season": season,
                "sportId": sport_id,
                "limit": 5000,
            }
        )
        logger.debug("MLB API responded %d", response.status_code)
        data = response.json()

        rows: list[dict[str, Any]] = []
        stats_list = data.get("stats", [])
        if stats_list:
            for split in stats_list[0].get("splits", []):
                player = split.get("player", {})
                stat = split.get("stat", {})
                team = split.get("team", {})
                league = split.get("league", {})

                rows.append(
                    {
                        "mlbam_id": player.get("id"),
                        "season": season,
                        "level": level,
                        "league": league.get("name", ""),
                        "team": team.get("name", ""),
                        "g": stat.get("gamesPlayed", 0),
                        "pa": stat.get("plateAppearances", 0),
                        "ab": stat.get("atBats", 0),
                        "h": stat.get("hits", 0),
                        "doubles": stat.get("doubles", 0),
                        "triples": stat.get("triples", 0),
                        "hr": stat.get("homeRuns", 0),
                        "r": stat.get("runs", 0),
                        "rbi": stat.get("rbi", 0),
                        "bb": stat.get("baseOnBalls", 0),
                        "so": stat.get("strikeOuts", 0),
                        "sb": stat.get("stolenBases", 0),
                        "cs": stat.get("caughtStealing", 0),
                        "avg": float(stat.get("avg", "0")),
                        "obp": float(stat.get("obp", "0")),
                        "slg": float(stat.get("slg", "0")),
                        "age": player.get("currentAge", 0),
                        "hbp": stat.get("hitByPitch"),
                        "sf": stat.get("sacFlies"),
                        "sh": stat.get("sacBunts"),
                        "first_name": player.get("firstName", ""),
                        "last_name": player.get("lastName", ""),
                    }
                )

        if not rows:
            return []
        logger.info("Fetched %d MiLB batting rows for %s %d", len(rows), level, season)
        return rows
