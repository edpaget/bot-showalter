from typing import Any

import httpx
import pandas as pd

_BASE_URL = "https://statsapi.mlb.com/api/v1/stats"

_SPORT_IDS: dict[str, int] = {
    "AAA": 11,
    "AA": 12,
    "A+": 13,
    "A": 14,
    "ROK": 16,
}

_COLUMNS = [
    "mlbam_id",
    "season",
    "level",
    "league",
    "team",
    "g",
    "pa",
    "ab",
    "h",
    "doubles",
    "triples",
    "hr",
    "r",
    "rbi",
    "bb",
    "so",
    "sb",
    "cs",
    "avg",
    "obp",
    "slg",
    "age",
    "hbp",
    "sf",
    "sh",
]


class MLBMinorLeagueBattingSource:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client()

    @property
    def source_type(self) -> str:
        return "mlb_api"

    @property
    def source_detail(self) -> str:
        return "milb_batting"

    def fetch(self, **params: Any) -> pd.DataFrame:
        season: int = params["season"]
        level: str = params["level"]
        sport_id = _SPORT_IDS[level]

        response = self._client.get(
            _BASE_URL,
            params={
                "group": "hitting",
                "type": "season",
                "season": season,
                "sportId": sport_id,
                "limit": 5000,
            },
        )
        response.raise_for_status()
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
                    }
                )

        if not rows:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.DataFrame(rows)
