from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.league.models import LeagueRosters, RosterPlayer, TeamRoster

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fantasy_baseball_manager.cache.protocol import CacheStore


class CachedPositionSource:
    def __init__(
        self,
        delegate: Any,
        cache: CacheStore,
        cache_key: str,
        ttl_seconds: int,
    ) -> None:
        self._delegate = delegate
        self._cache = cache
        self._cache_key = cache_key
        self._ttl_seconds = ttl_seconds

    def fetch_positions(self) -> dict[str, tuple[str, ...]]:
        cached = self._cache.get("positions", self._cache_key)
        if cached is not None:
            result = _deserialize_positions(cached)
            logger.debug("Cache hit for positions [key=%s] (%d players)", self._cache_key, len(result))
            return result
        logger.debug("Cache miss for positions [key=%s], fetching from source", self._cache_key)
        result = self._delegate.fetch_positions()
        self._cache.put("positions", self._cache_key, _serialize_positions(result), self._ttl_seconds)
        logger.debug("Cached %d positions [key=%s, ttl=%ds]", len(result), self._cache_key, self._ttl_seconds)
        return result


class CachedRosterSource:
    def __init__(
        self,
        delegate: Any,
        cache: CacheStore,
        cache_key: str,
        ttl_seconds: int,
    ) -> None:
        self._delegate = delegate
        self._cache = cache
        self._cache_key = cache_key
        self._ttl_seconds = ttl_seconds

    def fetch_rosters(self) -> LeagueRosters:
        cached = self._cache.get("rosters", self._cache_key)
        if cached is not None:
            result = _deserialize_rosters(cached)
            logger.debug("Cache hit for rosters [key=%s] (%d teams)", self._cache_key, len(result.teams))
            return result
        logger.debug("Cache miss for rosters [key=%s], fetching from source", self._cache_key)
        result = self._delegate.fetch_rosters()
        self._cache.put("rosters", self._cache_key, _serialize_rosters(result), self._ttl_seconds)
        logger.debug("Cached %d teams [key=%s, ttl=%ds]", len(result.teams), self._cache_key, self._ttl_seconds)
        return result


def _serialize_positions(positions: dict[str, tuple[str, ...]]) -> str:
    return json.dumps({pid: list(pos) for pid, pos in positions.items()})


def _deserialize_positions(data: str) -> dict[str, tuple[str, ...]]:
    raw: dict[str, list[str]] = json.loads(data)
    return {pid: tuple(pos) for pid, pos in raw.items()}


def _serialize_rosters(rosters: LeagueRosters) -> str:
    return json.dumps(
        {
            "league_key": rosters.league_key,
            "teams": [
                {
                    "team_key": team.team_key,
                    "team_name": team.team_name,
                    "players": [
                        {
                            "yahoo_id": p.yahoo_id,
                            "name": p.name,
                            "position_type": p.position_type,
                            "eligible_positions": list(p.eligible_positions),
                        }
                        for p in team.players
                    ],
                }
                for team in rosters.teams
            ],
        }
    )


def _deserialize_rosters(data: str) -> LeagueRosters:
    raw: dict[str, Any] = json.loads(data)
    teams: list[TeamRoster] = []
    for team_data in raw["teams"]:
        players: list[RosterPlayer] = []
        for p in team_data["players"]:
            players.append(
                RosterPlayer(
                    yahoo_id=p["yahoo_id"],
                    name=p["name"],
                    position_type=p["position_type"],
                    eligible_positions=tuple(p["eligible_positions"]),
                )
            )
        teams.append(
            TeamRoster(
                team_key=team_data["team_key"],
                team_name=team_data["team_name"],
                players=tuple(players),
            )
        )
    return LeagueRosters(league_key=raw["league_key"], teams=tuple(teams))
