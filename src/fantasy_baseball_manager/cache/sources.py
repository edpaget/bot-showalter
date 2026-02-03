from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.draft.results import DraftStatus, YahooDraftPick
from fantasy_baseball_manager.league.models import LeagueRosters, RosterPlayer, TeamRoster

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fantasy_baseball_manager.cache.protocol import CacheStore


def _cached_fetch(
    cache: CacheStore,
    namespace: str,
    cache_key: str,
    ttl_seconds: int,
    fetch_fn: Callable[[], Any],
    serialize: Callable[[Any], str],
    deserialize: Callable[[str], Any],
    count_fn: Callable[[Any], int],
    count_label: str,
) -> Any:
    """Shared cache-or-fetch logic for all cached sources."""
    cached = cache.get(namespace, cache_key)
    if cached is not None:
        result = deserialize(cached)
        logger.debug("Cache hit for %s [key=%s] (%d %s)", namespace, cache_key, count_fn(result), count_label)
        return result
    logger.debug("Cache miss for %s [key=%s], fetching from source", namespace, cache_key)
    result = fetch_fn()
    cache.put(namespace, cache_key, serialize(result), ttl_seconds)
    logger.debug("Cached %d %s [key=%s, ttl=%ds]", count_fn(result), count_label, cache_key, ttl_seconds)
    return result


class CachedPositionSource:
    def __init__(self, delegate: Any, cache: CacheStore, cache_key: str, ttl_seconds: int) -> None:
        self._delegate = delegate
        self._cache = cache
        self._cache_key = cache_key
        self._ttl_seconds = ttl_seconds

    def fetch_positions(self) -> dict[str, tuple[str, ...]]:
        return _cached_fetch(
            self._cache, "positions", self._cache_key, self._ttl_seconds,
            self._delegate.fetch_positions,
            _serialize_positions, _deserialize_positions, len, "players",
        )


class CachedRosterSource:
    def __init__(self, delegate: Any, cache: CacheStore, cache_key: str, ttl_seconds: int) -> None:
        self._delegate = delegate
        self._cache = cache
        self._cache_key = cache_key
        self._ttl_seconds = ttl_seconds

    def fetch_rosters(self) -> LeagueRosters:
        return _cached_fetch(
            self._cache, "rosters", self._cache_key, self._ttl_seconds,
            self._delegate.fetch_rosters,
            _serialize_rosters, _deserialize_rosters, lambda r: len(r.teams), "teams",
        )


class CachedDraftResultsSource:
    def __init__(self, delegate: Any, cache: CacheStore, cache_key: str, ttl_seconds: int) -> None:
        self._delegate = delegate
        self._cache = cache
        self._cache_key = cache_key
        self._ttl_seconds = ttl_seconds

    def fetch_draft_results(self) -> list[YahooDraftPick]:
        return _cached_fetch(
            self._cache, "draft_results", self._cache_key, self._ttl_seconds,
            self._delegate.fetch_draft_results,
            _serialize_draft_results, _deserialize_draft_results, len, "draft picks",
        )

    def fetch_draft_status(self) -> DraftStatus:
        return self._delegate.fetch_draft_status()

    def fetch_user_team_key(self) -> str:
        return self._delegate.fetch_user_team_key()


# Serializers for each cached type

def _serialize_positions(positions: dict[str, tuple[str, ...]]) -> str:
    return json.dumps({pid: list(pos) for pid, pos in positions.items()})


def _deserialize_positions(data: str) -> dict[str, tuple[str, ...]]:
    raw: dict[str, list[str]] = json.loads(data)
    return {pid: tuple(pos) for pid, pos in raw.items()}


def _serialize_rosters(rosters: LeagueRosters) -> str:
    return json.dumps({
        "league_key": rosters.league_key,
        "teams": [
            {
                "team_key": t.team_key,
                "team_name": t.team_name,
                "players": [
                    {"yahoo_id": p.yahoo_id, "name": p.name, "position_type": p.position_type,
                     "eligible_positions": list(p.eligible_positions)}
                    for p in t.players
                ],
            }
            for t in rosters.teams
        ],
    })


def _deserialize_rosters(data: str) -> LeagueRosters:
    raw: dict[str, Any] = json.loads(data)
    return LeagueRosters(
        league_key=raw["league_key"],
        teams=tuple(
            TeamRoster(
                team_key=t["team_key"],
                team_name=t["team_name"],
                players=tuple(
                    RosterPlayer(
                        yahoo_id=p["yahoo_id"], name=p["name"],
                        position_type=p["position_type"], eligible_positions=tuple(p["eligible_positions"]),
                    )
                    for p in t["players"]
                ),
            )
            for t in raw["teams"]
        ),
    )


def _serialize_draft_results(picks: list[YahooDraftPick]) -> str:
    return json.dumps([
        {"player_id": p.player_id, "team_key": p.team_key, "round": p.round, "pick": p.pick}
        for p in picks
    ])


def _deserialize_draft_results(data: str) -> list[YahooDraftPick]:
    raw: list[dict[str, object]] = json.loads(data)
    return [
        YahooDraftPick(
            player_id=str(i["player_id"]), team_key=str(i["team_key"]),
            round=int(i["round"]), pick=int(i["pick"]),
        )
        for i in raw
    ]
