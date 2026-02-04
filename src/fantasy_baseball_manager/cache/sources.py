from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypedDict, TypeVar

from fantasy_baseball_manager.adp.models import ADPData, ADPEntry
from fantasy_baseball_manager.draft.results import DraftStatus, YahooDraftPick
from fantasy_baseball_manager.league.models import LeagueRosters, RosterPlayer, TeamRoster

logger = logging.getLogger(__name__)


class _RosterPlayerDict(TypedDict):
    yahoo_id: str
    name: str
    position_type: str
    eligible_positions: list[str]


class _TeamRosterDict(TypedDict):
    team_key: str
    team_name: str
    players: list[_RosterPlayerDict]


class _LeagueRostersDict(TypedDict):
    league_key: str
    teams: list[_TeamRosterDict]


class _DraftPickDict(TypedDict):
    player_id: str
    team_key: str
    round: int
    pick: int


class _ADPEntryDict(TypedDict):
    name: str
    adp: float
    positions: list[str]
    percent_drafted: float | None


class _ADPDataDict(TypedDict):
    entries: list[_ADPEntryDict]
    fetched_at: str
    source: str


if TYPE_CHECKING:
    from fantasy_baseball_manager.cache.protocol import CacheStore

_T = TypeVar("_T")


class PositionSource(Protocol):
    """Protocol for sources that provide player position data."""

    def fetch_positions(self) -> dict[str, tuple[str, ...]]: ...


class RosterSource(Protocol):
    """Protocol for sources that provide league roster data."""

    def fetch_rosters(self) -> LeagueRosters: ...


class DraftResultsSource(Protocol):
    """Protocol for sources that provide draft results data."""

    def fetch_draft_results(self) -> list[YahooDraftPick]: ...
    def fetch_draft_status(self) -> DraftStatus: ...
    def fetch_user_team_key(self) -> str: ...


def _cached_fetch(
    cache: CacheStore,
    namespace: str,
    cache_key: str,
    ttl_seconds: int,
    fetch_fn: Callable[[], _T],
    serialize: Callable[[_T], str],
    deserialize: Callable[[str], _T],
    count_fn: Callable[..., int],
    count_label: str,
) -> _T:
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
    def __init__(self, delegate: PositionSource, cache: CacheStore, cache_key: str, ttl_seconds: int) -> None:
        self._delegate = delegate
        self._cache = cache
        self._cache_key = cache_key
        self._ttl_seconds = ttl_seconds

    def fetch_positions(self) -> dict[str, tuple[str, ...]]:
        return _cached_fetch(
            self._cache,
            "positions",
            self._cache_key,
            self._ttl_seconds,
            self._delegate.fetch_positions,
            _serialize_positions,
            _deserialize_positions,
            len,
            "players",
        )


class CachedRosterSource:
    def __init__(self, delegate: RosterSource, cache: CacheStore, cache_key: str, ttl_seconds: int) -> None:
        self._delegate = delegate
        self._cache = cache
        self._cache_key = cache_key
        self._ttl_seconds = ttl_seconds

    def fetch_rosters(self) -> LeagueRosters:
        return _cached_fetch(
            self._cache,
            "rosters",
            self._cache_key,
            self._ttl_seconds,
            self._delegate.fetch_rosters,
            _serialize_rosters,
            _deserialize_rosters,
            lambda r: len(r.teams),
            "teams",
        )


class CachedDraftResultsSource:
    def __init__(self, delegate: DraftResultsSource, cache: CacheStore, cache_key: str, ttl_seconds: int) -> None:
        self._delegate = delegate
        self._cache = cache
        self._cache_key = cache_key
        self._ttl_seconds = ttl_seconds

    def fetch_draft_results(self) -> list[YahooDraftPick]:
        return _cached_fetch(
            self._cache,
            "draft_results",
            self._cache_key,
            self._ttl_seconds,
            self._delegate.fetch_draft_results,
            _serialize_draft_results,
            _deserialize_draft_results,
            len,
            "draft picks",
        )

    def fetch_draft_status(self) -> DraftStatus:
        return self._delegate.fetch_draft_status()

    def fetch_user_team_key(self) -> str:
        return self._delegate.fetch_user_team_key()


class ADPSource(Protocol):
    """Protocol for sources that provide ADP data."""

    def fetch_adp(self) -> ADPData: ...


class CachedADPSource:
    """Cached wrapper for ADP data sources."""

    def __init__(
        self,
        delegate: ADPSource,
        cache: CacheStore,
        cache_key: str = "yahoo",
        ttl_seconds: int = 86400,
    ) -> None:
        self._delegate = delegate
        self._cache = cache
        self._cache_key = cache_key
        self._ttl_seconds = ttl_seconds

    def fetch_adp(self) -> ADPData:
        return _cached_fetch(
            self._cache,
            "adp_data",
            self._cache_key,
            self._ttl_seconds,
            self._delegate.fetch_adp,
            _serialize_adp_data,
            _deserialize_adp_data,
            lambda d: len(d.entries),
            "ADP entries",
        )


# Serializers for each cached type


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
                    "team_key": t.team_key,
                    "team_name": t.team_name,
                    "players": [
                        {
                            "yahoo_id": p.yahoo_id,
                            "name": p.name,
                            "position_type": p.position_type,
                            "eligible_positions": list(p.eligible_positions),
                        }
                        for p in t.players
                    ],
                }
                for t in rosters.teams
            ],
        }
    )


def _deserialize_rosters(data: str) -> LeagueRosters:
    raw: _LeagueRostersDict = json.loads(data)
    return LeagueRosters(
        league_key=raw["league_key"],
        teams=tuple(
            TeamRoster(
                team_key=t["team_key"],
                team_name=t["team_name"],
                players=tuple(
                    RosterPlayer(
                        yahoo_id=p["yahoo_id"],
                        name=p["name"],
                        position_type=p["position_type"],
                        eligible_positions=tuple(p["eligible_positions"]),
                    )
                    for p in t["players"]
                ),
            )
            for t in raw["teams"]
        ),
    )


def _serialize_draft_results(picks: list[YahooDraftPick]) -> str:
    return json.dumps(
        [{"player_id": p.player_id, "team_key": p.team_key, "round": p.round, "pick": p.pick} for p in picks]
    )


def _deserialize_draft_results(data: str) -> list[YahooDraftPick]:
    raw: list[_DraftPickDict] = json.loads(data)
    return [
        YahooDraftPick(
            player_id=i["player_id"],
            team_key=i["team_key"],
            round=i["round"],
            pick=i["pick"],
        )
        for i in raw
    ]


def _serialize_adp_data(adp_data: ADPData) -> str:
    return json.dumps(
        {
            "entries": [
                {
                    "name": e.name,
                    "adp": e.adp,
                    "positions": list(e.positions),
                    "percent_drafted": e.percent_drafted,
                }
                for e in adp_data.entries
            ],
            "fetched_at": adp_data.fetched_at.isoformat(),
            "source": adp_data.source,
        }
    )


def _deserialize_adp_data(data: str) -> ADPData:
    from datetime import datetime

    raw: _ADPDataDict = json.loads(data)
    return ADPData(
        entries=tuple(
            ADPEntry(
                name=e["name"],
                adp=e["adp"],
                positions=tuple(e["positions"]),
                percent_drafted=e["percent_drafted"],
            )
            for e in raw["entries"]
        ),
        fetched_at=datetime.fromisoformat(raw["fetched_at"]),
        source=raw["source"],
    )
