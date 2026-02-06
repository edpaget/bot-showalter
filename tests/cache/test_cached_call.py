"""Tests for cached_call() and new serializers."""

from __future__ import annotations

from fantasy_baseball_manager.cache.serialization import (
    DataclassListSerializer,
    LeagueRostersSerializer,
    PositionDictSerializer,
)
from fantasy_baseball_manager.draft.results import YahooDraftPick
from fantasy_baseball_manager.league.models import LeagueRosters, RosterPlayer, TeamRoster


class FakeCacheStore:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], str] = {}

    def get(self, namespace: str, key: str) -> str | None:
        return self._data.get((namespace, key))

    def put(self, namespace: str, key: str, value: str, ttl_seconds: int) -> None:
        self._data[(namespace, key)] = value

    def invalidate(self, namespace: str, key: str | None = None) -> None:
        if key is not None:
            self._data.pop((namespace, key), None)
        else:
            to_remove = [k for k in self._data if k[0] == namespace]
            for k in to_remove:
                del self._data[k]


class TestCachedCall:
    def test_cache_miss_delegates_and_stores(self) -> None:
        from fantasy_baseball_manager.cache.wrapper import cached_call

        store = FakeCacheStore()
        calls: list[int] = []

        def fetch() -> dict[str, tuple[str, ...]]:
            calls.append(1)
            return {"p1": ("C", "1B"), "p2": ("OF",)}

        result = cached_call(
            fetch,
            store=store,
            namespace="positions",
            cache_key="test_key",
            ttl_seconds=300,
            serializer=PositionDictSerializer(),
        )

        assert result == {"p1": ("C", "1B"), "p2": ("OF",)}
        assert len(calls) == 1
        assert store.get("positions", "test_key") is not None

    def test_cache_hit_returns_without_delegating(self) -> None:
        from fantasy_baseball_manager.cache.wrapper import cached_call

        store = FakeCacheStore()
        calls: list[int] = []

        def fetch() -> dict[str, tuple[str, ...]]:
            calls.append(1)
            return {"p1": ("C", "1B"), "p2": ("OF",)}

        serializer = PositionDictSerializer()

        cached_call(
            fetch,
            store=store,
            namespace="positions",
            cache_key="test_key",
            ttl_seconds=300,
            serializer=serializer,
        )  # populate cache
        result = cached_call(
            fetch,
            store=store,
            namespace="positions",
            cache_key="test_key",
            ttl_seconds=300,
            serializer=serializer,
        )  # should hit cache

        assert result == {"p1": ("C", "1B"), "p2": ("OF",)}
        assert len(calls) == 1

    def test_round_trip_with_complex_type(self) -> None:
        from fantasy_baseball_manager.cache.wrapper import cached_call

        store = FakeCacheStore()
        rosters = LeagueRosters(
            league_key="mlb.l.123",
            teams=(
                TeamRoster(
                    team_key="mlb.l.123.t.1",
                    team_name="Team One",
                    players=(
                        RosterPlayer(
                            yahoo_id="1001",
                            name="Player A",
                            position_type="B",
                            eligible_positions=("C", "1B"),
                        ),
                    ),
                ),
            ),
        )

        cached_call(
            lambda: rosters,
            store=store,
            namespace="rosters",
            cache_key="k",
            ttl_seconds=3600,
            serializer=LeagueRostersSerializer(),
        )

        # Fetch from cache with a different fetch_fn (should not be called)
        sentinel = LeagueRosters(league_key="unused", teams=())
        result = cached_call(
            lambda: sentinel,
            store=store,
            namespace="rosters",
            cache_key="k",
            ttl_seconds=3600,
            serializer=LeagueRostersSerializer(),
        )

        assert result == rosters
        assert result.teams[0].players[0].eligible_positions == ("C", "1B")


class TestPositionDictSerializer:
    def test_round_trip(self) -> None:
        serializer = PositionDictSerializer()
        original = {"p1": ("C", "1B", "OF"), "p2": ("SS",), "p3": ()}

        data = serializer.serialize(original)
        restored = serializer.deserialize(data)

        assert restored == original

    def test_empty_dict(self) -> None:
        serializer = PositionDictSerializer()
        original: dict[str, tuple[str, ...]] = {}

        data = serializer.serialize(original)
        restored = serializer.deserialize(data)

        assert restored == original


class TestLeagueRostersSerializer:
    def test_round_trip(self) -> None:
        serializer = LeagueRostersSerializer()
        original = LeagueRosters(
            league_key="mlb.l.12345",
            teams=(
                TeamRoster(
                    team_key="mlb.l.12345.t.1",
                    team_name="Team One",
                    players=(
                        RosterPlayer(
                            yahoo_id="1001",
                            name="Player A",
                            position_type="B",
                            eligible_positions=("C", "1B"),
                        ),
                        RosterPlayer(
                            yahoo_id="1002",
                            name="Player B",
                            position_type="P",
                            eligible_positions=("SP",),
                        ),
                    ),
                ),
                TeamRoster(
                    team_key="mlb.l.12345.t.2",
                    team_name="Team Two",
                    players=(),
                ),
            ),
        )

        data = serializer.serialize(original)
        restored = serializer.deserialize(data)

        assert restored == original
        assert restored.league_key == "mlb.l.12345"
        assert len(restored.teams) == 2
        assert restored.teams[0].players[0].eligible_positions == ("C", "1B")

    def test_empty_teams(self) -> None:
        serializer = LeagueRostersSerializer()
        original = LeagueRosters(league_key="empty", teams=())

        data = serializer.serialize(original)
        restored = serializer.deserialize(data)

        assert restored == original


class TestDraftPickListSerializer:
    def test_round_trip(self) -> None:
        serializer = DataclassListSerializer(YahooDraftPick)
        original = [
            YahooDraftPick(player_id="10660", team_key="422.l.12345.t.1", round=1, pick=1),
            YahooDraftPick(player_id="9542", team_key="422.l.12345.t.2", round=1, pick=2),
        ]

        data = serializer.serialize(original)
        restored = serializer.deserialize(data)

        assert restored == original
        assert restored[0].player_id == "10660"
        assert restored[1].team_key == "422.l.12345.t.2"
