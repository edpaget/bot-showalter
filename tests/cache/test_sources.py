from __future__ import annotations

from fantasy_baseball_manager.cache.sources import (
    CachedDraftResultsSource,
    CachedPositionSource,
    CachedRosterSource,
)
from fantasy_baseball_manager.draft.results import DraftStatus, YahooDraftPick
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


class FakePositionSource:
    def __init__(self, positions: dict[str, tuple[str, ...]]) -> None:
        self._positions = positions
        self.call_count = 0

    def fetch_positions(self) -> dict[str, tuple[str, ...]]:
        self.call_count += 1
        return self._positions


class FakeRosterSource:
    def __init__(self, rosters: LeagueRosters) -> None:
        self._rosters = rosters
        self.call_count = 0

    def fetch_rosters(self) -> LeagueRosters:
        self.call_count += 1
        return self._rosters


class TestCachedPositionSource:
    def test_cache_miss_delegates_and_stores(self) -> None:
        positions = {"p1": ("C", "1B"), "p2": ("OF",)}
        delegate = FakePositionSource(positions)
        cache = FakeCacheStore()
        source = CachedPositionSource(delegate, cache, "test_key", ttl_seconds=300)

        result = source.fetch_positions()

        assert result == positions
        assert delegate.call_count == 1
        assert cache.get("positions", "test_key") is not None

    def test_cache_hit_returns_without_delegating(self) -> None:
        positions = {"p1": ("C", "1B"), "p2": ("OF",)}
        delegate = FakePositionSource(positions)
        cache = FakeCacheStore()
        source = CachedPositionSource(delegate, cache, "test_key", ttl_seconds=300)

        source.fetch_positions()  # populate cache
        result = source.fetch_positions()  # should hit cache

        assert result == positions
        assert delegate.call_count == 1

    def test_round_trip_fidelity(self) -> None:
        positions = {"p1": ("C", "1B", "OF"), "p2": ("SS",), "p3": ()}
        delegate = FakePositionSource(positions)
        cache = FakeCacheStore()
        source = CachedPositionSource(delegate, cache, "k", ttl_seconds=300)

        original = source.fetch_positions()
        # Create new source with same cache to test deserialization
        delegate2 = FakePositionSource({})
        source2 = CachedPositionSource(delegate2, cache, "k", ttl_seconds=300)
        restored = source2.fetch_positions()

        assert restored == original


class TestCachedRosterSource:
    def _sample_rosters(self) -> LeagueRosters:
        return LeagueRosters(
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

    def test_cache_miss_delegates_and_stores(self) -> None:
        rosters = self._sample_rosters()
        delegate = FakeRosterSource(rosters)
        cache = FakeCacheStore()
        source = CachedRosterSource(delegate, cache, "test_key", ttl_seconds=3600)

        result = source.fetch_rosters()

        assert result == rosters
        assert delegate.call_count == 1
        assert cache.get("rosters", "test_key") is not None

    def test_cache_hit_returns_without_delegating(self) -> None:
        rosters = self._sample_rosters()
        delegate = FakeRosterSource(rosters)
        cache = FakeCacheStore()
        source = CachedRosterSource(delegate, cache, "test_key", ttl_seconds=3600)

        source.fetch_rosters()  # populate
        result = source.fetch_rosters()  # cache hit

        assert result == rosters
        assert delegate.call_count == 1

    def test_round_trip_fidelity(self) -> None:
        rosters = self._sample_rosters()
        delegate = FakeRosterSource(rosters)
        cache = FakeCacheStore()
        source = CachedRosterSource(delegate, cache, "k", ttl_seconds=3600)

        original = source.fetch_rosters()
        delegate2 = FakeRosterSource(LeagueRosters(league_key="unused", teams=()))
        source2 = CachedRosterSource(delegate2, cache, "k", ttl_seconds=3600)
        restored = source2.fetch_rosters()

        assert restored == original
        assert restored.league_key == "mlb.l.12345"
        assert len(restored.teams) == 2
        assert restored.teams[0].players[0].eligible_positions == ("C", "1B")


class FakeDraftResultsSource:
    def __init__(self, picks: list[YahooDraftPick]) -> None:
        self._picks = picks
        self.call_count = 0

    def fetch_draft_results(self) -> list[YahooDraftPick]:
        self.call_count += 1
        return self._picks

    def fetch_draft_status(self) -> DraftStatus:
        return DraftStatus.POST_DRAFT

    def fetch_user_team_key(self) -> str:
        return "422.l.12345.t.1"


class TestCachedDraftResultsSource:
    def _sample_picks(self) -> list[YahooDraftPick]:
        return [
            YahooDraftPick(player_id="10660", team_key="422.l.12345.t.1", round=1, pick=1),
            YahooDraftPick(player_id="9542", team_key="422.l.12345.t.2", round=1, pick=2),
        ]

    def test_cache_miss_delegates_and_stores(self) -> None:
        picks = self._sample_picks()
        delegate = FakeDraftResultsSource(picks)
        cache = FakeCacheStore()
        source = CachedDraftResultsSource(delegate, cache, "test_key", ttl_seconds=300)

        result = source.fetch_draft_results()

        assert result == picks
        assert delegate.call_count == 1
        assert cache.get("draft_results", "test_key") is not None

    def test_cache_hit_returns_without_delegating(self) -> None:
        picks = self._sample_picks()
        delegate = FakeDraftResultsSource(picks)
        cache = FakeCacheStore()
        source = CachedDraftResultsSource(delegate, cache, "test_key", ttl_seconds=300)

        source.fetch_draft_results()  # populate cache
        result = source.fetch_draft_results()  # should hit cache

        assert result == picks
        assert delegate.call_count == 1

    def test_round_trip_fidelity(self) -> None:
        picks = self._sample_picks()
        delegate = FakeDraftResultsSource(picks)
        cache = FakeCacheStore()
        source = CachedDraftResultsSource(delegate, cache, "k", ttl_seconds=300)

        original = source.fetch_draft_results()
        delegate2 = FakeDraftResultsSource([])
        source2 = CachedDraftResultsSource(delegate2, cache, "k", ttl_seconds=300)
        restored = source2.fetch_draft_results()

        assert restored == original
        assert len(restored) == 2
        assert restored[0].player_id == "10660"
        assert restored[1].team_key == "422.l.12345.t.2"

    def test_status_passes_through(self) -> None:
        delegate = FakeDraftResultsSource([])
        cache = FakeCacheStore()
        source = CachedDraftResultsSource(delegate, cache, "k", ttl_seconds=300)

        assert source.fetch_draft_status() == DraftStatus.POST_DRAFT

    def test_user_team_key_passes_through(self) -> None:
        delegate = FakeDraftResultsSource([])
        cache = FakeCacheStore()
        source = CachedDraftResultsSource(delegate, cache, "k", ttl_seconds=300)

        assert source.fetch_user_team_key() == "422.l.12345.t.1"
