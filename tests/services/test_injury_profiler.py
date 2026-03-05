from __future__ import annotations

from fantasy_baseball_manager.domain import ILStint, Player
from fantasy_baseball_manager.services.injury_profiler import InjuryProfiler, build_profiles


def _stint(
    player_id: int = 1,
    season: int = 2023,
    start_date: str = "2023-05-01",
    il_type: str = "10-day",
    end_date: str | None = None,
    days: int | None = None,
    injury_location: str | None = None,
) -> ILStint:
    return ILStint(
        player_id=player_id,
        season=season,
        start_date=start_date,
        il_type=il_type,
        end_date=end_date,
        days=days,
        injury_location=injury_location,
    )


class TestBuildProfiles:
    def test_empty_stints_returns_empty(self) -> None:
        result = build_profiles([], [2022, 2023, 2024])
        assert result == {}

    def test_empty_seasons_returns_empty(self) -> None:
        result = build_profiles([_stint()], [])
        assert result == {}

    def test_healthy_player_not_in_output(self) -> None:
        """Players with no stints don't appear — caller handles clean profiles."""
        result = build_profiles([], [2022, 2023, 2024])
        assert 1 not in result

    def test_single_incident_player_with_days_field(self) -> None:
        stints = [_stint(days=60, il_type="60-day", injury_location="elbow")]
        result = build_profiles(stints, [2021, 2022, 2023])
        profile = result[1]
        assert profile.total_stints == 1
        assert profile.total_days_lost == 60
        assert profile.avg_days_per_season == 60 / 3
        assert profile.max_days_in_season == 60
        assert profile.pct_seasons_with_il == 1 / 3
        assert profile.injury_locations == {"elbow": 1}

    def test_days_calculation_date_fallback(self) -> None:
        stints = [_stint(start_date="2023-05-01", end_date="2023-05-21", days=None)]
        result = build_profiles(stints, [2023])
        assert result[1].total_days_lost == 20

    def test_days_calculation_il_type_default_fallback(self) -> None:
        stints = [_stint(days=None, end_date=None, il_type="60-day")]
        result = build_profiles(stints, [2023])
        assert result[1].total_days_lost == 60

    def test_days_calculation_unknown_il_type_defaults_to_15(self) -> None:
        stints = [_stint(days=None, end_date=None, il_type="unknown")]
        result = build_profiles(stints, [2023])
        assert result[1].total_days_lost == 15

    def test_chronically_injured_player(self) -> None:
        stints = [
            _stint(season=2021, start_date="2021-04-15", days=15, injury_location="hamstring"),
            _stint(season=2022, start_date="2022-06-01", days=30, injury_location="shoulder"),
            _stint(season=2022, start_date="2022-08-15", days=10, injury_location="hamstring"),
            _stint(season=2023, start_date="2023-05-01", days=20, injury_location="hamstring"),
            _stint(season=2024, start_date="2024-07-01", days=45, injury_location="knee"),
        ]
        seasons = [2021, 2022, 2023, 2024]
        result = build_profiles(stints, seasons)
        profile = result[1]
        assert profile.total_stints == 5
        assert profile.total_days_lost == 120
        assert profile.avg_days_per_season == 30.0
        assert profile.max_days_in_season == 45  # 2024: single 45-day stint
        assert profile.pct_seasons_with_il == 1.0  # all 4 seasons
        assert profile.injury_locations == {"hamstring": 3, "shoulder": 1, "knee": 1}

    def test_recent_stints_only_last_two_seasons(self) -> None:
        stints = [
            _stint(season=2020, start_date="2020-05-01", days=10),
            _stint(season=2021, start_date="2021-05-01", days=10),
            _stint(season=2022, start_date="2022-05-01", days=10),
            _stint(season=2023, start_date="2023-05-01", days=10),
            _stint(season=2024, start_date="2024-05-01", days=10),
        ]
        seasons = [2020, 2021, 2022, 2023, 2024]
        result = build_profiles(stints, seasons)
        recent = result[1].recent_stints
        assert len(recent) == 2
        assert all(s.season in (2023, 2024) for s in recent)

    def test_recent_stints_sorted_by_start_date(self) -> None:
        stints = [
            _stint(season=2024, start_date="2024-08-01", days=5),
            _stint(season=2024, start_date="2024-04-01", days=10),
            _stint(season=2023, start_date="2023-06-15", days=15),
        ]
        seasons = [2023, 2024]
        result = build_profiles(stints, seasons)
        recent = result[1].recent_stints
        assert [s.start_date for s in recent] == ["2023-06-15", "2024-04-01", "2024-08-01"]

    def test_non_consecutive_seasons(self) -> None:
        stints = [
            _stint(season=2020, start_date="2020-05-01", days=30),
            _stint(season=2024, start_date="2024-05-01", days=20),
        ]
        seasons = [2020, 2024]
        result = build_profiles(stints, seasons)
        profile = result[1]
        assert profile.seasons_tracked == 2
        assert profile.total_days_lost == 50
        assert profile.pct_seasons_with_il == 1.0

    def test_multiple_stints_same_season(self) -> None:
        stints = [
            _stint(season=2023, start_date="2023-04-15", days=10),
            _stint(season=2023, start_date="2023-07-01", days=15),
            _stint(season=2023, start_date="2023-09-01", days=10),
        ]
        seasons = [2023]
        result = build_profiles(stints, seasons)
        profile = result[1]
        assert profile.total_stints == 3
        assert profile.total_days_lost == 35
        assert profile.max_days_in_season == 35
        assert profile.pct_seasons_with_il == 1.0

    def test_stints_outside_seasons_excluded(self) -> None:
        stints = [
            _stint(season=2021, start_date="2021-05-01", days=30),
            _stint(season=2023, start_date="2023-05-01", days=20),
        ]
        # Only tracking 2023
        result = build_profiles(stints, [2023])
        profile = result[1]
        assert profile.total_stints == 1
        assert profile.total_days_lost == 20

    def test_multiple_players(self) -> None:
        stints = [
            _stint(player_id=1, season=2023, days=30),
            _stint(player_id=2, season=2023, days=10, start_date="2023-06-01"),
            _stint(player_id=2, season=2023, days=15, start_date="2023-08-01"),
        ]
        result = build_profiles(stints, [2023])
        assert result[1].total_days_lost == 30
        assert result[2].total_days_lost == 25

    def test_injury_location_none_excluded(self) -> None:
        stints = [_stint(injury_location=None, days=10)]
        result = build_profiles(stints, [2023])
        assert result[1].injury_locations == {}

    def test_pct_seasons_with_il_partial(self) -> None:
        stints = [_stint(season=2022, start_date="2022-05-01", days=10)]
        result = build_profiles(stints, [2020, 2021, 2022, 2023, 2024])
        assert result[1].pct_seasons_with_il == 1 / 5


class _FakePlayerRepo:
    """Fake PlayerRepo for testing."""

    def __init__(self, players: list[Player]) -> None:
        self._players = players

    def search_by_name(self, name: str) -> list[Player]:
        name_lower = name.lower()
        return [p for p in self._players if name_lower in f"{p.name_first} {p.name_last}".lower()]

    def get_by_ids(self, player_ids: list[int]) -> list[Player]:
        id_set = set(player_ids)
        return [p for p in self._players if p.id in id_set]

    # Protocol stubs
    def upsert(self, player: Player) -> int:
        raise NotImplementedError

    def get_by_id(self, player_id: int) -> Player | None:
        raise NotImplementedError

    def get_by_mlbam_id(self, mlbam_id: int) -> Player | None:
        raise NotImplementedError

    def get_by_bbref_id(self, bbref_id: str) -> Player | None:
        raise NotImplementedError

    def get_by_last_name(self, last_name: str) -> list[Player]:
        raise NotImplementedError

    def search_by_last_name_normalized(self, last_name: str) -> list[Player]:
        raise NotImplementedError

    def all(self) -> list[Player]:
        raise NotImplementedError


class _FakeILStintRepo:
    """Fake ILStintRepo for testing."""

    def __init__(self, stints: list[ILStint]) -> None:
        self._stints = stints

    def upsert(self, stint: ILStint) -> int:
        raise NotImplementedError

    def get_by_player(self, player_id: int) -> list[ILStint]:
        return [s for s in self._stints if s.player_id == player_id]

    def get_by_player_season(self, player_id: int, season: int) -> list[ILStint]:
        return [s for s in self._stints if s.player_id == player_id and s.season == season]

    def get_by_season(self, season: int) -> list[ILStint]:
        return [s for s in self._stints if s.season == season]


class TestInjuryProfiler:
    def _make_profiler(self, players: list[Player], stints: list[ILStint]) -> InjuryProfiler:
        return InjuryProfiler(
            player_repo=_FakePlayerRepo(players),
            il_stint_repo=_FakeILStintRepo(stints),
        )

    def test_lookup_profile_found(self) -> None:
        player = Player(id=1, name_first="Mike", name_last="Trout", bats="R", birth_date="1991-08-07")
        stints = [_stint(player_id=1, season=2023, days=30, injury_location="knee")]
        profiler = self._make_profiler([player], stints)

        result = profiler.lookup_profile("Trout", [2022, 2023])
        assert result is not None
        profile, name = result
        assert name == "Mike Trout"
        assert profile.total_stints == 1
        assert profile.total_days_lost == 30

    def test_lookup_profile_not_found(self) -> None:
        profiler = self._make_profiler([], [])
        assert profiler.lookup_profile("Nobody", [2023]) is None

    def test_lookup_profile_clean_history(self) -> None:
        player = Player(id=1, name_first="Healthy", name_last="Guy", bats="R", birth_date="1990-01-01")
        profiler = self._make_profiler([player], [])

        result = profiler.lookup_profile("Healthy", [2022, 2023])
        assert result is not None
        profile, name = result
        assert name == "Healthy Guy"
        assert profile.total_stints == 0
        assert profile.total_days_lost == 0

    def test_list_high_risk(self) -> None:
        players = [
            Player(id=1, name_first="Fragile", name_last="Fred", bats="R", birth_date="1990-01-01"),
            Player(id=2, name_first="Somewhat", name_last="Hurt", bats="L", birth_date="1992-01-01"),
            Player(id=3, name_first="Iron", name_last="Man", bats="R", birth_date="1988-01-01"),
        ]
        stints = [
            _stint(player_id=1, season=2023, days=60, start_date="2023-05-01"),
            _stint(player_id=1, season=2023, days=30, start_date="2023-08-01"),
            _stint(player_id=2, season=2023, days=15, start_date="2023-06-01"),
        ]
        profiler = self._make_profiler(players, stints)
        results = profiler.list_high_risk([2023])

        assert len(results) == 2
        assert results[0][1] == "Fragile Fred"
        assert results[0][0].total_days_lost == 90
        assert results[1][1] == "Somewhat Hurt"

    def test_list_high_risk_min_stints_filter(self) -> None:
        players = [
            Player(id=1, name_first="Chronic", name_last="Injury", bats="R", birth_date="1990-01-01"),
            Player(id=2, name_first="One", name_last="Time", bats="L", birth_date="1992-01-01"),
        ]
        stints = [
            _stint(player_id=1, season=2023, days=10, start_date="2023-04-01"),
            _stint(player_id=1, season=2023, days=10, start_date="2023-06-01"),
            _stint(player_id=1, season=2023, days=10, start_date="2023-08-01"),
            _stint(player_id=2, season=2023, days=60, start_date="2023-05-01"),
        ]
        profiler = self._make_profiler(players, stints)

        # min_stints=2 should exclude player 2 who has only 1 stint
        results = profiler.list_high_risk([2023], min_stints=2)
        assert len(results) == 1
        assert results[0][1] == "Chronic Injury"

    def test_list_high_risk_top_n(self) -> None:
        players = [
            Player(id=i, name_first="Player", name_last=str(i), bats="R", birth_date="1990-01-01") for i in range(1, 6)
        ]
        stints = [_stint(player_id=i, season=2023, days=i * 10, start_date=f"2023-0{i}-01") for i in range(1, 6)]
        profiler = self._make_profiler(players, stints)
        results = profiler.list_high_risk([2023], top_n=3)
        assert len(results) == 3
        assert results[0][0].total_days_lost == 50

    def test_estimate_player_games_lost_found(self) -> None:
        player = Player(id=1, name_first="Mike", name_last="Trout", bats="R", birth_date="1991-08-07")
        stints = [
            _stint(player_id=1, season=2023, days=30, injury_location="knee"),
            _stint(player_id=1, season=2024, days=20, start_date="2024-05-01", injury_location="calf"),
        ]
        profiler = self._make_profiler([player], stints)

        result = profiler.estimate_player_games_lost("Trout", [2022, 2023, 2024], projection_season=2026)
        assert result is not None
        estimate, profile, name = result
        assert name == "Mike Trout"
        assert estimate.expected_days_lost > 0
        assert profile.total_stints == 2

    def test_estimate_player_games_lost_not_found(self) -> None:
        profiler = self._make_profiler([], [])
        assert profiler.estimate_player_games_lost("Nobody", [2023], projection_season=2026) is None

    def test_list_games_lost_estimates(self) -> None:
        players = [
            Player(id=1, name_first="Fragile", name_last="Fred", bats="R", birth_date="1990-01-01"),
            Player(id=2, name_first="Somewhat", name_last="Hurt", bats="L", birth_date="1992-01-01"),
        ]
        stints = [
            _stint(player_id=1, season=2023, days=60, start_date="2023-05-01"),
            _stint(player_id=1, season=2024, days=30, start_date="2024-05-01"),
            _stint(player_id=2, season=2024, days=15, start_date="2024-06-01"),
        ]
        profiler = self._make_profiler(players, stints)
        results = profiler.list_games_lost_estimates([2023, 2024], projection_season=2026)

        assert len(results) == 2
        # Sorted by expected_days_lost desc
        assert results[0][2] == "Fragile Fred"
        assert results[0][0].expected_days_lost > results[1][0].expected_days_lost

    def test_list_games_lost_estimates_top_n(self) -> None:
        players = [
            Player(id=1, name_first="A", name_last="Player", bats="R", birth_date="1990-01-01"),
            Player(id=2, name_first="B", name_last="Player", bats="R", birth_date="1990-01-01"),
        ]
        stints = [
            _stint(player_id=1, season=2024, days=50, start_date="2024-05-01"),
            _stint(player_id=2, season=2024, days=20, start_date="2024-06-01"),
        ]
        profiler = self._make_profiler(players, stints)
        results = profiler.list_games_lost_estimates([2024], projection_season=2026, top_n=1)
        assert len(results) == 1
