from __future__ import annotations

from fantasy_baseball_manager.domain import InjuryValueDelta, PlayerValuation
from fantasy_baseball_manager.domain.il_stint import ILStint
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.services.injury_profiler import InjuryProfiler
from fantasy_baseball_manager.services.injury_valuation import (
    compute_injury_adjusted_deltas,
    compute_injury_adjusted_valuations_list,
)
from fantasy_baseball_manager.services.player_eligibility import PlayerEligibilityService
from tests.fakes.repos import (
    FakePlayerRepo,
    FakePositionAppearanceRepo,
    FakeProjectionRepo,
    FakeValuationRepo,
)


def _league() -> LeagueSettings:
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=2,
        budget=260,
        roster_batters=2,
        roster_pitchers=1,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="r", name="Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="sv", name="Saves", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        roster_util=1,
        positions={"of": 1},
    )


def _projections() -> list[Projection]:
    return [
        Projection(
            player_id=1,
            season=2025,
            system="steamer",
            version="v1",
            player_type="batter",
            stat_json={"pa": 600, "hr": 40.0, "r": 100.0},
        ),
        Projection(
            player_id=2,
            season=2025,
            system="steamer",
            version="v1",
            player_type="batter",
            stat_json={"pa": 500, "hr": 20.0, "r": 70.0},
        ),
        Projection(
            player_id=10,
            season=2025,
            system="steamer",
            version="v1",
            player_type="batter",
            stat_json={"pa": 450, "hr": 15.0, "r": 60.0},
        ),
        Projection(
            player_id=11,
            season=2025,
            system="steamer",
            version="v1",
            player_type="batter",
            stat_json={"pa": 400, "hr": 12.0, "r": 50.0},
        ),
        Projection(
            player_id=12,
            season=2025,
            system="steamer",
            version="v1",
            player_type="batter",
            stat_json={"pa": 350, "hr": 8.0, "r": 40.0},
        ),
        Projection(
            player_id=3,
            season=2025,
            system="steamer",
            version="v1",
            player_type="pitcher",
            stat_json={"ip": 200, "w": 15.0, "sv": 0.0},
        ),
        Projection(
            player_id=13,
            season=2025,
            system="steamer",
            version="v1",
            player_type="pitcher",
            stat_json={"ip": 100, "w": 6.0, "sv": 20.0},
        ),
    ]


def _players() -> list[Player]:
    return [
        Player(id=1, name_first="Mike", name_last="Trout"),
        Player(id=2, name_first="Aaron", name_last="Judge"),
        Player(id=3, name_first="Gerrit", name_last="Cole"),
    ]


def _appearances() -> list[PositionAppearance]:
    return [
        PositionAppearance(player_id=1, season=2025, position="of", games=100),
        PositionAppearance(player_id=2, season=2025, position="of", games=150),
    ]


def _il_stints() -> list[ILStint]:
    """Give player 1 some IL history so they get an injury discount."""
    return [
        ILStint(player_id=1, season=2023, start_date="2023-05-01", il_type="10-day", days=30),
        ILStint(player_id=1, season=2024, start_date="2024-06-01", il_type="10-day", days=45),
    ]


class FakeILStintRepo:
    def __init__(self, stints: list[ILStint] | None = None) -> None:
        self._stints = stints or []

    def upsert(self, stint: ILStint) -> int:
        return 1

    def get_by_player(self, player_id: int) -> list[ILStint]:
        return [s for s in self._stints if s.player_id == player_id]

    def get_by_player_season(self, player_id: int, season: int) -> list[ILStint]:
        return [s for s in self._stints if s.player_id == player_id and s.season == season]

    def get_by_season(self, season: int) -> list[ILStint]:
        return [s for s in self._stints if s.season == season]


def _build_deps(
    stints: list[ILStint] | None = None,
) -> tuple[
    FakeProjectionRepo,
    FakePlayerRepo,
    FakePositionAppearanceRepo,
    FakeValuationRepo,
    PlayerEligibilityService,
    InjuryProfiler,
]:
    projection_repo = FakeProjectionRepo(_projections())
    player_repo = FakePlayerRepo(_players())
    position_repo = FakePositionAppearanceRepo(_appearances())
    valuation_repo = FakeValuationRepo()
    eligibility_service = PlayerEligibilityService(position_repo)
    il_stint_repo = FakeILStintRepo(stints or _il_stints())
    profiler = InjuryProfiler(player_repo=player_repo, il_stint_repo=il_stint_repo)
    return projection_repo, player_repo, position_repo, valuation_repo, eligibility_service, profiler


class TestComputeInjuryAdjustedValuationsList:
    def test_returns_player_valuations(self) -> None:
        projection_repo, player_repo, _, _, eligibility_service, profiler = _build_deps()
        result = compute_injury_adjusted_valuations_list(
            season=2025,
            league=_league(),
            projection_system="steamer",
            projection_version=None,
            season_list=[2023, 2024, 2025],
            profiler=profiler,
            projection_repo=projection_repo,
            player_repo=player_repo,
            eligibility_service=eligibility_service,
        )
        assert len(result) > 0
        assert all(isinstance(v, PlayerValuation) for v in result)

    def test_valuations_have_injury_adjusted_version(self) -> None:
        projection_repo, player_repo, _, _, eligibility_service, profiler = _build_deps()
        result = compute_injury_adjusted_valuations_list(
            season=2025,
            league=_league(),
            projection_system="steamer",
            projection_version=None,
            season_list=[2023, 2024, 2025],
            profiler=profiler,
            projection_repo=projection_repo,
            player_repo=player_repo,
            eligibility_service=eligibility_service,
        )
        for v in result:
            assert v.version == "injury-adjusted"

    def test_player_names_are_resolved(self) -> None:
        projection_repo, player_repo, _, _, eligibility_service, profiler = _build_deps()
        result = compute_injury_adjusted_valuations_list(
            season=2025,
            league=_league(),
            projection_system="steamer",
            projection_version=None,
            season_list=[2023, 2024, 2025],
            profiler=profiler,
            projection_repo=projection_repo,
            player_repo=player_repo,
            eligibility_service=eligibility_service,
        )
        names = {v.player_name for v in result}
        assert "Mike Trout" in names

    def test_no_il_stints_returns_unmodified_valuations(self) -> None:
        projection_repo, player_repo, _, _, eligibility_service, profiler = _build_deps(stints=[])
        result = compute_injury_adjusted_valuations_list(
            season=2025,
            league=_league(),
            projection_system="steamer",
            projection_version=None,
            season_list=[2023, 2024, 2025],
            profiler=profiler,
            projection_repo=projection_repo,
            player_repo=player_repo,
            eligibility_service=eligibility_service,
        )
        # Should still return valid valuations even with no injury data
        assert len(result) > 0


class TestComputeInjuryAdjustedDeltas:
    def _seed_original_valuations(self, valuation_repo: FakeValuationRepo) -> None:
        """Seed valuations that match our projection players."""
        valuation_repo.upsert(
            Valuation(
                player_id=1,
                season=2025,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="batter",
                position="of",
                value=30.0,
                rank=1,
                category_scores={"hr": 1.5},
            )
        )
        valuation_repo.upsert(
            Valuation(
                player_id=2,
                season=2025,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="batter",
                position="of",
                value=20.0,
                rank=2,
                category_scores={"hr": 0.5},
            )
        )
        valuation_repo.upsert(
            Valuation(
                player_id=3,
                season=2025,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="pitcher",
                position="p",
                value=15.0,
                rank=3,
                category_scores={"w": 1.0},
            )
        )

    def test_returns_deltas_for_matching_players(self) -> None:
        projection_repo, player_repo, _, valuation_repo, eligibility_service, profiler = _build_deps()
        self._seed_original_valuations(valuation_repo)

        deltas = compute_injury_adjusted_deltas(
            season=2025,
            league=_league(),
            projection_system="steamer",
            projection_version=None,
            season_list=[2023, 2024, 2025],
            profiler=profiler,
            projection_repo=projection_repo,
            player_repo=player_repo,
            valuation_repo=valuation_repo,
            eligibility_service=eligibility_service,
        )
        assert len(deltas) > 0
        assert all(isinstance(d, InjuryValueDelta) for d in deltas)

    def test_deltas_sorted_ascending_by_value_delta(self) -> None:
        projection_repo, player_repo, _, valuation_repo, eligibility_service, profiler = _build_deps()
        self._seed_original_valuations(valuation_repo)

        deltas = compute_injury_adjusted_deltas(
            season=2025,
            league=_league(),
            projection_system="steamer",
            projection_version=None,
            season_list=[2023, 2024, 2025],
            profiler=profiler,
            projection_repo=projection_repo,
            player_repo=player_repo,
            valuation_repo=valuation_repo,
            eligibility_service=eligibility_service,
        )
        values = [d.value_delta for d in deltas]
        assert values == sorted(values)

    def test_no_original_valuations_returns_empty(self) -> None:
        projection_repo, player_repo, _, valuation_repo, eligibility_service, profiler = _build_deps()
        # Don't seed any valuations

        deltas = compute_injury_adjusted_deltas(
            season=2025,
            league=_league(),
            projection_system="steamer",
            projection_version=None,
            season_list=[2023, 2024, 2025],
            profiler=profiler,
            projection_repo=projection_repo,
            player_repo=player_repo,
            valuation_repo=valuation_repo,
            eligibility_service=eligibility_service,
        )
        assert deltas == []

    def test_injured_player_has_lower_adjusted_value(self) -> None:
        projection_repo, player_repo, _, _, eligibility_service, profiler = _build_deps()

        # Compute non-injured valuations first
        no_injury = compute_injury_adjusted_valuations_list(
            season=2025,
            league=_league(),
            projection_system="steamer",
            projection_version=None,
            season_list=[2023, 2024, 2025],
            profiler=InjuryProfiler(
                player_repo=player_repo,
                il_stint_repo=FakeILStintRepo([]),
            ),
            projection_repo=projection_repo,
            player_repo=player_repo,
            eligibility_service=eligibility_service,
        )
        # Compute injury-adjusted valuations
        with_injury = compute_injury_adjusted_valuations_list(
            season=2025,
            league=_league(),
            projection_system="steamer",
            projection_version=None,
            season_list=[2023, 2024, 2025],
            profiler=profiler,
            projection_repo=projection_repo,
            player_repo=player_repo,
            eligibility_service=eligibility_service,
        )

        baseline = {v.player_name: v.value for v in no_injury}
        adjusted = {v.player_name: v.value for v in with_injury}
        # Trout has IL history, so injury-adjusted value should be lower
        assert adjusted["Mike Trout"] < baseline["Mike Trout"]
