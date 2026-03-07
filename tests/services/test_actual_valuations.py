from fantasy_baseball_manager.domain import Position
from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.model_protocol import EligibilityProvider
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.services.actual_valuations import compute_actual_valuations
from tests.fakes.repos import (
    FakeBattingStatsRepo,
    FakePitchingStatsRepo,
    FakePositionAppearanceRepo,
    FakeValuationRepo,
)


class FakeEligibilityProvider:
    """Fake that returns pre-configured pitcher positions."""

    def __init__(self, pitcher_positions: dict[int, list[str]]) -> None:
        self._pitcher_positions = pitcher_positions

    def get_batter_positions(
        self,
        season: int,
        league: LeagueSettings,
        *,
        min_games: int = 10,
    ) -> dict[int, list[str]]:
        return {}

    def get_pitcher_positions(
        self,
        season: int,
        league: LeagueSettings,
        pitcher_ids: list[int],
    ) -> dict[int, list[str]]:
        return {pid: self._pitcher_positions.get(pid, ["p"]) for pid in pitcher_ids}


assert isinstance(FakeEligibilityProvider({}), EligibilityProvider)


def _counting_league() -> LeagueSettings:
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


class TestComputeActualValuations:
    def test_produces_ranked_valuations_from_actual_stats(self) -> None:
        batting_repo = FakeBattingStatsRepo(
            [
                BattingStats(player_id=1, season=2023, source="fangraphs", pa=500, hr=30, r=80, h=120, bb=50, hbp=5),
                BattingStats(player_id=2, season=2023, source="fangraphs", pa=400, hr=15, r=50, h=100, bb=30, hbp=3),
            ]
        )
        pitching_repo = FakePitchingStatsRepo(
            [
                PitchingStats(player_id=3, season=2023, source="fangraphs", ip=150.0, w=12, sv=0),
            ]
        )
        position_repo = FakePositionAppearanceRepo(
            [
                PositionAppearance(player_id=1, season=2023, position="of", games=100),
                PositionAppearance(player_id=2, season=2023, position="1b", games=80),
            ]
        )
        valuation_repo = FakeValuationRepo()

        result = compute_actual_valuations(
            season=2023,
            league=_counting_league(),
            batting_repo=batting_repo,
            pitching_repo=pitching_repo,
            position_repo=position_repo,
            valuation_repo=valuation_repo,
        )

        assert len(result) == 3
        # All ranked
        ranks = [v.rank for v in result]
        assert ranks == [1, 2, 3]
        # System and projection_system set correctly
        for v in result:
            assert v.system == "zar"
            assert v.projection_system == "actual"
            assert v.projection_version == "2023"
            assert v.season == 2023
        # Persisted
        assert len(valuation_repo.upserted) == 3

    def test_returns_empty_when_no_stats(self) -> None:
        result = compute_actual_valuations(
            season=2023,
            league=_counting_league(),
            batting_repo=FakeBattingStatsRepo(),
            pitching_repo=FakePitchingStatsRepo(),
            position_repo=FakePositionAppearanceRepo(),
            valuation_repo=FakeValuationRepo(),
        )
        assert result == []

    def test_sp_rp_split_when_pitcher_positions_defined(self) -> None:
        """With eligibility_provider and pitcher_positions, pitchers get SP/RP/P."""
        league = LeagueSettings(
            name="Test League",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=2,
            budget=260,
            roster_batters=2,
            roster_pitchers=4,
            batting_categories=(
                CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            ),
            pitching_categories=(
                CategoryConfig(key="w", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
                CategoryConfig(key="sv", name="Saves", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            ),
            roster_util=0,
            positions={"of": 2},
            pitcher_positions={"sp": 2, "rp": 2, "p": 4},
        )
        pitching_repo = FakePitchingStatsRepo(
            [
                PitchingStats(player_id=10, season=2023, source="fangraphs", ip=180.0, w=15, sv=0),
                PitchingStats(player_id=20, season=2023, source="fangraphs", ip=60.0, w=3, sv=25),
            ]
        )
        eligibility = FakeEligibilityProvider(
            pitcher_positions={10: ["sp", "p"], 20: ["rp", "p"]},
        )
        valuation_repo = FakeValuationRepo()

        result = compute_actual_valuations(
            season=2023,
            league=league,
            batting_repo=FakeBattingStatsRepo(),
            pitching_repo=pitching_repo,
            position_repo=FakePositionAppearanceRepo(),
            valuation_repo=valuation_repo,
            eligibility_provider=eligibility,
        )

        pitcher_positions = {v.player_id: v.position for v in result}
        # Pitchers should get SP or RP, not just P
        assert pitcher_positions[10] in ("sp", "p")
        assert pitcher_positions[20] in ("rp", "p")
        # At least one should have a specific position (not all "p")
        all_positions = set(pitcher_positions.values())
        assert all_positions != {"p"}, "Expected SP/RP positions, got all P"

    def test_fallback_to_single_pool_when_no_eligibility_provider(self) -> None:
        """With pitcher_positions defined but no eligibility_provider, all pitchers get P."""
        league = LeagueSettings(
            name="Test League",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=2,
            budget=260,
            roster_batters=2,
            roster_pitchers=4,
            batting_categories=(
                CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            ),
            pitching_categories=(
                CategoryConfig(key="w", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
                CategoryConfig(key="sv", name="Saves", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            ),
            roster_util=0,
            positions={"of": 2},
            pitcher_positions={"sp": 2, "rp": 2, "p": 4},
        )
        pitching_repo = FakePitchingStatsRepo(
            [
                PitchingStats(player_id=10, season=2023, source="fangraphs", ip=180.0, w=15, sv=0),
                PitchingStats(player_id=20, season=2023, source="fangraphs", ip=60.0, w=3, sv=25),
            ]
        )
        valuation_repo = FakeValuationRepo()

        result = compute_actual_valuations(
            season=2023,
            league=league,
            batting_repo=FakeBattingStatsRepo(),
            pitching_repo=pitching_repo,
            position_repo=FakePositionAppearanceRepo(),
            valuation_repo=valuation_repo,
            # No eligibility_provider — should fall back to single P pool
        )

        for v in result:
            assert v.position == Position.P, f"Expected Position.P, got '{v.position}'"

    def test_batters_only_when_no_pitching_stats(self) -> None:
        batting_repo = FakeBattingStatsRepo(
            [
                BattingStats(player_id=1, season=2023, source="fangraphs", pa=500, hr=30, r=80, h=120, bb=50, hbp=5),
            ]
        )
        valuation_repo = FakeValuationRepo()

        result = compute_actual_valuations(
            season=2023,
            league=_counting_league(),
            batting_repo=batting_repo,
            pitching_repo=FakePitchingStatsRepo(),
            position_repo=FakePositionAppearanceRepo(),
            valuation_repo=valuation_repo,
        )

        assert len(result) == 1
        assert result[0].player_type == "batter"
