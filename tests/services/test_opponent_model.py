from __future__ import annotations

import math

from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.domain.league_settings import (
    LeagueFormat,
    LeagueSettings,
)
from fantasy_baseball_manager.services.draft_state import (
    DraftConfig,
    DraftFormat,
    DraftPick,
    DraftState,
)
from fantasy_baseball_manager.services.opponent_model import compute_league_needs


def _row(
    player_id: int,
    name: str,
    position: str,
    player_type: str,
    value: float,
) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=player_id,
        player_type=player_type,
        position=position,
        value=value,
        category_z_scores={},
    )


def _league(teams: int = 4) -> LeagueSettings:
    """Minimal league with C, SS, SP, UTIL, BN slots."""
    return LeagueSettings(
        name="test",
        teams=teams,
        format=LeagueFormat.ROTO,
        budget=260,
        positions={"C": 1, "SS": 1},
        roster_batters=2,
        roster_pitchers=1,
        roster_util=1,
        roster_bench=1,
        pitcher_positions={"SP": 1},
        batting_categories=(),
        pitching_categories=(),
    )


def _config(teams: int = 4) -> DraftConfig:
    return DraftConfig(
        teams=teams,
        roster_slots={"C": 1, "SS": 1, "SP": 1, "UTIL": 1, "BN": 1},
        format=DraftFormat.SNAKE,
        user_team=1,
        season=2026,
    )


def _make_pool(*rows: DraftBoardRow) -> dict[int, DraftBoardRow]:
    return {r.player_id: r for r in rows}


class TestEmptyDraft:
    def test_all_slots_unfilled(self) -> None:
        pool_rows = [
            _row(1, "Catcher A", "C", "batter", 10.0),
            _row(2, "SS A", "SS", "batter", 15.0),
            _row(3, "SP A", "SP", "pitcher", 12.0),
            _row(4, "SS B", "SS", "batter", 8.0),
            _row(5, "C B", "C", "batter", 5.0),
            _row(6, "SP B", "SP", "pitcher", 7.0),
        ]
        state = DraftState(
            config=_config(),
            picks=[],
            available_pool=_make_pool(*pool_rows),
            team_rosters={i: [] for i in range(1, 5)},
            team_budgets={},
        )

        result = compute_league_needs(state, _league())

        assert len(result.teams) == 4
        for team in result.teams:
            assert team.unfilled == {"C": 1, "SS": 1, "SP": 1, "UTIL": 1, "BN": 1}
            assert team.filled == {}
            assert team.total_value == 0.0

        # 4 teams * 1 slot each
        assert result.demand_by_position["C"] == 4
        assert result.demand_by_position["SS"] == 4
        assert result.demand_by_position["SP"] == 4

        # Supply: 2 catchers, 2 shortstops, 2 pitchers
        assert result.supply_by_position["C"] == 2
        assert result.supply_by_position["SS"] == 2
        assert result.supply_by_position["SP"] == 2

        # Scarcity: demand/supply
        assert result.scarcity_ratio["C"] == 4 / 2
        assert result.scarcity_ratio["SS"] == 4 / 2


class TestMidDraft:
    def test_partial_fills(self) -> None:
        pool_rows = [
            _row(3, "SP A", "SP", "pitcher", 12.0),
            _row(4, "SS B", "SS", "batter", 8.0),
            _row(5, "C B", "C", "batter", 5.0),
            _row(6, "SP B", "SP", "pitcher", 7.0),
        ]
        # Team 1 drafted C, team 2 drafted SS
        picks_team1 = [DraftPick(pick_number=1, team=1, player_id=1, player_name="Catcher A", position="C")]
        picks_team2 = [DraftPick(pick_number=2, team=2, player_id=2, player_name="SS A", position="SS")]

        state = DraftState(
            config=_config(),
            picks=picks_team1 + picks_team2,
            available_pool=_make_pool(*pool_rows),
            team_rosters={
                1: picks_team1,
                2: picks_team2,
                3: [],
                4: [],
            },
            team_budgets={},
        )

        result = compute_league_needs(state, _league())

        team1 = next(t for t in result.teams if t.team_idx == 1)
        assert team1.filled == {"C": 1}
        assert "C" not in team1.unfilled
        assert team1.unfilled["SS"] == 1

        team2 = next(t for t in result.teams if t.team_idx == 2)
        assert team2.filled == {"SS": 1}
        assert "SS" not in team2.unfilled
        assert team2.unfilled["C"] == 1

        # C demand: teams 2,3,4 still need one = 3
        assert result.demand_by_position["C"] == 3
        # SS demand: teams 1,3,4 still need one = 3
        assert result.demand_by_position["SS"] == 3


class TestLateDraft:
    def test_scarcity_increases(self) -> None:
        """When most catchers are gone, C scarcity should be high."""
        pool_rows = [
            _row(10, "Last C", "C", "batter", 2.0),
            _row(11, "SP C", "SP", "pitcher", 3.0),
        ]
        # All 4 teams still need C (only 1 left in pool)
        state = DraftState(
            config=_config(),
            picks=[],
            available_pool=_make_pool(*pool_rows),
            team_rosters={i: [] for i in range(1, 5)},
            team_budgets={},
        )

        result = compute_league_needs(state, _league())

        # 4 teams need C, only 1 available
        assert result.demand_by_position["C"] == 4
        assert result.supply_by_position["C"] == 1
        assert result.scarcity_ratio["C"] == 4.0


class TestNoSupply:
    def test_inf_scarcity_when_demand_positive(self) -> None:
        """scarcity_ratio = inf when no supply but positive demand."""
        # Empty pool — no players at all
        state = DraftState(
            config=_config(),
            picks=[],
            available_pool={},
            team_rosters={i: [] for i in range(1, 5)},
            team_budgets={},
        )

        result = compute_league_needs(state, _league())

        assert result.supply_by_position["C"] == 0
        assert result.demand_by_position["C"] == 4
        assert math.isinf(result.scarcity_ratio["C"])

    def test_zero_scarcity_when_no_demand(self) -> None:
        """scarcity_ratio = 0 when no demand and no supply."""
        # All teams have C filled, no catchers in pool
        picks = [DraftPick(pick_number=i, team=i, player_id=i, player_name=f"C{i}", position="C") for i in range(1, 5)]
        state = DraftState(
            config=_config(),
            picks=picks,
            available_pool={},
            team_rosters={i: [picks[i - 1]] for i in range(1, 5)},
            team_budgets={},
        )

        result = compute_league_needs(state, _league())

        assert result.demand_by_position.get("C", 0) == 0
        assert result.supply_by_position["C"] == 0
        assert result.scarcity_ratio["C"] == 0.0


class TestCompositeSlots:
    def test_util_supply_counts_all_batters(self) -> None:
        pool_rows = [
            _row(1, "C A", "C", "batter", 10.0),
            _row(2, "SS A", "SS", "batter", 8.0),
            _row(3, "SP A", "SP", "pitcher", 12.0),
        ]
        state = DraftState(
            config=_config(),
            picks=[],
            available_pool=_make_pool(*pool_rows),
            team_rosters={i: [] for i in range(1, 5)},
            team_budgets={},
        )

        result = compute_league_needs(state, _league())

        # UTIL supply = all batters = C + SS = 2
        assert result.supply_by_position["UTIL"] == 2
        # BN supply = all players = 3
        assert result.supply_by_position["BN"] == 3

    def test_mi_supply(self) -> None:
        """MI slot counts 2B + SS players."""
        league = LeagueSettings(
            name="test",
            teams=2,
            format=LeagueFormat.ROTO,
            budget=260,
            positions={"MI": 1},
            roster_batters=1,
            roster_pitchers=1,
            roster_util=0,
            roster_bench=0,
            pitcher_positions={"SP": 1},
            batting_categories=(),
            pitching_categories=(),
        )
        config = DraftConfig(
            teams=2,
            roster_slots={"MI": 1, "SP": 1},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        pool_rows = [
            _row(1, "2B A", "2B", "batter", 10.0),
            _row(2, "SS A", "SS", "batter", 9.0),
            _row(3, "1B A", "1B", "batter", 8.0),
            _row(4, "SP A", "SP", "pitcher", 7.0),
        ]
        state = DraftState(
            config=config,
            picks=[],
            available_pool=_make_pool(*pool_rows),
            team_rosters={1: [], 2: []},
            team_budgets={},
        )

        result = compute_league_needs(state, league)

        # MI supply = 2B + SS = 2
        assert result.supply_by_position["MI"] == 2

    def test_ci_supply(self) -> None:
        """CI slot counts 1B + 3B players."""
        league = LeagueSettings(
            name="test",
            teams=2,
            format=LeagueFormat.ROTO,
            budget=260,
            positions={"CI": 1},
            roster_batters=1,
            roster_pitchers=1,
            roster_util=0,
            roster_bench=0,
            pitcher_positions={"SP": 1},
            batting_categories=(),
            pitching_categories=(),
        )
        config = DraftConfig(
            teams=2,
            roster_slots={"CI": 1, "SP": 1},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        pool_rows = [
            _row(1, "1B A", "1B", "batter", 10.0),
            _row(2, "3B A", "3B", "batter", 9.0),
            _row(3, "SS A", "SS", "batter", 8.0),
            _row(4, "SP A", "SP", "pitcher", 7.0),
        ]
        state = DraftState(
            config=config,
            picks=[],
            available_pool=_make_pool(*pool_rows),
            team_rosters={1: [], 2: []},
            team_budgets={},
        )

        result = compute_league_needs(state, league)

        # CI supply = 1B + 3B = 2
        assert result.supply_by_position["CI"] == 2


class TestTotalValue:
    def test_sums_from_player_values(self) -> None:
        pool_rows = [_row(3, "SP A", "SP", "pitcher", 12.0)]
        picks_team1 = [
            DraftPick(pick_number=1, team=1, player_id=1, player_name="C A", position="C"),
            DraftPick(pick_number=3, team=1, player_id=2, player_name="SS A", position="SS"),
        ]
        state = DraftState(
            config=_config(),
            picks=picks_team1,
            available_pool=_make_pool(*pool_rows),
            team_rosters={1: picks_team1, 2: [], 3: [], 4: []},
            team_budgets={},
        )

        player_values = {1: 10.0, 2: 15.0, 3: 12.0}
        result = compute_league_needs(state, _league(), player_values=player_values)

        team1 = next(t for t in result.teams if t.team_idx == 1)
        assert team1.total_value == 25.0

    def test_zero_when_no_values(self) -> None:
        state = DraftState(
            config=_config(),
            picks=[],
            available_pool={},
            team_rosters={i: [] for i in range(1, 5)},
            team_budgets={},
        )

        result = compute_league_needs(state, _league())

        for team in result.teams:
            assert team.total_value == 0.0
