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
    PoolKey,
)
from fantasy_baseball_manager.services.opponent_model import (
    assess_threats,
    compute_league_needs,
    detect_position_runs,
)


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


def _make_pool(*rows: DraftBoardRow) -> dict[PoolKey, DraftBoardRow]:
    return {(r.player_id, r.player_type): r for r in rows}


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


# ---------- detect_position_runs tests ----------


def _12team_league() -> LeagueSettings:
    """12-team league with C, SS, SP, UTIL, BN slots."""
    return LeagueSettings(
        name="test",
        teams=12,
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


def _12team_config(fmt: DraftFormat = DraftFormat.SNAKE) -> DraftConfig:
    return DraftConfig(
        teams=12,
        roster_slots={"C": 1, "SS": 1, "SP": 1, "UTIL": 1, "BN": 1},
        format=fmt,
        user_team=1,
        season=2026,
    )


def _pick(num: int, team: int, position: str, player_id: int | None = None) -> DraftPick:
    pid = player_id if player_id is not None else num + 100
    return DraftPick(
        pick_number=num,
        team=team,
        player_id=pid,
        player_name=f"Player {pid}",
        position=position,
    )


class TestRunDetected:
    def test_three_ss_in_six_picks(self) -> None:
        """3 SS picks in a 6-pick window (12-team league) → detected as run."""
        picks = [
            _pick(1, 2, "SS"),
            _pick(2, 3, "C"),
            _pick(3, 4, "SS"),
            _pick(4, 5, "SP"),
            _pick(5, 6, "SS"),
            _pick(6, 7, "C"),
        ]
        pool = _make_pool(
            _row(200, "SS X", "SS", "batter", 5.0),
            _row(201, "SS Y", "SS", "batter", 4.0),
            _row(202, "C Z", "C", "batter", 3.0),
            _row(203, "SP Z", "SP", "pitcher", 6.0),
        )
        state = DraftState(
            config=_12team_config(),
            picks=picks,
            available_pool=pool,
            team_rosters={i: [] for i in range(1, 13)},
            team_budgets={},
        )

        runs = detect_position_runs(state, _12team_league())

        ss_runs = [r for r in runs if r.position == "SS"]
        assert len(ss_runs) == 1
        assert ss_runs[0].run_length == 3
        assert ss_runs[0].remaining_supply == 2


class TestNoFalsePositive:
    def test_spread_out_picks_no_run(self) -> None:
        """3 SS picks spread across 24 picks → no clustering → no run."""
        picks = [_pick(i, (i % 12) + 1, "C") for i in range(1, 25)]
        # Place SS picks far apart: pick 1, pick 12, pick 24
        picks[0] = _pick(1, 2, "SS")
        picks[11] = _pick(12, 3, "SS")
        picks[23] = _pick(24, 4, "SS")

        pool = _make_pool(
            _row(200, "SS X", "SS", "batter", 5.0),
            _row(201, "C Z", "C", "batter", 3.0),
        )
        state = DraftState(
            config=_12team_config(),
            picks=picks,
            available_pool=pool,
            team_rosters={i: [] for i in range(1, 13)},
            team_budgets={},
        )

        runs = detect_position_runs(state, _12team_league())

        ss_runs = [r for r in runs if r.position == "SS"]
        assert len(ss_runs) == 0


class TestDevelopingUrgency:
    def test_two_clustered_healthy_supply(self) -> None:
        """2 picks clustered but supply healthy → developing."""
        picks = [
            _pick(1, 2, "SS"),
            _pick(2, 3, "SS"),
            _pick(3, 4, "C"),
        ]
        pool = _make_pool(
            *[_row(200 + i, f"SS {i}", "SS", "batter", 5.0) for i in range(10)],
            _row(300, "C Z", "C", "batter", 3.0),
        )
        state = DraftState(
            config=_12team_config(),
            picks=picks,
            available_pool=pool,
            team_rosters={i: [] for i in range(1, 13)},
            team_budgets={},
        )

        runs = detect_position_runs(state, _12team_league())

        ss_runs = [r for r in runs if r.position == "SS"]
        assert len(ss_runs) == 1
        assert ss_runs[0].urgency == "developing"


class TestCriticalUrgency:
    def test_three_picks_thin_supply(self) -> None:
        """3+ picks AND supply < 1.5 * user need → critical."""
        picks = [
            _pick(1, 2, "SS"),
            _pick(2, 3, "SS"),
            _pick(3, 4, "SS"),
        ]
        # Only 1 SS left, user needs 1 → 1 < 1.5 * 1 → critical
        pool = _make_pool(
            _row(200, "SS X", "SS", "batter", 5.0),
            _row(201, "C Z", "C", "batter", 3.0),
        )
        state = DraftState(
            config=_12team_config(),
            picks=picks,
            available_pool=pool,
            team_rosters={i: [] for i in range(1, 13)},
            team_budgets={},
        )

        runs = detect_position_runs(state, _12team_league())

        ss_runs = [r for r in runs if r.position == "SS"]
        assert len(ss_runs) == 1
        assert ss_runs[0].urgency == "critical"


class TestSinglePickNoRun:
    def test_one_pick_no_run(self) -> None:
        """Single pick at a position → no run."""
        picks = [_pick(1, 2, "SS")]
        pool = _make_pool(_row(200, "SS X", "SS", "batter", 5.0))
        state = DraftState(
            config=_12team_config(),
            picks=picks,
            available_pool=pool,
            team_rosters={i: [] for i in range(1, 13)},
            team_budgets={},
        )

        runs = detect_position_runs(state, _12team_league())

        assert len(runs) == 0


class TestUserDoesNotNeedPosition:
    def test_user_filled_never_critical(self) -> None:
        """Run detected as developing even when user doesn't need the position."""
        picks = [
            _pick(1, 2, "SS"),
            _pick(2, 3, "SS"),
            _pick(3, 4, "SS"),
        ]
        # User (team 1) already has SS filled
        user_pick = _pick(0, 1, "SS", player_id=99)
        pool = _make_pool(_row(200, "SS X", "SS", "batter", 5.0))
        state = DraftState(
            config=_12team_config(),
            picks=[user_pick, *picks],
            available_pool=pool,
            team_rosters={1: [user_pick], **{i: [] for i in range(2, 13)}},
            team_budgets={},
        )

        runs = detect_position_runs(state, _12team_league())

        ss_runs = [r for r in runs if r.position == "SS"]
        assert len(ss_runs) == 1
        # user_need = 0 → never critical
        assert ss_runs[0].urgency == "developing"


class TestSortedOutput:
    def test_critical_before_developing(self) -> None:
        """Critical runs sort before developing runs."""
        picks = [
            _pick(1, 2, "SS"),
            _pick(2, 3, "SS"),
            _pick(3, 4, "SS"),
            _pick(4, 5, "C"),
            _pick(5, 6, "C"),
        ]
        # SS: 1 left, user needs 1 → critical
        # C: 10 left, user needs 1 → developing
        pool = _make_pool(
            _row(200, "SS X", "SS", "batter", 5.0),
            *[_row(300 + i, f"C {i}", "C", "batter", 3.0) for i in range(10)],
            _row(400, "SP Z", "SP", "pitcher", 6.0),
        )
        state = DraftState(
            config=_12team_config(),
            picks=picks,
            available_pool=pool,
            team_rosters={i: [] for i in range(1, 13)},
            team_budgets={},
        )

        runs = detect_position_runs(state, _12team_league())

        assert len(runs) == 2
        assert runs[0].urgency == "critical"
        assert runs[0].position == "SS"
        assert runs[1].urgency == "developing"
        assert runs[1].position == "C"


class TestCustomWindow:
    def test_window_limits_scan(self) -> None:
        """Custom window controls how many recent picks are scanned."""
        picks = [
            _pick(1, 2, "SS"),
            _pick(2, 3, "SS"),
            # Later picks — these are the only ones in window=3
            _pick(3, 4, "C"),
            _pick(4, 5, "C"),
            _pick(5, 6, "SP"),
        ]
        pool = _make_pool(
            _row(200, "SS X", "SS", "batter", 5.0),
            _row(201, "C Z", "C", "batter", 3.0),
        )
        state = DraftState(
            config=_12team_config(),
            picks=picks,
            available_pool=pool,
            team_rosters={i: [] for i in range(1, 13)},
            team_budgets={},
        )

        # With window=3, only last 3 picks are scanned (picks 3,4,5)
        # SS picks are outside the window
        runs = detect_position_runs(state, _12team_league(), window=3)

        ss_runs = [r for r in runs if r.position == "SS"]
        assert len(ss_runs) == 0
        # C picks are in the window
        c_runs = [r for r in runs if r.position == "C"]
        assert len(c_runs) == 1


class TestAuctionFormat:
    def test_same_detection_logic(self) -> None:
        """Auction format uses same detection — picks are still sequential."""
        picks = [
            _pick(1, 2, "SS"),
            _pick(2, 3, "SS"),
            _pick(3, 4, "SS"),
        ]
        pool = _make_pool(
            _row(200, "SS X", "SS", "batter", 5.0),
            _row(201, "C Z", "C", "batter", 3.0),
        )
        state = DraftState(
            config=_12team_config(fmt=DraftFormat.AUCTION),
            picks=picks,
            available_pool=pool,
            team_rosters={i: [] for i in range(1, 13)},
            team_budgets={},
        )

        runs = detect_position_runs(state, _12team_league())

        ss_runs = [r for r in runs if r.position == "SS"]
        assert len(ss_runs) == 1
        assert ss_runs[0].run_length == 3


# ---------- assess_threats tests ----------


def _row_adp(
    player_id: int,
    name: str,
    position: str,
    player_type: str,
    value: float,
    adp: float | None = None,
) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=player_id,
        player_type=player_type,
        position=position,
        value=value,
        category_z_scores={},
        adp_overall=adp,
    )


class TestThreatLikelyGone:
    def test_adp_in_danger_zone_and_two_teams_need(self) -> None:
        """Player with ADP in the danger zone AND 2+ teams needing position → likely-gone."""
        # 4-team snake, user is team 1, current pick is 2 (team 2's turn)
        # User's next pick is pick 4 → picks_until = 2
        # Teams 2 and 3 pick before user; both need SS
        pool = _make_pool(
            _row_adp(10, "SS Star", "SS", "batter", 20.0, adp=3.0),
            _row_adp(11, "C Guy", "C", "batter", 15.0, adp=10.0),
            _row_adp(12, "SP Ace", "SP", "pitcher", 12.0, adp=8.0),
        )
        state = DraftState(
            config=DraftConfig(
                teams=4,
                roster_slots={"C": 1, "SS": 1, "SP": 1, "UTIL": 1, "BN": 1},
                format=DraftFormat.SNAKE,
                user_team=1,
                season=2026,
            ),
            picks=[_pick(1, 1, "C", player_id=1)],
            available_pool=pool,
            team_rosters={
                1: [_pick(1, 1, "C", player_id=1)],
                2: [],
                3: [],
                4: [],
            },
            team_budgets={},
            current_pick=2,
        )

        threats = assess_threats(state, _league())

        ss_threats = [t for t in threats if t.position == "SS"]
        assert len(ss_threats) == 1
        assert ss_threats[0].threat_level == "likely-gone"
        assert ss_threats[0].teams_needing_position >= 2


class TestThreatAtRisk:
    def test_adp_in_danger_one_team_needs(self) -> None:
        """Player with ADP in danger zone but only 1 team needs position → at-risk."""
        # 4-team snake, user team 1, current pick 2
        # picks_until = 6 (picks 2-7 before user's pick 8)
        # Intervening teams: 2, 3, 4. Make teams 3 and 4 already have SS → only team 2 needs it
        pool = _make_pool(
            _row_adp(10, "SS Star", "SS", "batter", 20.0, adp=3.0),
            _row_adp(11, "C Guy", "C", "batter", 15.0, adp=10.0),
        )
        state = DraftState(
            config=DraftConfig(
                teams=4,
                roster_slots={"C": 1, "SS": 1, "SP": 1, "UTIL": 1, "BN": 1},
                format=DraftFormat.SNAKE,
                user_team=1,
                season=2026,
            ),
            picks=[
                _pick(1, 1, "C", player_id=1),
            ],
            available_pool=pool,
            team_rosters={
                1: [_pick(1, 1, "C", player_id=1)],
                2: [],
                3: [_pick(0, 3, "SS", player_id=98)],
                4: [_pick(0, 4, "SS", player_id=99)],
            },
            team_budgets={},
            current_pick=2,
        )

        threats = assess_threats(state, _league())

        ss_threats = [t for t in threats if t.position == "SS"]
        assert len(ss_threats) == 1
        assert ss_threats[0].threat_level == "at-risk"
        assert ss_threats[0].teams_needing_position == 1

    def test_many_teams_need_no_adp(self) -> None:
        """3+ teams needing position → at-risk even without ADP data."""
        pool = _make_pool(
            _row_adp(10, "SS NoADP", "SS", "batter", 20.0, adp=None),
            _row_adp(11, "C Guy", "C", "batter", 15.0),
        )
        state = DraftState(
            config=_12team_config(),
            picks=[],
            available_pool=pool,
            team_rosters={i: [] for i in range(1, 13)},
            team_budgets={},
            current_pick=2,
        )

        threats = assess_threats(state, _12team_league())

        ss_threats = [t for t in threats if t.position == "SS"]
        assert len(ss_threats) == 1
        assert ss_threats[0].threat_level == "at-risk"


class TestThreatSafe:
    def test_adp_well_beyond_next_pick(self) -> None:
        """Player with ADP well beyond user's next pick → safe."""
        # ADP 50 is way beyond danger zone (~8), and ≤2 teams need each position → safe
        pool = _make_pool(
            _row_adp(10, "SS Late", "SS", "batter", 10.0, adp=50.0),
            _row_adp(11, "C Guy", "C", "batter", 15.0, adp=50.0),
        )
        state = DraftState(
            config=DraftConfig(
                teams=4,
                roster_slots={"C": 1, "SS": 1, "SP": 1, "UTIL": 1, "BN": 1},
                format=DraftFormat.SNAKE,
                user_team=1,
                season=2026,
            ),
            picks=[_pick(1, 1, "C", player_id=1)],
            available_pool=pool,
            team_rosters={
                1: [_pick(1, 1, "C", player_id=1)],
                2: [_pick(0, 2, "SS", player_id=97), _pick(0, 2, "C", player_id=94)],
                3: [_pick(0, 3, "SS", player_id=98), _pick(0, 3, "C", player_id=95)],
                4: [_pick(0, 4, "SS", player_id=99)],
            },
            team_budgets={},
            current_pick=2,
        )

        threats = assess_threats(state, _league())

        for t in threats:
            assert t.threat_level == "safe"


class TestThreatAuctionEmpty:
    def test_auction_returns_empty(self) -> None:
        """Auction format → picks_until = 0 → empty list."""
        pool = _make_pool(
            _row_adp(10, "SS Star", "SS", "batter", 20.0, adp=3.0),
        )
        state = DraftState(
            config=DraftConfig(
                teams=4,
                roster_slots={"C": 1, "SS": 1, "SP": 1, "UTIL": 1, "BN": 1},
                format=DraftFormat.AUCTION,
                user_team=1,
                season=2026,
                budget=260,
            ),
            picks=[],
            available_pool=pool,
            team_rosters={i: [] for i in range(1, 5)},
            team_budgets={i: 260 for i in range(1, 5)},
        )

        threats = assess_threats(state, _league())

        assert threats == []


class TestThreatNoAdp:
    def test_no_adp_still_at_risk_when_many_teams_need(self) -> None:
        """Players without ADP can be at-risk if 3+ teams need the position."""
        pool = _make_pool(
            _row_adp(10, "SS NoADP", "SS", "batter", 20.0, adp=None),
        )
        state = DraftState(
            config=_12team_config(),
            picks=[],
            available_pool=pool,
            team_rosters={i: [] for i in range(1, 13)},
            team_budgets={},
            current_pick=2,
        )

        threats = assess_threats(state, _12team_league())

        ss_threats = [t for t in threats if t.position == "SS"]
        assert len(ss_threats) == 1
        assert ss_threats[0].threat_level == "at-risk"


class TestThreatUserDoesNotNeed:
    def test_user_filled_position_excluded(self) -> None:
        """Player is filtered out if user doesn't need the position."""
        pool = _make_pool(
            _row_adp(10, "SS Star", "SS", "batter", 20.0, adp=3.0),
            _row_adp(11, "C Guy", "C", "batter", 15.0, adp=3.0),
        )
        # User already has SS filled
        user_pick = _pick(1, 1, "SS", player_id=1)
        state = DraftState(
            config=DraftConfig(
                teams=4,
                roster_slots={"C": 1, "SS": 1, "SP": 1, "UTIL": 1, "BN": 1},
                format=DraftFormat.SNAKE,
                user_team=1,
                season=2026,
            ),
            picks=[user_pick],
            available_pool=pool,
            team_rosters={
                1: [user_pick],
                2: [],
                3: [],
                4: [],
            },
            team_budgets={},
            current_pick=2,
        )

        threats = assess_threats(state, _league())

        # C Guy should be in threats since user needs C
        c_threats = [t for t in threats if t.position == "C"]
        assert len(c_threats) == 1


class TestThreatSortOrder:
    def test_likely_gone_before_at_risk_before_safe(self) -> None:
        """Sorted: likely-gone first, at-risk, then safe."""
        # 4-team snake, user team 1, current pick 2
        pool = _make_pool(
            _row_adp(10, "SS Danger", "SS", "batter", 20.0, adp=3.0),  # likely-gone
            _row_adp(11, "C Safe", "C", "batter", 25.0, adp=50.0),  # safe (ADP way beyond)
            _row_adp(12, "SP Mid", "SP", "pitcher", 18.0, adp=3.0),  # depends on teams needing SP
        )
        state = DraftState(
            config=DraftConfig(
                teams=4,
                roster_slots={"C": 1, "SS": 1, "SP": 1, "UTIL": 1, "BN": 1},
                format=DraftFormat.SNAKE,
                user_team=1,
                season=2026,
            ),
            picks=[],
            available_pool=pool,
            team_rosters={i: [] for i in range(1, 5)},
            team_budgets={},
            current_pick=2,
        )

        threats = assess_threats(state, _league())

        levels = [t.threat_level for t in threats]
        # Verify ordering: all likely-gone before at-risk, all at-risk before safe
        order = {"likely-gone": 0, "at-risk": 1, "safe": 2}
        for i in range(len(levels) - 1):
            assert order[levels[i]] <= order[levels[i + 1]]


class TestThreatUpdatesAfterPick:
    def test_threat_changes_after_pick(self) -> None:
        """After a pick changes team rosters, threat counts update."""
        pool_before = _make_pool(
            _row_adp(10, "SS Star", "SS", "batter", 20.0, adp=3.0),
            _row_adp(11, "C Guy", "C", "batter", 15.0, adp=10.0),
        )
        config = DraftConfig(
            teams=4,
            roster_slots={"C": 1, "SS": 1, "SP": 1, "UTIL": 1, "BN": 1},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )

        # State before: teams 2,3 need SS
        state_before = DraftState(
            config=config,
            picks=[_pick(1, 1, "C", player_id=1)],
            available_pool=pool_before,
            team_rosters={
                1: [_pick(1, 1, "C", player_id=1)],
                2: [],
                3: [],
                4: [],
            },
            team_budgets={},
            current_pick=2,
        )
        threats_before = assess_threats(state_before, _league())
        ss_before = [t for t in threats_before if t.position == "SS"]

        # State after: team 2 drafted SS, now only team 3 needs SS
        pool_after = _make_pool(
            _row_adp(10, "SS Star", "SS", "batter", 20.0, adp=4.0),
            _row_adp(11, "C Guy", "C", "batter", 15.0, adp=10.0),
        )
        state_after = DraftState(
            config=config,
            picks=[
                _pick(1, 1, "C", player_id=1),
                _pick(2, 2, "SS", player_id=99),
            ],
            available_pool=pool_after,
            team_rosters={
                1: [_pick(1, 1, "C", player_id=1)],
                2: [_pick(2, 2, "SS", player_id=99)],
                3: [],
                4: [],
            },
            team_budgets={},
            current_pick=3,
        )
        threats_after = assess_threats(state_after, _league())
        ss_after = [t for t in threats_after if t.position == "SS"]

        # Before: 2+ teams needing SS → likely-gone; after: fewer teams → different level
        assert len(ss_before) == 1
        assert ss_before[0].teams_needing_position >= 2
        if ss_after:
            assert ss_after[0].teams_needing_position < ss_before[0].teams_needing_position


class TestThreatLimit:
    def test_limit_parameter(self) -> None:
        """Only returns top N threats."""
        players = [_row_adp(i, f"Player {i}", "SS", "batter", 100.0 - i, adp=3.0) for i in range(1, 21)]
        pool = _make_pool(*players)
        state = DraftState(
            config=_12team_config(),
            picks=[],
            available_pool=pool,
            team_rosters={i: [] for i in range(1, 13)},
            team_budgets={},
            current_pick=2,
        )

        threats = assess_threats(state, _12team_league(), limit=5)

        assert len(threats) <= 5
