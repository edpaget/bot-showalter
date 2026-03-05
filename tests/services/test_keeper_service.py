import datetime

import pytest

from fantasy_baseball_manager.domain import (
    CategoryConfig,
    Direction,
    KeeperCost,
    LeagueFormat,
    LeagueSettings,
    Player,
    Projection,
    Roster,
    RosterEntry,
    StatType,
    Valuation,
)
from fantasy_baseball_manager.domain.result import Err, Ok
from fantasy_baseball_manager.services.keeper_service import (
    adjust_valuations_for_league_keepers,
    compute_adjusted_valuations,
    compute_surplus,
    estimate_other_keepers,
    evaluate_trade,
    set_keeper_cost,
)
from tests.fakes.repos import FakeKeeperCostRepo, FakePlayerRepo


class TestSetKeeperCost:
    def test_single_match_returns_ok(self) -> None:
        player = Player(name_first="Mike", name_last="Trout", id=1)
        player_repo = FakePlayerRepo([player])
        keeper_repo = FakeKeeperCostRepo()

        result = set_keeper_cost(
            "Trout", cost=25.0, season=2026, league="dynasty", player_repo=player_repo, keeper_repo=keeper_repo
        )

        assert isinstance(result, Ok)
        kc = result.value
        assert kc.player_id == 1
        assert kc.cost == 25.0
        assert kc.season == 2026
        assert kc.league == "dynasty"
        assert kc.source == "auction"
        assert kc.years_remaining == 1

    def test_single_match_upserts_to_repo(self) -> None:
        player = Player(name_first="Mike", name_last="Trout", id=1)
        player_repo = FakePlayerRepo([player])
        keeper_repo = FakeKeeperCostRepo()

        set_keeper_cost(
            "Trout", cost=25.0, season=2026, league="dynasty", player_repo=player_repo, keeper_repo=keeper_repo
        )

        stored = keeper_repo.find_by_season_league(2026, "dynasty")
        assert len(stored) == 1
        assert stored[0].player_id == 1

    def test_custom_years_and_source(self) -> None:
        player = Player(name_first="Mike", name_last="Trout", id=1)
        player_repo = FakePlayerRepo([player])
        keeper_repo = FakeKeeperCostRepo()

        result = set_keeper_cost(
            "Trout",
            cost=30.0,
            season=2026,
            league="dynasty",
            player_repo=player_repo,
            keeper_repo=keeper_repo,
            years_remaining=3,
            source="trade",
        )

        assert isinstance(result, Ok)
        assert result.value.years_remaining == 3
        assert result.value.source == "trade"

    def test_no_match_returns_err(self) -> None:
        player_repo = FakePlayerRepo()
        keeper_repo = FakeKeeperCostRepo()

        result = set_keeper_cost(
            "Nobody", cost=10.0, season=2026, league="dynasty", player_repo=player_repo, keeper_repo=keeper_repo
        )

        assert isinstance(result, Err)
        assert "no player found matching 'Nobody'" in result.error

    def test_no_match_does_not_upsert(self) -> None:
        player_repo = FakePlayerRepo()
        keeper_repo = FakeKeeperCostRepo()

        set_keeper_cost(
            "Nobody", cost=10.0, season=2026, league="dynasty", player_repo=player_repo, keeper_repo=keeper_repo
        )

        assert keeper_repo.find_by_season_league(2026, "dynasty") == []

    def test_ambiguous_match_returns_err(self) -> None:
        players = [
            Player(name_first="Mike", name_last="Smith", id=1),
            Player(name_first="John", name_last="Smith", id=2),
        ]
        player_repo = FakePlayerRepo(players)
        keeper_repo = FakeKeeperCostRepo()

        result = set_keeper_cost(
            "Smith", cost=10.0, season=2026, league="dynasty", player_repo=player_repo, keeper_repo=keeper_repo
        )

        assert isinstance(result, Err)
        assert "ambiguous name 'Smith'" in result.error
        assert "Mike Smith" in result.error
        assert "John Smith" in result.error


def _valuation(player_id: int, value: float, position: str = "SS") -> Valuation:
    return Valuation(
        player_id=player_id,
        season=2026,
        system="zar",
        version="v1",
        projection_system="composite",
        projection_version="v1",
        player_type="batter",
        position=position,
        value=value,
        rank=1,
        category_scores={},
    )


def _keeper_cost(player_id: int, cost: float, years: int = 1) -> KeeperCost:
    return KeeperCost(
        player_id=player_id, season=2026, league="dynasty", cost=cost, source="auction", years_remaining=years
    )


class TestSetKeeperCostOriginalRound:
    def test_set_keeper_cost_with_original_round(self) -> None:
        player = Player(name_first="Mike", name_last="Trout", id=1)
        player_repo = FakePlayerRepo([player])
        keeper_repo = FakeKeeperCostRepo()

        result = set_keeper_cost(
            "Trout",
            cost=18.0,
            season=2026,
            league="dynasty",
            player_repo=player_repo,
            keeper_repo=keeper_repo,
            source="draft_round",
            original_round=3,
        )

        assert isinstance(result, Ok)
        kc = result.value
        assert kc.original_round == 3
        assert kc.source == "draft_round"
        assert kc.cost == 18.0


class TestComputeSurplus:
    def test_positive_surplus(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 10.0)]
        valuations = [_valuation(1, 25.0)]

        result = compute_surplus(keepers, valuations, players)

        assert len(result) == 1
        assert result[0].surplus == 15.0
        assert result[0].recommendation == "keep"

    def test_negative_surplus(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 30.0)]
        valuations = [_valuation(1, 10.0)]

        result = compute_surplus(keepers, valuations, players)

        assert len(result) == 1
        assert result[0].surplus == -20.0
        assert result[0].recommendation == "release"

    def test_threshold_filters(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 18.0)]
        valuations = [_valuation(1, 20.0)]

        result = compute_surplus(keepers, valuations, players, threshold=5.0)

        assert len(result) == 1
        assert result[0].surplus == 2.0
        assert result[0].recommendation == "release"

    def test_multi_year_discount(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 15.0, years=3)]
        valuations = [_valuation(1, 25.0)]

        result = compute_surplus(keepers, valuations, players, decay=0.85)

        # single-year surplus = 25 - 15 = 10
        # total = 10 + 10*0.85 + 10*0.85^2 = 10 + 8.5 + 7.225 = 25.725
        assert len(result) == 1
        assert result[0].surplus == pytest.approx(25.725)
        assert result[0].recommendation == "keep"

    def test_sorted_by_surplus_descending(self) -> None:
        players = [
            Player(name_first="Mike", name_last="Trout", id=1),
            Player(name_first="Shohei", name_last="Ohtani", id=2),
            Player(name_first="Aaron", name_last="Judge", id=3),
        ]
        keepers = [_keeper_cost(1, 10.0), _keeper_cost(2, 5.0), _keeper_cost(3, 20.0)]
        valuations = [_valuation(1, 25.0), _valuation(2, 30.0), _valuation(3, 22.0)]

        result = compute_surplus(keepers, valuations, players)

        surpluses = [d.surplus for d in result]
        assert surpluses == sorted(surpluses, reverse=True)
        assert surpluses == [25.0, 15.0, 2.0]

    def test_missing_valuation(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 10.0)]
        valuations: list[Valuation] = []

        result = compute_surplus(keepers, valuations, players)

        assert len(result) == 1
        assert result[0].projected_value == 0.0
        assert result[0].surplus == -10.0
        assert result[0].recommendation == "release"

    def test_empty_inputs(self) -> None:
        result = compute_surplus([], [], [])
        assert result == []

    def test_player_name_and_position(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 10.0)]
        valuations = [_valuation(1, 25.0, position="CF")]

        result = compute_surplus(keepers, valuations, players)

        assert result[0].player_name == "Mike Trout"
        assert result[0].position == "CF"

    def test_missing_player_uses_unknown(self) -> None:
        keepers = [_keeper_cost(1, 10.0)]
        valuations = [_valuation(1, 25.0)]

        result = compute_surplus(keepers, valuations, [])

        assert result[0].player_name == "Unknown"
        assert result[0].position == "SS"

    def test_highest_valuation_used(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 10.0)]
        valuations = [_valuation(1, 20.0), _valuation(1, 25.0)]

        result = compute_surplus(keepers, valuations, players)

        assert result[0].projected_value == 25.0
        assert result[0].surplus == 15.0


class TestComputeSurplusOriginalRound:
    def test_propagates_original_round(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [
            KeeperCost(player_id=1, season=2026, league="dynasty", cost=18.0, source="draft_round", original_round=3)
        ]
        valuations = [_valuation(1, 25.0)]

        result = compute_surplus(keepers, valuations, players)

        assert len(result) == 1
        assert result[0].original_round == 3

    def test_null_original_round_propagated(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 10.0)]
        valuations = [_valuation(1, 25.0)]

        result = compute_surplus(keepers, valuations, players)

        assert len(result) == 1
        assert result[0].original_round is None


def _adj_league() -> LeagueSettings:
    """Minimal league for adjusted valuation tests."""
    return LeagueSettings(
        name="test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=2,
        budget=100,
        roster_batters=3,
        roster_pitchers=0,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(),
        positions={"c": 1, "of": 1},
        roster_util=0,
    )


def _adj_proj(player_id: int, hr: float) -> Projection:
    return Projection(
        player_id=player_id,
        season=2026,
        system="composite",
        version="v1",
        player_type="batter",
        stat_json={"pa": 600, "hr": hr},
    )


def _adj_projections() -> list[Projection]:
    """4 catchers + 2 outfielders with varied HR values."""
    return [
        _adj_proj(1, hr=40),  # best catcher
        _adj_proj(2, hr=30),
        _adj_proj(3, hr=20),
        _adj_proj(4, hr=10),  # worst catcher
        _adj_proj(5, hr=25),  # best OF
        _adj_proj(6, hr=15),  # 2nd OF
    ]


def _adj_batter_positions() -> dict[int, list[str]]:
    return {1: ["c"], 2: ["c"], 3: ["c"], 4: ["c"], 5: ["of"], 6: ["of"]}


def _adj_players() -> list[Player]:
    return [
        Player(name_first="P", name_last="One", id=1),
        Player(name_first="P", name_last="Two", id=2),
        Player(name_first="P", name_last="Three", id=3),
        Player(name_first="P", name_last="Four", id=4),
        Player(name_first="P", name_last="Five", id=5),
        Player(name_first="P", name_last="Six", id=6),
    ]


class TestComputeAdjustedValuations:
    def test_kept_players_excluded(self) -> None:
        original_vals = [_valuation(pid, 10.0) for pid in range(1, 7)]

        result = compute_adjusted_valuations(
            kept_player_ids={1},
            projections=_adj_projections(),
            batter_positions=_adj_batter_positions(),
            pitcher_positions={},
            league=_adj_league(),
            original_valuations=original_vals,
            players=_adj_players(),
        )

        result_ids = {r.player_id for r in result}
        assert 1 not in result_ids
        assert len(result) == 5

    def test_replacement_level_shifts(self) -> None:
        league = _adj_league()
        projections = _adj_projections()
        batter_positions = _adj_batter_positions()
        players = _adj_players()

        # Get baseline values (no keepers) to use as original valuations
        baseline = compute_adjusted_valuations(
            kept_player_ids=set(),
            projections=projections,
            batter_positions=batter_positions,
            pitcher_positions={},
            league=league,
            original_valuations=[],
            players=players,
        )
        original_vals = [
            Valuation(
                player_id=a.player_id,
                season=2026,
                system="zar",
                version="v1",
                projection_system="composite",
                projection_version="v1",
                player_type=a.player_type,
                position=a.position,
                value=a.adjusted_value,
                rank=0,
                category_scores={},
            )
            for a in baseline
        ]

        # Keep best catcher (pid 1)
        adjusted = compute_adjusted_valuations(
            kept_player_ids={1},
            projections=projections,
            batter_positions=batter_positions,
            pitcher_positions={},
            league=league,
            original_valuations=original_vals,
            players=players,
        )

        # Above-replacement catchers should see value increase
        catcher_results = [a for a in adjusted if a.player_id in {2, 3}]
        assert len(catcher_results) == 2
        for c in catcher_results:
            assert c.value_change > 0, f"Catcher {c.player_id} should see positive value change"

    def test_value_change_computed(self) -> None:
        original_vals = [_valuation(pid, float(pid * 5)) for pid in range(1, 7)]

        result = compute_adjusted_valuations(
            kept_player_ids={1},
            projections=_adj_projections(),
            batter_positions=_adj_batter_positions(),
            pitcher_positions={},
            league=_adj_league(),
            original_valuations=original_vals,
            players=_adj_players(),
        )

        for a in result:
            assert a.value_change == pytest.approx(a.adjusted_value - a.original_value)

    def test_sorted_by_adjusted_value(self) -> None:
        original_vals = [_valuation(pid, 10.0) for pid in range(1, 7)]

        result = compute_adjusted_valuations(
            kept_player_ids=set(),
            projections=_adj_projections(),
            batter_positions=_adj_batter_positions(),
            pitcher_positions={},
            league=_adj_league(),
            original_valuations=original_vals,
            players=_adj_players(),
        )

        values = [r.adjusted_value for r in result]
        assert values == sorted(values, reverse=True)

    def test_empty_kept_ids(self) -> None:
        league = _adj_league()
        projections = _adj_projections()
        batter_positions = _adj_batter_positions()
        players = _adj_players()

        # First call to get baseline adjusted values
        baseline = compute_adjusted_valuations(
            kept_player_ids=set(),
            projections=projections,
            batter_positions=batter_positions,
            pitcher_positions={},
            league=league,
            original_valuations=[],
            players=players,
        )
        original_vals = [
            Valuation(
                player_id=a.player_id,
                season=2026,
                system="zar",
                version="v1",
                projection_system="composite",
                projection_version="v1",
                player_type=a.player_type,
                position=a.position,
                value=a.adjusted_value,
                rank=0,
                category_scores={},
            )
            for a in baseline
        ]

        # Second call: same empty keepers, but with original vals set
        result = compute_adjusted_valuations(
            kept_player_ids=set(),
            projections=projections,
            batter_positions=batter_positions,
            pitcher_positions={},
            league=league,
            original_valuations=original_vals,
            players=players,
        )

        assert len(result) == 6
        for r in result:
            assert r.value_change == pytest.approx(0.0, abs=0.01)

    def test_all_kept(self) -> None:
        all_ids = {p.player_id for p in _adj_projections()}
        original_vals = [_valuation(pid, 10.0) for pid in range(1, 7)]

        result = compute_adjusted_valuations(
            kept_player_ids=all_ids,
            projections=_adj_projections(),
            batter_positions=_adj_batter_positions(),
            pitcher_positions={},
            league=_adj_league(),
            original_valuations=original_vals,
            players=_adj_players(),
        )

        assert result == []


class TestEvaluateTrade:
    def test_surplus_exchanged(self) -> None:
        """Team A gives player with surplus $10, receives player with surplus $20."""
        players = [
            Player(name_first="Mike", name_last="Trout", id=1),
            Player(name_first="Shohei", name_last="Ohtani", id=2),
        ]
        keeper_costs = [_keeper_cost(1, 15.0), _keeper_cost(2, 10.0)]
        valuations = [_valuation(1, 25.0), _valuation(2, 30.0)]

        result = evaluate_trade(
            team_a_gives=[1],
            team_b_gives=[2],
            keeper_costs=keeper_costs,
            valuations=valuations,
            players=players,
        )

        # Player 1 surplus = 25 - 15 = 10, Player 2 surplus = 30 - 10 = 20
        # Team A gives surplus 10, receives surplus 20 → delta = +10
        assert result.team_a_surplus_delta == pytest.approx(10.0)
        assert result.team_b_surplus_delta == pytest.approx(-10.0)

    def test_multi_player_trade(self) -> None:
        """2-for-1 trade with known surpluses."""
        players = [
            Player(name_first="Mike", name_last="Trout", id=1),
            Player(name_first="Shohei", name_last="Ohtani", id=2),
            Player(name_first="Aaron", name_last="Judge", id=3),
        ]
        keeper_costs = [_keeper_cost(1, 10.0), _keeper_cost(2, 5.0), _keeper_cost(3, 20.0)]
        valuations = [_valuation(1, 25.0), _valuation(2, 15.0), _valuation(3, 30.0)]

        result = evaluate_trade(
            team_a_gives=[1, 2],
            team_b_gives=[3],
            keeper_costs=keeper_costs,
            valuations=valuations,
            players=players,
        )

        # A gives: surplus 15 + 10 = 25. B gives: surplus 10
        # A delta = 10 - 25 = -15
        assert result.team_a_surplus_delta == pytest.approx(-15.0)
        assert result.team_b_surplus_delta == pytest.approx(15.0)
        assert len(result.team_a_gives) == 2
        assert len(result.team_b_gives) == 1

    def test_winner_determination(self) -> None:
        """Positive delta → team_a wins, negative → team_b, zero → even."""
        players = [
            Player(name_first="P", name_last="One", id=1),
            Player(name_first="P", name_last="Two", id=2),
        ]

        # Team A wins: receives higher surplus
        costs = [_keeper_cost(1, 20.0), _keeper_cost(2, 5.0)]
        vals = [_valuation(1, 25.0), _valuation(2, 25.0)]
        result = evaluate_trade([1], [2], costs, vals, players)
        assert result.winner == "team_a"

        # Team B wins: receives lower surplus
        costs = [_keeper_cost(1, 5.0), _keeper_cost(2, 20.0)]
        vals = [_valuation(1, 25.0), _valuation(2, 25.0)]
        result = evaluate_trade([1], [2], costs, vals, players)
        assert result.winner == "team_b"

        # Even: equal surplus
        costs = [_keeper_cost(1, 10.0), _keeper_cost(2, 10.0)]
        vals = [_valuation(1, 25.0), _valuation(2, 25.0)]
        result = evaluate_trade([1], [2], costs, vals, players)
        assert result.winner == "even"

    def test_multi_year_contracts(self) -> None:
        """Multi-year contract uses discounted surplus (same decay as compute_surplus)."""
        players = [
            Player(name_first="P", name_last="One", id=1),
            Player(name_first="P", name_last="Two", id=2),
        ]
        costs = [_keeper_cost(1, 15.0, years=3), _keeper_cost(2, 10.0)]
        vals = [_valuation(1, 25.0), _valuation(2, 30.0)]

        result = evaluate_trade([1], [2], costs, vals, players, decay=0.85)

        # Player 1: surplus = 10/yr, total = 10 + 8.5 + 7.225 = 25.725
        # Player 2: surplus = 20/yr, total = 20
        # A delta = 20 - 25.725 = -5.725
        assert result.team_a_surplus_delta == pytest.approx(-5.725)
        assert result.team_b_surplus_delta == pytest.approx(5.725)
        assert result.team_a_gives[0].surplus == pytest.approx(25.725)

    def test_missing_valuation(self) -> None:
        """Player without valuation → projected_value = 0, surplus = -cost."""
        players = [
            Player(name_first="P", name_last="One", id=1),
            Player(name_first="P", name_last="Two", id=2),
        ]
        costs = [_keeper_cost(1, 10.0), _keeper_cost(2, 5.0)]
        vals = [_valuation(2, 20.0)]  # No valuation for player 1

        result = evaluate_trade([1], [2], costs, vals, players)

        a_detail = result.team_a_gives[0]
        assert a_detail.projected_value == 0.0
        assert a_detail.surplus == pytest.approx(-10.0)

    def test_missing_keeper_cost(self) -> None:
        """Player without keeper cost → cost = 0, surplus = full value."""
        players = [
            Player(name_first="P", name_last="One", id=1),
            Player(name_first="P", name_last="Two", id=2),
        ]
        costs = [_keeper_cost(2, 5.0)]  # No cost for player 1
        vals = [_valuation(1, 25.0), _valuation(2, 20.0)]

        result = evaluate_trade([1], [2], costs, vals, players)

        a_detail = result.team_a_gives[0]
        assert a_detail.cost == 0.0
        assert a_detail.surplus == pytest.approx(25.0)

    def test_player_details_populated(self) -> None:
        """Verify name, position, cost, value fields on TradePlayerDetail."""
        players = [
            Player(name_first="Mike", name_last="Trout", id=1),
            Player(name_first="Shohei", name_last="Ohtani", id=2),
        ]
        costs = [_keeper_cost(1, 15.0, years=2), _keeper_cost(2, 10.0)]
        vals = [_valuation(1, 25.0, position="CF"), _valuation(2, 30.0, position="DH")]

        result = evaluate_trade([1], [2], costs, vals, players)

        a = result.team_a_gives[0]
        assert a.player_id == 1
        assert a.player_name == "Mike Trout"
        assert a.position == "CF"
        assert a.cost == 15.0
        assert a.projected_value == 25.0
        assert a.years_remaining == 2

        b = result.team_b_gives[0]
        assert b.player_id == 2
        assert b.player_name == "Shohei Ohtani"
        assert b.position == "DH"


def _roster(team_key: str, player_ids: list[int]) -> Roster:
    """Build a minimal roster for testing estimate_other_keepers."""
    entries = tuple(
        RosterEntry(
            player_id=pid,
            yahoo_player_key=f"449.p.{pid}",
            player_name=f"Player {pid}",
            position="UTIL",
            roster_status="active",
            acquisition_type="draft",
        )
        for pid in player_ids
    )
    return Roster(
        team_key=team_key,
        league_key="449.l.100",
        season=2025,
        week=1,
        as_of=datetime.date(2025, 10, 1),
        entries=entries,
    )


class TestEstimateOtherKeepers:
    def test_top_n_from_each_team(self) -> None:
        """Two teams with 3 players each, max_keepers=2 → returns top 2 from each (4 IDs)."""
        rosters = [
            _roster("t.1", [1, 2, 3]),
            _roster("t.2", [4, 5, 6]),
        ]
        valuations = [
            _valuation(1, 30.0),
            _valuation(2, 20.0),
            _valuation(3, 10.0),
            _valuation(4, 25.0),
            _valuation(5, 15.0),
            _valuation(6, 5.0),
        ]

        result = estimate_other_keepers(rosters, valuations, max_keepers=2)

        assert result == {1, 2, 4, 5}

    def test_no_valuation_treated_as_zero(self) -> None:
        """Players with no valuation are sorted last (value 0)."""
        rosters = [_roster("t.1", [1, 2, 3])]
        valuations = [_valuation(1, 10.0)]  # no vals for 2, 3

        result = estimate_other_keepers(rosters, valuations, max_keepers=2)

        # Player 1 (val 10) is top, then either 2 or 3 (val 0) fills second slot
        assert 1 in result
        assert len(result) == 2

    def test_fewer_players_than_max_keepers(self) -> None:
        """Team with fewer qualifying players than max_keepers keeps all."""
        rosters = [_roster("t.1", [1])]
        valuations = [_valuation(1, 20.0)]

        result = estimate_other_keepers(rosters, valuations, max_keepers=3)

        assert result == {1}

    def test_empty_roster_list(self) -> None:
        """Empty roster list → empty set."""
        result = estimate_other_keepers([], [], max_keepers=2)

        assert result == set()

    def test_skips_entries_with_no_player_id(self) -> None:
        """Roster entries with player_id=None are excluded."""
        entries = (
            RosterEntry(
                player_id=1,
                yahoo_player_key="449.p.1",
                player_name="Player 1",
                position="UTIL",
                roster_status="active",
                acquisition_type="draft",
            ),
            RosterEntry(
                player_id=None,
                yahoo_player_key="449.p.999",
                player_name="Unknown",
                position="UTIL",
                roster_status="active",
                acquisition_type="add",
            ),
        )
        rosters = [
            Roster(
                team_key="t.1",
                league_key="449.l.100",
                season=2025,
                week=1,
                as_of=datetime.date(2025, 10, 1),
                entries=entries,
            )
        ]
        valuations = [_valuation(1, 20.0)]

        result = estimate_other_keepers(rosters, valuations, max_keepers=2)

        assert result == {1}


class TestAdjustValuationsForLeagueKeepers:
    """Tests for the combined estimation + valuation adjustment pipeline."""

    def _league(self) -> LeagueSettings:
        return _adj_league()

    def test_adjusts_valuations_for_estimated_keepers(self) -> None:
        """Valuations are adjusted when league keepers are estimated."""
        rosters = [_roster("t.1", [1, 5])]  # other team has catcher 1 + OF 5
        projections = _adj_projections()
        batter_positions = _adj_batter_positions()
        players = _adj_players()
        league = self._league()

        # Build baseline valuations from all players
        baseline = compute_adjusted_valuations(
            kept_player_ids=set(),
            projections=projections,
            batter_positions=batter_positions,
            pitcher_positions={},
            league=league,
            original_valuations=[],
            players=players,
        )
        valuations = [
            Valuation(
                player_id=a.player_id,
                season=2026,
                system="zar",
                version="v1",
                projection_system="composite",
                projection_version="v1",
                player_type=a.player_type,
                position=a.position,
                value=a.adjusted_value,
                rank=0,
                category_scores={},
            )
            for a in baseline
        ]

        result = adjust_valuations_for_league_keepers(
            rosters=rosters,
            valuations=valuations,
            projections=projections,
            batter_positions=batter_positions,
            pitcher_positions={},
            league=league,
            players=players,
            max_keepers=1,
        )

        # Estimated keeper is player 1 (highest val on the roster)
        result_ids = {v.player_id for v in result}
        assert 1 not in result_ids  # kept player excluded
        assert len(result) < len(valuations)

    def test_returns_original_when_no_keepers_estimated(self) -> None:
        """Empty rosters → no keepers estimated → original valuations returned unchanged."""
        valuations = [_valuation(1, 25.0), _valuation(2, 15.0)]

        result = adjust_valuations_for_league_keepers(
            rosters=[],
            valuations=valuations,
            projections=[],
            batter_positions={},
            pitcher_positions={},
            league=_adj_league(),
            players=[],
            max_keepers=2,
        )

        assert result == valuations
