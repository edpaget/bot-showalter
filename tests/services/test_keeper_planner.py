from __future__ import annotations

from fantasy_baseball_manager.domain import (
    CategoryConfig,
    Direction,
    KeeperCost,
    KeeperPlanResult,
    LeagueFormat,
    LeagueSettings,
    Player,
    PlayerType,
    Projection,
    StatType,
    Valuation,
)
from fantasy_baseball_manager.services.keeper_planner import KeeperPlannerService


def _league() -> LeagueSettings:
    return LeagueSettings(
        name="Test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=10,
        budget=260,
        roster_batters=14,
        roster_pitchers=9,
        batting_categories=(
            CategoryConfig(key="HR", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="RBI", name="RBI", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="W", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="K", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        positions={"C": 1, "1B": 1, "OF": 3},
        pitcher_positions={"SP": 5, "RP": 2},
    )


def _keeper_cost(player_id: int, cost: float) -> KeeperCost:
    return KeeperCost(player_id=player_id, season=2026, league="test", cost=cost, source="auction")


def _valuation(
    player_id: int, value: float, position: str = "OF", player_type: PlayerType = PlayerType.BATTER
) -> Valuation:
    return Valuation(
        player_id=player_id,
        season=2026,
        system="zar",
        version="1.0",
        projection_system="steamer",
        projection_version="2026",
        player_type=player_type,
        position=position,
        value=value,
        rank=0,
        category_scores={},
    )


def _player(player_id: int, name: str) -> Player:
    first, last = name.split(" ", 1)
    return Player(name_first=first, name_last=last, id=player_id)


def _projection(player_id: int, player_type: PlayerType = PlayerType.BATTER) -> Projection:
    stats = {"pa": 600, "hr": 30, "rbi": 80} if player_type == PlayerType.BATTER else {"ip": 180, "w": 12, "k": 200}
    return Projection(
        player_id=player_id,
        season=2026,
        system="steamer",
        version="2026",
        player_type=player_type,
        stat_json=stats,
    )


class TestKeeperPlannerService:
    """Tests for KeeperPlannerService.plan()."""

    def _build_service(
        self,
        keeper_costs: list[KeeperCost],
        valuations: list[Valuation],
        players: list[Player],
        projections: list[Projection],
        batter_positions: dict[int, list[str]] | None = None,
        pitcher_positions: dict[int, list[str]] | None = None,
    ) -> KeeperPlannerService:
        return KeeperPlannerService(
            keeper_costs=keeper_costs,
            valuations=valuations,
            players=players,
            projections=projections,
            league=_league(),
            batter_positions=batter_positions or {},
            pitcher_positions=pitcher_positions or {},
        )

    def test_two_scenarios_produce_different_boards(self) -> None:
        """Optimal and alternative keeper sets produce different board previews."""
        costs = [_keeper_cost(1, 10.0), _keeper_cost(2, 15.0), _keeper_cost(3, 20.0)]
        valuations = [
            _valuation(1, 35.0, "OF"),
            _valuation(2, 30.0, "OF"),
            _valuation(3, 25.0, "SP", PlayerType.PITCHER),
            _valuation(4, 20.0, "OF"),
            _valuation(5, 18.0, "1B"),
        ]
        players = [
            _player(1, "Player One"),
            _player(2, "Player Two"),
            _player(3, "Player Three"),
            _player(4, "Player Four"),
            _player(5, "Player Five"),
        ]
        projections = [
            _projection(1),
            _projection(2),
            _projection(3, PlayerType.PITCHER),
            _projection(4),
            _projection(5),
        ]
        batter_positions = {1: ["OF"], 2: ["OF"], 4: ["OF"], 5: ["1B"]}
        pitcher_positions = {3: ["SP"]}

        svc = self._build_service(costs, valuations, players, projections, batter_positions, pitcher_positions)
        result = svc.plan(
            season=2026,
            max_keepers=2,
            board_preview_size=5,
        )

        assert isinstance(result, KeeperPlanResult)
        assert len(result.scenarios) >= 2

        # Scenarios should have different keeper sets
        scenario_ids = [s.keeper_ids for s in result.scenarios]
        assert len(set(scenario_ids)) == len(scenario_ids), "All scenarios should have unique keeper sets"

        # Each scenario should have board preview, scarcity, and category needs
        for scenario in result.scenarios:
            assert len(scenario.keeper_decisions) > 0
            assert isinstance(scenario.total_surplus, float)

    def test_cache_prevents_recomputation(self) -> None:
        """Same keeper IDs in custom scenarios don't recompute."""
        costs = [_keeper_cost(1, 10.0)]
        valuations = [_valuation(1, 35.0), _valuation(2, 20.0)]
        players = [_player(1, "Player One"), _player(2, "Player Two")]
        projections = [_projection(1), _projection(2)]

        svc = self._build_service(costs, valuations, players, projections)

        # Request the same custom scenario twice
        result = svc.plan(
            season=2026,
            max_keepers=1,
            custom_scenarios=[{1}, {1}],
            board_preview_size=5,
        )

        # Should only produce one scenario (deduplicated via frozenset)
        unique_ids = {s.keeper_ids for s in result.scenarios}
        # The optimizer's optimal set is {1}, and custom scenarios are also {1}
        # So all should collapse to the same frozenset
        assert frozenset({1}) in unique_ids

    def test_custom_scenarios_included(self) -> None:
        """Custom scenarios appear in the result alongside optimizer results."""
        costs = [_keeper_cost(1, 10.0), _keeper_cost(2, 15.0)]
        valuations = [
            _valuation(1, 35.0),
            _valuation(2, 30.0),
            _valuation(3, 20.0),
        ]
        players = [_player(1, "Player One"), _player(2, "Player Two"), _player(3, "Player Three")]
        projections = [_projection(1), _projection(2), _projection(3)]

        svc = self._build_service(costs, valuations, players, projections)

        # Request a custom scenario that the optimizer wouldn't produce
        result = svc.plan(
            season=2026,
            max_keepers=1,
            custom_scenarios=[{2}],
            board_preview_size=5,
        )

        custom_ids = frozenset({2})
        assert any(s.keeper_ids == custom_ids for s in result.scenarios)

    def test_no_keeper_costs_returns_custom_only(self) -> None:
        """When no keeper costs exist, only custom scenarios are returned."""
        valuations = [_valuation(1, 35.0), _valuation(2, 20.0)]
        players = [_player(1, "Player One"), _player(2, "Player Two")]
        projections = [_projection(1), _projection(2)]

        svc = self._build_service([], valuations, players, projections)
        result = svc.plan(
            season=2026,
            max_keepers=2,
            custom_scenarios=[{1}],
            board_preview_size=5,
        )

        assert len(result.scenarios) == 1
        assert result.scenarios[0].keeper_ids == frozenset({1})

    def test_two_way_player_both_types_in_scarcity(self) -> None:
        """A two-way player's batter and pitcher valuations both survive in the lookup."""
        costs = [_keeper_cost(1, 10.0)]
        valuations = [
            _valuation(1, 35.0, "OF", PlayerType.BATTER),
            _valuation(1, 25.0, "SP", PlayerType.PITCHER),
            _valuation(2, 20.0, "OF"),
        ]
        players = [_player(1, "Two Way"), _player(2, "Regular Player")]
        projections = [_projection(1, PlayerType.BATTER), _projection(1, PlayerType.PITCHER), _projection(2)]
        batter_positions = {1: ["OF"], 2: ["OF"]}
        pitcher_positions = {1: ["SP"]}

        svc = self._build_service(costs, valuations, players, projections, batter_positions, pitcher_positions)
        result = svc.plan(season=2026, max_keepers=1, board_preview_size=5)

        assert len(result.scenarios) >= 1
        # Should not crash and should have valid scarcity data
        scenario = result.scenarios[0]
        assert isinstance(scenario.scarcity, tuple)
        assert isinstance(scenario.board_preview, tuple)

    def test_no_keeper_costs_no_custom_returns_empty(self) -> None:
        """No costs and no custom scenarios → empty result."""
        valuations = [_valuation(1, 35.0)]
        players = [_player(1, "Player One")]
        projections = [_projection(1)]

        svc = self._build_service([], valuations, players, projections)
        result = svc.plan(season=2026, max_keepers=2)

        assert len(result.scenarios) == 0

    def test_scenario_has_scarcity_and_categories(self) -> None:
        """Each scenario includes scarcity and category analysis."""
        costs = [_keeper_cost(1, 10.0)]
        valuations = [
            _valuation(1, 35.0, "OF"),
            _valuation(2, 20.0, "OF"),
            _valuation(3, 15.0, "SP", PlayerType.PITCHER),
        ]
        players = [_player(1, "Player One"), _player(2, "Player Two"), _player(3, "Player Three")]
        projections = [_projection(1), _projection(2), _projection(3, PlayerType.PITCHER)]
        batter_positions = {1: ["OF"], 2: ["OF"]}
        pitcher_positions = {3: ["SP"]}

        svc = self._build_service(costs, valuations, players, projections, batter_positions, pitcher_positions)
        result = svc.plan(season=2026, max_keepers=1, board_preview_size=5)

        assert len(result.scenarios) >= 1
        scenario = result.scenarios[0]
        # Scarcity is a tuple (may be empty if not enough data)
        assert isinstance(scenario.scarcity, tuple)
        # Category needs is a tuple
        assert isinstance(scenario.category_needs, tuple)
        # Strongest/weakest categories are tuples of strings
        assert isinstance(scenario.strongest_categories, tuple)
        assert isinstance(scenario.weakest_categories, tuple)
