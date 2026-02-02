from fantasy_baseball_manager.draft.models import RosterConfig, RosterSlot
from fantasy_baseball_manager.draft.rules import PositionNotBeforeRound
from fantasy_baseball_manager.draft.simulation import generate_snake_order, simulate_draft
from fantasy_baseball_manager.draft.simulation_models import (
    DraftStrategy,
    SimulationConfig,
    TeamConfig,
)
from fantasy_baseball_manager.valuation.models import CategoryValue, PlayerValue, StatCategory


def _make_player(
    player_id: str,
    name: str,
    hr_value: float = 1.0,
    sb_value: float = 0.5,
    k_value: float = 0.0,
    era_value: float = 0.0,
    position_type: str = "B",
    hr_raw: float = 20.0,
    sb_raw: float = 10.0,
    k_raw: float = 0.0,
    era_raw: float = 0.0,
) -> PlayerValue:
    cats = [
        CategoryValue(category=StatCategory.HR, raw_stat=hr_raw, value=hr_value),
        CategoryValue(category=StatCategory.SB, raw_stat=sb_raw, value=sb_value),
    ]
    if k_value or era_value:
        cats.append(CategoryValue(category=StatCategory.K, raw_stat=k_raw, value=k_value))
        cats.append(CategoryValue(category=StatCategory.ERA, raw_stat=era_raw, value=era_value))
    return PlayerValue(
        player_id=player_id,
        name=name,
        category_values=tuple(cats),
        total_value=hr_value + sb_value + k_value + era_value,
        position_type=position_type,
    )


def _simple_roster() -> RosterConfig:
    return RosterConfig(
        slots=(
            RosterSlot(position="1B", count=1),
            RosterSlot(position="OF", count=2),
            RosterSlot(position="SP", count=2),
            RosterSlot(position="RP", count=1),
            RosterSlot(position="Util", count=1),
            RosterSlot(position="BN", count=2),
        )
    )


def _balanced_strategy() -> DraftStrategy:
    return DraftStrategy(name="balanced", category_weights={}, rules=(), noise_scale=0.0)


def _make_players(count: int = 20) -> list[PlayerValue]:
    """Create a pool of batters and pitchers with descending values."""
    players: list[PlayerValue] = []
    for i in range(count // 2):
        players.append(
            _make_player(
                player_id=f"b{i}",
                name=f"Batter {i}",
                hr_value=float(count // 2 - i),
                sb_value=float(count // 2 - i) * 0.5,
                position_type="B",
                hr_raw=float((count // 2 - i) * 10),
                sb_raw=float((count // 2 - i) * 5),
            )
        )
    for i in range(count // 2):
        players.append(
            _make_player(
                player_id=f"p{i}",
                name=f"Pitcher {i}",
                hr_value=0.0,
                sb_value=0.0,
                k_value=float(count // 2 - i),
                era_value=float(count // 2 - i) * 0.5,
                position_type="P",
                k_raw=float((count // 2 - i) * 50),
                era_raw=3.5 + i * 0.2,
            )
        )
    return players


def _make_positions(players: list[PlayerValue]) -> dict[tuple[str, str], tuple[str, ...]]:
    positions: dict[tuple[str, str], tuple[str, ...]] = {}
    for pv in players:
        if pv.position_type == "B":
            positions[(pv.player_id, "B")] = ("OF",)
        else:
            positions[(pv.player_id, "P")] = ("SP",)
    return positions


class TestSnakeOrder:
    def test_two_teams_two_rounds(self) -> None:
        order = generate_snake_order(2, 2)
        assert order == [0, 1, 1, 0]

    def test_three_teams_three_rounds(self) -> None:
        order = generate_snake_order(3, 3)
        assert order == [0, 1, 2, 2, 1, 0, 0, 1, 2]

    def test_single_team(self) -> None:
        order = generate_snake_order(1, 3)
        assert order == [0, 0, 0]

    def test_length_equals_teams_times_rounds(self) -> None:
        order = generate_snake_order(4, 5)
        assert len(order) == 20


class TestSimulateDraft:
    def test_two_team_draft_completes(self) -> None:
        players = _make_players(20)
        positions = _make_positions(players)
        roster = _simple_roster()
        strategy = _balanced_strategy()
        config = SimulationConfig(
            teams=(
                TeamConfig(team_id=1, name="Team 1", strategy=strategy),
                TeamConfig(team_id=2, name="Team 2", strategy=strategy),
            ),
            roster_config=roster,
            total_rounds=3,
            seed=42,
        )
        result = simulate_draft(config, players, positions)
        assert len(result.pick_log) == 6  # 2 teams * 3 rounds
        assert len(result.team_results) == 2

    def test_all_picks_unique(self) -> None:
        players = _make_players(20)
        positions = _make_positions(players)
        roster = _simple_roster()
        strategy = _balanced_strategy()
        config = SimulationConfig(
            teams=(
                TeamConfig(team_id=1, name="Team 1", strategy=strategy),
                TeamConfig(team_id=2, name="Team 2", strategy=strategy),
            ),
            roster_config=roster,
            total_rounds=5,
            seed=42,
        )
        result = simulate_draft(config, players, positions)
        player_ids = [pick.player_id for pick in result.pick_log]
        assert len(player_ids) == len(set(player_ids))

    def test_keepers_excluded_from_pool(self) -> None:
        players = _make_players(20)
        positions = _make_positions(players)
        roster = _simple_roster()
        strategy = _balanced_strategy()
        config = SimulationConfig(
            teams=(
                TeamConfig(team_id=1, name="Team 1", strategy=strategy, keepers=("b0",)),
                TeamConfig(team_id=2, name="Team 2", strategy=strategy),
            ),
            roster_config=roster,
            total_rounds=3,
            seed=42,
        )
        result = simulate_draft(config, players, positions)
        # b0 should be on Team 1's roster
        team1 = next(tr for tr in result.team_results if tr.team_id == 1)
        team1_ids = {p.player_id for p in team1.picks}
        assert "b0" in team1_ids
        # b0 should NOT be on Team 2's roster
        team2 = next(tr for tr in result.team_results if tr.team_id == 2)
        team2_ids = {p.player_id for p in team2.picks}
        assert "b0" not in team2_ids

    def test_category_weights_affect_picks(self) -> None:
        """HR-weighted team should draft HR-heavy players earlier."""
        players = _make_players(20)
        positions = _make_positions(players)
        roster = _simple_roster()
        hr_strategy = DraftStrategy(
            name="hr_heavy",
            category_weights={StatCategory.HR: 5.0},
            rules=(),
            noise_scale=0.0,
        )
        k_strategy = DraftStrategy(
            name="k_heavy",
            category_weights={StatCategory.K: 5.0},
            rules=(),
            noise_scale=0.0,
        )
        config = SimulationConfig(
            teams=(
                TeamConfig(team_id=1, name="HR Team", strategy=hr_strategy),
                TeamConfig(team_id=2, name="K Team", strategy=k_strategy),
            ),
            roster_config=roster,
            total_rounds=3,
            seed=42,
        )
        result = simulate_draft(config, players, positions)
        team1 = next(tr for tr in result.team_results if tr.team_id == 1)
        team2 = next(tr for tr in result.team_results if tr.team_id == 2)
        # HR team should have picked batters (b0, b1, ...) — high HR value
        team1_batters = sum(1 for p in team1.picks if p.player_id.startswith("b"))
        # K team should have picked pitchers (p0, p1, ...) — high K value
        team2_pitchers = sum(1 for p in team2.picks if p.player_id.startswith("p"))
        assert team1_batters >= 2
        assert team2_pitchers >= 2

    def test_rules_prevent_picks(self) -> None:
        """PositionNotBeforeRound should block early RP picks."""
        players: list[PlayerValue] = []
        # Create an RP who is very valuable
        players.append(
            _make_player(
                player_id="rp1",
                name="Top Closer",
                hr_value=0.0,
                sb_value=0.0,
                k_value=10.0,
                era_value=5.0,
                position_type="P",
                k_raw=100.0,
                era_raw=2.5,
            )
        )
        # Add some batters with lower value
        for i in range(10):
            players.append(
                _make_player(
                    player_id=f"b{i}",
                    name=f"Batter {i}",
                    hr_value=float(5 - i % 5),
                    sb_value=float(3 - i % 3),
                    position_type="B",
                )
            )
        positions: dict[tuple[str, str], tuple[str, ...]] = {
            ("rp1", "P"): ("RP",),
        }
        for i in range(10):
            positions[(f"b{i}", "B")] = ("OF",)

        roster = _simple_roster()
        strategy = DraftStrategy(
            name="no_early_rp",
            category_weights={},
            rules=(PositionNotBeforeRound(position="RP", earliest_round=5),),
            noise_scale=0.0,
        )
        config = SimulationConfig(
            teams=(
                TeamConfig(team_id=1, name="Team 1", strategy=strategy),
                TeamConfig(team_id=2, name="Team 2", strategy=strategy),
            ),
            roster_config=roster,
            total_rounds=3,
            seed=42,
        )
        result = simulate_draft(config, players, positions)
        # RP should not be drafted in rounds 1-3
        for pick in result.pick_log:
            if pick.player_id == "rp1":
                assert pick.round_number >= 5, f"RP1 drafted in round {pick.round_number}"

    def test_noise_with_seed_reproducible(self) -> None:
        players = _make_players(20)
        positions = _make_positions(players)
        roster = _simple_roster()
        strategy = DraftStrategy(name="noisy", category_weights={}, rules=(), noise_scale=0.15)
        config = SimulationConfig(
            teams=(
                TeamConfig(team_id=1, name="Team 1", strategy=strategy),
                TeamConfig(team_id=2, name="Team 2", strategy=strategy),
            ),
            roster_config=roster,
            total_rounds=5,
            seed=42,
        )
        result1 = simulate_draft(config, players, positions)
        result2 = simulate_draft(config, players, positions)
        ids1 = [p.player_id for p in result1.pick_log]
        ids2 = [p.player_id for p in result2.pick_log]
        assert ids1 == ids2

    def test_noise_with_different_seeds_differs(self) -> None:
        players = _make_players(40)
        positions = _make_positions(players)
        roster = _simple_roster()
        strategy = DraftStrategy(name="noisy", category_weights={}, rules=(), noise_scale=0.5)
        config1 = SimulationConfig(
            teams=(
                TeamConfig(team_id=1, name="Team 1", strategy=strategy),
                TeamConfig(team_id=2, name="Team 2", strategy=strategy),
            ),
            roster_config=roster,
            total_rounds=10,
            seed=42,
        )
        config2 = SimulationConfig(
            teams=config1.teams,
            roster_config=roster,
            total_rounds=10,
            seed=999,
        )
        result1 = simulate_draft(config1, players, positions)
        result2 = simulate_draft(config2, players, positions)
        ids1 = [p.player_id for p in result1.pick_log]
        ids2 = [p.player_id for p in result2.pick_log]
        assert ids1 != ids2

    def test_pick_log_has_correct_overall_pick_numbers(self) -> None:
        players = _make_players(20)
        positions = _make_positions(players)
        roster = _simple_roster()
        strategy = _balanced_strategy()
        config = SimulationConfig(
            teams=(
                TeamConfig(team_id=1, name="Team 1", strategy=strategy),
                TeamConfig(team_id=2, name="Team 2", strategy=strategy),
            ),
            roster_config=roster,
            total_rounds=3,
            seed=42,
        )
        result = simulate_draft(config, players, positions)
        overall_picks = [p.overall_pick for p in result.pick_log]
        assert overall_picks == list(range(1, 7))

    def test_team_results_have_category_totals(self) -> None:
        players = _make_players(20)
        positions = _make_positions(players)
        roster = _simple_roster()
        strategy = _balanced_strategy()
        config = SimulationConfig(
            teams=(
                TeamConfig(team_id=1, name="Team 1", strategy=strategy),
                TeamConfig(team_id=2, name="Team 2", strategy=strategy),
            ),
            roster_config=roster,
            total_rounds=3,
            seed=42,
        )
        result = simulate_draft(config, players, positions)
        for tr in result.team_results:
            assert len(tr.category_totals) > 0

    def test_snake_order_first_pick_alternates(self) -> None:
        players = _make_players(20)
        positions = _make_positions(players)
        roster = _simple_roster()
        strategy = _balanced_strategy()
        config = SimulationConfig(
            teams=(
                TeamConfig(team_id=1, name="Team 1", strategy=strategy),
                TeamConfig(team_id=2, name="Team 2", strategy=strategy),
            ),
            roster_config=roster,
            total_rounds=4,
            seed=42,
        )
        result = simulate_draft(config, players, positions)
        # Round 1: Team 1, Team 2
        # Round 2: Team 2, Team 1
        # Round 3: Team 1, Team 2
        # Round 4: Team 2, Team 1
        team_ids = [p.team_id for p in result.pick_log]
        assert team_ids == [1, 2, 2, 1, 1, 2, 2, 1]
