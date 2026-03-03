from fantasy_baseball_manager.domain import RosterEntry, YahooDraftPick
from fantasy_baseball_manager.services.keeper_cost_derivation import derive_keeper_costs


def _make_pick(
    player_id: int,
    *,
    team_key: str = "449.l.12345.t.1",
    round: int = 1,
    pick: int = 1,
    cost: int | None = 25,
) -> YahooDraftPick:
    return YahooDraftPick(
        league_key="449.l.12345",
        season=2025,
        round=round,
        pick=pick,
        team_key=team_key,
        yahoo_player_key=f"449.p.{player_id}",
        player_id=player_id,
        player_name=f"Player {player_id}",
        position="OF",
        cost=cost,
    )


def _make_entry(
    player_id: int | None,
    *,
    acquisition_type: str = "draft",
) -> RosterEntry:
    return RosterEntry(
        player_id=player_id,
        yahoo_player_key=f"449.p.{player_id}" if player_id else "449.p.unknown",
        player_name=f"Player {player_id}" if player_id else "Unknown",
        position="OF",
        roster_status="active",
        acquisition_type=acquisition_type,
    )


class TestDeriveKeeperCosts:
    def test_drafted_player_uses_draft_price(self) -> None:
        picks = [_make_pick(100, cost=25)]
        entries = [_make_entry(100, acquisition_type="draft")]

        result = derive_keeper_costs(picks, entries, "keeper", 2026)

        assert len(result) == 1
        assert result[0].player_id == 100
        assert result[0].cost == 25.0
        assert result[0].source == "yahoo_draft"

    def test_trade_acquired_retains_original_draft_cost(self) -> None:
        # Player drafted by team 1 but now on team 2 (traded)
        picks = [_make_pick(100, team_key="449.l.12345.t.1", cost=30)]
        entries = [_make_entry(100, acquisition_type="trade")]

        result = derive_keeper_costs(picks, entries, "keeper", 2026)

        assert len(result) == 1
        assert result[0].cost == 30.0
        assert result[0].source == "yahoo_trade"

    def test_free_agent_pickup_uses_cost_floor(self) -> None:
        picks = []  # Not drafted
        entries = [_make_entry(200, acquisition_type="add")]

        result = derive_keeper_costs(picks, entries, "keeper", 2026)

        assert len(result) == 1
        assert result[0].cost == 1.0
        assert result[0].source == "yahoo_fa"

    def test_custom_cost_floor(self) -> None:
        picks = []
        entries = [_make_entry(200, acquisition_type="add")]

        result = derive_keeper_costs(picks, entries, "keeper", 2026, cost_floor=5.0)

        assert len(result) == 1
        assert result[0].cost == 5.0

    def test_player_id_none_skipped(self) -> None:
        picks = [_make_pick(100, cost=25)]
        entries = [_make_entry(None, acquisition_type="draft")]

        result = derive_keeper_costs(picks, entries, "keeper", 2026)

        assert result == []

    def test_drafted_player_not_in_draft_picks_uses_floor(self) -> None:
        # Player is on roster with acquisition_type "draft" but not found in draft picks
        picks = []
        entries = [_make_entry(100, acquisition_type="draft")]

        result = derive_keeper_costs(picks, entries, "keeper", 2026)

        assert len(result) == 1
        assert result[0].cost == 1.0
        assert result[0].source == "yahoo_fa"

    def test_mixed_acquisition_types(self) -> None:
        picks = [_make_pick(100, cost=25), _make_pick(200, cost=15, pick=2)]
        entries = [
            _make_entry(100, acquisition_type="draft"),
            _make_entry(200, acquisition_type="trade"),
            _make_entry(300, acquisition_type="add"),
        ]

        result = derive_keeper_costs(picks, entries, "keeper", 2026)

        costs_by_player = {kc.player_id: kc for kc in result}
        assert costs_by_player[100].source == "yahoo_draft"
        assert costs_by_player[100].cost == 25.0
        assert costs_by_player[200].source == "yahoo_trade"
        assert costs_by_player[200].cost == 15.0
        assert costs_by_player[300].source == "yahoo_fa"
        assert costs_by_player[300].cost == 1.0

    def test_empty_roster_returns_empty(self) -> None:
        picks = [_make_pick(100, cost=25)]
        entries: list[RosterEntry] = []

        result = derive_keeper_costs(picks, entries, "keeper", 2026)

        assert result == []

    def test_empty_draft_picks_all_get_floor(self) -> None:
        picks: list[YahooDraftPick] = []
        entries = [_make_entry(100, acquisition_type="draft"), _make_entry(200, acquisition_type="add")]

        result = derive_keeper_costs(picks, entries, "keeper", 2026)

        assert len(result) == 2
        assert all(kc.cost == 1.0 for kc in result)

    def test_season_and_league_propagated(self) -> None:
        picks = [_make_pick(100, cost=25)]
        entries = [_make_entry(100, acquisition_type="draft")]

        result = derive_keeper_costs(picks, entries, "my-league", 2027)

        assert result[0].season == 2027
        assert result[0].league == "my-league"
        assert result[0].years_remaining == 1

    def test_snake_draft_uses_round_number(self) -> None:
        picks = [_make_pick(100, cost=None, round=5)]
        entries = [_make_entry(100, acquisition_type="draft")]

        result = derive_keeper_costs(picks, entries, "keeper", 2026)

        assert len(result) == 1
        assert result[0].cost == 5.0
        assert result[0].source == "yahoo_draft"
