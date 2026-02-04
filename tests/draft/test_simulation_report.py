from io import StringIO

from rich.console import Console

from fantasy_baseball_manager.draft.models import RosterConfig
from fantasy_baseball_manager.draft.simulation_models import (
    DraftStrategy,
    SimulationConfig,
    SimulationPick,
    SimulationResult,
    TeamConfig,
    TeamResult,
)
from fantasy_baseball_manager.draft import simulation_report
from fantasy_baseball_manager.valuation.models import StatCategory


def _make_pick(
    overall: int,
    round_num: int,
    pick_in_round: int,
    team_id: int,
    team_name: str,
    player_id: str,
    player_name: str,
    position: str = "OF",
    value: float = 5.0,
) -> SimulationPick:
    return SimulationPick(
        overall_pick=overall,
        round_number=round_num,
        pick_in_round=pick_in_round,
        team_id=team_id,
        team_name=team_name,
        player_id=player_id,
        player_name=player_name,
        position=position,
        adjusted_value=value,
    )


def _make_result() -> SimulationResult:
    strategy = DraftStrategy(name="balanced", category_weights={}, rules=())
    picks = (
        _make_pick(1, 1, 1, 1, "Team A", "p1", "Player 1", "OF", 10.0),
        _make_pick(2, 1, 2, 2, "Team B", "p2", "Player 2", "SP", 9.5),
        _make_pick(3, 2, 1, 2, "Team B", "p3", "Player 3", "OF", 8.0),
        _make_pick(4, 2, 2, 1, "Team A", "p4", "Player 4", "SP", 7.5),
    )
    team_results = (
        TeamResult(
            team_id=1,
            team_name="Team A",
            strategy_name="balanced",
            picks=(picks[0], picks[3]),
            category_totals={
                StatCategory.HR: 30.0,
                StatCategory.SB: 15.0,
                StatCategory.K: 100.0,
                StatCategory.ERA: 3.50,
            },
        ),
        TeamResult(
            team_id=2,
            team_name="Team B",
            strategy_name="balanced",
            picks=(picks[1], picks[2]),
            category_totals={
                StatCategory.HR: 25.0,
                StatCategory.SB: 20.0,
                StatCategory.K: 120.0,
                StatCategory.ERA: 3.20,
            },
        ),
    )
    config = SimulationConfig(
        teams=(
            TeamConfig(team_id=1, name="Team A", strategy=strategy),
            TeamConfig(team_id=2, name="Team B", strategy=strategy),
        ),
        roster_config=RosterConfig(slots=()),
        total_rounds=2,
    )
    return SimulationResult(pick_log=picks, team_results=team_results, config=config)


def _capture_output(func, *args):
    """Capture output from a print function by temporarily replacing the console."""
    output = StringIO()
    test_console = Console(file=output, force_terminal=True)
    original_console = simulation_report.console
    simulation_report.console = test_console
    try:
        func(*args)
    finally:
        simulation_report.console = original_console
    return output.getvalue()


class TestPrintPickLog:
    def test_contains_header(self) -> None:
        result = _make_result()
        output = _capture_output(simulation_report.print_pick_log, result)
        assert "Pick" in output
        assert "Team" in output
        assert "Player" in output

    def test_contains_all_picks(self) -> None:
        result = _make_result()
        output = _capture_output(simulation_report.print_pick_log, result)
        assert "Player 1" in output
        assert "Player 2" in output
        assert "Player 3" in output
        assert "Player 4" in output

    def test_shows_round_numbers(self) -> None:
        result = _make_result()
        output = _capture_output(simulation_report.print_pick_log, result)
        assert "Rd" in output or "1" in output


class TestPrintTeamRoster:
    def test_contains_team_name(self) -> None:
        result = _make_result()
        team = result.team_results[0]
        output = _capture_output(simulation_report.print_team_roster, team)
        assert "Team A" in output

    def test_contains_player_names(self) -> None:
        result = _make_result()
        team = result.team_results[0]
        output = _capture_output(simulation_report.print_team_roster, team)
        assert "Player 1" in output
        assert "Player 4" in output

    def test_contains_position(self) -> None:
        result = _make_result()
        team = result.team_results[0]
        output = _capture_output(simulation_report.print_team_roster, team)
        assert "OF" in output
        assert "SP" in output


class TestPrintStandings:
    def test_contains_team_names(self) -> None:
        result = _make_result()
        output = _capture_output(simulation_report.print_standings, result)
        assert "Team A" in output
        assert "Team B" in output

    def test_contains_category_headers(self) -> None:
        result = _make_result()
        output = _capture_output(simulation_report.print_standings, result)
        assert "HR" in output

    def test_contains_total_column(self) -> None:
        result = _make_result()
        output = _capture_output(simulation_report.print_standings, result)
        assert "Total" in output


class TestPrintFullReport:
    def test_combines_sections(self) -> None:
        result = _make_result()
        output = _capture_output(simulation_report.print_full_report, result)
        # Should contain elements from all sections
        assert "Team A" in output
        assert "Team B" in output
        assert "Player 1" in output
        assert "Total" in output
