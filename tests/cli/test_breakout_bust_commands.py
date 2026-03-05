from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.cli.factory import BreakoutBustReportContext
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.models.protocols import ModelConfig, PredictResult

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pytest

runner = CliRunner()


def _fake_predictions() -> list[dict[str, Any]]:
    return [
        {
            "player_id": 1,
            "player_name": "Mike Trout",
            "player_type": "batter",
            "position": "OF",
            "p_breakout": 0.65,
            "p_bust": 0.10,
            "p_neutral": 0.25,
            "top_features": [("exit_velo", 0.3), ("age", 0.2)],
        },
        {
            "player_id": 2,
            "player_name": "Aaron Judge",
            "player_type": "batter",
            "position": "OF",
            "p_breakout": 0.45,
            "p_bust": 0.20,
            "p_neutral": 0.35,
            "top_features": [("barrel_rate", 0.4)],
        },
        {
            "player_id": 3,
            "player_name": "Gerrit Cole",
            "player_type": "pitcher",
            "position": "SP",
            "p_breakout": 0.30,
            "p_bust": 0.50,
            "p_neutral": 0.20,
            "top_features": [("whiff_rate", 0.5)],
        },
    ]


class _FakeModel:
    @property
    def name(self) -> str:
        return "breakout-bust"

    @property
    def description(self) -> str:
        return "fake"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"predict"})

    @property
    def artifact_type(self) -> str:
        return "breakout-bust-classifier"

    def predict(self, config: ModelConfig) -> PredictResult:
        return PredictResult(
            model_name="breakout-bust",
            predictions=_fake_predictions(),
            output_path="",
        )


@contextmanager
def _fake_context(data_dir: str) -> Iterator[BreakoutBustReportContext]:
    conn = create_connection(":memory:")
    try:
        yield BreakoutBustReportContext(conn=conn, model=_FakeModel())  # type: ignore[arg-type]
    finally:
        conn.close()


class TestBreakoutCandidatesCommand:
    def test_help(self) -> None:
        result = runner.invoke(app, ["report", "breakout-candidates", "--help"])
        assert result.exit_code == 0
        assert "breakout" in result.output.lower()

    def test_prints_ranked_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.report.build_breakout_bust_report_context",
            _fake_context,
        )
        result = runner.invoke(app, ["report", "breakout-candidates", "--season", "2023"])
        assert result.exit_code == 0, result.output
        assert "Mike Trout" in result.output
        assert "Aaron Judge" in result.output

    def test_filter_player_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.report.build_breakout_bust_report_context",
            _fake_context,
        )
        result = runner.invoke(app, ["report", "breakout-candidates", "--season", "2023", "--player-type", "pitcher"])
        assert result.exit_code == 0, result.output
        assert "Gerrit Cole" in result.output
        assert "Mike Trout" not in result.output

    def test_min_probability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.report.build_breakout_bust_report_context",
            _fake_context,
        )
        result = runner.invoke(app, ["report", "breakout-candidates", "--season", "2023", "--min-probability", "0.5"])
        assert result.exit_code == 0, result.output
        assert "Mike Trout" in result.output
        assert "Gerrit Cole" not in result.output

    def test_top_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.report.build_breakout_bust_report_context",
            _fake_context,
        )
        result = runner.invoke(app, ["report", "breakout-candidates", "--season", "2023", "--top", "1"])
        assert result.exit_code == 0, result.output
        assert "Mike Trout" in result.output
        # Second highest p_breakout should be excluded
        assert "Gerrit Cole" not in result.output


class TestBustRisksCommand:
    def test_help(self) -> None:
        result = runner.invoke(app, ["report", "bust-risks", "--help"])
        assert result.exit_code == 0
        assert "bust" in result.output.lower()

    def test_prints_ranked_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.report.build_breakout_bust_report_context",
            _fake_context,
        )
        result = runner.invoke(app, ["report", "bust-risks", "--season", "2023"])
        assert result.exit_code == 0, result.output
        # Gerrit Cole has highest p_bust (0.50), should be first
        assert "Gerrit Cole" in result.output
        assert "Bust Risks" in result.output

    def test_filter_player_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.report.build_breakout_bust_report_context",
            _fake_context,
        )
        result = runner.invoke(app, ["report", "bust-risks", "--season", "2023", "--player-type", "batter"])
        assert result.exit_code == 0, result.output
        assert "Aaron Judge" in result.output
        assert "Gerrit Cole" not in result.output
