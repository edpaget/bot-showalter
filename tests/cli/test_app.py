from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.models.batting.marcel import MarcelModel
from fantasy_baseball_manager.models.registry import _clear, register

runner = CliRunner()


def _ensure_marcel_registered() -> None:
    """Clear and re-register marcel so each test starts with a known state."""
    _clear()
    register("marcel")(MarcelModel)


class TestListCommand:
    def test_list_shows_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "marcel" in result.output

    def test_list_empty_registry(self) -> None:
        _clear()
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No models registered" in result.output


class TestInfoCommand:
    def test_info_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["info", "marcel"])
        assert result.exit_code == 0
        assert "marcel" in result.output
        assert "batting" in result.output
        assert "prepare" in result.output
        assert "train" in result.output
        assert "evaluate" in result.output

    def test_info_unknown_model(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["info", "nonexistent"])
        assert result.exit_code != 0


class TestActionCommands:
    def test_train_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["train", "marcel"])
        assert result.exit_code == 0
        assert "marcel" in result.output.lower()

    def test_prepare_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["prepare", "marcel"])
        assert result.exit_code == 0

    def test_evaluate_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["evaluate", "marcel"])
        assert result.exit_code == 0

    def test_finetune_marcel_fails(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["finetune", "marcel"])
        assert result.exit_code != 0
        assert "does not support" in result.output.lower()

    def test_predict_marcel_fails(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["predict", "marcel"])
        assert result.exit_code != 0

    def test_ablate_marcel_fails(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["ablate", "marcel"])
        assert result.exit_code != 0

    def test_train_with_output_dir(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["train", "marcel", "--output-dir", "/tmp/out"])
        assert result.exit_code == 0

    def test_train_with_seasons(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["train", "marcel", "--season", "2023", "--season", "2024"])
        assert result.exit_code == 0
