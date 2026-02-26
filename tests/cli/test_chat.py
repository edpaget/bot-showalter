from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


class TestChatCommand:
    def test_chat_command_exists(self) -> None:
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "chat" in result.output.lower() or "model" in result.output.lower()

    def test_missing_api_key_shows_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.output
