from __future__ import annotations

from typing import TYPE_CHECKING

from config import ConfigurationSet

from fantasy_baseball_manager.config import create_config

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_create_config_returns_defaults() -> None:
    cfg = create_config(yaml_path="/nonexistent/config.yaml")
    assert isinstance(cfg, ConfigurationSet)
    assert cfg["yahoo.client_id"] == ""
    assert cfg["yahoo.client_secret"] == ""
    assert cfg["yahoo.token_file"] == "oauth2.json"
    assert cfg["league.id"] == ""
    assert cfg["league.game_code"] == "mlb"
    assert cfg["league.season"] == 2025


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        "yahoo:\n" "  client_id: yaml_id\n" "  client_secret: yaml_secret\n" "league:\n" "  season: 2026\n"
    )
    cfg = create_config(yaml_path=str(yaml_file))
    assert cfg["yahoo.client_id"] == "yaml_id"
    assert cfg["yahoo.client_secret"] == "yaml_secret"
    assert cfg["league.season"] == 2026
    # Defaults still apply for unset keys
    assert cfg["yahoo.token_file"] == "oauth2.json"
    assert cfg["league.game_code"] == "mlb"


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("yahoo:\n  client_id: yaml_id\n")

    monkeypatch.setenv("FANTASY__YAHOO__CLIENT_ID", "env_id")
    monkeypatch.setenv("FANTASY__LEAGUE__SEASON", "2030")

    cfg = create_config(yaml_path=str(yaml_file))
    assert cfg["yahoo.client_id"] == "env_id"
    assert cfg["league.season"] == "2030"  # env vars are strings


def test_missing_yaml_falls_back_to_defaults() -> None:
    cfg = create_config(yaml_path="/does/not/exist.yaml")
    assert cfg["yahoo.client_id"] == ""
    assert cfg["league.season"] == 2025
