from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from config import ConfigurationSet

from fantasy_baseball_manager.config import clear_cli_overrides, create_config, load_league_settings, set_cli_overrides
from fantasy_baseball_manager.valuation.models import ScoringStyle, StatCategory

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all FANTASY__ env vars so tests are isolated from .envrc."""
    for key in list(os.environ):
        if key.startswith("FANTASY__"):
            monkeypatch.delenv(key)


def test_create_config_returns_defaults() -> None:
    cfg = create_config(yaml_path="/nonexistent/config.yaml")
    assert isinstance(cfg, ConfigurationSet)
    assert cfg["yahoo.client_id"] == ""
    assert cfg["yahoo.client_secret"] == ""
    assert cfg["yahoo.credentials_file"] == "~/.config/fbm/credentials.json"
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
    assert cfg["yahoo.credentials_file"] == "~/.config/fbm/credentials.json"
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


class TestCliOverrides:
    def teardown_method(self) -> None:
        clear_cli_overrides()

    def test_set_cli_overrides_league_id(self) -> None:
        set_cli_overrides({"league": {"id": "99999"}})
        cfg = create_config(yaml_path="/nonexistent/config.yaml")
        assert cfg["league.id"] == "99999"

    def test_set_cli_overrides_season(self) -> None:
        set_cli_overrides({"league": {"season": 2030}})
        cfg = create_config(yaml_path="/nonexistent/config.yaml")
        assert cfg["league.season"] == 2030

    def test_cli_overrides_take_priority_over_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("league:\n  id: yaml_league\n  season: 2026\n")
        set_cli_overrides({"league": {"id": "cli_league", "season": 2028}})
        cfg = create_config(yaml_path=str(yaml_file))
        assert cfg["league.id"] == "cli_league"
        assert cfg["league.season"] == 2028

    def test_clear_cli_overrides_resets(self) -> None:
        set_cli_overrides({"league": {"id": "override"}})
        clear_cli_overrides()
        cfg = create_config(yaml_path="/nonexistent/config.yaml")
        assert cfg["league.id"] == ""

    def test_no_overrides_does_not_add_layer(self) -> None:
        cfg = create_config(yaml_path="/nonexistent/config.yaml")
        assert cfg["league.id"] == ""
        assert cfg["league.season"] == 2025


class TestLoadLeagueSettings:
    def test_defaults(self) -> None:
        settings = load_league_settings(create_config(yaml_path="/nonexistent/config.yaml"))
        assert settings.team_count == 12
        assert settings.scoring_style is ScoringStyle.H2H_EACH_CATEGORY
        assert settings.batting_categories == (StatCategory.HR, StatCategory.SB, StatCategory.OBP)
        assert settings.pitching_categories == (StatCategory.K, StatCategory.ERA, StatCategory.WHIP)

    def test_yaml_overrides(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "league:\n"
            "  team_count: 10\n"
            "  scoring_style: roto\n"
            "  batting_categories:\n"
            "    - HR\n"
            "    - R\n"
            "    - RBI\n"
            "    - SB\n"
            "    - OBP\n"
            "  pitching_categories:\n"
            "    - K\n"
            "    - ERA\n"
            "    - WHIP\n"
            "    - NSVH\n"
        )
        settings = load_league_settings(create_config(yaml_path=str(yaml_file)))
        assert settings.team_count == 10
        assert settings.scoring_style is ScoringStyle.ROTO
        assert settings.batting_categories == (
            StatCategory.HR,
            StatCategory.R,
            StatCategory.RBI,
            StatCategory.SB,
            StatCategory.OBP,
        )
        assert settings.pitching_categories == (
            StatCategory.K,
            StatCategory.ERA,
            StatCategory.WHIP,
            StatCategory.NSVH,
        )

    def test_creates_config_when_none_passed(self) -> None:
        settings = load_league_settings()
        assert settings.team_count == 12
        assert settings.scoring_style is ScoringStyle.H2H_EACH_CATEGORY
