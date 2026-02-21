from pathlib import Path

import pytest

from fantasy_baseball_manager.config_yahoo import (
    YahooConfig,
    YahooConfigError,
    YahooLeagueConfig,
    load_yahoo_config,
    resolve_default_league,
)


def _write_toml(tmp_path: Path, content: str) -> None:
    path = tmp_path / "fbm.toml"
    path.write_text(content)


class TestLoadYahooConfig:
    def test_parses_valid_config(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "my-client-id"
client_secret = "my-client-secret"
default_league = "keeper"

[yahoo.leagues.keeper]
league_id = 12345

[yahoo.leagues.redraft]
league_id = 67890
""",
        )
        config = load_yahoo_config(tmp_path)
        assert config.client_id == "my-client-id"
        assert config.client_secret == "my-client-secret"
        assert config.default_league == "keeper"
        assert len(config.leagues) == 2
        assert config.leagues["keeper"].name == "keeper"
        assert config.leagues["keeper"].league_id == 12345
        assert config.leagues["redraft"].league_id == 67890

    def test_missing_yahoo_section_raises(self, tmp_path: Path) -> None:
        _write_toml(tmp_path, "[common]\ndata_dir = './data'\n")
        with pytest.raises(YahooConfigError, match="No \\[yahoo\\] section"):
            load_yahoo_config(tmp_path)

    def test_missing_client_id_raises(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[yahoo]
client_secret = "secret"
""",
        )
        with pytest.raises(YahooConfigError, match="client_id"):
            load_yahoo_config(tmp_path)

    def test_missing_client_secret_raises(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
""",
        )
        with pytest.raises(YahooConfigError, match="client_secret"):
            load_yahoo_config(tmp_path)

    def test_missing_toml_raises(self, tmp_path: Path) -> None:
        with pytest.raises(YahooConfigError, match="fbm.toml not found"):
            load_yahoo_config(tmp_path)

    def test_no_leagues_section_returns_empty_leagues(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"
""",
        )
        config = load_yahoo_config(tmp_path)
        assert config.leagues == {}
        assert config.default_league is None

    def test_league_missing_league_id_raises(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"

[yahoo.leagues.keeper]
name = "keeper"
""",
        )
        with pytest.raises(YahooConfigError, match="league_id"):
            load_yahoo_config(tmp_path)


class TestResolveDefaultLeague:
    def test_env_var_overrides_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FBM_YAHOO_LEAGUE", "redraft")
        config = YahooConfig(
            client_id="id",
            client_secret="secret",
            default_league="keeper",
            leagues={
                "keeper": YahooLeagueConfig(name="keeper", league_id=12345),
                "redraft": YahooLeagueConfig(name="redraft", league_id=67890),
            },
        )
        assert resolve_default_league(config) == "redraft"

    def test_falls_back_to_config_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FBM_YAHOO_LEAGUE", raising=False)
        config = YahooConfig(
            client_id="id",
            client_secret="secret",
            default_league="keeper",
            leagues={"keeper": YahooLeagueConfig(name="keeper", league_id=12345)},
        )
        assert resolve_default_league(config) == "keeper"

    def test_no_default_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FBM_YAHOO_LEAGUE", raising=False)
        config = YahooConfig(
            client_id="id",
            client_secret="secret",
            default_league=None,
            leagues={"keeper": YahooLeagueConfig(name="keeper", league_id=12345)},
        )
        with pytest.raises(YahooConfigError, match="No default league"):
            resolve_default_league(config)

    def test_env_var_unknown_league_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FBM_YAHOO_LEAGUE", "nonexistent")
        config = YahooConfig(
            client_id="id",
            client_secret="secret",
            default_league=None,
            leagues={"keeper": YahooLeagueConfig(name="keeper", league_id=12345)},
        )
        with pytest.raises(YahooConfigError, match="nonexistent"):
            resolve_default_league(config)
