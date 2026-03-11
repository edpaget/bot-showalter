from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.config_yahoo import (
    YahooConfig,
    YahooConfigError,
    YahooLeagueConfig,
    load_yahoo_config,
    load_yahoo_league,
    resolve_default_league,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_toml(tmp_path: Path, content: str) -> None:
    path = tmp_path / "fbm.toml"
    path.write_text(content)


class TestLoadYahooConfig:
    def test_parses_valid_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FBM_YAHOO_CLIENT_ID", raising=False)
        monkeypatch.delenv("FBM_YAHOO_CLIENT_SECRET", raising=False)
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "my-client-id"
client_secret = "my-client-secret"
default_league = "keeper"

[yahoo.leagues.keeper]
league_id = 12345
keeper = true

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
        assert config.leagues["keeper"].keeper is True
        assert config.leagues["redraft"].league_id == 67890
        assert config.leagues["redraft"].keeper is False

    def test_missing_yahoo_section_raises(self, tmp_path: Path) -> None:
        _write_toml(tmp_path, "[common]\ndata_dir = './data'\n")
        with pytest.raises(YahooConfigError, match="No \\[yahoo\\] section"):
            load_yahoo_config(tmp_path)

    def test_missing_client_id_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FBM_YAHOO_CLIENT_ID", raising=False)
        monkeypatch.delenv("FBM_YAHOO_CLIENT_SECRET", raising=False)
        _write_toml(
            tmp_path,
            """
[yahoo]
client_secret = "secret"
""",
        )
        with pytest.raises(YahooConfigError, match="client_id"):
            load_yahoo_config(tmp_path)

    def test_missing_client_secret_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FBM_YAHOO_CLIENT_ID", raising=False)
        monkeypatch.delenv("FBM_YAHOO_CLIENT_SECRET", raising=False)
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
        with pytest.raises(YahooConfigError, match=r"No \[yahoo\] section"):
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

    def test_env_var_overrides_client_id(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FBM_YAHOO_CLIENT_ID", "env-client-id")
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "toml-client-id"
client_secret = "secret"
""",
        )
        config = load_yahoo_config(tmp_path)
        assert config.client_id == "env-client-id"

    def test_env_var_overrides_client_secret(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FBM_YAHOO_CLIENT_SECRET", "env-secret")
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "toml-secret"
""",
        )
        config = load_yahoo_config(tmp_path)
        assert config.client_secret == "env-secret"

    def test_env_var_supplies_client_id_when_missing_from_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FBM_YAHOO_CLIENT_ID", "env-client-id")
        _write_toml(
            tmp_path,
            """
[yahoo]
client_secret = "secret"
""",
        )
        config = load_yahoo_config(tmp_path)
        assert config.client_id == "env-client-id"

    def test_env_var_supplies_client_secret_when_missing_from_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FBM_YAHOO_CLIENT_SECRET", "env-secret")
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
""",
        )
        config = load_yahoo_config(tmp_path)
        assert config.client_secret == "env-secret"

    def test_local_toml_provides_yahoo_secrets(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FBM_YAHOO_CLIENT_ID", raising=False)
        monkeypatch.delenv("FBM_YAHOO_CLIENT_SECRET", raising=False)
        _write_toml(
            tmp_path,
            """
[yahoo]
default_league = "keeper"

[yahoo.leagues.keeper]
league_id = 12345
""",
        )
        (tmp_path / "fbm.local.toml").write_text('[yahoo]\nclient_id = "local-id"\nclient_secret = "local-secret"\n')
        config = load_yahoo_config(tmp_path)
        assert config.client_id == "local-id"
        assert config.client_secret == "local-secret"
        assert config.default_league == "keeper"
        assert config.leagues["keeper"].league_id == 12345

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

    def test_parses_keeper_format_and_max_keepers(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"

[yahoo.leagues.keeper]
league_id = 40214
keeper = true
keeper_format = "best_n"
max_keepers = 4
""",
        )
        config = load_yahoo_config(tmp_path)
        lc = config.leagues["keeper"]
        assert lc.keeper is True
        assert lc.keeper_format == "best_n"
        assert lc.max_keepers == 4

    def test_keeper_format_defaults_to_auction(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"

[yahoo.leagues.keeper]
league_id = 12345
keeper = true
""",
        )
        config = load_yahoo_config(tmp_path)
        lc = config.leagues["keeper"]
        assert lc.keeper_format == "auction"
        assert lc.max_keepers is None

    def test_best_n_without_max_keepers_raises(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"

[yahoo.leagues.keeper]
league_id = 12345
keeper_format = "best_n"
""",
        )
        with pytest.raises(YahooConfigError, match="max_keepers is required"):
            load_yahoo_config(tmp_path)

    def test_invalid_keeper_format_raises(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"

[yahoo.leagues.keeper]
league_id = 12345
keeper_format = "invalid"
""",
        )
        with pytest.raises(YahooConfigError, match="keeper_format must be"):
            load_yahoo_config(tmp_path)


class TestLoadYahooConfigNewFormat:
    """Tests for loading yahoo config from [leagues.*.yahoo] sub-tables."""

    def test_parses_from_leagues_yahoo_subtable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FBM_YAHOO_CLIENT_ID", raising=False)
        monkeypatch.delenv("FBM_YAHOO_CLIENT_SECRET", raising=False)
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "my-id"
client_secret = "my-secret"
default_league = "h2h"

[leagues.h2h]
format = "h2h_categories"
teams = 12
budget = 260
roster_batters = 9
roster_pitchers = 8
batting_categories = []
pitching_categories = []

[leagues.h2h.yahoo]
league_id = 12345
keeper = true
""",
        )
        config = load_yahoo_config(tmp_path)
        assert "h2h" in config.leagues
        assert config.leagues["h2h"].league_id == 12345
        assert config.leagues["h2h"].keeper is True
        assert config.leagues["h2h"].name == "h2h"

    def test_validates_keeper_fields_new_format(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"

[leagues.h2h.yahoo]
league_id = 12345
keeper = true
keeper_format = "best_n"
max_keepers = 4
""",
        )
        config = load_yahoo_config(tmp_path)
        lc = config.leagues["h2h"]
        assert lc.keeper_format == "best_n"
        assert lc.max_keepers == 4

    def test_missing_league_id_new_format_raises(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"

[leagues.h2h.yahoo]
keeper = true
""",
        )
        with pytest.raises(YahooConfigError, match="league_id"):
            load_yahoo_config(tmp_path)

    def test_invalid_keeper_format_new_format_raises(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"

[leagues.h2h.yahoo]
league_id = 12345
keeper_format = "invalid"
""",
        )
        with pytest.raises(YahooConfigError, match="keeper_format must be"):
            load_yahoo_config(tmp_path)

    def test_best_n_without_max_keepers_new_format_raises(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"

[leagues.h2h.yahoo]
league_id = 12345
keeper_format = "best_n"
""",
        )
        with pytest.raises(YahooConfigError, match="max_keepers is required"):
            load_yahoo_config(tmp_path)

    def test_old_format_still_works(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FBM_YAHOO_CLIENT_ID", raising=False)
        monkeypatch.delenv("FBM_YAHOO_CLIENT_SECRET", raising=False)
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"

[yahoo.leagues.keeper]
league_id = 12345
keeper = true
""",
        )
        config = load_yahoo_config(tmp_path)
        assert "keeper" in config.leagues
        assert config.leagues["keeper"].league_id == 12345

    def test_new_format_takes_precedence(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FBM_YAHOO_CLIENT_ID", raising=False)
        monkeypatch.delenv("FBM_YAHOO_CLIENT_SECRET", raising=False)
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"

[yahoo.leagues.h2h]
league_id = 99999

[leagues.h2h.yahoo]
league_id = 12345
""",
        )
        config = load_yahoo_config(tmp_path)
        assert config.leagues["h2h"].league_id == 12345

    def test_both_formats_merged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Old-format leagues not in new format are still picked up."""
        monkeypatch.delenv("FBM_YAHOO_CLIENT_ID", raising=False)
        monkeypatch.delenv("FBM_YAHOO_CLIENT_SECRET", raising=False)
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"

[yahoo.leagues.redraft]
league_id = 67890

[leagues.h2h.yahoo]
league_id = 12345
""",
        )
        config = load_yahoo_config(tmp_path)
        assert config.leagues["h2h"].league_id == 12345
        assert config.leagues["redraft"].league_id == 67890

    def test_no_yahoo_section_with_leagues_yahoo_subtable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If [yahoo] section has credentials but no leagues, new-format leagues are still found."""
        monkeypatch.delenv("FBM_YAHOO_CLIENT_ID", raising=False)
        monkeypatch.delenv("FBM_YAHOO_CLIENT_SECRET", raising=False)
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"

[leagues.h2h.yahoo]
league_id = 12345
""",
        )
        config = load_yahoo_config(tmp_path)
        assert len(config.leagues) == 1
        assert config.leagues["h2h"].league_id == 12345


class TestLoadYahooLeague:
    def test_returns_config_when_yahoo_exists(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[yahoo]
client_id = "id"
client_secret = "secret"

[leagues.h2h.yahoo]
league_id = 12345
keeper = true
""",
        )
        result = load_yahoo_league("h2h", tmp_path)
        assert result is not None
        assert result.league_id == 12345
        assert result.keeper is True
        assert result.name == "h2h"

    def test_returns_none_when_no_yahoo_subtable(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[leagues.h2h]
format = "h2h_categories"
teams = 12
budget = 260
roster_batters = 9
roster_pitchers = 8
batting_categories = []
pitching_categories = []
""",
        )
        result = load_yahoo_league("h2h", tmp_path)
        assert result is None

    def test_raises_when_league_not_found(self, tmp_path: Path) -> None:
        _write_toml(
            tmp_path,
            """
[leagues.h2h]
format = "h2h_categories"
teams = 12
budget = 260
roster_batters = 9
roster_pitchers = 8
batting_categories = []
pitching_categories = []
""",
        )
        with pytest.raises(YahooConfigError, match="nonexistent"):
            load_yahoo_league("nonexistent", tmp_path)

    def test_returns_none_when_no_leagues_section(self, tmp_path: Path) -> None:
        _write_toml(tmp_path, "[common]\ndata_dir = './data'\n")
        with pytest.raises(YahooConfigError, match="nonexistent"):
            load_yahoo_league("nonexistent", tmp_path)


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
