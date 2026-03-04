from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.config_toml import deep_merge, load_toml

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestDeepMerge:
    def test_flat_merge(self) -> None:
        result = deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_override_flat_value(self) -> None:
        result = deep_merge({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_nested_merge(self) -> None:
        base: dict[str, Any] = {"pitcher": {"n_estimators": 100, "max_depth": 6}}
        override: dict[str, Any] = {"pitcher": {"learning_rate": 0.05}}
        result = deep_merge(base, override)
        assert result == {"pitcher": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.05}}

    def test_new_nested_key(self) -> None:
        base: dict[str, Any] = {"pitcher": {"n_estimators": 100}}
        override: dict[str, Any] = {"batter": {"lags": 3}}
        result = deep_merge(base, override)
        assert result == {"pitcher": {"n_estimators": 100}, "batter": {"lags": 3}}

    def test_new_top_level_key(self) -> None:
        result = deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self) -> None:
        base: dict[str, Any] = {"pitcher": {"n_estimators": 100}}
        override: dict[str, Any] = {"pitcher": {"learning_rate": 0.05}}
        deep_merge(base, override)
        assert base == {"pitcher": {"n_estimators": 100}}

    def test_empty_base(self) -> None:
        result = deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self) -> None:
        result = deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_replaces_non_dict_with_dict(self) -> None:
        base: dict[str, Any] = {"pitcher": "old_value"}
        override: dict[str, Any] = {"pitcher": {"learning_rate": 0.05}}
        result = deep_merge(base, override)
        assert result == {"pitcher": {"learning_rate": 0.05}}


class TestLoadToml:
    def test_base_only(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text('[common]\ndata_dir = "./mydata"\n')
        result = load_toml(tmp_path)
        assert result == {"common": {"data_dir": "./mydata"}}

    def test_local_only(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.local.toml").write_text('[common]\ndata_dir = "./local"\n')
        result = load_toml(tmp_path)
        assert result == {"common": {"data_dir": "./local"}}

    def test_both_files_merged(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text('[common]\ndata_dir = "./base"\n')
        (tmp_path / "fbm.local.toml").write_text("[models.marcel.params]\nlags = 3\n")
        result = load_toml(tmp_path)
        assert result == {
            "common": {"data_dir": "./base"},
            "models": {"marcel": {"params": {"lags": 3}}},
        }

    def test_local_overrides_base(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text('[common]\ndata_dir = "./base"\n')
        (tmp_path / "fbm.local.toml").write_text('[common]\ndata_dir = "./override"\n')
        result = load_toml(tmp_path)
        assert result["common"]["data_dir"] == "./override"

    def test_nested_merge(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text('[yahoo]\nclient_id = "base-id"\ndefault_league = "keeper"\n')
        (tmp_path / "fbm.local.toml").write_text('[yahoo]\nclient_id = "secret-id"\nclient_secret = "secret"\n')
        result = load_toml(tmp_path)
        assert result["yahoo"]["client_id"] == "secret-id"
        assert result["yahoo"]["client_secret"] == "secret"
        assert result["yahoo"]["default_league"] == "keeper"

    def test_no_files_returns_empty(self, tmp_path: Path) -> None:
        result = load_toml(tmp_path)
        assert result == {}

    def test_none_config_dir_uses_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "fbm.toml").write_text("[common]\nseasons = [2024]\n")
        result = load_toml(None)
        assert result == {"common": {"seasons": [2024]}}
