from pathlib import Path
from typing import Any

from fantasy_baseball_manager.config import _deep_merge, load_config


class TestLoadConfigDefaults:
    def test_no_toml_returns_defaults(self, tmp_path: Path) -> None:
        config = load_config(model_name="marcel", config_dir=tmp_path)
        assert config.data_dir == "./data"
        assert config.artifacts_dir == "./artifacts"
        assert config.seasons == []
        assert config.model_params == {}

    def test_cli_overrides_applied(self, tmp_path: Path) -> None:
        config = load_config(
            model_name="marcel",
            config_dir=tmp_path,
            output_dir="/tmp/out",
            seasons=[2023, 2024],
        )
        assert config.output_dir == "/tmp/out"
        assert config.seasons == [2023, 2024]


class TestLoadConfigFromToml:
    def test_common_section_loaded(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "fbm.toml"
        toml_path.write_text(
            '[common]\ndata_dir = "./mydata"\nartifacts_dir = "./myartifacts"\nseasons = [2021, 2022]\n'
        )
        config = load_config(model_name="marcel", config_dir=tmp_path)
        assert config.data_dir == "./mydata"
        assert config.artifacts_dir == "./myartifacts"
        assert config.seasons == [2021, 2022]

    def test_model_params_loaded(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "fbm.toml"
        toml_path.write_text("[models.marcel.params]\nweights = [5, 4, 3]\nregression_pct = 0.4\n")
        config = load_config(model_name="marcel", config_dir=tmp_path)
        assert config.model_params == {"weights": [5, 4, 3], "regression_pct": 0.4}

    def test_cli_overrides_beat_toml(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "fbm.toml"
        toml_path.write_text('[common]\ndata_dir = "./toml_data"\nseasons = [2020]\n')
        config = load_config(
            model_name="marcel",
            config_dir=tmp_path,
            output_dir="/cli/out",
            seasons=[2024],
        )
        assert config.data_dir == "./toml_data"
        assert config.output_dir == "/cli/out"
        assert config.seasons == [2024]

    def test_unrelated_model_params_ignored(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "fbm.toml"
        toml_path.write_text("[models.steamer.params]\nsmoothing = 0.9\n")
        config = load_config(model_name="marcel", config_dir=tmp_path)
        assert config.model_params == {}

    def test_version_loaded_from_toml(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "fbm.toml"
        toml_path.write_text('[models.marcel]\nversion = "v2.1"\n')
        config = load_config(model_name="marcel", config_dir=tmp_path)
        assert config.version == "v2.1"

    def test_tags_loaded_from_toml(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "fbm.toml"
        toml_path.write_text('[models.marcel]\n\n[models.marcel.tags]\nenv = "dev"\nowner = "bob"\n')
        config = load_config(model_name="marcel", config_dir=tmp_path)
        assert config.tags == {"env": "dev", "owner": "bob"}


class TestEnsembleConfigWithStatcastGBM:
    def test_ensemble_config_with_statcast_gbm_component(self, tmp_path: Path) -> None:
        """Recommended config: 60% Marcel / 40% statcast-gbm."""
        toml_path = tmp_path / "fbm.toml"
        toml_path.write_text('[models.ensemble.params.components]\nmarcel = 0.6\n"statcast-gbm" = 0.4\n')
        config = load_config(model_name="ensemble", config_dir=tmp_path)
        assert config.model_params["components"] == {"marcel": 0.6, "statcast-gbm": 0.4}

    def test_ensemble_config_blend_rates_mode(self, tmp_path: Path) -> None:
        """blend_rates mode with explicit stats list round-trips through TOML."""
        toml_path = tmp_path / "fbm.toml"
        toml_path.write_text(
            "[models.ensemble.params]\n"
            'mode = "blend_rates"\n'
            'pt_stat = "pa"\n'
            'stats = ["avg", "obp", "slg"]\n'
            "\n"
            "[models.ensemble.params.components]\n"
            "marcel = 0.6\n"
            '"statcast-gbm" = 0.4\n'
        )
        config = load_config(model_name="ensemble", config_dir=tmp_path)
        assert config.model_params["mode"] == "blend_rates"
        assert config.model_params["pt_stat"] == "pa"
        assert config.model_params["stats"] == ["avg", "obp", "slg"]
        assert config.model_params["components"] == {"marcel": 0.6, "statcast-gbm": 0.4}


class TestLoadConfigCliOverrides:
    def test_cli_version_overrides_toml(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "fbm.toml"
        toml_path.write_text('[models.marcel]\nversion = "1.0"\n')
        config = load_config(model_name="marcel", config_dir=tmp_path, version="2.0")
        assert config.version == "2.0"

    def test_cli_version_when_toml_has_none(self, tmp_path: Path) -> None:
        config = load_config(model_name="marcel", config_dir=tmp_path, version="3.0")
        assert config.version == "3.0"

    def test_cli_tags_merge_with_toml(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "fbm.toml"
        toml_path.write_text('[models.marcel.tags]\na = "1"\nb = "2"\n')
        config = load_config(model_name="marcel", config_dir=tmp_path, tags={"b": "override", "c": "3"})
        assert config.tags == {"a": "1", "b": "override", "c": "3"}

    def test_cli_tags_when_toml_has_none(self, tmp_path: Path) -> None:
        config = load_config(model_name="marcel", config_dir=tmp_path, tags={"x": "1"})
        assert config.tags == {"x": "1"}

    def test_cli_params_override_toml(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "fbm.toml"
        toml_path.write_text("[models.marcel.params]\nlags = 3\n")
        config = load_config(model_name="marcel", config_dir=tmp_path, model_params={"lags": 5})
        assert config.model_params["lags"] == 5

    def test_cli_params_merge_with_toml(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "fbm.toml"
        toml_path.write_text("[models.marcel.params]\na = 1\n")
        config = load_config(model_name="marcel", config_dir=tmp_path, model_params={"b": 2})
        assert config.model_params == {"a": 1, "b": 2}

    def test_cli_params_when_toml_has_none(self, tmp_path: Path) -> None:
        config = load_config(model_name="marcel", config_dir=tmp_path, model_params={"x": 1})
        assert config.model_params == {"x": 1}


class TestDeepMerge:
    def test_flat_merge(self) -> None:
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_override_flat_value(self) -> None:
        result = _deep_merge({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_nested_merge(self) -> None:
        base: dict[str, Any] = {"pitcher": {"n_estimators": 100, "max_depth": 6}}
        override: dict[str, Any] = {"pitcher": {"learning_rate": 0.05}}
        result = _deep_merge(base, override)
        assert result == {"pitcher": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.05}}

    def test_new_nested_key(self) -> None:
        base: dict[str, Any] = {"pitcher": {"n_estimators": 100}}
        override: dict[str, Any] = {"batter": {"lags": 3}}
        result = _deep_merge(base, override)
        assert result == {"pitcher": {"n_estimators": 100}, "batter": {"lags": 3}}

    def test_new_top_level_key(self) -> None:
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self) -> None:
        base: dict[str, Any] = {"pitcher": {"n_estimators": 100}}
        override: dict[str, Any] = {"pitcher": {"learning_rate": 0.05}}
        _deep_merge(base, override)
        assert base == {"pitcher": {"n_estimators": 100}}

    def test_empty_base(self) -> None:
        result = _deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self) -> None:
        result = _deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_replaces_non_dict_with_dict(self) -> None:
        base: dict[str, Any] = {"pitcher": "old_value"}
        override: dict[str, Any] = {"pitcher": {"learning_rate": 0.05}}
        result = _deep_merge(base, override)
        assert result == {"pitcher": {"learning_rate": 0.05}}


class TestDeepMergeConfig:
    def test_dotted_param_overrides_nested_toml(self, tmp_path: Path) -> None:
        """CLI --param pitcher.learning_rate=0.1 overrides nested TOML pitcher section."""
        toml_path = tmp_path / "fbm.toml"
        toml_path.write_text("[models.statcast-gbm.params.pitcher]\nlearning_rate = 0.05\nn_estimators = 200\n")
        config = load_config(
            model_name="statcast-gbm",
            config_dir=tmp_path,
            model_params={"pitcher": {"learning_rate": 0.1}},
        )
        assert config.model_params["pitcher"]["learning_rate"] == 0.1
        assert config.model_params["pitcher"]["n_estimators"] == 200

    def test_flat_param_backward_compat(self, tmp_path: Path) -> None:
        """Flat params still override flat TOML values."""
        toml_path = tmp_path / "fbm.toml"
        toml_path.write_text("[models.marcel.params]\nlags = 3\n")
        config = load_config(
            model_name="marcel",
            config_dir=tmp_path,
            model_params={"lags": 5},
        )
        assert config.model_params["lags"] == 5

    def test_nested_param_no_toml_section(self, tmp_path: Path) -> None:
        """Dotted CLI param works even when TOML has no params section."""
        config = load_config(
            model_name="statcast-gbm",
            config_dir=tmp_path,
            model_params={"pitcher": {"learning_rate": 0.1}},
        )
        assert config.model_params == {"pitcher": {"learning_rate": 0.1}}
