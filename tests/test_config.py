from pathlib import Path

from fantasy_baseball_manager.config import load_config


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
