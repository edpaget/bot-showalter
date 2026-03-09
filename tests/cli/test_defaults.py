from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.cli._defaults import CliDefaults, load_cli_defaults

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadCliDefaults:
    def test_loads_from_toml(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text('[common]\nsystem = "custom-sys"\nversion = "v2"\n')
        defaults = load_cli_defaults(tmp_path)
        assert defaults.system == "custom-sys"
        assert defaults.version == "v2"

    def test_local_overrides_base(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text('[common]\nsystem = "base-sys"\nversion = "base-ver"\n')
        (tmp_path / "fbm.local.toml").write_text('[common]\nversion = "local-ver"\n')
        defaults = load_cli_defaults(tmp_path)
        assert defaults.system == "base-sys"
        assert defaults.version == "local-ver"

    def test_fallback_when_keys_missing(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text("[common]\n")
        defaults = load_cli_defaults(tmp_path)
        assert defaults.system == "zar"
        assert defaults.version == "production"

    def test_fallback_when_no_common_section(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text("")
        defaults = load_cli_defaults(tmp_path)
        assert defaults.system == "zar"
        assert defaults.version == "production"

    def test_season_populated(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text("[common]\n")
        defaults = load_cli_defaults(tmp_path)
        assert isinstance(defaults.season, int)
        assert defaults.season >= 2025

    def test_returns_frozen_dataclass(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text("[common]\n")
        defaults = load_cli_defaults(tmp_path)
        assert isinstance(defaults, CliDefaults)
