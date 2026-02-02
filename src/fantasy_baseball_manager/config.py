from __future__ import annotations

from typing import Protocol

from config import ConfigurationSet, config_from_dict, config_from_env, config_from_yaml


class AppConfig(Protocol):
    def __getitem__(self, key: str) -> object: ...


_cli_overrides: dict[str, object] = {}


def set_cli_overrides(overrides: dict[str, object]) -> None:
    global _cli_overrides
    _cli_overrides = overrides


def clear_cli_overrides() -> None:
    global _cli_overrides
    _cli_overrides = {}


def apply_cli_overrides(league_id: str | None, season: int | None) -> None:
    overrides: dict[str, object] = {}
    if league_id is not None:
        overrides["league"] = {"id": league_id}
    if season is not None:
        league_dict: dict[str, object] = overrides.get("league", {})  # type: ignore[assignment]
        league_dict["season"] = season
        overrides["league"] = league_dict
    if overrides:
        set_cli_overrides(overrides)


_DEFAULTS: dict[str, object] = {
    "yahoo": {
        "client_id": "",
        "client_secret": "",
        "credentials_file": "~/.config/fbm/credentials.json",
    },
    "league": {
        "id": "",
        "game_code": "mlb",
        "season": 2025,
        "is_keeper": False,
    },
    "cache": {
        "db_path": "~/.config/fbm/cache.db",
        "positions_ttl": 86400,
        "rosters_ttl": 3600,
        "id_mappings_ttl": 604800,
        "draft_results_ttl": 86400,
    },
}


def create_config(
    yaml_path: str = "config.yaml",
    env_prefix: str = "FANTASY",
    defaults: dict[str, object] | None = None,
) -> ConfigurationSet:
    """Create a layered configuration.

    Priority (highest to lowest): env vars > YAML file > defaults dict.
    """
    if defaults is None:
        defaults = _DEFAULTS

    layers = [
        config_from_env(env_prefix, separator="__", lowercase_keys=True),
        config_from_yaml(yaml_path, read_from_file=True, ignore_missing_paths=True),
        config_from_dict(defaults),
    ]
    if _cli_overrides:
        layers.insert(0, config_from_dict(_cli_overrides))

    return ConfigurationSet(*layers)
