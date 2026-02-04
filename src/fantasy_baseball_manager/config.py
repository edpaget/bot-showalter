from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

from config import ConfigurationSet, config_from_dict, config_from_env, config_from_yaml

from fantasy_baseball_manager.valuation.models import LeagueSettings, ScoringStyle, StatCategory


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
        league_dict = cast("dict[str, object]", overrides.get("league", {}))
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
        "team_count": 12,
        "scoring_style": "h2h_each_category",
        "batting_categories": ["HR", "SB", "OBP"],
        "pitching_categories": ["K", "ERA", "WHIP"],
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
    *,
    league_id: str | None = None,
    season: int | None = None,
) -> ConfigurationSet:
    """Create a layered configuration.

    Priority (highest to lowest): explicit overrides > env vars > YAML file > defaults dict.

    Args:
        yaml_path: Path to the YAML config file.
        env_prefix: Prefix for environment variables.
        defaults: Default configuration values.
        league_id: Override the league ID (passed explicitly, not from global state).
        season: Override the season (passed explicitly, not from global state).
    """
    if defaults is None:
        defaults = _DEFAULTS

    layers = [
        config_from_env(env_prefix, separator="__", lowercase_keys=True),
        config_from_yaml(yaml_path, read_from_file=True, ignore_missing_paths=True),
        config_from_dict(defaults),
    ]

    # Build overrides from explicit parameters (preferred) or legacy global state
    overrides = _build_overrides(league_id, season)
    if not overrides and _cli_overrides:
        # Fallback to legacy global state for backwards compatibility
        overrides = _cli_overrides
    if overrides:
        layers.insert(0, config_from_dict(overrides))

    return ConfigurationSet(*layers)


def _build_overrides(league_id: str | None, season: int | None) -> dict[str, object]:
    """Build override dict from explicit parameters."""
    if league_id is None and season is None:
        return {}
    overrides: dict[str, object] = {}
    league_dict: dict[str, object] = {}
    if league_id is not None:
        league_dict["id"] = league_id
    if season is not None:
        league_dict["season"] = season
    if league_dict:
        overrides["league"] = league_dict
    return overrides


def _parse_stat_categories(raw: list[str]) -> tuple[StatCategory, ...]:
    return tuple(StatCategory(name) for name in raw)


def load_league_settings(cfg: ConfigurationSet | None = None) -> LeagueSettings:
    if cfg is None:
        cfg = create_config()
    team_count = int(str(cfg["league.team_count"]))
    scoring_style = ScoringStyle(str(cfg["league.scoring_style"]))
    batting_categories = _parse_stat_categories(list(cast("Iterable[str]", cfg["league.batting_categories"])))
    pitching_categories = _parse_stat_categories(list(cast("Iterable[str]", cfg["league.pitching_categories"])))
    return LeagueSettings(
        team_count=team_count,
        scoring_style=scoring_style,
        batting_categories=batting_categories,
        pitching_categories=pitching_categories,
    )
