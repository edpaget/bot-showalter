import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.config_toml import load_toml
from fantasy_baseball_manager.exceptions import FbmException

if TYPE_CHECKING:
    from pathlib import Path


class YahooConfigError(FbmException):
    """Raised when Yahoo configuration is invalid or missing."""


@dataclass(frozen=True)
class YahooLeagueConfig:
    name: str
    league_id: int
    keeper: bool = False
    keeper_format: str = "auction"
    max_keepers: int | None = None


@dataclass(frozen=True)
class YahooConfig:
    client_id: str
    client_secret: str
    default_league: str | None
    leagues: dict[str, YahooLeagueConfig]


def _require_field(raw: dict[str, Any], field: str, context: str) -> Any:
    if field not in raw:
        raise YahooConfigError(f"{context}: missing required field '{field}'")
    return raw[field]


def _parse_yahoo_league(name: str, raw: dict[str, Any], context: str) -> YahooLeagueConfig:
    """Parse a single yahoo league config dict into a YahooLeagueConfig."""
    league_id: int = _require_field(raw, "league_id", context)
    keeper: bool = bool(raw.get("keeper", False))
    keeper_format: str = raw.get("keeper_format", "auction")
    if keeper_format not in ("auction", "best_n"):
        raise YahooConfigError(f"{context}: keeper_format must be 'auction' or 'best_n', got '{keeper_format}'")
    max_keepers: int | None = raw.get("max_keepers")
    if keeper_format == "best_n" and max_keepers is None:
        raise YahooConfigError(f"{context}: max_keepers is required when keeper_format is 'best_n'")
    return YahooLeagueConfig(
        name=name,
        league_id=league_id,
        keeper=keeper,
        keeper_format=keeper_format,
        max_keepers=max_keepers,
    )


def load_yahoo_config(config_dir: Path) -> YahooConfig:
    """Load Yahoo Fantasy configuration from fbm.toml (+ fbm.local.toml overlay).

    Looks for yahoo league config in two places (new format takes precedence):
    1. ``[leagues.*.yahoo]`` sub-tables (new format — shares canonical league name)
    2. ``[yahoo.leagues.*]`` (old format — backward compatibility)
    """
    data = load_toml(config_dir)

    yahoo = data.get("yahoo")
    if yahoo is None:
        raise YahooConfigError("No [yahoo] section in fbm.toml")

    client_id: str = os.environ.get("FBM_YAHOO_CLIENT_ID") or _require_field(yahoo, "client_id", "[yahoo]")
    client_secret: str = os.environ.get("FBM_YAHOO_CLIENT_SECRET") or _require_field(yahoo, "client_secret", "[yahoo]")
    default_league: str | None = yahoo.get("default_league")

    leagues: dict[str, YahooLeagueConfig] = {}

    # New format: [leagues.*.yahoo] sub-tables
    all_leagues = data.get("leagues", {})
    for name, league_data in all_leagues.items():
        if isinstance(league_data, dict) and "yahoo" in league_data:
            raw_yahoo = league_data["yahoo"]
            if isinstance(raw_yahoo, dict):
                leagues[name] = _parse_yahoo_league(name, raw_yahoo, f"[leagues.{name}.yahoo]")

    # Old format: [yahoo.leagues.*] — skip leagues already found in new format
    raw_leagues = yahoo.get("leagues", {})
    for name, raw_league in raw_leagues.items():
        if name not in leagues:
            leagues[name] = _parse_yahoo_league(name, raw_league, f"[yahoo.leagues.{name}]")

    return YahooConfig(
        client_id=client_id,
        client_secret=client_secret,
        default_league=default_league,
        leagues=leagues,
    )


def load_yahoo_league(name: str, config_dir: Path) -> YahooLeagueConfig | None:
    """Load a single league's yahoo config from ``[leagues.<name>.yahoo]``.

    Returns ``None`` if the league exists but has no ``yahoo`` sub-table.
    Raises ``YahooConfigError`` if the league name doesn't exist.
    """
    data = load_toml(config_dir)
    all_leagues = data.get("leagues", {})

    if name not in all_leagues:
        raise YahooConfigError(f"League '{name}' not found in fbm.toml")

    league_data = all_leagues[name]
    if not isinstance(league_data, dict) or "yahoo" not in league_data:
        return None

    raw_yahoo = league_data["yahoo"]
    return _parse_yahoo_league(name, raw_yahoo, f"[leagues.{name}.yahoo]")


def resolve_default_league(config: YahooConfig) -> str:
    """Resolve the default league name: FBM_YAHOO_LEAGUE env var -> config.default_league -> error."""
    env_league = os.environ.get("FBM_YAHOO_LEAGUE")
    if env_league is not None:
        if env_league not in config.leagues:
            raise YahooConfigError(f"League '{env_league}' (from FBM_YAHOO_LEAGUE) not found in config")
        return env_league

    if config.default_league is not None:
        return config.default_league

    raise YahooConfigError("No default league configured. Set FBM_YAHOO_LEAGUE or default_league in fbm.toml")
