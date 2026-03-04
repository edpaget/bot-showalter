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


def load_yahoo_config(config_dir: Path) -> YahooConfig:
    """Load Yahoo Fantasy configuration from fbm.toml (+ fbm.local.toml overlay)."""
    data = load_toml(config_dir)

    yahoo = data.get("yahoo")
    if yahoo is None:
        raise YahooConfigError("No [yahoo] section in fbm.toml")

    client_id: str = os.environ.get("FBM_YAHOO_CLIENT_ID") or _require_field(yahoo, "client_id", "[yahoo]")
    client_secret: str = os.environ.get("FBM_YAHOO_CLIENT_SECRET") or _require_field(yahoo, "client_secret", "[yahoo]")
    default_league: str | None = yahoo.get("default_league")

    leagues: dict[str, YahooLeagueConfig] = {}
    raw_leagues = yahoo.get("leagues", {})
    for name, raw_league in raw_leagues.items():
        league_id: int = _require_field(raw_league, "league_id", f"[yahoo.leagues.{name}]")
        keeper: bool = bool(raw_league.get("keeper", False))
        leagues[name] = YahooLeagueConfig(name=name, league_id=league_id, keeper=keeper)

    return YahooConfig(
        client_id=client_id,
        client_secret=client_secret,
        default_league=default_league,
        leagues=leagues,
    )


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
