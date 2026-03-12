from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.config_toml import load_toml
from fantasy_baseball_manager.domain import (
    BudgetSplitMode,
    CategoryConfig,
    Direction,
    EligibilityRules,
    LeagueFormat,
    LeagueSettings,
    StatType,
    position_from_raw,
)
from fantasy_baseball_manager.exceptions import FbmException

if TYPE_CHECKING:
    from pathlib import Path


class LeagueConfigError(FbmException):
    """Raised when league configuration is invalid or missing."""


# -- Validation --------------------------------------------------------------


def validate_category(cat: CategoryConfig) -> None:
    if cat.stat_type is StatType.RATE:
        if cat.numerator is None:
            raise LeagueConfigError(f"Rate category '{cat.key}' requires numerator")
        if cat.denominator is None:
            raise LeagueConfigError(f"Rate category '{cat.key}' requires denominator")
    else:
        if cat.numerator is not None:
            raise LeagueConfigError(f"Counting category '{cat.key}' must not have numerator")
        if cat.denominator is not None:
            raise LeagueConfigError(f"Counting category '{cat.key}' must not have denominator")


def validate_league(settings: LeagueSettings) -> None:
    if settings.teams <= 0:
        raise LeagueConfigError(f"League '{settings.name}': teams must be > 0, got {settings.teams}")
    if settings.budget < 0:
        raise LeagueConfigError(f"League '{settings.name}': budget must be >= 0, got {settings.budget}")
    if not settings.batting_categories:
        raise LeagueConfigError(f"League '{settings.name}': batting_categories must not be empty")
    if not settings.pitching_categories:
        raise LeagueConfigError(f"League '{settings.name}': pitching_categories must not be empty")

    batting_keys = [c.key for c in settings.batting_categories]
    if len(batting_keys) != len(set(batting_keys)):
        raise LeagueConfigError(f"League '{settings.name}': duplicate batting category keys")

    pitching_keys = [c.key for c in settings.pitching_categories]
    if len(pitching_keys) != len(set(pitching_keys)):
        raise LeagueConfigError(f"League '{settings.name}': duplicate pitching category keys")

    for cat in settings.batting_categories:
        validate_category(cat)
    for cat in settings.pitching_categories:
        validate_category(cat)


# -- Parsing -----------------------------------------------------------------


def _require_field(raw: dict[str, Any], field: str, context: str) -> Any:
    if field not in raw:
        raise LeagueConfigError(f"{context}: missing required field '{field}'")
    return raw[field]


def parse_category(raw: dict[str, Any]) -> CategoryConfig:
    key = _require_field(raw, "key", "category")
    name = _require_field(raw, "name", "category")
    raw_stat_type = _require_field(raw, "stat_type", "category")
    raw_direction = _require_field(raw, "direction", "category")

    try:
        stat_type = StatType(raw_stat_type)
    except ValueError:
        raise LeagueConfigError(f"Category '{key}': invalid stat_type '{raw_stat_type}'") from None

    try:
        direction = Direction(raw_direction)
    except ValueError:
        raise LeagueConfigError(f"Category '{key}': invalid direction '{raw_direction}'") from None

    return CategoryConfig(
        key=key,
        name=name,
        stat_type=stat_type,
        direction=direction,
        numerator=raw.get("numerator"),
        denominator=raw.get("denominator"),
    )


def _parse_positions(raw: dict[str, Any], context: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in raw.items():
        try:
            pos = position_from_raw(key)
        except ValueError:
            raise LeagueConfigError(f"{context}: unknown position '{key}'") from None
        result[pos] = value
    return result


def parse_league(name: str, raw: dict[str, Any]) -> LeagueSettings:
    context = f"League '{name}'"

    raw_format = _require_field(raw, "format", context)
    try:
        league_format = LeagueFormat(raw_format)
    except ValueError:
        raise LeagueConfigError(f"{context}: invalid format '{raw_format}'") from None

    teams: int = _require_field(raw, "teams", context)
    budget: int = _require_field(raw, "budget", context)
    roster_batters: int = _require_field(raw, "roster_batters", context)
    roster_pitchers: int = _require_field(raw, "roster_pitchers", context)

    raw_batting = _require_field(raw, "batting_categories", context)
    raw_pitching = _require_field(raw, "pitching_categories", context)

    batting_categories = tuple(parse_category(c) for c in raw_batting)
    pitching_categories = tuple(parse_category(c) for c in raw_pitching)

    eligibility_raw = raw.get("eligibility", {})
    eligibility = EligibilityRules(**eligibility_raw)

    budget_split = BudgetSplitMode.ROSTER_SPOTS
    raw_budget_split = raw.get("budget_split")
    if raw_budget_split is not None:
        try:
            budget_split = BudgetSplitMode(raw_budget_split)
        except ValueError:
            raise LeagueConfigError(f"{context}: invalid budget_split '{raw_budget_split}'") from None

    budget_hitter_pct: float | None = None
    raw_budget_hitter_pct = raw.get("budget_hitter_pct")
    if raw_budget_hitter_pct is not None:
        budget_hitter_pct = float(raw_budget_hitter_pct)

    # Validate budget_hitter_pct / budget_split consistency
    if budget_split is BudgetSplitMode.FIXED_RATIO and budget_hitter_pct is None:
        raise LeagueConfigError(f"{context}: budget_hitter_pct is required when budget_split is 'fixed_ratio'")
    if budget_split is not BudgetSplitMode.FIXED_RATIO and budget_hitter_pct is not None:
        raise LeagueConfigError(f"{context}: budget_hitter_pct must not be set when budget_split is '{budget_split}'")

    settings = LeagueSettings(
        name=name,
        format=league_format,
        teams=teams,
        budget=budget,
        roster_batters=roster_batters,
        roster_pitchers=roster_pitchers,
        batting_categories=batting_categories,
        pitching_categories=pitching_categories,
        roster_util=raw.get("roster_util", 0),
        roster_bench=raw.get("roster_bench", 0),
        positions=_parse_positions(raw.get("positions", {}), context),
        pitcher_positions=_parse_positions(raw.get("pitcher_positions", {}), context),
        budget_split=budget_split,
        budget_hitter_pct=budget_hitter_pct,
        eligibility=eligibility,
    )

    validate_league(settings)
    return settings


# -- TOML loading ------------------------------------------------------------


def load_league(name: str, config_dir: Path) -> LeagueSettings:
    data = load_toml(config_dir)

    leagues = data.get("leagues")
    if leagues is None:
        raise LeagueConfigError("No [leagues] section in fbm.toml")

    if name not in leagues:
        raise LeagueConfigError(f"League '{name}' not found in fbm.toml")

    return parse_league(name, leagues[name])


def list_leagues(config_dir: Path) -> list[str]:
    data = load_toml(config_dir)

    leagues = data.get("leagues")
    if leagues is None:
        return []

    return sorted(leagues.keys())
