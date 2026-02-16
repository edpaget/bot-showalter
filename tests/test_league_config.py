from pathlib import Path

import pytest

from fantasy_baseball_manager.config_league import (
    LeagueConfigError,
    list_leagues,
    load_league,
    parse_category,
    parse_league,
    validate_category,
    validate_league,
)
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)


# -- Fixtures ----------------------------------------------------------------


def _counting_cat(key: str = "hr", name: str = "Home Runs") -> CategoryConfig:
    return CategoryConfig(key=key, name=name, stat_type=StatType.COUNTING, direction=Direction.HIGHER)


def _rate_cat(
    key: str = "avg",
    name: str = "Batting Average",
    numerator: str = "h",
    denominator: str = "ab",
) -> CategoryConfig:
    return CategoryConfig(
        key=key,
        name=name,
        stat_type=StatType.RATE,
        direction=Direction.HIGHER,
        numerator=numerator,
        denominator=denominator,
    )


def _minimal_settings(**overrides: object) -> LeagueSettings:
    defaults: dict[str, object] = {
        "name": "test",
        "format": LeagueFormat.H2H_CATEGORIES,
        "teams": 12,
        "budget": 260,
        "roster_batters": 14,
        "roster_pitchers": 9,
        "batting_categories": (_counting_cat(),),
        "pitching_categories": (_rate_cat(key="era", name="ERA", numerator="er", denominator="ip"),),
    }
    defaults.update(overrides)
    return LeagueSettings(**defaults)  # type: ignore[arg-type]


# -- validate_category -------------------------------------------------------


class TestValidateCategory:
    def test_valid_counting(self) -> None:
        validate_category(_counting_cat())

    def test_valid_rate(self) -> None:
        validate_category(_rate_cat())

    def test_rate_missing_numerator(self) -> None:
        cat = CategoryConfig(
            key="avg",
            name="Batting Average",
            stat_type=StatType.RATE,
            direction=Direction.HIGHER,
            denominator="ab",
        )
        with pytest.raises(LeagueConfigError, match="numerator"):
            validate_category(cat)

    def test_rate_missing_denominator(self) -> None:
        cat = CategoryConfig(
            key="avg",
            name="Batting Average",
            stat_type=StatType.RATE,
            direction=Direction.HIGHER,
            numerator="h",
        )
        with pytest.raises(LeagueConfigError, match="denominator"):
            validate_category(cat)

    def test_counting_with_numerator(self) -> None:
        cat = CategoryConfig(
            key="hr",
            name="Home Runs",
            stat_type=StatType.COUNTING,
            direction=Direction.HIGHER,
            numerator="hr",
        )
        with pytest.raises(LeagueConfigError, match="numerator"):
            validate_category(cat)

    def test_counting_with_denominator(self) -> None:
        cat = CategoryConfig(
            key="hr",
            name="Home Runs",
            stat_type=StatType.COUNTING,
            direction=Direction.HIGHER,
            denominator="ab",
        )
        with pytest.raises(LeagueConfigError, match="denominator"):
            validate_category(cat)


# -- validate_league ---------------------------------------------------------


class TestValidateLeague:
    def test_valid(self) -> None:
        validate_league(_minimal_settings())

    def test_teams_zero(self) -> None:
        with pytest.raises(LeagueConfigError, match="teams"):
            validate_league(_minimal_settings(teams=0))

    def test_teams_negative(self) -> None:
        with pytest.raises(LeagueConfigError, match="teams"):
            validate_league(_minimal_settings(teams=-1))

    def test_budget_negative(self) -> None:
        with pytest.raises(LeagueConfigError, match="budget"):
            validate_league(_minimal_settings(budget=-1))

    def test_empty_batting_categories(self) -> None:
        with pytest.raises(LeagueConfigError, match="batting_categories"):
            validate_league(_minimal_settings(batting_categories=()))

    def test_empty_pitching_categories(self) -> None:
        with pytest.raises(LeagueConfigError, match="pitching_categories"):
            validate_league(_minimal_settings(pitching_categories=()))

    def test_duplicate_batting_keys(self) -> None:
        cats = (_counting_cat("hr"), _counting_cat("hr"))
        with pytest.raises(LeagueConfigError, match="duplicate"):
            validate_league(_minimal_settings(batting_categories=cats))

    def test_duplicate_pitching_keys(self) -> None:
        cats = (
            _rate_cat(key="era", name="ERA", numerator="er", denominator="ip"),
            _rate_cat(key="era", name="ERA2", numerator="er", denominator="ip"),
        )
        with pytest.raises(LeagueConfigError, match="duplicate"):
            validate_league(_minimal_settings(pitching_categories=cats))

    def test_invalid_category_caught(self) -> None:
        bad_rate = CategoryConfig(
            key="avg",
            name="AVG",
            stat_type=StatType.RATE,
            direction=Direction.HIGHER,
        )
        with pytest.raises(LeagueConfigError):
            validate_league(_minimal_settings(batting_categories=(bad_rate,)))


# -- parse_category ----------------------------------------------------------


class TestParseCategory:
    def test_counting(self) -> None:
        raw = {"key": "hr", "name": "Home Runs", "stat_type": "counting", "direction": "higher"}
        cat = parse_category(raw)
        assert cat == _counting_cat()

    def test_rate(self) -> None:
        raw = {
            "key": "avg",
            "name": "Batting Average",
            "stat_type": "rate",
            "direction": "higher",
            "numerator": "h",
            "denominator": "ab",
        }
        cat = parse_category(raw)
        assert cat == _rate_cat()

    def test_missing_key(self) -> None:
        raw = {"name": "HR", "stat_type": "counting", "direction": "higher"}
        with pytest.raises(LeagueConfigError, match="key"):
            parse_category(raw)

    def test_missing_name(self) -> None:
        raw = {"key": "hr", "stat_type": "counting", "direction": "higher"}
        with pytest.raises(LeagueConfigError, match="name"):
            parse_category(raw)

    def test_missing_stat_type(self) -> None:
        raw = {"key": "hr", "name": "HR", "direction": "higher"}
        with pytest.raises(LeagueConfigError, match="stat_type"):
            parse_category(raw)

    def test_missing_direction(self) -> None:
        raw = {"key": "hr", "name": "HR", "stat_type": "counting"}
        with pytest.raises(LeagueConfigError, match="direction"):
            parse_category(raw)

    def test_invalid_stat_type(self) -> None:
        raw = {"key": "hr", "name": "HR", "stat_type": "bad", "direction": "higher"}
        with pytest.raises(LeagueConfigError, match="stat_type"):
            parse_category(raw)

    def test_invalid_direction(self) -> None:
        raw = {"key": "hr", "name": "HR", "stat_type": "counting", "direction": "bad"}
        with pytest.raises(LeagueConfigError, match="direction"):
            parse_category(raw)


# -- parse_league ------------------------------------------------------------


class TestParseLeague:
    def test_minimal(self) -> None:
        raw = {
            "format": "h2h_categories",
            "teams": 12,
            "budget": 260,
            "roster_batters": 14,
            "roster_pitchers": 9,
            "batting_categories": [
                {"key": "hr", "name": "Home Runs", "stat_type": "counting", "direction": "higher"},
            ],
            "pitching_categories": [
                {
                    "key": "era",
                    "name": "ERA",
                    "stat_type": "rate",
                    "direction": "lower",
                    "numerator": "er",
                    "denominator": "ip",
                },
            ],
        }
        settings = parse_league("main", raw)
        assert settings.name == "main"
        assert settings.format is LeagueFormat.H2H_CATEGORIES
        assert settings.teams == 12
        assert settings.budget == 260
        assert settings.roster_batters == 14
        assert settings.roster_pitchers == 9
        assert len(settings.batting_categories) == 1
        assert len(settings.pitching_categories) == 1
        assert settings.roster_util == 0
        assert settings.positions == {}

    def test_with_optional_fields(self) -> None:
        raw = {
            "format": "roto",
            "teams": 10,
            "budget": 260,
            "roster_batters": 14,
            "roster_pitchers": 9,
            "roster_util": 2,
            "positions": {"c": 1, "ss": 1, "of": 5},
            "batting_categories": [
                {"key": "hr", "name": "Home Runs", "stat_type": "counting", "direction": "higher"},
            ],
            "pitching_categories": [
                {
                    "key": "era",
                    "name": "ERA",
                    "stat_type": "rate",
                    "direction": "lower",
                    "numerator": "er",
                    "denominator": "ip",
                },
            ],
        }
        settings = parse_league("side", raw)
        assert settings.roster_util == 2
        assert settings.positions == {"c": 1, "ss": 1, "of": 5}

    def test_missing_required_field(self) -> None:
        raw = {
            "format": "h2h_categories",
            # teams missing
            "budget": 260,
            "roster_batters": 14,
            "roster_pitchers": 9,
            "batting_categories": [
                {"key": "hr", "name": "HR", "stat_type": "counting", "direction": "higher"},
            ],
            "pitching_categories": [
                {
                    "key": "era",
                    "name": "ERA",
                    "stat_type": "rate",
                    "direction": "lower",
                    "numerator": "er",
                    "denominator": "ip",
                },
            ],
        }
        with pytest.raises(LeagueConfigError, match="teams"):
            parse_league("main", raw)

    def test_invalid_format(self) -> None:
        raw = {
            "format": "points",
            "teams": 12,
            "budget": 260,
            "roster_batters": 14,
            "roster_pitchers": 9,
            "batting_categories": [
                {"key": "hr", "name": "HR", "stat_type": "counting", "direction": "higher"},
            ],
            "pitching_categories": [
                {
                    "key": "era",
                    "name": "ERA",
                    "stat_type": "rate",
                    "direction": "lower",
                    "numerator": "er",
                    "denominator": "ip",
                },
            ],
        }
        with pytest.raises(LeagueConfigError, match="format"):
            parse_league("main", raw)

    def test_validation_runs(self) -> None:
        """parse_league calls validate_league, so invalid data is caught."""
        raw = {
            "format": "h2h_categories",
            "teams": 0,
            "budget": 260,
            "roster_batters": 14,
            "roster_pitchers": 9,
            "batting_categories": [
                {"key": "hr", "name": "HR", "stat_type": "counting", "direction": "higher"},
            ],
            "pitching_categories": [
                {
                    "key": "era",
                    "name": "ERA",
                    "stat_type": "rate",
                    "direction": "lower",
                    "numerator": "er",
                    "denominator": "ip",
                },
            ],
        }
        with pytest.raises(LeagueConfigError, match="teams"):
            parse_league("main", raw)


# -- TOML round-trip helpers -------------------------------------------------


_FULL_TOML = """\
[leagues.main]
format = "h2h_categories"
teams = 12
budget = 260
roster_batters = 14
roster_pitchers = 9
roster_util = 1

[leagues.main.positions]
c = 1
ss = 1
of = 5

[[leagues.main.batting_categories]]
key = "hr"
name = "Home Runs"
stat_type = "counting"
direction = "higher"

[[leagues.main.batting_categories]]
key = "avg"
name = "Batting Average"
stat_type = "rate"
direction = "higher"
numerator = "h"
denominator = "ab"

[[leagues.main.pitching_categories]]
key = "era"
name = "ERA"
stat_type = "rate"
direction = "lower"
numerator = "er"
denominator = "ip"

[leagues.side]
format = "roto"
teams = 10
budget = 260
roster_batters = 14
roster_pitchers = 9

[[leagues.side.batting_categories]]
key = "r"
name = "Runs"
stat_type = "counting"
direction = "higher"

[[leagues.side.pitching_categories]]
key = "k"
name = "Strikeouts"
stat_type = "counting"
direction = "higher"
"""


# -- load_league -------------------------------------------------------------


class TestLoadLeague:
    def test_load_existing(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text(_FULL_TOML)
        settings = load_league("main", tmp_path)
        assert settings.name == "main"
        assert settings.teams == 12
        assert settings.roster_util == 1
        assert settings.positions == {"c": 1, "ss": 1, "of": 5}
        assert len(settings.batting_categories) == 2
        assert len(settings.pitching_categories) == 1

    def test_load_second_league(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text(_FULL_TOML)
        settings = load_league("side", tmp_path)
        assert settings.name == "side"
        assert settings.format is LeagueFormat.ROTO
        assert settings.teams == 10

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(LeagueConfigError, match="fbm.toml"):
            load_league("main", tmp_path)

    def test_no_leagues_section(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text("[common]\ndata_dir = './data'\n")
        with pytest.raises(LeagueConfigError, match="leagues"):
            load_league("main", tmp_path)

    def test_league_not_found(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text(_FULL_TOML)
        with pytest.raises(LeagueConfigError, match="missing"):
            load_league("missing", tmp_path)


# -- list_leagues ------------------------------------------------------------


class TestListLeagues:
    def test_lists_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text(_FULL_TOML)
        assert list_leagues(tmp_path) == ["main", "side"]

    def test_no_file(self, tmp_path: Path) -> None:
        assert list_leagues(tmp_path) == []

    def test_no_leagues_section(self, tmp_path: Path) -> None:
        (tmp_path / "fbm.toml").write_text("[common]\ndata_dir = './data'\n")
        assert list_leagues(tmp_path) == []
