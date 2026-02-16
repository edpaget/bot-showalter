import pytest

from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)


class TestStatType:
    def test_from_string(self) -> None:
        assert StatType("counting") is StatType.COUNTING
        assert StatType("rate") is StatType.RATE

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            StatType("invalid")


class TestDirection:
    def test_from_string(self) -> None:
        assert Direction("higher") is Direction.HIGHER
        assert Direction("lower") is Direction.LOWER

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            Direction("invalid")


class TestLeagueFormat:
    def test_from_string(self) -> None:
        assert LeagueFormat("h2h_categories") is LeagueFormat.H2H_CATEGORIES
        assert LeagueFormat("roto") is LeagueFormat.ROTO

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            LeagueFormat("invalid")


class TestCategoryConfig:
    def test_counting_stat(self) -> None:
        cat = CategoryConfig(
            key="hr",
            name="Home Runs",
            stat_type=StatType.COUNTING,
            direction=Direction.HIGHER,
        )
        assert cat.key == "hr"
        assert cat.name == "Home Runs"
        assert cat.stat_type is StatType.COUNTING
        assert cat.direction is Direction.HIGHER
        assert cat.numerator is None
        assert cat.denominator is None

    def test_rate_stat(self) -> None:
        cat = CategoryConfig(
            key="avg",
            name="Batting Average",
            stat_type=StatType.RATE,
            direction=Direction.HIGHER,
            numerator="h",
            denominator="ab",
        )
        assert cat.numerator == "h"
        assert cat.denominator == "ab"

    def test_frozen(self) -> None:
        cat = CategoryConfig(
            key="hr",
            name="Home Runs",
            stat_type=StatType.COUNTING,
            direction=Direction.HIGHER,
        )
        with pytest.raises(AttributeError):
            cat.key = "sb"  # type: ignore[misc]


class TestLeagueSettings:
    @pytest.fixture
    def batting_categories(self) -> tuple[CategoryConfig, ...]:
        return (
            CategoryConfig(
                key="hr",
                name="Home Runs",
                stat_type=StatType.COUNTING,
                direction=Direction.HIGHER,
            ),
        )

    @pytest.fixture
    def pitching_categories(self) -> tuple[CategoryConfig, ...]:
        return (
            CategoryConfig(
                key="era",
                name="ERA",
                stat_type=StatType.RATE,
                direction=Direction.LOWER,
                numerator="er",
                denominator="ip",
            ),
        )

    def test_construct(
        self,
        batting_categories: tuple[CategoryConfig, ...],
        pitching_categories: tuple[CategoryConfig, ...],
    ) -> None:
        settings = LeagueSettings(
            name="main",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=12,
            budget=260,
            roster_batters=14,
            roster_pitchers=9,
            batting_categories=batting_categories,
            pitching_categories=pitching_categories,
        )
        assert settings.name == "main"
        assert settings.format is LeagueFormat.H2H_CATEGORIES
        assert settings.teams == 12
        assert settings.budget == 260
        assert settings.roster_batters == 14
        assert settings.roster_pitchers == 9
        assert settings.batting_categories == batting_categories
        assert settings.pitching_categories == pitching_categories

    def test_defaults(
        self,
        batting_categories: tuple[CategoryConfig, ...],
        pitching_categories: tuple[CategoryConfig, ...],
    ) -> None:
        settings = LeagueSettings(
            name="main",
            format=LeagueFormat.ROTO,
            teams=10,
            budget=260,
            roster_batters=14,
            roster_pitchers=9,
            batting_categories=batting_categories,
            pitching_categories=pitching_categories,
        )
        assert settings.roster_util == 0
        assert settings.positions == {}

    def test_frozen(
        self,
        batting_categories: tuple[CategoryConfig, ...],
        pitching_categories: tuple[CategoryConfig, ...],
    ) -> None:
        settings = LeagueSettings(
            name="main",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=12,
            budget=260,
            roster_batters=14,
            roster_pitchers=9,
            batting_categories=batting_categories,
            pitching_categories=pitching_categories,
        )
        with pytest.raises(AttributeError):
            settings.teams = 10  # type: ignore[misc]
