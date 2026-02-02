from dataclasses import FrozenInstanceError

import pytest

from fantasy_baseball_manager.valuation.models import (
    CategoryValue,
    LeagueSettings,
    PlayerValue,
    ScoringStyle,
    SGPDenominators,
    StatCategory,
)


class TestStatCategory:
    def test_batting_categories_exist(self) -> None:
        assert StatCategory.HR.value == "HR"
        assert StatCategory.R.value == "R"
        assert StatCategory.RBI.value == "RBI"
        assert StatCategory.SB.value == "SB"
        assert StatCategory.OBP.value == "OBP"

    def test_pitching_categories_exist(self) -> None:
        assert StatCategory.W.value == "W"
        assert StatCategory.K.value == "K"
        assert StatCategory.ERA.value == "ERA"
        assert StatCategory.WHIP.value == "WHIP"
        assert StatCategory.NSVH.value == "NSVH"


class TestScoringStyle:
    def test_values(self) -> None:
        assert ScoringStyle.H2H_CATEGORIES.value == "h2h_categories"
        assert ScoringStyle.ROTO.value == "roto"
        assert ScoringStyle.H2H_POINTS.value == "h2h_points"

    def test_from_string(self) -> None:
        assert ScoringStyle("h2h_categories") is ScoringStyle.H2H_CATEGORIES
        assert ScoringStyle("roto") is ScoringStyle.ROTO
        assert ScoringStyle("h2h_points") is ScoringStyle.H2H_POINTS

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ScoringStyle("invalid")


class TestLeagueSettings:
    def test_construction(self) -> None:
        settings = LeagueSettings(
            team_count=12,
            batting_categories=(StatCategory.HR, StatCategory.SB),
            pitching_categories=(StatCategory.K, StatCategory.ERA),
        )
        assert settings.team_count == 12
        assert settings.batting_categories == (StatCategory.HR, StatCategory.SB)
        assert settings.pitching_categories == (StatCategory.K, StatCategory.ERA)

    def test_scoring_style_default(self) -> None:
        settings = LeagueSettings(
            team_count=12,
            batting_categories=(StatCategory.HR,),
            pitching_categories=(StatCategory.K,),
        )
        assert settings.scoring_style is ScoringStyle.H2H_CATEGORIES

    def test_scoring_style_explicit(self) -> None:
        settings = LeagueSettings(
            team_count=12,
            batting_categories=(StatCategory.HR,),
            pitching_categories=(StatCategory.K,),
            scoring_style=ScoringStyle.ROTO,
        )
        assert settings.scoring_style is ScoringStyle.ROTO

    def test_frozen(self) -> None:
        settings = LeagueSettings(
            team_count=12,
            batting_categories=(StatCategory.HR,),
            pitching_categories=(StatCategory.K,),
        )
        with pytest.raises(FrozenInstanceError):
            settings.team_count = 10  # type: ignore[misc]


class TestCategoryValue:
    def test_construction(self) -> None:
        cv = CategoryValue(category=StatCategory.HR, raw_stat=30.0, value=1.5)
        assert cv.category == StatCategory.HR
        assert cv.raw_stat == 30.0
        assert cv.value == 1.5

    def test_frozen(self) -> None:
        cv = CategoryValue(category=StatCategory.HR, raw_stat=30.0, value=1.5)
        with pytest.raises(FrozenInstanceError):
            cv.value = 2.0  # type: ignore[misc]


class TestPlayerValue:
    def test_construction(self) -> None:
        cv = CategoryValue(category=StatCategory.HR, raw_stat=30.0, value=1.5)
        pv = PlayerValue(
            player_id="123",
            name="Test Player",
            category_values=(cv,),
            total_value=1.5,
        )
        assert pv.player_id == "123"
        assert pv.name == "Test Player"
        assert pv.category_values == (cv,)
        assert pv.total_value == 1.5

    def test_frozen(self) -> None:
        pv = PlayerValue(
            player_id="123",
            name="Test Player",
            category_values=(),
            total_value=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            pv.total_value = 5.0  # type: ignore[misc]


class TestSGPDenominators:
    def test_construction(self) -> None:
        denoms = SGPDenominators(denominators={StatCategory.HR: 5.0, StatCategory.SB: 3.0})
        assert denoms.denominators[StatCategory.HR] == 5.0
        assert denoms.denominators[StatCategory.SB] == 3.0

    def test_frozen(self) -> None:
        denoms = SGPDenominators(denominators={StatCategory.HR: 5.0})
        with pytest.raises(FrozenInstanceError):
            denoms.denominators = {}  # type: ignore[misc]
