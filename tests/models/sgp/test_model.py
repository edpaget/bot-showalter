import dataclasses
from typing import Any

import pytest

from fantasy_baseball_manager.domain import (
    Evaluable,
    FineTunable,
    Model,
    ModelConfig,
    Predictable,
    SgpDenominators,
    Trainable,
)
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    EligibilityRules,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.sgp.model import SgpModel
from fantasy_baseball_manager.services.player_eligibility import PlayerEligibilityService
from tests.fakes.repos import (
    FakePlayerRepo,
    FakePositionAppearanceRepo,
    FakeProjectionRepo,
    FakeValuationRepo,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _standard_league() -> LeagueSettings:
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=2,
        budget=260,
        roster_batters=3,
        roster_pitchers=2,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="r", name="Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(
                key="avg",
                name="AVG",
                stat_type=StatType.RATE,
                direction=Direction.HIGHER,
                numerator="h",
                denominator="ab",
            ),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="sv", name="Saves", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        roster_util=1,
        positions={"C": 1, "OF": 1},
    )


def _standard_denominators() -> dict[str, float]:
    return {
        "hr": 8.0,
        "r": 16.0,
        "avg": 0.005,
        "w": 3.0,
        "sv": 5.0,
    }


def _build_projections() -> list[Projection]:
    return [
        Projection(
            player_id=1,
            season=2025,
            system="steamer",
            version="v1",
            player_type="batter",
            stat_json={"pa": 600, "hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0, "avg": 0.291},
        ),
        Projection(
            player_id=2,
            season=2025,
            system="steamer",
            version="v1",
            player_type="batter",
            stat_json={"pa": 550, "hr": 20.0, "r": 70.0, "h": 140.0, "ab": 500.0, "avg": 0.280},
        ),
        Projection(
            player_id=3,
            season=2025,
            system="steamer",
            version="v1",
            player_type="batter",
            stat_json={"pa": 570, "hr": 10.0, "r": 50.0, "h": 130.0, "ab": 520.0, "avg": 0.250},
        ),
        Projection(
            player_id=4,
            season=2025,
            system="steamer",
            version="v1",
            player_type="pitcher",
            stat_json={"ip": 200, "w": 15.0, "sv": 0.0},
        ),
        Projection(
            player_id=5,
            season=2025,
            system="steamer",
            version="v1",
            player_type="pitcher",
            stat_json={"ip": 70, "w": 8.0, "sv": 30.0},
        ),
    ]


def _build_appearances() -> list[PositionAppearance]:
    return [
        PositionAppearance(player_id=1, season=2025, position="C", games=100),
        PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        PositionAppearance(player_id=3, season=2025, position="OF", games=140),
    ]


def _build_model(
    projections: list[Projection] | None = None,
    appearances: list[PositionAppearance] | None = None,
    valuation_repo: FakeValuationRepo | None = None,
) -> tuple[SgpModel, FakeValuationRepo]:
    val_repo = valuation_repo or FakeValuationRepo()
    pos_repo = FakePositionAppearanceRepo(_build_appearances() if appearances is None else appearances)
    return (
        SgpModel(
            projection_repo=FakeProjectionRepo(_build_projections() if projections is None else projections),
            player_repo=FakePlayerRepo(),
            position_repo=pos_repo,
            valuation_repo=val_repo,
            eligibility_service=PlayerEligibilityService(pos_repo),
        ),
        val_repo,
    )


def _standard_config() -> ModelConfig:
    return ModelConfig(
        seasons=[2025],
        model_params={
            "league": _standard_league(),
            "projection_system": "steamer",
            "denominators": _standard_denominators(),
        },
        version="1.0",
    )


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------


class TestSgpModelProtocol:
    def test_is_model(self) -> None:
        model, _ = _build_model()
        assert isinstance(model, Model)

    def test_is_predictable(self) -> None:
        model, _ = _build_model()
        assert isinstance(model, Predictable)

    def test_is_not_trainable(self) -> None:
        model, _ = _build_model()
        assert not isinstance(model, Trainable)

    def test_is_not_evaluable(self) -> None:
        model, _ = _build_model()
        assert not isinstance(model, Evaluable)

    def test_is_not_finetuneable(self) -> None:
        model, _ = _build_model()
        assert not isinstance(model, FineTunable)

    def test_name(self) -> None:
        model, _ = _build_model()
        assert model.name == "sgp"

    def test_supported_operations(self) -> None:
        model, _ = _build_model()
        assert model.supported_operations == frozenset({"predict"})

    def test_artifact_type(self) -> None:
        model, _ = _build_model()
        assert model.artifact_type == "none"


# ---------------------------------------------------------------------------
# Predict tests
# ---------------------------------------------------------------------------


class TestSgpModelPredict:
    def test_predict_returns_predict_result(self) -> None:
        model, _ = _build_model()
        result = model.predict(_standard_config())
        assert result.model_name == "sgp"

    def test_predict_produces_predictions_for_all_players(self) -> None:
        model, _ = _build_model()
        result = model.predict(_standard_config())
        player_ids = {p["player_id"] for p in result.predictions}
        assert player_ids == {1, 2, 3, 4, 5}

    def test_predict_persists_valuations(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        assert len(val_repo.upserted) == 5

    def test_predict_valuations_have_ranks(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        ranks = sorted(v.rank for v in val_repo.upserted)
        assert ranks == [1, 2, 3, 4, 5]

    def test_predict_valuations_ranked_by_value_descending(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        by_rank = sorted(val_repo.upserted, key=lambda v: v.rank)
        values = [v.value for v in by_rank]
        assert values == sorted(values, reverse=True)

    def test_predict_category_scores_present(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        for v in val_repo.upserted:
            assert isinstance(v.category_scores, dict)
            assert len(v.category_scores) > 0

    def test_predict_batter_categories_match_league(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        batter_vals = [v for v in val_repo.upserted if v.player_type == "batter"]
        for v in batter_vals:
            assert set(v.category_scores.keys()) == {"hr", "r", "avg"}

    def test_predict_pitcher_categories_match_league(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        pitcher_vals = [v for v in val_repo.upserted if v.player_type == "pitcher"]
        for v in pitcher_vals:
            assert set(v.category_scores.keys()) == {"w", "sv"}

    def test_predict_valuation_metadata(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        for v in val_repo.upserted:
            assert v.system == "sgp"
            assert v.version == "1.0"
            assert v.projection_system == "steamer"
            assert v.season == 2025

    def test_predict_assigns_valid_positions(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        valid_positions = {"C", "OF", "UTIL"}
        batter_vals = [v for v in val_repo.upserted if v.player_type == "batter"]
        for v in batter_vals:
            assert v.position in valid_positions, f"Player {v.player_id} got position {v.position!r}"

    def test_predict_pitcher_position(self) -> None:
        model, val_repo = _build_model()
        model.predict(_standard_config())
        pitcher_vals = [v for v in val_repo.upserted if v.player_type == "pitcher"]
        for v in pitcher_vals:
            assert v.position == "P"

    def test_predict_empty_projections(self) -> None:
        model, val_repo = _build_model(projections=[], appearances=[])
        result = model.predict(_standard_config())
        assert result.predictions == []
        assert len(val_repo.upserted) == 0


class TestSgpModelDollarSum:
    def test_dollar_sum_equals_budget(self) -> None:
        """Dollar values should sum to total league budget within $1."""
        model, val_repo = _build_model()
        model.predict(_standard_config())
        total = sum(v.value for v in val_repo.upserted)
        league = _standard_league()
        expected_budget = league.budget * league.teams
        assert total == pytest.approx(expected_budget, abs=1.0)


class TestSgpModelRateStatIndependence:
    def test_same_era_different_ip_same_era_sgp(self) -> None:
        """Pitchers with the same ERA but different IP should get the same ERA SGP."""
        league = LeagueSettings(
            name="Test",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=2,
            budget=260,
            roster_batters=0,
            roster_pitchers=2,
            batting_categories=(),
            pitching_categories=(
                CategoryConfig(
                    key="era",
                    name="ERA",
                    stat_type=StatType.RATE,
                    direction=Direction.LOWER,
                    numerator="er",
                    denominator="ip",
                ),
                CategoryConfig(key="w", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            ),
        )
        projections = [
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v1",
                player_type="pitcher",
                stat_json={"ip": 200, "er": 60.0, "w": 15.0},  # ERA rate = 0.3
            ),
            Projection(
                player_id=2,
                season=2025,
                system="steamer",
                version="v1",
                player_type="pitcher",
                stat_json={"ip": 100, "er": 30.0, "w": 8.0},  # ERA rate = 0.3 (same)
            ),
        ]
        denominators = {"era": 0.1, "w": 3.0}
        model, val_repo = _build_model(projections=projections, appearances=[])
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "league": league,
                "projection_system": "steamer",
                "denominators": denominators,
            },
            version="1.0",
        )
        model.predict(config)
        vals = {v.player_id: v for v in val_repo.upserted}
        # ERA SGP should be identical for both
        assert vals[1].category_scores["era"] == pytest.approx(vals[2].category_scores["era"])


class TestSgpModelDenominatorSources:
    def test_sgp_denominators_object_used(self) -> None:
        """When denominators is an SgpDenominators object, its averages are used."""
        denoms = SgpDenominators(per_season=(), averages=_standard_denominators())
        model, val_repo = _build_model()
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "league": _standard_league(),
                "projection_system": "steamer",
                "denominators": denoms,
            },
            version="1.0",
        )
        model.predict(config)
        assert len(val_repo.upserted) == 5

    def test_denominator_provider_called_when_no_denominators(self) -> None:
        """When no denominators in model_params, the provider is called."""
        calls: list[Any] = []

        def fake_provider(league: Any) -> SgpDenominators:
            calls.append(league)
            return SgpDenominators(per_season=(), averages=_standard_denominators())

        pos_repo = FakePositionAppearanceRepo(_build_appearances())
        model = SgpModel(
            projection_repo=FakeProjectionRepo(_build_projections()),
            position_repo=pos_repo,
            valuation_repo=FakeValuationRepo(),
            eligibility_service=PlayerEligibilityService(pos_repo),
            denominator_provider=fake_provider,
        )
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "league": _standard_league(),
                "projection_system": "steamer",
            },
            version="1.0",
        )
        model.predict(config)
        assert len(calls) == 1

    def test_no_denominators_no_provider_raises(self) -> None:
        """TypeError raised when no denominators and no provider."""
        model, _ = _build_model()
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "league": _standard_league(),
                "projection_system": "steamer",
            },
            version="1.0",
        )
        with pytest.raises(TypeError, match="denominators"):
            model.predict(config)

    def test_no_eligibility_service_raises(self) -> None:
        """TypeError raised when eligibility_service is None."""
        model = SgpModel(
            projection_repo=FakeProjectionRepo(),
            position_repo=FakePositionAppearanceRepo(),
        )
        with pytest.raises(TypeError, match="eligibility_service"):
            model.predict(_standard_config())

    def test_projection_version_filters(self) -> None:
        """When projection_version is set, only that version is used."""
        projections = [
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 600, "hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0},
            ),
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v2",
                player_type="batter",
                stat_json={"pa": 580, "hr": 35.0, "r": 90.0, "h": 150.0, "ab": 530.0},
            ),
            Projection(
                player_id=4,
                season=2025,
                system="steamer",
                version="v1",
                player_type="pitcher",
                stat_json={"ip": 200, "w": 15.0, "sv": 0.0},
            ),
        ]
        model, val_repo = _build_model(projections=projections)
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "league": _standard_league(),
                "projection_system": "steamer",
                "projection_version": "v1",
                "denominators": _standard_denominators(),
            },
            version="1.0",
        )
        model.predict(config)
        player_ids = {v.player_id for v in val_repo.upserted}
        assert player_ids == {1, 4}
        assert all(v.projection_version == "v1" for v in val_repo.upserted)


class TestSgpModelPlayingTimeCutoffs:
    def test_min_pa_filters_low_pa_batters(self) -> None:
        league = dataclasses.replace(
            _standard_league(),
            eligibility=EligibilityRules(min_pa=100),
        )
        projections = [
            Projection(
                player_id=1,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 600, "hr": 40.0, "r": 100.0, "h": 160.0, "ab": 550.0},
            ),
            Projection(
                player_id=2,
                season=2025,
                system="steamer",
                version="v1",
                player_type="batter",
                stat_json={"pa": 50, "hr": 2.0, "r": 5.0, "h": 10.0, "ab": 45.0},
            ),
            Projection(
                player_id=4,
                season=2025,
                system="steamer",
                version="v1",
                player_type="pitcher",
                stat_json={"ip": 200, "w": 15.0, "sv": 0.0},
            ),
        ]
        model, val_repo = _build_model(projections=projections, appearances=[])
        config = ModelConfig(
            seasons=[2025],
            model_params={
                "league": league,
                "projection_system": "steamer",
                "denominators": _standard_denominators(),
            },
            version="1.0",
        )
        model.predict(config)
        player_ids = {v.player_id for v in val_repo.upserted}
        assert 1 in player_ids
        assert 2 not in player_ids
