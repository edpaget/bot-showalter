import numpy as np

from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.valuation.features import BATTER_FEATURE_NAMES, PITCHER_FEATURE_NAMES
from fantasy_baseball_manager.valuation.ml_valuate import ml_valuate_batting, ml_valuate_pitching
from fantasy_baseball_manager.valuation.ridge_model import RidgeValuationModel


def _fitted_batter_model() -> RidgeValuationModel:
    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, len(BATTER_FEATURE_NAMES)))
    y = rng.standard_normal(n)
    model = RidgeValuationModel(player_type="batter")
    model.fit(X, y, BATTER_FEATURE_NAMES)
    return model


def _fitted_pitcher_model() -> RidgeValuationModel:
    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, len(PITCHER_FEATURE_NAMES)))
    y = rng.standard_normal(n)
    model = RidgeValuationModel(player_type="pitcher")
    model.fit(X, y, PITCHER_FEATURE_NAMES)
    return model


def _batting_projection(player_id: str = "b1", name: str = "Hitter") -> BattingProjection:
    return BattingProjection(
        player_id=player_id,
        name=name,
        year=2026,
        age=28,
        pa=600.0,
        ab=540.0,
        h=150.0,
        singles=100.0,
        doubles=30.0,
        triples=5.0,
        hr=15.0,
        bb=50.0,
        so=100.0,
        hbp=5.0,
        sf=3.0,
        sh=2.0,
        sb=20.0,
        cs=5.0,
        r=80.0,
        rbi=70.0,
    )


def _pitching_projection(player_id: str = "p1", name: str = "Ace") -> PitchingProjection:
    return PitchingProjection(
        player_id=player_id,
        name=name,
        year=2026,
        age=27,
        ip=200.0,
        g=33.0,
        gs=33.0,
        er=60.0,
        h=170.0,
        bb=50.0,
        so=200.0,
        hr=20.0,
        hbp=8.0,
        era=2.70,
        whip=1.10,
        w=15.0,
        nsvh=0.0,
    )


class TestMlValuateBatting:
    def test_returns_player_values(self) -> None:
        model = _fitted_batter_model()
        projections = [_batting_projection("b1", "A"), _batting_projection("b2", "B")]
        result = ml_valuate_batting(projections, model)
        assert len(result) == 2
        assert result[0].player_id == "b1"
        assert result[1].player_id == "b2"

    def test_position_type_is_B(self) -> None:
        model = _fitted_batter_model()
        result = ml_valuate_batting([_batting_projection()], model)
        assert result[0].position_type == "B"

    def test_category_values_empty(self) -> None:
        model = _fitted_batter_model()
        result = ml_valuate_batting([_batting_projection()], model)
        assert result[0].category_values == ()

    def test_values_are_finite(self) -> None:
        model = _fitted_batter_model()
        projections = [_batting_projection(f"b{i}") for i in range(10)]
        result = ml_valuate_batting(projections, model)
        for pv in result:
            assert np.isfinite(pv.total_value)

    def test_empty_projections(self) -> None:
        model = _fitted_batter_model()
        result = ml_valuate_batting([], model)
        assert result == []

    def test_deterministic(self) -> None:
        model = _fitted_batter_model()
        projections = [_batting_projection()]
        r1 = ml_valuate_batting(projections, model)
        r2 = ml_valuate_batting(projections, model)
        assert r1[0].total_value == r2[0].total_value

    def test_not_pre_sorted(self) -> None:
        """Values should not be sorted — caller is responsible for sorting."""
        model = _fitted_batter_model()
        projections = [
            _batting_projection("b1", "A"),
            _batting_projection("b2", "B"),
            _batting_projection("b3", "C"),
        ]
        result = ml_valuate_batting(projections, model)
        # Just verify the order matches input order
        assert [pv.player_id for pv in result] == ["b1", "b2", "b3"]


class TestMlValuatePitching:
    def test_returns_player_values(self) -> None:
        model = _fitted_pitcher_model()
        projections = [_pitching_projection("p1", "X"), _pitching_projection("p2", "Y")]
        result = ml_valuate_pitching(projections, model)
        assert len(result) == 2

    def test_position_type_is_P(self) -> None:
        model = _fitted_pitcher_model()
        result = ml_valuate_pitching([_pitching_projection()], model)
        assert result[0].position_type == "P"

    def test_empty_projections(self) -> None:
        model = _fitted_pitcher_model()
        result = ml_valuate_pitching([], model)
        assert result == []

    def test_values_are_finite(self) -> None:
        model = _fitted_pitcher_model()
        projections = [_pitching_projection(f"p{i}") for i in range(10)]
        result = ml_valuate_pitching(projections, model)
        for pv in result:
            assert np.isfinite(pv.total_value)
