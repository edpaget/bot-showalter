from unittest.mock import MagicMock, patch

import pytest

from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.valuation.models import StatCategory, ValuationResult, Valuator
from fantasy_baseball_manager.valuation.valuator import VALUATORS, RidgeValuator, ZScoreValuator


def _make_batting_projection(
    player_id: str = "b1",
    name: str = "Test Hitter",
    hr: float = 25.0,
    sb: float = 10.0,
) -> BattingProjection:
    return BattingProjection(
        player_id=player_id,
        name=name,
        year=2025,
        age=28,
        pa=600.0,
        ab=540.0,
        h=160.0,
        singles=100.0,
        doubles=30.0,
        triples=5.0,
        hr=hr,
        bb=50.0,
        so=120.0,
        hbp=5.0,
        sf=3.0,
        sh=2.0,
        sb=sb,
        cs=3.0,
        r=80.0,
        rbi=90.0,
    )


def _make_pitching_projection(
    player_id: str = "p1",
    name: str = "Test Pitcher",
    so: float = 200.0,
    er: float = 70.0,
) -> PitchingProjection:
    return PitchingProjection(
        player_id=player_id,
        name=name,
        year=2025,
        age=28,
        ip=180.0,
        g=32.0,
        gs=32.0,
        er=er,
        h=150.0,
        bb=50.0,
        so=so,
        hr=20.0,
        hbp=5.0,
        era=3.50,
        whip=1.11,
        w=12.0,
        nsvh=0.0,
    )


class TestZScoreValuator:
    def test_valuate_batting_returns_valuation_result(self) -> None:
        valuator = ZScoreValuator()
        projections = [
            _make_batting_projection("b1", "Slugger", hr=40.0, sb=5.0),
            _make_batting_projection("b2", "Speedy", hr=10.0, sb=30.0),
        ]
        cats = (StatCategory.HR, StatCategory.SB)

        result = valuator.valuate_batting(projections, cats)

        assert isinstance(result, ValuationResult)
        assert len(result.values) == 2
        assert result.categories == cats
        assert result.label == "Z-score"

    def test_valuate_pitching_returns_valuation_result(self) -> None:
        valuator = ZScoreValuator()
        projections = [
            _make_pitching_projection("p1", "Ace", so=250.0, er=50.0),
            _make_pitching_projection("p2", "Bullpen", so=150.0, er=80.0),
        ]
        cats = (StatCategory.K, StatCategory.ERA)

        result = valuator.valuate_pitching(projections, cats)

        assert isinstance(result, ValuationResult)
        assert len(result.values) == 2
        assert result.categories == cats
        assert result.label == "Z-score"

    def test_valuate_batting_preserves_player_ids(self) -> None:
        valuator = ZScoreValuator()
        projections = [
            _make_batting_projection("b1", "Slugger", hr=40.0),
            _make_batting_projection("b2", "Speedy", hr=10.0),
        ]
        cats = (StatCategory.HR,)

        result = valuator.valuate_batting(projections, cats)

        player_ids = {pv.player_id for pv in result.values}
        assert player_ids == {"b1", "b2"}

    def test_satisfies_valuator_protocol(self) -> None:
        valuator: Valuator = ZScoreValuator()
        assert valuator is not None


class TestRidgeValuator:
    @patch("fantasy_baseball_manager.valuation.ml_valuate.ml_valuate_batting")
    @patch("fantasy_baseball_manager.valuation.ridge_model.load_model")
    def test_valuate_batting_returns_valuation_result(
        self, mock_load: MagicMock, mock_ml_valuate: MagicMock
    ) -> None:
        from fantasy_baseball_manager.valuation.models import PlayerValue

        mock_load.return_value = MagicMock()
        mock_ml_valuate.return_value = [
            PlayerValue(player_id="b1", name="Slugger", category_values=(), total_value=5.0, position_type="B"),
        ]

        valuator = RidgeValuator()
        projections = [_make_batting_projection("b1", "Slugger")]
        cats = (StatCategory.HR,)

        result = valuator.valuate_batting(projections, cats)

        assert isinstance(result, ValuationResult)
        assert result.categories == ()
        assert result.label == "ML Ridge"
        assert len(result.values) == 1
        mock_ml_valuate.assert_called_once()

    @patch("fantasy_baseball_manager.valuation.ml_valuate.ml_valuate_pitching")
    @patch("fantasy_baseball_manager.valuation.ridge_model.load_model")
    def test_valuate_pitching_returns_valuation_result(
        self, mock_load: MagicMock, mock_ml_valuate: MagicMock
    ) -> None:
        from fantasy_baseball_manager.valuation.models import PlayerValue

        mock_load.return_value = MagicMock()
        mock_ml_valuate.return_value = [
            PlayerValue(player_id="p1", name="Ace", category_values=(), total_value=3.0, position_type="P"),
        ]

        valuator = RidgeValuator()
        projections = [_make_pitching_projection("p1", "Ace")]
        cats = (StatCategory.K,)

        result = valuator.valuate_pitching(projections, cats)

        assert isinstance(result, ValuationResult)
        assert result.categories == ()
        assert result.label == "ML Ridge"
        assert len(result.values) == 1
        mock_ml_valuate.assert_called_once()

    @patch("fantasy_baseball_manager.valuation.ridge_model.load_model")
    def test_lazy_loads_models(self, mock_load: MagicMock) -> None:
        mock_load.return_value = MagicMock()
        valuator = RidgeValuator()

        mock_load.assert_not_called()

        _ = valuator._batter_model
        mock_load.assert_called_once_with("default", "batter")

    def test_satisfies_valuator_protocol(self) -> None:
        valuator: Valuator = RidgeValuator()
        assert valuator is not None


class TestVALUATORSRegistry:
    def test_contains_zscore(self) -> None:
        assert "zscore" in VALUATORS

    def test_contains_ml_ridge(self) -> None:
        assert "ml-ridge" in VALUATORS

    def test_zscore_factory_returns_zscore_valuator(self) -> None:
        valuator = VALUATORS["zscore"]()
        assert isinstance(valuator, ZScoreValuator)

    def test_ml_ridge_factory_returns_ridge_valuator(self) -> None:
        valuator = VALUATORS["ml-ridge"]()
        assert isinstance(valuator, RidgeValuator)

    def test_unknown_method_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            VALUATORS["unknown"]()
