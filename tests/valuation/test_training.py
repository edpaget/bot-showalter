import numpy as np

from fantasy_baseball_manager.adp.training_dataset import BatterTrainingRow, PitcherTrainingRow
from fantasy_baseball_manager.valuation.ridge_model import RidgeValuationConfig
from fantasy_baseball_manager.valuation.training import (
    ValuationEvaluation,
    _compute_top_k_precision,
    train_ridge_valuation,
)


def _batter_row(
    player_id: str = "b1",
    year: int = 2023,
    pa: int = 600,
    hr: int = 30,
    r: int = 90,
    rbi: int = 100,
    sb: int = 15,
    adp: float = 25.0,
) -> BatterTrainingRow:
    return BatterTrainingRow(
        player_id=player_id,
        name=f"Batter {player_id}",
        team="NYY",
        year=year,
        position="SS",
        pa=pa,
        hr=hr,
        r=r,
        rbi=rbi,
        sb=sb,
        bb=60,
        so=120,
        obp=0.350,
        slg=0.500,
        woba=0.370,
        war=5.0,
        adp=adp,
    )


def _pitcher_row(
    player_id: str = "p1",
    year: int = 2023,
    ip: float = 200.0,
    so: int = 200,
    era: float = 2.70,
    adp: float = 30.0,
) -> PitcherTrainingRow:
    return PitcherTrainingRow(
        player_id=player_id,
        name=f"Pitcher {player_id}",
        team="LAD",
        year=year,
        ip=ip,
        w=15,
        sv=0,
        hld=0,
        so=so,
        bb=50,
        h=160,
        er=60,
        hr=20,
        era=era,
        whip=1.05,
        fip=2.80,
        war=5.0,
        adp=adp,
    )


def _make_batter_rows(n: int = 50, year: int = 2023) -> list[BatterTrainingRow]:
    """Generate n batter rows with varying stats."""
    rows = []
    for i in range(n):
        rows.append(
            _batter_row(
                player_id=f"b{i}",
                year=year,
                pa=400 + i * 5,
                hr=10 + i,
                r=50 + i * 2,
                rbi=40 + i * 2,
                sb=5 + i,
                adp=float(1 + i * 5),
            )
        )
    return rows


def _make_pitcher_rows(n: int = 50, year: int = 2023) -> list[PitcherTrainingRow]:
    """Generate n pitcher rows with varying stats."""
    rows = []
    for i in range(n):
        rows.append(
            _pitcher_row(
                player_id=f"p{i}",
                year=year,
                ip=100.0 + i * 3,
                so=80 + i * 4,
                era=2.5 + i * 0.05,
                adp=float(1 + i * 6),
            )
        )
    return rows


class TestTrainRidgeValuation:
    def test_produces_fitted_model(self) -> None:
        train = _make_batter_rows(40, year=2023)
        test = _make_batter_rows(10, year=2024)
        model, _evaluation = train_ridge_valuation(train, test, "batter")
        assert model.is_fitted
        assert model.player_type == "batter"

    def test_evaluation_fields_populated(self) -> None:
        train = _make_batter_rows(40, year=2023)
        test = _make_batter_rows(10, year=2024)
        _, evaluation = train_ridge_valuation(train, test, "batter")
        assert isinstance(evaluation, ValuationEvaluation)
        assert evaluation.n_train == 40
        assert evaluation.n_test == 10
        assert evaluation.training_years == (2023,)
        assert evaluation.test_years == (2024,)
        assert np.isfinite(evaluation.spearman_rho)
        assert evaluation.rmse > 0
        assert evaluation.mae > 0

    def test_top_50_precision_in_range(self) -> None:
        train = _make_batter_rows(60, year=2023)
        test = _make_batter_rows(60, year=2024)
        _, evaluation = train_ridge_valuation(train, test, "batter")
        assert 0.0 <= evaluation.top_50_precision <= 1.0

    def test_pitcher_model(self) -> None:
        train = _make_pitcher_rows(40, year=2023)
        test = _make_pitcher_rows(10, year=2024)
        model, evaluation = train_ridge_valuation(train, test, "pitcher")
        assert model.is_fitted
        assert model.player_type == "pitcher"
        assert evaluation.n_train == 40

    def test_year_split_no_leakage(self) -> None:
        """Training and test years should not overlap."""
        train = _make_batter_rows(30, year=2022) + _make_batter_rows(30, year=2023)
        test = _make_batter_rows(20, year=2024)
        _, evaluation = train_ridge_valuation(train, test, "batter")
        assert 2024 not in evaluation.training_years
        assert 2022 not in evaluation.test_years
        assert 2023 not in evaluation.test_years

    def test_coefficients_populated(self) -> None:
        train = _make_batter_rows(40, year=2023)
        test = _make_batter_rows(10, year=2024)
        _, evaluation = train_ridge_valuation(train, test, "batter")
        assert len(evaluation.coefficient_analysis) > 0

    def test_custom_config(self) -> None:
        train = _make_batter_rows(40, year=2023)
        test = _make_batter_rows(10, year=2024)
        config = RidgeValuationConfig(alpha=10.0)
        model, _ = train_ridge_valuation(train, test, "batter", config)
        assert model.config.alpha == 10.0


class TestTopKPrecision:
    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        precision = _compute_top_k_precision(y, y, k=3)
        assert precision == 1.0

    def test_completely_wrong(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        precision = _compute_top_k_precision(y_true, y_pred, k=2)
        assert precision == 0.0

    def test_partial_overlap(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 3.0, 2.0, 4.0, 5.0])
        precision = _compute_top_k_precision(y_true, y_pred, k=3)
        # True top 3 = {0, 1, 2}, Pred top 3 = {0, 1, 2} → 3/3
        assert precision == 1.0

    def test_k_larger_than_array(self) -> None:
        y = np.array([1.0, 2.0])
        precision = _compute_top_k_precision(y, y, k=10)
        assert precision == 1.0

    def test_empty_arrays(self) -> None:
        y = np.array([])
        precision = _compute_top_k_precision(y, y, k=5)
        assert precision == 0.0
