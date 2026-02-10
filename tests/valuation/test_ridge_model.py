import numpy as np
import pytest

from fantasy_baseball_manager.valuation.ridge_model import (
    RidgeValuationConfig,
    RidgeValuationModel,
    load_model,
    save_model,
)


def _synthetic_data(
    n: int = 100,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    # y = linear combination + noise
    coef = rng.standard_normal(n_features)
    y = X @ coef + rng.standard_normal(n) * 0.1
    return X, y


FEATURE_NAMES: tuple[str, ...] = ("f1", "f2", "f3", "f4", "f5")


class TestRidgeValuationModel:
    def test_fit_predict_produces_finite(self) -> None:
        X, y = _synthetic_data()
        model = RidgeValuationModel(player_type="batter")
        model.fit(X, y, FEATURE_NAMES)
        preds = model.predict(X)
        assert preds.shape == (100,)
        assert np.all(np.isfinite(preds))

    def test_predict_raises_when_not_fitted(self) -> None:
        model = RidgeValuationModel(player_type="batter")
        X, _ = _synthetic_data(n=5)
        with pytest.raises(ValueError, match="fitted"):
            model.predict(X)

    def test_is_fitted_property(self) -> None:
        model = RidgeValuationModel(player_type="batter")
        assert model.is_fitted is False
        X, y = _synthetic_data()
        model.fit(X, y, FEATURE_NAMES)
        assert model.is_fitted is True

    def test_coefficients_keys(self) -> None:
        X, y = _synthetic_data()
        model = RidgeValuationModel(player_type="batter")
        model.fit(X, y, FEATURE_NAMES)
        coefs = model.coefficients()
        assert set(coefs.keys()) == set(FEATURE_NAMES)

    def test_predict_value_is_negated(self) -> None:
        X, y = _synthetic_data()
        model = RidgeValuationModel(player_type="batter")
        model.fit(X, y, FEATURE_NAMES)
        raw = model.predict(X)
        values = model.predict_value(X)
        np.testing.assert_array_almost_equal(values, -raw)

    def test_scaler_is_applied(self) -> None:
        """StandardScaler should center features around 0."""
        X, y = _synthetic_data()
        model = RidgeValuationModel(player_type="batter")
        model.fit(X, y, FEATURE_NAMES)
        assert model._scaler is not None
        scaled = model._scaler.transform(X)
        np.testing.assert_array_almost_equal(scaled.mean(axis=0), 0.0, decimal=10)

    def test_custom_alpha(self) -> None:
        X, y = _synthetic_data()
        config = RidgeValuationConfig(alpha=100.0)
        model = RidgeValuationModel(player_type="pitcher", config=config)
        model.fit(X, y, FEATURE_NAMES)
        preds = model.predict(X)
        assert np.all(np.isfinite(preds))

    def test_coefficients_raises_when_not_fitted(self) -> None:
        model = RidgeValuationModel(player_type="batter")
        with pytest.raises(ValueError, match="fitted"):
            model.coefficients()


class TestSerializationRoundTrip:
    def test_get_params_from_params(self) -> None:
        X, y = _synthetic_data()
        model = RidgeValuationModel(player_type="batter")
        model.fit(X, y, FEATURE_NAMES)
        model.training_years = (2022, 2023)

        params = model.get_params()
        restored = RidgeValuationModel.from_params(params)

        assert restored.player_type == "batter"
        assert restored.feature_names == FEATURE_NAMES
        assert restored.training_years == (2022, 2023)
        assert restored.is_fitted is True

        np.testing.assert_array_almost_equal(
            model.predict(X), restored.predict(X)
        )

    def test_save_load_roundtrip(self, tmp_path: object) -> None:
        from pathlib import Path

        directory = Path(str(tmp_path))
        X, y = _synthetic_data()
        model = RidgeValuationModel(player_type="batter")
        model.fit(X, y, FEATURE_NAMES)
        model.training_years = (2022, 2023)

        save_model(model, "test_model", directory)
        loaded = load_model("test_model", "batter", directory)

        assert loaded.player_type == "batter"
        assert loaded.feature_names == FEATURE_NAMES
        assert loaded.is_fitted is True

        np.testing.assert_array_almost_equal(
            model.predict(X), loaded.predict(X)
        )

    def test_load_missing_model_raises(self, tmp_path: object) -> None:
        from pathlib import Path

        directory = Path(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent", "batter", directory)
