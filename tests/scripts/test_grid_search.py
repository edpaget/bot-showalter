from unittest.mock import MagicMock, patch

from fantasy_baseball_manager.pipeline.stages.regression_constants import (
    BATTING_REGRESSION_PA,
    PITCHING_REGRESSION_OUTS,
)
from scripts.grid_search import (
    COARSE_SEARCH_SPACE,
    FULL_SEARCH_SPACE,
    SearchPoint,
    generate_grid,
    search_point_to_config,
)


class TestGenerateGrid:
    def test_full_grid_size(self) -> None:
        grid = generate_grid(FULL_SEARCH_SPACE)
        expected = 1
        for values in FULL_SEARCH_SPACE.values():
            expected *= len(values)
        assert len(grid) == expected

    def test_coarse_grid_size(self) -> None:
        grid = generate_grid(COARSE_SEARCH_SPACE)
        expected = 1
        for values in COARSE_SEARCH_SPACE.values():
            expected *= len(values)
        assert len(grid) == expected

    def test_all_elements_are_search_points(self) -> None:
        grid = generate_grid(COARSE_SEARCH_SPACE)
        assert all(isinstance(p, SearchPoint) for p in grid)

    def test_grid_contains_expected_corners(self) -> None:
        space = {"babip_regression_weight": [0.3, 0.7], "lob_regression_weight": [0.4, 0.8]}
        # Provide remaining fields with single values
        full_space = {
            **space,
            "batting_hr_pa": [500],
            "batting_singles_pa": [800],
            "pitching_h_outs": [200],
            "pitching_er_outs": [150],
        }
        grid = generate_grid(full_space)
        assert len(grid) == 4
        weights = {(p.babip_regression_weight, p.lob_regression_weight) for p in grid}
        assert weights == {(0.3, 0.4), (0.3, 0.8), (0.7, 0.4), (0.7, 0.8)}


class TestSearchPointToConfig:
    def test_overrides_batting_hr(self) -> None:
        point = SearchPoint(
            babip_regression_weight=0.5,
            lob_regression_weight=0.6,
            batting_hr_pa=999.0,
            batting_singles_pa=800.0,
            pitching_h_outs=200.0,
            pitching_er_outs=150.0,
        )
        config = search_point_to_config(point)
        assert config.batting_regression_pa["hr"] == 999.0

    def test_overrides_batting_singles(self) -> None:
        point = SearchPoint(
            babip_regression_weight=0.5,
            lob_regression_weight=0.6,
            batting_hr_pa=500.0,
            batting_singles_pa=999.0,
            pitching_h_outs=200.0,
            pitching_er_outs=150.0,
        )
        config = search_point_to_config(point)
        assert config.batting_regression_pa["singles"] == 999.0

    def test_overrides_pitching_h(self) -> None:
        point = SearchPoint(
            babip_regression_weight=0.5,
            lob_regression_weight=0.6,
            batting_hr_pa=500.0,
            batting_singles_pa=800.0,
            pitching_h_outs=999.0,
            pitching_er_outs=150.0,
        )
        config = search_point_to_config(point)
        assert config.pitching_regression_outs["h"] == 999.0

    def test_overrides_pitching_er(self) -> None:
        point = SearchPoint(
            babip_regression_weight=0.5,
            lob_regression_weight=0.6,
            batting_hr_pa=500.0,
            batting_singles_pa=800.0,
            pitching_h_outs=200.0,
            pitching_er_outs=999.0,
        )
        config = search_point_to_config(point)
        assert config.pitching_regression_outs["er"] == 999.0

    def test_overrides_normalization_weights(self) -> None:
        point = SearchPoint(
            babip_regression_weight=0.42,
            lob_regression_weight=0.73,
            batting_hr_pa=500.0,
            batting_singles_pa=800.0,
            pitching_h_outs=200.0,
            pitching_er_outs=150.0,
        )
        config = search_point_to_config(point)
        assert config.pitcher_normalization.babip_regression_weight == 0.42
        assert config.pitcher_normalization.lob_regression_weight == 0.73

    def test_preserves_non_overridden_batting_stats(self) -> None:
        point = SearchPoint(
            babip_regression_weight=0.5,
            lob_regression_weight=0.6,
            batting_hr_pa=999.0,
            batting_singles_pa=999.0,
            pitching_h_outs=200.0,
            pitching_er_outs=150.0,
        )
        config = search_point_to_config(point)
        assert config.batting_regression_pa["so"] == BATTING_REGRESSION_PA["so"]
        assert config.batting_regression_pa["bb"] == BATTING_REGRESSION_PA["bb"]
        assert config.batting_regression_pa["doubles"] == BATTING_REGRESSION_PA["doubles"]

    def test_preserves_non_overridden_pitching_stats(self) -> None:
        point = SearchPoint(
            babip_regression_weight=0.5,
            lob_regression_weight=0.6,
            batting_hr_pa=500.0,
            batting_singles_pa=800.0,
            pitching_h_outs=999.0,
            pitching_er_outs=999.0,
        )
        config = search_point_to_config(point)
        assert config.pitching_regression_outs["so"] == PITCHING_REGRESSION_OUTS["so"]
        assert config.pitching_regression_outs["bb"] == PITCHING_REGRESSION_OUTS["bb"]

    def test_does_not_mutate_module_constants(self) -> None:
        point = SearchPoint(
            babip_regression_weight=0.5,
            lob_regression_weight=0.6,
            batting_hr_pa=999.0,
            batting_singles_pa=999.0,
            pitching_h_outs=999.0,
            pitching_er_outs=999.0,
        )
        original_batting_hr = BATTING_REGRESSION_PA["hr"]
        original_pitching_h = PITCHING_REGRESSION_OUTS["h"]
        search_point_to_config(point)
        assert BATTING_REGRESSION_PA["hr"] == original_batting_hr
        assert PITCHING_REGRESSION_OUTS["h"] == original_pitching_h


class TestEvaluatePoint:
    @patch("scripts.grid_search.evaluate_source")
    @patch("scripts.grid_search.PybaseballDataSource")
    @patch("scripts.grid_search.load_league_settings")
    def test_returns_expected_structure(
        self,
        mock_settings: MagicMock,
        mock_ds: MagicMock,
        mock_eval: MagicMock,
    ) -> None:
        from scripts.grid_search import evaluate_point

        # Setup mocks
        mock_settings.return_value = MagicMock(
            batting_categories=("HR",),
            pitching_categories=("K",),
        )
        mock_ds.return_value = MagicMock()

        mock_bat_rank = MagicMock()
        mock_bat_rank.spearman_rho = 0.8
        mock_pitch_rank = MagicMock()
        mock_pitch_rank.spearman_rho = 0.7

        mock_bat_stat = MagicMock()
        mock_bat_stat.rmse = 5.0
        mock_bat_stat.mae = 3.0
        mock_pitch_stat = MagicMock()
        mock_pitch_stat.rmse = 2.0
        mock_pitch_stat.mae = 1.0

        mock_evaluation = MagicMock()
        mock_evaluation.batting_rank_accuracy = mock_bat_rank
        mock_evaluation.pitching_rank_accuracy = mock_pitch_rank
        mock_evaluation.batting_stat_accuracy = [mock_bat_stat]
        mock_evaluation.pitching_stat_accuracy = [mock_pitch_stat]
        mock_eval.return_value = mock_evaluation

        point = SearchPoint(
            babip_regression_weight=0.5,
            lob_regression_weight=0.6,
            batting_hr_pa=500.0,
            batting_singles_pa=800.0,
            pitching_h_outs=200.0,
            pitching_er_outs=150.0,
        )

        result = evaluate_point(
            point, eval_years=[2023], pipeline_name="marcel_norm", min_pa=200, min_ip=50.0, top_n=20
        )

        assert "params" in result
        assert "metrics" in result
        assert result["params"]["babip_regression_weight"] == 0.5
        assert result["metrics"]["avg_spearman_rho"] == round((0.8 + 0.7) / 2, 5)
        assert result["metrics"]["avg_batting_rho"] == 0.8
        assert result["metrics"]["avg_pitching_rho"] == 0.7
