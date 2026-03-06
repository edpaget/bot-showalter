import sys
import types

from fantasy_baseball_manager.cli._output._bio import print_player_summaries
from fantasy_baseball_manager.cli._output._common import console, err_console, print_error
from fantasy_baseball_manager.cli._output._datasets import print_dataset_list
from fantasy_baseball_manager.cli._output._draft import (
    print_cascade_result,
    print_category_needs,
    print_draft_board,
    print_draft_report,
    print_draft_tiers,
    print_opportunity_costs,
    print_pick_trade_evaluation,
    print_pick_value_curve,
    print_position_check,
    print_scarcity_rankings,
    print_scarcity_report,
    print_tier_summary,
    print_upgrades,
    print_value_curve,
)
from fantasy_baseball_manager.cli._output._evaluation import (
    print_comparison_result,
    print_gate_result,
    print_regression_check_result,
    print_stratified_comparison_result,
    print_system_metrics,
)
from fantasy_baseball_manager.cli._output._experiments import (
    print_checkpoint_detail,
    print_checkpoint_list,
    print_compare_features_result,
    print_experiment_detail,
    print_experiment_search_results,
    print_experiment_summary,
)
from fantasy_baseball_manager.cli._output._feature_factory import (
    print_bin_target_means,
    print_binned_summary,
    print_candidate_values,
    print_interaction_scan_results,
)
from fantasy_baseball_manager.cli._output._features import print_features
from fantasy_baseball_manager.cli._output._ingest import print_import_result, print_ingest_result
from fantasy_baseball_manager.cli._output._injury import (
    print_games_lost_leaderboard,
    print_injury_estimate,
    print_injury_profile,
    print_injury_risk_leaderboard,
    print_injury_value_deltas,
)
from fantasy_baseball_manager.cli._output._keeper import (
    print_adjusted_rankings,
    print_keeper_decisions,
    print_keeper_scenarios,
    print_keeper_solution,
    print_keeper_trade_impact,
    print_league_keeper_overview,
    print_trade_evaluation,
)
from fantasy_baseball_manager.cli._output._marginal_value import print_marginal_value_results
from fantasy_baseball_manager.cli._output._mock_draft import (
    print_batch_simulation_result,
    print_mock_draft_result,
    print_strategy_comparison_table,
)
from fantasy_baseball_manager.cli._output._model import (
    print_ablation_result,
    print_coverage_matrix,
    print_predict_result,
    print_prepare_result,
    print_routing_table,
    print_train_result,
    print_tune_result,
)
from fantasy_baseball_manager.cli._output._profile import (
    print_column_profiles,
    print_column_ranking,
    print_correlation_results,
    print_stability_result,
)
from fantasy_baseball_manager.cli._output._projections import (
    print_player_projections,
    print_pt_sources,
    print_system_summaries,
)
from fantasy_baseball_manager.cli._output._quick_eval import print_quick_eval_result
from fantasy_baseball_manager.cli._output._reports import (
    print_adp_accuracy_report,
    print_adp_movers_report,
    print_breakout_candidates,
    print_classifier_evaluation,
    print_performance_report,
    print_projection_confidence,
    print_residual_analysis_report,
    print_residual_persistence_report,
    print_system_disagreements,
    print_talent_delta_report,
    print_talent_quality_report,
    print_value_over_adp,
    print_variance_targets,
)
from fantasy_baseball_manager.cli._output._residuals import (
    print_cohort_bias_report,
    print_cohort_bias_summary,
    print_error_decomposition_report,
    print_feature_gap_report,
)
from fantasy_baseball_manager.cli._output._runs import (
    diff_records,
    print_run_detail,
    print_run_diff,
    print_run_inspect,
    print_run_list,
)
from fantasy_baseball_manager.cli._output._validate import print_preflight_result, print_validation_result
from fantasy_baseball_manager.cli._output._valuations import (
    print_player_valuations,
    print_valuation_eval_result,
    print_valuation_rankings,
)

_SUBMODULE_NAMES = (
    "_bio",
    "_common",
    "_datasets",
    "_draft",
    "_evaluation",
    "_experiments",
    "_feature_factory",
    "_features",
    "_ingest",
    "_injury",
    "_keeper",
    "_marginal_value",
    "_mock_draft",
    "_model",
    "_profile",
    "_projections",
    "_quick_eval",
    "_reports",
    "_residuals",
    "_runs",
    "_validate",
    "_valuations",
)


class _OutputModule(types.ModuleType):
    """Custom module class that propagates ``console`` patches to submodules.

    Tests monkeypatch ``_output.console`` to inject a custom-width or
    force-terminal console.  Each submodule binds ``console`` via
    ``from _common import console``, creating a local name.  This class
    intercepts ``setattr`` so the new value is written into every
    submodule's ``__dict__``, keeping them in sync.
    """

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name in ("console", "err_console"):
            for sub in _SUBMODULE_NAMES:
                mod = sys.modules.get(f"{self.__name__}.{sub}")
                if mod is not None and name in mod.__dict__:
                    mod.__dict__[name] = value


sys.modules[__name__].__class__ = _OutputModule

__all__ = [
    "console",
    "err_console",
    "print_ablation_result",
    "print_adjusted_rankings",
    "print_adp_accuracy_report",
    "print_adp_movers_report",
    "print_bin_target_means",
    "print_binned_summary",
    "print_breakout_candidates",
    "print_candidate_values",
    "print_interaction_scan_results",
    "print_cascade_result",
    "print_category_needs",
    "print_cohort_bias_report",
    "print_cohort_bias_summary",
    "print_checkpoint_detail",
    "print_checkpoint_list",
    "print_classifier_evaluation",
    "print_column_profiles",
    "print_column_ranking",
    "print_compare_features_result",
    "print_comparison_result",
    "print_correlation_results",
    "print_dataset_list",
    "print_draft_board",
    "print_draft_report",
    "print_draft_tiers",
    "print_error",
    "print_error_decomposition_report",
    "print_feature_gap_report",
    "print_experiment_detail",
    "print_experiment_search_results",
    "print_experiment_summary",
    "print_features",
    "print_gate_result",
    "print_import_result",
    "print_ingest_result",
    "print_games_lost_leaderboard",
    "print_injury_estimate",
    "print_injury_profile",
    "print_injury_risk_leaderboard",
    "print_injury_value_deltas",
    "print_keeper_decisions",
    "print_league_keeper_overview",
    "print_keeper_scenarios",
    "print_keeper_solution",
    "print_keeper_trade_impact",
    "print_batch_simulation_result",
    "print_marginal_value_results",
    "print_mock_draft_result",
    "print_opportunity_costs",
    "print_player_summaries",
    "print_performance_report",
    "print_position_check",
    "print_preflight_result",
    "print_pick_trade_evaluation",
    "print_pick_value_curve",
    "print_player_projections",
    "print_pt_sources",
    "print_player_valuations",
    "print_coverage_matrix",
    "print_predict_result",
    "print_prepare_result",
    "print_projection_confidence",
    "print_quick_eval_result",
    "print_regression_check_result",
    "print_routing_table",
    "print_scarcity_rankings",
    "print_scarcity_report",
    "print_residual_analysis_report",
    "print_residual_persistence_report",
    "diff_records",
    "print_run_detail",
    "print_run_diff",
    "print_run_inspect",
    "print_run_list",
    "print_stability_result",
    "print_strategy_comparison_table",
    "print_stratified_comparison_result",
    "print_system_disagreements",
    "print_system_metrics",
    "print_system_summaries",
    "print_talent_delta_report",
    "print_talent_quality_report",
    "print_tier_summary",
    "print_trade_evaluation",
    "print_train_result",
    "print_tune_result",
    "print_upgrades",
    "print_validation_result",
    "print_valuation_eval_result",
    "print_valuation_rankings",
    "print_value_curve",
    "print_value_over_adp",
    "print_variance_targets",
]
