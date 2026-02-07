#!/usr/bin/env python
"""Train MLE model and run holdout evaluation.

This script:
1. Trains an MLE model on 2022-2023 call-up data
2. Evaluates on 2024 call-ups (held-out test set)
3. Compares against traditional MLE and no-translation baselines
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    from fantasy_baseball_manager.cache.factory import create_cache_store
    from fantasy_baseball_manager.cache.serialization import DataclassListSerializer
    from fantasy_baseball_manager.cache.wrapper import cached
    from fantasy_baseball_manager.marcel.data_source import create_batting_source
    from fantasy_baseball_manager.marcel.models import BattingSeasonStats
    from fantasy_baseball_manager.minors.data_source import (
        MiLBBatterStatsSerializer,
        create_milb_batting_source,
    )
    from fantasy_baseball_manager.minors.evaluation import (
        MLEEvaluator,
        print_evaluation_report,
    )
    from fantasy_baseball_manager.minors.persistence import MLEModelStore
    from fantasy_baseball_manager.minors.training import MLEModelTrainer, MLETrainingConfig
    from fantasy_baseball_manager.player_id.mapper import build_cached_sfbb_mapper

    # Set up data sources with caching
    logger.info("Setting up data sources...")
    cache = create_cache_store()

    # Build ID mapper for MLBAM <-> FanGraphs conversion
    logger.info("Building player ID mapper...")
    id_mapper = build_cached_sfbb_mapper(cache, cache_key="sfbb_2024", ttl=60 * 60 * 24 * 7)

    milb_source = cached(
        create_milb_batting_source(),
        namespace="milb_batting",
        ttl_seconds=30 * 86400,
        serializer=MiLBBatterStatsSerializer(),
    )
    mlb_batting_source = cached(
        create_batting_source(),
        namespace="stats_batting",
        ttl_seconds=30 * 86400,
        serializer=DataclassListSerializer(BattingSeasonStats),
    )

    # Check if model already exists
    model_store = MLEModelStore()
    model_name = "default"

    if model_store.exists(model_name, "batter"):
        logger.info("Loading existing MLE model...")
        model = model_store.load(model_name, "batter")
        logger.info(
            "Loaded model trained on years %s with stats %s",
            model.training_years,
            model.get_stats(),
        )
    else:
        # Train a new model
        logger.info("Training new MLE model on 2022-2023 data...")
        config = MLETrainingConfig(
            min_milb_pa=200,
            min_mlb_pa=100,
            max_prior_mlb_pa=200,
        )
        trainer = MLEModelTrainer(
            milb_source=milb_source,
            mlb_batting_source=mlb_batting_source,
            config=config,
            id_mapper=id_mapper,
        )

        # Train on 2022-2023 (MiLB features from 2021-2022, MLB targets from 2022-2023)
        training_years = (2022, 2023)
        model = trainer.train_batter_models(target_years=training_years)

        # Save the model
        logger.info("Saving trained model...")
        model_store.save(model, model_name)
        logger.info("Model saved to %s", model_store.model_dir)

    # Run evaluation on 2024 holdout
    logger.info("Evaluating on 2024 holdout set...")
    evaluator = MLEEvaluator(
        milb_source=milb_source,
        mlb_batting_source=mlb_batting_source,
        min_milb_pa=200,
        min_mlb_pa=100,
        max_prior_mlb_pa=200,
        id_mapper=id_mapper,
    )

    test_years = (2024,)
    model_report, baseline_reports = evaluator.evaluate_with_baselines(
        model=model,
        test_years=test_years,
        include_traditional=True,
        include_no_translation=True,
    )

    # Print results
    print_evaluation_report(model_report, baseline_reports)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if baseline_reports.get("traditional_mle") and baseline_reports["traditional_mle"].baseline_comparisons:
        trad_comps = baseline_reports["traditional_mle"].baseline_comparisons
        avg_improvement = sum(c.rmse_improvement_pct for c in trad_comps) / len(trad_comps)
        print(f"\nAverage RMSE improvement vs traditional MLE: {avg_improvement:.1f}%")

        # Count wins
        wins = sum(1 for c in trad_comps if c.rmse_improvement > 0)
        print(f"Stats where ML MLE beats traditional: {wins}/{len(trad_comps)}")

    if baseline_reports.get("no_translation") and baseline_reports["no_translation"].baseline_comparisons:
        no_trans_comps = baseline_reports["no_translation"].baseline_comparisons
        avg_improvement = sum(c.rmse_improvement_pct for c in no_trans_comps) / len(no_trans_comps)
        print(f"\nAverage RMSE improvement vs no translation: {avg_improvement:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
