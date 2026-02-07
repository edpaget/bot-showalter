"""Composable pipeline builder for constructing custom projection pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fantasy_baseball_manager.cache.factory import create_cache_store
from fantasy_baseball_manager.cache.protocol import CacheStore  # noqa: TC001
from fantasy_baseball_manager.pipeline.batted_ball_data import (
    CachedBattedBallDataSource,
    PitcherBattedBallDataSource,
    PybaseballBattedBallDataSource,
)
from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.pipeline.park_factors import (
    CachedParkFactorProvider,
    FanGraphsParkFactorProvider,
)
from fantasy_baseball_manager.pipeline.skill_data import (
    CachedSkillDataSource,
    CompositeSkillDataSource,
    FanGraphsSkillDataSource,
    SkillDataSource,
    StatcastSprintSpeedSource,
)
from fantasy_baseball_manager.pipeline.stages.adjusters import RebaselineAdjuster
from fantasy_baseball_manager.pipeline.stages.batter_babip_adjuster import (
    BatterBabipAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.component_aging import (
    ComponentAgingAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.enhanced_playing_time import (
    EnhancedPlayingTimeProjector,
)
from fantasy_baseball_manager.pipeline.stages.finalizers import StandardFinalizer
from fantasy_baseball_manager.pipeline.stages.gb_residual_adjuster import (
    GBResidualAdjuster,
    GBResidualConfig,
)
from fantasy_baseball_manager.pipeline.stages.mtl_blender import MTLBlender
from fantasy_baseball_manager.pipeline.stages.mtl_rate_computer import MTLRateComputer
from fantasy_baseball_manager.pipeline.stages.park_factor_adjuster import (
    ParkFactorAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.pitcher_babip_skill_adjuster import (
    PitcherBabipSkillAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.pitcher_normalization import (
    PitcherNormalizationAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.pitcher_statcast_adjuster import (
    PitcherStatcastAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.platoon_rate_computer import (
    PlatoonRateComputer,
)
from fantasy_baseball_manager.pipeline.stages.playing_time import MarcelPlayingTime
from fantasy_baseball_manager.pipeline.stages.regression_config import RegressionConfig
from fantasy_baseball_manager.pipeline.stages.skill_change_adjuster import (
    SkillChangeAdjuster,
    SkillChangeConfig,
    SkillDeltaComputer,
)
from fantasy_baseball_manager.pipeline.stages.split_data_source import (
    CachedSplitDataSource,
    PybaseballSplitDataSource,
    SplitStatsDataSource,
)
from fantasy_baseball_manager.pipeline.stages.stat_specific_rate_computer import (
    StatSpecificRegressionRateComputer,
)
from fantasy_baseball_manager.pipeline.stages.statcast_adjuster import (
    StatcastRateAdjuster,
)
from fantasy_baseball_manager.pipeline.statcast_data import (
    CachedStatcastDataSource,
    FullStatcastDataSource,
    PitcherStatcastDataSource,
    PybaseballStatcastDataSource,
    StatcastDataSource,
)
from fantasy_baseball_manager.player_id.mapper import (
    PlayerIdMapper,
    build_cached_sfbb_mapper,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.data.protocol import DataSource
    from fantasy_baseball_manager.minors.rate_computer import MLERateComputerConfig
    from fantasy_baseball_manager.minors.types import MinorLeagueBatterSeasonStats
    from fantasy_baseball_manager.ml.mtl.config import MTLBlenderConfig, MTLRateComputerConfig
    from fantasy_baseball_manager.pipeline.protocols import RateAdjuster
    from fantasy_baseball_manager.pipeline.stages.playing_time_config import (
        PlayingTimeConfig,
    )


class PipelineBuilder:
    """Fluent builder for composing projection pipelines.

    Always includes RebaselineAdjuster and ComponentAgingAdjuster.
    Optional adjusters are added via builder methods and ordered
    automatically: park -> pitcher_norm -> pitcher_statcast ->
    pitcher_babip_skill -> statcast -> batter_babip -> rebaseline -> aging.
    """

    def __init__(
        self,
        name: str = "custom",
        config: RegressionConfig | None = None,
        cache_store: CacheStore | None = None,
    ) -> None:
        self._name = name
        self._config = config or RegressionConfig()
        self._cache_store = cache_store
        self._rate_computer_type: str = "stat_specific"
        self._park_factors: bool = False
        self._pitcher_normalization: bool = False
        self._statcast: bool = False
        self._batter_babip: bool = False
        self._pitcher_statcast: bool = False
        self._statcast_source: StatcastDataSource | None = None
        self._pitcher_statcast_source: PitcherStatcastDataSource | None = None
        self._id_mapper: PlayerIdMapper | None = None
        self._pitcher_babip_skill: bool = False
        self._pitcher_babip_source: PitcherBattedBallDataSource | None = None
        self._split_source: SplitStatsDataSource | None = None
        self._enhanced_playing_time: bool = False
        self._playing_time_config: PlayingTimeConfig | None = None
        self._gb_residual: bool = False
        self._gb_residual_config: GBResidualConfig | None = None
        self._skill_change: bool = False
        self._skill_change_config: SkillChangeConfig | None = None
        self._skill_data_source: SkillDataSource | None = None
        self._mtl_rate_computer: bool = False
        self._mtl_rate_computer_config: MTLRateComputerConfig | None = None
        self._mtl_blender: bool = False
        self._mtl_blender_config: MTLBlenderConfig | None = None
        self._mle_rate_computer: bool = False
        self._mle_rate_computer_config: MLERateComputerConfig | None = None
        self._mle_for_rookies: bool = False
        self._mle_for_rookies_config: MLERateComputerConfig | None = None

    def with_cache_store(self, cache_store: CacheStore) -> PipelineBuilder:
        """Set the cache store to use for all cached data sources.

        If not set, a new cache store is created via create_cache_store().
        This is useful for testing or when you want to share a cache across
        multiple pipelines.
        """
        self._cache_store = cache_store
        return self

    def rate_computer(self, kind: str) -> PipelineBuilder:
        """Set the rate computer: 'stat_specific' (default) or 'platoon'."""
        if kind not in ("stat_specific", "platoon"):
            raise ValueError(f"Unknown rate computer: {kind!r}")
        self._rate_computer_type = kind
        return self

    def with_park_factors(self) -> PipelineBuilder:
        self._park_factors = True
        return self

    def with_pitcher_normalization(self) -> PipelineBuilder:
        self._pitcher_normalization = True
        return self

    def with_statcast(
        self,
        statcast_source: StatcastDataSource | None = None,
        id_mapper: PlayerIdMapper | None = None,
    ) -> PipelineBuilder:
        self._statcast = True
        if statcast_source is not None:
            self._statcast_source = statcast_source
        if id_mapper is not None:
            self._id_mapper = id_mapper
        return self

    def with_batter_babip(
        self,
        statcast_source: StatcastDataSource | None = None,
        id_mapper: PlayerIdMapper | None = None,
    ) -> PipelineBuilder:
        self._batter_babip = True
        if statcast_source is not None:
            self._statcast_source = statcast_source
        if id_mapper is not None:
            self._id_mapper = id_mapper
        return self

    def with_pitcher_statcast(
        self,
        pitcher_statcast_source: PitcherStatcastDataSource | None = None,
        id_mapper: PlayerIdMapper | None = None,
    ) -> PipelineBuilder:
        self._pitcher_statcast = True
        if pitcher_statcast_source is not None:
            self._pitcher_statcast_source = pitcher_statcast_source
        if id_mapper is not None:
            self._id_mapper = id_mapper
        return self

    def with_pitcher_babip_skill(
        self,
        source: PitcherBattedBallDataSource | None = None,
    ) -> PipelineBuilder:
        self._pitcher_babip_skill = True
        if source is not None:
            self._pitcher_babip_source = source
        return self

    def with_split_source(self, source: SplitStatsDataSource) -> PipelineBuilder:
        self._split_source = source
        return self

    def with_enhanced_playing_time(
        self,
        config: PlayingTimeConfig | None = None,
    ) -> PipelineBuilder:
        """Enable enhanced playing time with injury, age, and volatility adjustments."""
        self._enhanced_playing_time = True
        if config is not None:
            self._playing_time_config = config
        return self

    def with_gb_residual(
        self,
        config: GBResidualConfig | None = None,
    ) -> PipelineBuilder:
        """Enable gradient boosting residual adjustments using trained ML models."""
        self._gb_residual = True
        if config is not None:
            self._gb_residual_config = config
        return self

    def with_skill_change_adjuster(
        self,
        skill_source: SkillDataSource | None = None,
        config: SkillChangeConfig | None = None,
    ) -> PipelineBuilder:
        """Enable skill change adjustments based on year-over-year skill changes.

        Detects changes in barrel rate, exit velocity, chase rate, whiff rate,
        sprint speed (batters) and fastball velocity, whiff rate, ground ball
        rate (pitchers) and applies targeted projection adjustments.
        """
        self._skill_change = True
        if skill_source is not None:
            self._skill_data_source = skill_source
        if config is not None:
            self._skill_change_config = config
        return self

    def with_mtl_rate_computer(
        self,
        config: MTLRateComputerConfig | None = None,
    ) -> PipelineBuilder:
        """Use MTL neural network for rate computation (replaces Marcel).

        The MTL model predicts raw stat rates directly using Statcast features.
        Players without sufficient Statcast data fall back to Marcel rates.
        """
        self._mtl_rate_computer = True
        if config is not None:
            self._mtl_rate_computer_config = config
        return self

    def with_mtl_blender(
        self,
        config: MTLBlenderConfig | None = None,
    ) -> PipelineBuilder:
        """Blend Marcel rates with MTL predictions.

        Ensemble mode: blended = (1 - weight) * marcel + weight * mtl
        Default weight is 0.3 (30% MTL, 70% Marcel).
        """
        self._mtl_blender = True
        if config is not None:
            self._mtl_blender_config = config
        return self

    def with_mle_rate_computer(
        self,
        config: MLERateComputerConfig | None = None,
    ) -> PipelineBuilder:
        """Use ML-based Minor League Equivalencies for rate computation.

        For players with limited MLB history (<200 PA), uses trained MLE models
        to translate their minor league stats to MLB equivalents and blends
        them with available MLB data.

        Players with sufficient MLB history fall back to Marcel rates.
        """
        self._mle_rate_computer = True
        if config is not None:
            self._mle_rate_computer_config = config
        return self

    def with_mle_for_rookies(
        self,
        config: MLERateComputerConfig | None = None,
    ) -> PipelineBuilder:
        """Augment rate computation with MLE for rookies.

        This wraps the existing rate computer (e.g., StatSpecificRegression)
        and augments it with MLE predictions for players with limited MLB
        history (<200 PA). Established players use the normal rate computer
        unchanged.

        This is different from with_mle_rate_computer() which replaces the
        rate computer entirely. Use this method when you want to keep the
        advanced rate computation (with all adjusters) but add MLE support
        for rookies.

        Requires a trained MLE model (run scripts/run_mle_evaluation.py).
        """
        self._mle_for_rookies = True
        if config is not None:
            self._mle_for_rookies_config = config
        return self

    def build(self) -> ProjectionPipeline:
        rate_computer = self._build_rate_computer()
        adjusters = self._build_adjusters()
        playing_time = self._build_playing_time()
        return ProjectionPipeline(
            name=self._name,
            rate_computer=rate_computer,
            adjusters=tuple(adjusters),
            playing_time=playing_time,
            finalizer=StandardFinalizer(),
            years_back=3,
        )

    def _build_playing_time(self) -> MarcelPlayingTime | EnhancedPlayingTimeProjector:
        if self._enhanced_playing_time:
            return EnhancedPlayingTimeProjector(config=self._playing_time_config)
        return MarcelPlayingTime()

    def _build_rate_computer(self) -> Any:  # TODO: restore -> RateComputer after all implementations migrated
        cfg = self._config

        # MTL rate computer (replaces Marcel entirely)
        if self._mtl_rate_computer:
            from fantasy_baseball_manager.ml.mtl.config import MTLRateComputerConfig

            return MTLRateComputer(
                statcast_source=self._resolve_full_statcast_source(),
                batted_ball_source=self._resolve_pitcher_babip_source(),
                skill_data_source=self._resolve_skill_data_source(),
                id_mapper=self._resolve_id_mapper(),
                config=self._mtl_rate_computer_config or MTLRateComputerConfig(),
            )

        # MLE rate computer (ML-based Minor League Equivalencies)
        if self._mle_rate_computer:
            from fantasy_baseball_manager.minors.rate_computer import (
                MLERateComputer,
                MLERateComputerConfig,
            )

            return MLERateComputer(
                milb_source=self._resolve_milb_source(),
                config=self._mle_rate_computer_config or MLERateComputerConfig(),
            )

        # Build base rate computer
        # TODO: restore RateComputer type after migrating all implementations
        base_computer: Any
        if self._rate_computer_type == "platoon":
            split_source = self._split_source or CachedSplitDataSource(
                delegate=PybaseballSplitDataSource(),
                cache=self._get_cache_store(),
            )
            pitching_delegate = StatSpecificRegressionRateComputer(
                batting_regression=cfg.batting_regression_pa,
                pitching_regression=cfg.pitching_regression_outs,
            )
            base_computer = PlatoonRateComputer(
                split_source=split_source,
                pitching_delegate=pitching_delegate,
                batting_regression=cfg.platoon.batting_split_regression_pa,
                pct_vs_rhp=cfg.platoon.pct_vs_rhp,
                pct_vs_lhp=cfg.platoon.pct_vs_lhp,
            )
        else:
            base_computer = StatSpecificRegressionRateComputer(
                batting_regression=cfg.batting_regression_pa,
                pitching_regression=cfg.pitching_regression_outs,
            )

        # Wrap with MLE augmentation for rookies if enabled
        if self._mle_for_rookies:
            from fantasy_baseball_manager.minors.rate_computer import (
                MLEAugmentedRateComputer,
                MLERateComputerConfig,
            )

            return MLEAugmentedRateComputer(
                delegate=base_computer,
                milb_source=self._resolve_milb_source(),
                id_mapper=self._resolve_id_mapper(),
                config=self._mle_for_rookies_config or MLERateComputerConfig(),
            )

        return base_computer

    def _build_adjusters(self) -> list[RateAdjuster]:
        adjusters: list[RateAdjuster] = []

        if self._park_factors:
            adjusters.append(
                ParkFactorAdjuster(
                    CachedParkFactorProvider(
                        delegate=FanGraphsParkFactorProvider(),
                        cache=self._get_cache_store(),
                    )
                )
            )

        if self._pitcher_normalization:
            adjusters.append(PitcherNormalizationAdjuster(self._config.pitcher_normalization))

        if self._pitcher_statcast:
            source = self._resolve_pitcher_statcast_source()
            mapper = self._resolve_id_mapper()
            adjusters.append(
                PitcherStatcastAdjuster(
                    statcast_source=source,
                    id_mapper=mapper,
                    config=self._config.pitcher_statcast,
                )
            )

        if self._pitcher_babip_skill:
            bb_source = self._resolve_pitcher_babip_source()
            adjusters.append(
                PitcherBabipSkillAdjuster(
                    source=bb_source,
                    config=self._config.pitcher_babip_skill,
                )
            )

        if self._statcast:
            source = self._resolve_statcast_source()
            mapper = self._resolve_id_mapper()
            adjusters.append(StatcastRateAdjuster(statcast_source=source, id_mapper=mapper))

        if self._batter_babip:
            source = self._resolve_statcast_source()
            mapper = self._resolve_id_mapper()
            adjusters.append(BatterBabipAdjuster(statcast_source=source, id_mapper=mapper))

        if self._gb_residual:
            full_source = self._resolve_full_statcast_source()
            bb_source = self._resolve_pitcher_babip_source()
            skill_source = self._resolve_skill_data_source()
            mapper = self._resolve_id_mapper()
            adjusters.append(
                GBResidualAdjuster(
                    statcast_source=full_source,
                    batted_ball_source=bb_source,
                    skill_data_source=skill_source,
                    id_mapper=mapper,
                    config=self._gb_residual_config or GBResidualConfig(),
                )
            )

        if self._skill_change:
            skill_source = self._resolve_skill_data_source()
            delta_computer = SkillDeltaComputer(skill_source)
            adjusters.append(
                SkillChangeAdjuster(
                    delta_computer=delta_computer,
                    config=self._skill_change_config or SkillChangeConfig(),
                )
            )

        if self._mtl_blender:
            from fantasy_baseball_manager.ml.mtl.config import MTLBlenderConfig

            full_source = self._resolve_full_statcast_source()
            bb_source = self._resolve_pitcher_babip_source()
            skill_source = self._resolve_skill_data_source()
            mapper = self._resolve_id_mapper()
            adjusters.append(
                MTLBlender(
                    statcast_source=full_source,
                    batted_ball_source=bb_source,
                    skill_data_source=skill_source,
                    id_mapper=mapper,
                    config=self._mtl_blender_config or MTLBlenderConfig(),
                )
            )

        # Always last: rebaseline then aging
        adjusters.append(RebaselineAdjuster())
        adjusters.append(ComponentAgingAdjuster())

        return adjusters

    def _resolve_statcast_source(self) -> StatcastDataSource:
        if self._statcast_source is not None:
            return self._statcast_source
        source = CachedStatcastDataSource(
            delegate=PybaseballStatcastDataSource(),
            cache=self._get_cache_store(),
        )
        self._statcast_source = source
        return source

    def _resolve_pitcher_statcast_source(self) -> PitcherStatcastDataSource:
        if self._pitcher_statcast_source is not None:
            return self._pitcher_statcast_source
        # CachedStatcastDataSource satisfies both protocols
        source = self._resolve_statcast_source()
        return cast("PitcherStatcastDataSource", source)

    def _resolve_pitcher_babip_source(self) -> PitcherBattedBallDataSource:
        if self._pitcher_babip_source is not None:
            return self._pitcher_babip_source
        source = CachedBattedBallDataSource(
            delegate=PybaseballBattedBallDataSource(),
            cache=self._get_cache_store(),
        )
        self._pitcher_babip_source = source
        return source

    def _resolve_id_mapper(self) -> PlayerIdMapper:
        if self._id_mapper is not None:
            return self._id_mapper
        mapper = build_cached_sfbb_mapper(
            cache=self._get_cache_store(),
            cache_key="builder",
            ttl=7 * 86400,
        )
        self._id_mapper = mapper
        return mapper

    def _resolve_full_statcast_source(self) -> FullStatcastDataSource:
        # CachedStatcastDataSource satisfies FullStatcastDataSource protocol
        source = self._resolve_statcast_source()
        return cast("FullStatcastDataSource", source)

    def _resolve_skill_data_source(self) -> SkillDataSource:
        """Resolve or create a skill data source."""
        if self._skill_data_source is not None:
            return self._skill_data_source
        mapper = self._resolve_id_mapper()
        fangraphs = FanGraphsSkillDataSource()
        sprint = StatcastSprintSpeedSource()
        composite = CompositeSkillDataSource(fangraphs, sprint, mapper)
        source = CachedSkillDataSource(composite, self._get_cache_store())
        self._skill_data_source = source
        return source

    def _resolve_milb_source(self) -> DataSource[MinorLeagueBatterSeasonStats]:
        """Resolve or create a minor league data source."""
        from fantasy_baseball_manager.cache.wrapper import cached
        from fantasy_baseball_manager.minors.data_source import (
            MiLBBatterStatsSerializer,
            create_milb_batting_source,
        )

        return cached(
            create_milb_batting_source(),
            namespace="milb_batting",
            ttl_seconds=30 * 86400,
            serializer=MiLBBatterStatsSerializer(),
        )

    def _get_cache_store(self) -> CacheStore:
        """Get the cache store, creating one if not injected."""
        if self._cache_store is None:
            self._cache_store = create_cache_store()
        return self._cache_store
