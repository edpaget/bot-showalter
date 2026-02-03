"""Composable pipeline builder for constructing custom projection pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fantasy_baseball_manager.cache.factory import create_cache_store
from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.pipeline.park_factors import (
    CachedParkFactorProvider,
    FanGraphsParkFactorProvider,
)
from fantasy_baseball_manager.pipeline.stages.adjusters import RebaselineAdjuster
from fantasy_baseball_manager.pipeline.stages.batter_babip_adjuster import (
    BatterBabipAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.component_aging import (
    ComponentAgingAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.finalizers import StandardFinalizer
from fantasy_baseball_manager.pipeline.stages.park_factor_adjuster import (
    ParkFactorAdjuster,
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
    PitcherStatcastDataSource,
    PybaseballStatcastDataSource,
    StatcastDataSource,
)
from fantasy_baseball_manager.player_id.mapper import (
    PlayerIdMapper,
    build_cached_sfbb_mapper,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.pipeline.protocols import RateAdjuster, RateComputer


class PipelineBuilder:
    """Fluent builder for composing projection pipelines.

    Always includes RebaselineAdjuster and ComponentAgingAdjuster.
    Optional adjusters are added via builder methods and ordered
    automatically: park -> pitcher_norm -> pitcher_statcast ->
    statcast -> batter_babip -> rebaseline -> aging.
    """

    def __init__(
        self,
        name: str = "custom",
        config: RegressionConfig | None = None,
    ) -> None:
        self._name = name
        self._config = config or RegressionConfig()
        self._rate_computer_type: str = "stat_specific"
        self._park_factors: bool = False
        self._pitcher_normalization: bool = False
        self._statcast: bool = False
        self._batter_babip: bool = False
        self._pitcher_statcast: bool = False
        self._statcast_source: StatcastDataSource | None = None
        self._pitcher_statcast_source: PitcherStatcastDataSource | None = None
        self._id_mapper: PlayerIdMapper | None = None
        self._split_source: SplitStatsDataSource | None = None

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

    def with_split_source(self, source: SplitStatsDataSource) -> PipelineBuilder:
        self._split_source = source
        return self

    def build(self) -> ProjectionPipeline:
        rate_computer = self._build_rate_computer()
        adjusters = self._build_adjusters()
        return ProjectionPipeline(
            name=self._name,
            rate_computer=rate_computer,
            adjusters=tuple(adjusters),
            playing_time=MarcelPlayingTime(),
            finalizer=StandardFinalizer(),
            years_back=3,
        )

    def _build_rate_computer(self) -> RateComputer:
        cfg = self._config
        if self._rate_computer_type == "platoon":
            split_source = self._split_source or CachedSplitDataSource(
                delegate=PybaseballSplitDataSource(),
                cache=create_cache_store(),
            )
            pitching_delegate = StatSpecificRegressionRateComputer(
                batting_regression=cfg.batting_regression_pa,
                pitching_regression=cfg.pitching_regression_outs,
            )
            return PlatoonRateComputer(
                split_source=split_source,
                pitching_delegate=pitching_delegate,
                batting_regression=cfg.platoon.batting_split_regression_pa,
                pct_vs_rhp=cfg.platoon.pct_vs_rhp,
                pct_vs_lhp=cfg.platoon.pct_vs_lhp,
            )
        return StatSpecificRegressionRateComputer(
            batting_regression=cfg.batting_regression_pa,
            pitching_regression=cfg.pitching_regression_outs,
        )

    def _build_adjusters(self) -> list[RateAdjuster]:
        adjusters: list[RateAdjuster] = []

        if self._park_factors:
            adjusters.append(
                ParkFactorAdjuster(
                    CachedParkFactorProvider(
                        delegate=FanGraphsParkFactorProvider(),
                        cache=create_cache_store(),
                    )
                )
            )

        if self._pitcher_normalization:
            adjusters.append(
                PitcherNormalizationAdjuster(self._config.pitcher_normalization)
            )

        if self._pitcher_statcast:
            source = self._resolve_pitcher_statcast_source()
            mapper = self._resolve_id_mapper()
            adjusters.append(PitcherStatcastAdjuster(
                statcast_source=source,
                id_mapper=mapper,
                config=self._config.pitcher_statcast,
            ))

        if self._statcast:
            source = self._resolve_statcast_source()
            mapper = self._resolve_id_mapper()
            adjusters.append(StatcastRateAdjuster(statcast_source=source, id_mapper=mapper))

        if self._batter_babip:
            source = self._resolve_statcast_source()
            mapper = self._resolve_id_mapper()
            adjusters.append(BatterBabipAdjuster(statcast_source=source, id_mapper=mapper))

        # Always last: rebaseline then aging
        adjusters.append(RebaselineAdjuster())
        adjusters.append(ComponentAgingAdjuster())

        return adjusters

    def _resolve_statcast_source(self) -> StatcastDataSource:
        if self._statcast_source is not None:
            return self._statcast_source
        source = CachedStatcastDataSource(
            delegate=PybaseballStatcastDataSource(),
            cache=create_cache_store(),
        )
        self._statcast_source = source
        return source

    def _resolve_pitcher_statcast_source(self) -> PitcherStatcastDataSource:
        if self._pitcher_statcast_source is not None:
            return self._pitcher_statcast_source
        # CachedStatcastDataSource satisfies both protocols
        source = self._resolve_statcast_source()
        return cast(PitcherStatcastDataSource, source)

    def _resolve_id_mapper(self) -> PlayerIdMapper:
        if self._id_mapper is not None:
            return self._id_mapper
        mapper = build_cached_sfbb_mapper(
            cache=create_cache_store(),
            cache_key="builder",
            ttl=7 * 86400,
        )
        self._id_mapper = mapper
        return mapper
