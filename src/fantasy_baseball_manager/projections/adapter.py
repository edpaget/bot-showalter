"""Adapter to use external projections with the pipeline interface."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSourceError
from fantasy_baseball_manager.marcel.models import (
    BattingProjection as MarcelBattingProjection,
)
from fantasy_baseball_manager.marcel.models import (
    PitchingProjection as MarcelPitchingProjection,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.data.protocol import DataSource
    from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats
    from fantasy_baseball_manager.player_id.mapper import SfbbMapper
    from fantasy_baseball_manager.projections.models import BattingProjection, PitchingProjection

logger = logging.getLogger(__name__)


class ExternalProjectionAdapter:
    """Adapts external projections to the ProjectionPipeline interface.

    This adapter allows external projection systems (Steamer, ZiPS) to be used
    anywhere a ProjectionPipeline is expected. It converts external projection
    models to the internal Marcel projection format.

    Accepts separate DataSource[BattingProjection] and DataSource[PitchingProjection]
    for integration with the unified DataSource/cached() infrastructure.
    """

    def __init__(
        self,
        batting_source: DataSource[BattingProjection],
        pitching_source: DataSource[PitchingProjection],
        *,
        name: str = "external",
        projection_year: int | None = None,
        id_mapper: SfbbMapper | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            batting_source: DataSource providing batting projections.
            pitching_source: DataSource providing pitching projections.
            name: Display name for this adapter (e.g., "steamer", "zips").
            projection_year: The year these projections are for. Defaults to current year.
            id_mapper: Optional mapper to resolve FanGraphs IDs from MLBAM IDs.
        """
        self._batting_source = batting_source
        self._pitching_source = pitching_source
        self._name = name
        self._projection_year = projection_year or datetime.now().year
        self._id_mapper = id_mapper

    @classmethod
    def from_projection_source(
        cls,
        source: ...,  # ProjectionSource (avoids import for rarely-used path)
        *,
        projection_year: int | None = None,
        id_mapper: SfbbMapper | None = None,
    ) -> ExternalProjectionAdapter:
        """Create from a legacy ProjectionSource.

        Convenience constructor for backward compatibility with ProjectionSource
        implementations (e.g. CSVProjectionSource).

        Args:
            source: A ProjectionSource with a fetch_projections() method.
            projection_year: The year these projections are for.
            id_mapper: Optional mapper to resolve FanGraphs IDs from MLBAM IDs.

        Returns:
            An ExternalProjectionAdapter wrapping the source.
        """
        from fantasy_baseball_manager.projections.data_source import (
            BattingProjectionDataSource,
            PitchingProjectionDataSource,
        )

        return cls(
            BattingProjectionDataSource(source),
            PitchingProjectionDataSource(source),
            name=getattr(source, "name", "external"),
            projection_year=projection_year,
            id_mapper=id_mapper,
        )

    @property
    def name(self) -> str:
        """Return the name of this adapter for display."""
        return self._name

    @property
    def years_back(self) -> int:
        """Return years of historical data used.

        For external projections this is not applicable, but we return 0
        for interface compatibility.
        """
        return 0

    def project_batters(
        self,
        batting_source: DataSource[BattingSeasonStats],
        team_batting_source: DataSource[BattingSeasonStats],
        year: int,
    ) -> list[MarcelBattingProjection]:
        """Convert external batting projections to internal format.

        Args:
            batting_source: Ignored - external projections are pre-computed.
            team_batting_source: Ignored - external projections are pre-computed.
            year: Ignored - uses projection_year from constructor.

        Returns:
            List of Marcel-format batting projections.
        """
        batting = self._fetch_batting()
        result: list[MarcelBattingProjection] = []

        for ext in batting:
            # Skip players with minimal playing time
            if ext.pa < 50:
                continue

            proj = MarcelBattingProjection(
                player_id=self._resolve_player_id(ext.mlbam_id, ext.player_id),
                name=ext.name,
                year=self._projection_year,
                age=0,  # Age not available from FanGraphs
                pa=float(ext.pa),
                ab=float(ext.ab),
                h=float(ext.h),
                singles=float(ext.singles),
                doubles=float(ext.doubles),
                triples=float(ext.triples),
                hr=float(ext.hr),
                bb=float(ext.bb),
                so=float(ext.so),
                hbp=float(ext.hbp),
                sf=float(ext.sf),
                sh=float(ext.sh),
                sb=float(ext.sb),
                cs=float(ext.cs),
                r=float(ext.r),
                rbi=float(ext.rbi),
            )
            result.append(proj)

        logger.info(
            "Converted %d external batting projections to internal format",
            len(result),
        )
        return result

    def project_pitchers(
        self,
        pitching_source: DataSource[PitchingSeasonStats],
        team_pitching_source: DataSource[PitchingSeasonStats],
        year: int,
    ) -> list[MarcelPitchingProjection]:
        """Convert external pitching projections to internal format.

        Args:
            pitching_source: Ignored - external projections are pre-computed.
            team_pitching_source: Ignored - external projections are pre-computed.
            year: Ignored - uses projection_year from constructor.

        Returns:
            List of Marcel-format pitching projections.
        """
        pitching = self._fetch_pitching()
        result: list[MarcelPitchingProjection] = []

        for ext in pitching:
            # Skip players with minimal innings
            if ext.ip < 10:
                continue

            proj = MarcelPitchingProjection(
                player_id=self._resolve_player_id(ext.mlbam_id, ext.player_id),
                name=ext.name,
                year=self._projection_year,
                age=0,  # Age not available from FanGraphs
                ip=ext.ip,
                g=float(ext.g),
                gs=float(ext.gs),
                er=float(ext.er),
                h=float(ext.h),
                bb=float(ext.bb),
                so=float(ext.so),
                hr=float(ext.hr),
                hbp=float(ext.hbp),
                era=ext.era,
                whip=ext.whip,
                w=float(ext.w),
                nsvh=float(ext.sv + ext.hld),  # Combined saves + holds
            )
            result.append(proj)

        logger.info(
            "Converted %d external pitching projections to internal format",
            len(result),
        )
        return result

    def _fetch_batting(self) -> list[BattingProjection]:
        """Fetch batting projections from the DataSource."""
        result = self._batting_source(ALL_PLAYERS)
        if result.is_err():
            raise DataSourceError(f"Failed to fetch batting projections: {result.unwrap_err()}")
        return result.unwrap()

    def _fetch_pitching(self) -> list[PitchingProjection]:
        """Fetch pitching projections from the DataSource."""
        result = self._pitching_source(ALL_PLAYERS)
        if result.is_err():
            raise DataSourceError(f"Failed to fetch pitching projections: {result.unwrap_err()}")
        return result.unwrap()

    def _resolve_player_id(
        self,
        mlbam_id: str | None,
        fangraphs_id: str,
    ) -> str:
        """Resolve to a consistent player ID.

        Prefers FanGraphs ID to match actuals (which use IDfg). When the
        FanGraphs ID is missing and an id_mapper is available, attempts to
        resolve from the MLBAM ID. Falls back to MLBAM ID as a last resort.

        Args:
            mlbam_id: MLB Advanced Media ID if available.
            fangraphs_id: FanGraphs player ID.

        Returns:
            Resolved player ID string.
        """
        if fangraphs_id:
            return fangraphs_id
        if mlbam_id and self._id_mapper is not None:
            resolved = self._id_mapper.mlbam_to_fangraphs(mlbam_id)
            if resolved:
                return resolved
        if mlbam_id:
            return mlbam_id
        logger.warning("Player has no usable ID (fangraphs_id and mlbam_id both empty)")
        return ""
