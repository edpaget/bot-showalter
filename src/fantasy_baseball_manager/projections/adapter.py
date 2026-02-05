"""Adapter to use external projections with the pipeline interface."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from fantasy_baseball_manager.marcel.models import (
    BattingProjection as MarcelBattingProjection,
    PitchingProjection as MarcelPitchingProjection,
)
from fantasy_baseball_manager.projections.models import ProjectionData

if TYPE_CHECKING:
    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
    from fantasy_baseball_manager.projections.protocol import ProjectionSource

logger = logging.getLogger(__name__)


class ExternalProjectionAdapter:
    """Adapts external projections to the ProjectionPipeline interface.

    This adapter allows external projection systems (Steamer, ZiPS) to be used
    anywhere a ProjectionPipeline is expected. It converts external projection
    models to the internal Marcel projection format.
    """

    def __init__(
        self,
        source: ProjectionSource,
        projection_year: int | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            source: The external projection source to fetch from.
            projection_year: The year these projections are for. Defaults to current year.
        """
        self._source = source
        self._projection_year = projection_year or datetime.now().year
        self._data: ProjectionData | None = None

    @property
    def name(self) -> str:
        """Return the name of this adapter for display."""
        if self._data is not None:
            return self._data.system.value
        return "external"

    @property
    def years_back(self) -> int:
        """Return years of historical data used.

        For external projections this is not applicable, but we return 0
        for interface compatibility.
        """
        return 0

    def project_batters(
        self,
        data_source: StatsDataSource,
        year: int,
    ) -> list[MarcelBattingProjection]:
        """Convert external batting projections to internal format.

        Args:
            data_source: Ignored - external projections are pre-computed.
            year: Ignored - uses projection_year from constructor.

        Returns:
            List of Marcel-format batting projections.
        """
        data = self._get_data()
        result: list[MarcelBattingProjection] = []

        for ext in data.batting:
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
        data_source: StatsDataSource,
        year: int,
    ) -> list[MarcelPitchingProjection]:
        """Convert external pitching projections to internal format.

        Args:
            data_source: Ignored - external projections are pre-computed.
            year: Ignored - uses projection_year from constructor.

        Returns:
            List of Marcel-format pitching projections.
        """
        data = self._get_data()
        result: list[MarcelPitchingProjection] = []

        for ext in data.pitching:
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

    def _get_data(self) -> ProjectionData:
        """Fetch and cache projection data."""
        if self._data is None:
            self._data = self._source.fetch_projections()
        return self._data

    def _resolve_player_id(
        self,
        mlbam_id: str | None,
        fangraphs_id: str,
    ) -> str:
        """Resolve to a consistent player ID.

        Prefers MLBAM ID for cross-system compatibility, falls back to
        FanGraphs ID prefixed with 'fg:' to avoid collisions.

        Args:
            mlbam_id: MLB Advanced Media ID if available.
            fangraphs_id: FanGraphs player ID.

        Returns:
            Resolved player ID string.
        """
        if mlbam_id:
            return mlbam_id
        return f"fg:{fangraphs_id}"
