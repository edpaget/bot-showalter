from typing import Protocol

from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingProjection,
    PitchingSeasonStats,
)


class ProjectionBlender(Protocol):
    def blend_batting(
        self,
        preseason: BattingProjection,
        actuals: BattingSeasonStats,
    ) -> BattingProjection: ...

    def blend_pitching(
        self,
        preseason: PitchingProjection,
        actuals: PitchingSeasonStats,
    ) -> PitchingProjection: ...
