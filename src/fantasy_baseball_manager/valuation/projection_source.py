from dataclasses import dataclass
from typing import Protocol

from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection


class ProjectionSource(Protocol):
    def batting_projections(self) -> list[BattingProjection]: ...

    def pitching_projections(self) -> list[PitchingProjection]: ...


@dataclass(frozen=True)
class SimpleProjectionSource:
    _batting: list[BattingProjection]
    _pitching: list[PitchingProjection]

    def batting_projections(self) -> list[BattingProjection]:
        return list(self._batting)

    def pitching_projections(self) -> list[PitchingProjection]:
        return list(self._pitching)
