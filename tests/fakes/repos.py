from typing import Any

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.valuation import Valuation


class FakePlayerRepo:
    def __init__(self, players: list[Player] | None = None) -> None:
        self._players = players or []

    def upsert(self, player: Player) -> int:
        return 1

    def get_by_id(self, player_id: int) -> Player | None:
        return next((p for p in self._players if p.id == player_id), None)

    def get_by_ids(self, player_ids: list[int]) -> list[Player]:
        return [p for p in self._players if p.id in player_ids]

    def get_by_mlbam_id(self, mlbam_id: int) -> Player | None:
        return None

    def get_by_bbref_id(self, bbref_id: str) -> Player | None:
        return None

    def search_by_name(self, name: str) -> list[Player]:
        return []

    def get_by_last_name(self, last_name: str) -> list[Player]:
        return []

    def all(self) -> list[Player]:
        return list(self._players)


class FakePositionAppearanceRepo:
    def __init__(self, appearances: list[PositionAppearance] | None = None) -> None:
        self._appearances = appearances or []

    def upsert(self, appearance: PositionAppearance) -> int:
        return 1

    def get_by_player(self, player_id: int) -> list[PositionAppearance]:
        return [a for a in self._appearances if a.player_id == player_id]

    def get_by_player_season(self, player_id: int, season: int) -> list[PositionAppearance]:
        return [a for a in self._appearances if a.player_id == player_id and a.season == season]

    def get_by_season(self, season: int) -> list[PositionAppearance]:
        return [a for a in self._appearances if a.season == season]


class FakeValuationRepo:
    def __init__(self, valuations: list[Valuation] | None = None) -> None:
        self._valuations: list[Valuation] = list(valuations) if valuations else []
        self.upserted: list[Valuation] = []

    def upsert(self, valuation: Valuation) -> int:
        self._valuations.append(valuation)
        self.upserted.append(valuation)
        return len(self._valuations)

    def get_by_player_season(self, player_id: int, season: int, system: str | None = None) -> list[Valuation]:
        result = [v for v in self._valuations if v.player_id == player_id and v.season == season]
        if system is not None:
            result = [v for v in result if v.system == system]
        return result

    def get_by_season(self, season: int, system: str | None = None) -> list[Valuation]:
        result = [v for v in self._valuations if v.season == season]
        if system is not None:
            result = [v for v in result if v.system == system]
        return result


class FakeProjectionRepo:
    def __init__(self, projections: list[Projection] | None = None) -> None:
        self._projections = projections or []

    def upsert(self, projection: Projection) -> int:
        return 1

    def get_by_player_season(
        self, player_id: int, season: int, system: str | None = None, *, include_distributions: bool = False
    ) -> list[Projection]:
        result = [p for p in self._projections if p.player_id == player_id and p.season == season]
        if system is not None:
            result = [p for p in result if p.system == system]
        return result

    def get_by_season(
        self, season: int, system: str | None = None, *, include_distributions: bool = False
    ) -> list[Projection]:
        result = [p for p in self._projections if p.season == season]
        if system is not None:
            result = [p for p in result if p.system == system]
        return result

    def get_by_system_version(self, system: str, version: str) -> list[Projection]:
        return [p for p in self._projections if p.system == system and p.version == version]

    def upsert_distributions(self, projection_id: int, distributions: list[Any]) -> None:
        pass

    def get_distributions(self, projection_id: int) -> list[Any]:
        return []


class FakeADPRepo:
    def __init__(self, adps: list[ADP] | None = None) -> None:
        self._adps = adps or []

    def upsert(self, adp: ADP) -> int:
        return 1

    def get_by_player_season(self, player_id: int, season: int) -> list[ADP]:
        return [a for a in self._adps if a.player_id == player_id and a.season == season]

    def get_by_season(self, season: int, provider: str | None = None) -> list[ADP]:
        result = [a for a in self._adps if a.season == season]
        if provider is not None:
            result = [a for a in result if a.provider == provider]
        return result


class FakeBattingStatsRepo:
    def __init__(self, stats: list[BattingStats] | None = None) -> None:
        self._stats = stats or []

    def upsert(self, stats: BattingStats) -> int:
        return 1

    def get_by_player_season(self, player_id: int, season: int, source: str | None = None) -> list[BattingStats]:
        result = [s for s in self._stats if s.player_id == player_id and s.season == season]
        if source is not None:
            result = [s for s in result if s.source == source]
        return result

    def get_by_season(self, season: int, source: str | None = None) -> list[BattingStats]:
        result = [s for s in self._stats if s.season == season]
        if source is not None:
            result = [s for s in result if s.source == source]
        return result


class FakePitchingStatsRepo:
    def __init__(self, stats: list[PitchingStats] | None = None) -> None:
        self._stats = stats or []

    def upsert(self, stats: PitchingStats) -> int:
        return 1

    def get_by_player_season(self, player_id: int, season: int, source: str | None = None) -> list[PitchingStats]:
        result = [s for s in self._stats if s.player_id == player_id and s.season == season]
        if source is not None:
            result = [s for s in result if s.source == source]
        return result

    def get_by_season(self, season: int, source: str | None = None) -> list[PitchingStats]:
        result = [s for s in self._stats if s.season == season]
        if source is not None:
            result = [s for s in result if s.source == source]
        return result
