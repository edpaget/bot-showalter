from __future__ import annotations

import datetime

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.player_bio import PlayerSummary
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.repos.protocols import (
    BattingStatsRepo,
    PitchingStatsRepo,
    PlayerRepo,
    PositionAppearanceRepo,
    RosterStintRepo,
    TeamRepo,
)

_EXPERIENCE_YEAR_RANGE = range(2000, 2030)


def _compute_age(birth_date: str | None, season: int) -> int | None:
    if birth_date is None:
        return None
    born = datetime.date.fromisoformat(birth_date)
    july_1 = datetime.date(season, 7, 1)
    return july_1.year - born.year - ((july_1.month, july_1.day) < (born.month, born.day))


def _primary_position(appearances: list[PositionAppearance]) -> str:
    if not appearances:
        return "DH"
    return max(appearances, key=lambda a: a.games).position


class PlayerBiographyService:
    def __init__(
        self,
        player_repo: PlayerRepo,
        team_repo: TeamRepo,
        roster_stint_repo: RosterStintRepo,
        batting_stats_repo: BattingStatsRepo,
        pitching_stats_repo: PitchingStatsRepo,
        position_appearance_repo: PositionAppearanceRepo,
    ) -> None:
        self._player_repo = player_repo
        self._team_repo = team_repo
        self._roster_stint_repo = roster_stint_repo
        self._batting_stats_repo = batting_stats_repo
        self._pitching_stats_repo = pitching_stats_repo
        self._position_appearance_repo = position_appearance_repo
        self._team_map: dict[int, str] | None = None

    def _get_team_map(self) -> dict[int, str]:
        if self._team_map is None:
            teams = self._team_repo.all()
            self._team_map = {t.id: t.abbreviation for t in teams if t.id is not None}
        return self._team_map

    def _count_experience(self, player_id: int) -> int:
        seasons: set[int] = set()
        for year in _EXPERIENCE_YEAR_RANGE:
            if self._batting_stats_repo.get_by_player_season(player_id, year, source="fangraphs"):
                seasons.add(year)
            if self._pitching_stats_repo.get_by_player_season(player_id, year, source="fangraphs"):
                seasons.add(year)
        return len(seasons)

    def _build_summary(self, player: Player, season: int) -> PlayerSummary:
        assert player.id is not None
        team_map = self._get_team_map()

        stints = self._roster_stint_repo.get_by_player_season(player.id, season)
        team = team_map.get(stints[-1].team_id, "FA") if stints else "FA"

        appearances = self._position_appearance_repo.get_by_player_season(player.id, season)

        return PlayerSummary(
            player_id=player.id,
            name=f"{player.name_first} {player.name_last}",
            team=team,
            age=_compute_age(player.birth_date, season),
            primary_position=_primary_position(appearances),
            bats=player.bats,
            throws=player.throws,
            experience=self._count_experience(player.id),
        )

    def search(self, name: str, season: int) -> list[PlayerSummary]:
        players = self._player_repo.search_by_name(name)
        return [self._build_summary(p, season) for p in players]

    def find(
        self,
        *,
        season: int,
        team: str | None = None,
        min_age: int | None = None,
        max_age: int | None = None,
        min_experience: int | None = None,
        max_experience: int | None = None,
        position: str | None = None,
    ) -> list[PlayerSummary]:
        if team is not None:
            team_obj = self._team_repo.get_by_abbreviation(team)
            if team_obj is None or team_obj.id is None:
                return []
            stints = self._roster_stint_repo.get_by_team_season(team_obj.id, season)
        else:
            stints = self._roster_stint_repo.get_by_season(season)

        player_ids = list({s.player_id for s in stints})
        players = self._player_repo.get_by_ids(player_ids)

        results: list[PlayerSummary] = []
        for player in players:
            assert player.id is not None

            if min_age is not None or max_age is not None:
                age = _compute_age(player.birth_date, season)
                if age is None:
                    continue
                if min_age is not None and age < min_age:
                    continue
                if max_age is not None and age > max_age:
                    continue

            if position is not None:
                appearances = self._position_appearance_repo.get_by_player_season(player.id, season)
                if _primary_position(appearances) != position:
                    continue

            if min_experience is not None or max_experience is not None:
                exp = self._count_experience(player.id)
                if min_experience is not None and exp < min_experience:
                    continue
                if max_experience is not None and exp > max_experience:
                    continue

            results.append(self._build_summary(player, season))

        results.sort(key=lambda s: s.name)
        return results

    def get_bio(self, player_id: int, season: int) -> PlayerSummary | None:
        player = self._player_repo.get_by_id(player_id)
        if player is None:
            return None
        return self._build_summary(player, season)
