from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import PlayerSummary, compute_age
from fantasy_baseball_manager.name_utils import resolve_players
from fantasy_baseball_manager.team_aliases import to_lahman, to_modern

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Player, PositionAppearance
    from fantasy_baseball_manager.repos import (
        BattingStatsRepo,
        PitchingStatsRepo,
        PlayerRepo,
        PlayerTeamProvider,
        PositionAppearanceRepo,
        RosterStintRepo,
        TeamRepo,
        TeamResolverProto,
    )
_EXPERIENCE_YEAR_RANGE = range(2000, 2030)


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
        player_team_provider: PlayerTeamProvider | None = None,
        team_resolver: TeamResolverProto | None = None,
    ) -> None:
        self._player_repo = player_repo
        self._team_repo = team_repo
        self._roster_stint_repo = roster_stint_repo
        self._batting_stats_repo = batting_stats_repo
        self._pitching_stats_repo = pitching_stats_repo
        self._position_appearance_repo = position_appearance_repo
        self._player_team_provider = player_team_provider
        self._team_resolver = team_resolver
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
        assert player.id is not None  # noqa: S101 - type narrowing
        team_map = self._get_team_map()

        stints = self._roster_stint_repo.get_by_player_season(player.id, season)
        if stints:
            team = to_modern(team_map.get(stints[-1].team_id, "FA"))
        elif self._player_team_provider is not None:
            provider_teams = self._player_team_provider.get_player_teams(season)
            team = provider_teams.get(player.id, "FA")
        else:
            team = "FA"

        appearances = self._position_appearance_repo.get_by_player_season(player.id, season)

        return PlayerSummary(
            player_id=player.id,
            name=f"{player.name_first} {player.name_last}",
            team=team,
            age=compute_age(player.birth_date, season),
            primary_position=_primary_position(appearances),
            bats=player.bats,
            throws=player.throws,
            experience=self._count_experience(player.id),
        )

    def search(self, name: str, season: int) -> list[PlayerSummary]:
        players = resolve_players(self._player_repo, name)
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
        player_ids: set[int] = set()

        # Try roster stints first (existing path)
        if team is not None:
            if self._team_resolver is not None:
                # Resolver path: handles abbreviations, full names, nicknames, fuzzy
                abbreviations = self._team_resolver.resolve(team)
                if not abbreviations:
                    msg = f"No team found matching '{team}'"
                    raise ValueError(msg)
                for abbrev in abbreviations:
                    team_obj = self._team_repo.get_by_abbreviation(abbrev)
                    if team_obj is not None and team_obj.id is not None:
                        stints = self._roster_stint_repo.get_by_team_season(team_obj.id, season)
                        player_ids.update(s.player_id for s in stints)
            else:
                # Legacy path: exact abbreviation + Lahman alias
                team_obj = self._team_repo.get_by_abbreviation(team)
                if team_obj is None:
                    lahman = to_lahman(team)
                    if lahman != team:
                        team_obj = self._team_repo.get_by_abbreviation(lahman)

                if team_obj is not None and team_obj.id is not None:
                    stints = self._roster_stint_repo.get_by_team_season(team_obj.id, season)
                    player_ids.update(s.player_id for s in stints)

            # Fallback: use provider if stints are empty
            if not player_ids and self._player_team_provider is not None:
                provider_teams = self._player_team_provider.get_player_teams(season)
                player_ids.update(pid for pid, abbrev in provider_teams.items() if abbrev == team)
        else:
            stints = self._roster_stint_repo.get_by_season(season)
            player_ids.update(s.player_id for s in stints)

            # Augment with provider if available
            if self._player_team_provider is not None:
                provider_teams = self._player_team_provider.get_player_teams(season)
                player_ids.update(provider_teams.keys())

        if not player_ids:
            return []

        players = self._player_repo.get_by_ids(list(player_ids))

        results: list[PlayerSummary] = []
        for player in players:
            assert player.id is not None  # noqa: S101 - type narrowing

            if min_age is not None or max_age is not None:
                age = compute_age(player.birth_date, season)
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
