"""Ingest roster stints from the MLB Stats API."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import RosterStint, Team

if TYPE_CHECKING:
    from fantasy_baseball_manager.repos import SqlitePlayerRepo, SqliteRosterStintRepo, SqliteTeamRepo


@dataclass(frozen=True)
class RosterApiResult:
    loaded: int
    skipped: int


def ingest_roster_api(
    fetcher: object,
    player_repo: SqlitePlayerRepo,
    team_repo: SqliteTeamRepo,
    roster_stint_repo: SqliteRosterStintRepo,
    season: int,
    as_of: str,
) -> RosterApiResult:
    """Fetch team assignments from the MLB API and upsert into roster_stint.

    Parameters
    ----------
    fetcher:
        Callable ``(season: int) -> dict[int, str]`` mapping mlbam_id to team abbreviation.
    player_repo:
        Resolves mlbam_id → player_id.
    team_repo:
        Resolves abbreviation → team_id (auto-upserts missing teams).
    roster_stint_repo:
        Target repo for upserting roster stints.
    season:
        Season year.
    as_of:
        ISO date string used as ``start_date`` for the roster stint.
    """
    # Build lookup from mlbam_id → player_id
    mlbam_to_player: dict[int, int] = {}
    for p in player_repo.all():
        if p.mlbam_id is not None and p.id is not None:
            mlbam_to_player[p.mlbam_id] = p.id

    # Fetch team assignments from MLB API
    mlbam_teams: dict[int, str] = fetcher(season)  # type: ignore[operator]

    # Build abbreviation → team_id cache
    abbrev_to_team_id: dict[str, int] = {}
    for t in team_repo.all():
        if t.id is not None:
            abbrev_to_team_id[t.abbreviation] = t.id

    loaded = 0
    skipped = 0
    for mlbam_id, abbrev in mlbam_teams.items():
        player_id = mlbam_to_player.get(mlbam_id)
        if player_id is None:
            skipped += 1
            continue

        # Resolve team, auto-upsert if missing
        team_id = abbrev_to_team_id.get(abbrev)
        if team_id is None:
            team_id = team_repo.upsert(Team(abbreviation=abbrev, name=abbrev, league="", division=""))
            abbrev_to_team_id[abbrev] = team_id

        stint = RosterStint(
            player_id=player_id,
            team_id=team_id,
            season=season,
            start_date=as_of,
            end_date=None,
        )
        roster_stint_repo.upsert(stint)
        loaded += 1

    return RosterApiResult(loaded=loaded, skipped=skipped)
