import datetime
from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.domain import (
    KeeperCost,
    Roster,
    RosterEntry,
    YahooDraftPick,
    YahooLeague,
    YahooTeam,
)
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.repos import (
    SqliteKeeperCostRepo,
    SqlitePlayerRepo,
    SqliteYahooLeagueRepo,
    SqliteYahooTeamRepo,
)
from fantasy_baseball_manager.services.yahoo_keeper import derive_and_store_keeper_costs

if TYPE_CHECKING:
    import sqlite3


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeDraftSource:
    """Returns canned draft picks."""

    def __init__(self, picks: list[YahooDraftPick]) -> None:
        self._picks = picks

    def fetch_draft_results(self, league_key: str, season: int) -> list[YahooDraftPick]:
        return self._picks


class FakeRosterSource:
    """Returns a canned roster."""

    def __init__(self, roster: Roster) -> None:
        self._roster = roster

    def fetch_team_roster(
        self,
        *,
        team_key: str,
        league_key: str,
        season: int,
        week: int,
        as_of: datetime.date,
    ) -> Roster:
        return self._roster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LEAGUE_KEY = "449.l.100"
_PRIOR_LEAGUE_KEY = "422.l.100"


def _seed_league(conn: sqlite3.Connection) -> None:
    league_repo = SqliteYahooLeagueRepo(conn)
    league_repo.upsert(
        YahooLeague(
            league_key=_LEAGUE_KEY,
            name="Test League",
            season=2026,
            num_teams=10,
            draft_type="live_standard_draft",
            is_keeper=True,
            game_key="449",
        )
    )
    conn.commit()


def _seed_team(conn: sqlite3.Connection, *, is_user: bool = True) -> None:
    _seed_league(conn)
    team_repo = SqliteYahooTeamRepo(conn)
    team_repo.upsert(
        YahooTeam(
            team_key="449.l.100.t.1",
            league_key=_LEAGUE_KEY,
            team_id=1,
            name="My Team",
            manager_name="Me",
            is_owned_by_user=is_user,
        )
    )
    conn.commit()


def _seed_player(conn: sqlite3.Connection) -> int:
    player_repo = SqlitePlayerRepo(conn)
    player_id = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
    conn.commit()
    return player_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeriveAndStoreKeeperCosts:
    def test_derives_and_upserts_costs(self, conn: sqlite3.Connection) -> None:
        _seed_team(conn)
        player_id = _seed_player(conn)

        pick = YahooDraftPick(
            league_key=_PRIOR_LEAGUE_KEY,
            season=2025,
            round=1,
            pick=1,
            team_key="449.l.100.t.1",
            yahoo_player_key="449.p.1000",
            player_id=player_id,
            player_name="Mike Trout",
            position="OF",
            cost=30,
        )

        roster = Roster(
            team_key="449.l.100.t.1",
            league_key=_LEAGUE_KEY,
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 1),
            entries=(
                RosterEntry(
                    player_id=player_id,
                    yahoo_player_key="449.p.1000",
                    player_name="Mike Trout",
                    position="OF",
                    roster_status="active",
                    acquisition_type="draft",
                ),
            ),
        )

        draft_source = FakeDraftSource([pick])
        roster_source = FakeRosterSource(roster)
        team_repo = SqliteYahooTeamRepo(conn)
        keeper_repo = SqliteKeeperCostRepo(conn)

        derive_and_store_keeper_costs(
            draft_source=draft_source,
            roster_source=roster_source,
            team_repo=team_repo,
            keeper_repo=keeper_repo,
            league_key=_LEAGUE_KEY,
            prior_league_key=_PRIOR_LEAGUE_KEY,
            prior_season=2025,
            season=2026,
            league_name="test_league",
            cost_floor=1.0,
        )

        costs = keeper_repo.find_by_season_league(2026, "test_league")
        assert len(costs) >= 1
        player_costs = [c for c in costs if c.player_id == player_id]
        assert len(player_costs) == 1

    def test_respects_manual_overrides(self, conn: sqlite3.Connection) -> None:
        _seed_team(conn)
        player_id = _seed_player(conn)

        # Pre-seed a manual override
        keeper_repo = SqliteKeeperCostRepo(conn)
        keeper_repo.upsert_batch(
            [
                KeeperCost(
                    player_id=player_id,
                    season=2026,
                    league="test_league",
                    cost=5.0,
                    source="manual",
                    years_remaining=1,
                )
            ]
        )
        conn.commit()

        pick = YahooDraftPick(
            league_key=_PRIOR_LEAGUE_KEY,
            season=2025,
            round=1,
            pick=1,
            team_key="449.l.100.t.1",
            yahoo_player_key="449.p.1000",
            player_id=player_id,
            player_name="Mike Trout",
            position="OF",
            cost=30,
        )

        roster = Roster(
            team_key="449.l.100.t.1",
            league_key=_LEAGUE_KEY,
            season=2026,
            week=1,
            as_of=datetime.date(2026, 3, 1),
            entries=(
                RosterEntry(
                    player_id=player_id,
                    yahoo_player_key="449.p.1000",
                    player_name="Mike Trout",
                    position="OF",
                    roster_status="active",
                    acquisition_type="draft",
                ),
            ),
        )

        draft_source = FakeDraftSource([pick])
        roster_source = FakeRosterSource(roster)
        team_repo = SqliteYahooTeamRepo(conn)

        derive_and_store_keeper_costs(
            draft_source=draft_source,
            roster_source=roster_source,
            team_repo=team_repo,
            keeper_repo=keeper_repo,
            league_key=_LEAGUE_KEY,
            prior_league_key=_PRIOR_LEAGUE_KEY,
            prior_season=2025,
            season=2026,
            league_name="test_league",
            cost_floor=1.0,
        )

        # Manual override should be preserved (source = "manual", not "yahoo_*")
        costs = keeper_repo.find_by_season_league(2026, "test_league")
        player_costs = [c for c in costs if c.player_id == player_id]
        assert len(player_costs) == 1
        assert player_costs[0].source == "manual"

    def test_raises_value_error_when_no_user_team(self, conn: sqlite3.Connection) -> None:
        # Don't seed any team
        draft_source = FakeDraftSource([])
        roster_source = FakeRosterSource(
            Roster(
                team_key="x",
                league_key=_LEAGUE_KEY,
                season=2026,
                week=1,
                as_of=datetime.date(2026, 3, 1),
                entries=(),
            )
        )
        team_repo = SqliteYahooTeamRepo(conn)
        keeper_repo = SqliteKeeperCostRepo(conn)

        with pytest.raises(ValueError, match="No user team found"):
            derive_and_store_keeper_costs(
                draft_source=draft_source,
                roster_source=roster_source,
                team_repo=team_repo,
                keeper_repo=keeper_repo,
                league_key=_LEAGUE_KEY,
                prior_league_key=_PRIOR_LEAGUE_KEY,
                prior_season=2025,
                season=2026,
                league_name="test_league",
                cost_floor=1.0,
            )
