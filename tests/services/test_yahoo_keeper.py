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
from fantasy_baseball_manager.services.yahoo_keeper import (
    derive_and_store_keeper_costs,
    derive_best_n_keeper_costs,
    ensure_prior_season_teams,
)

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


class FakeLeagueSource:
    """Returns canned league + teams, and tracks calls."""

    def __init__(self, league: YahooLeague, teams: list[YahooTeam]) -> None:
        self._league = league
        self._teams = teams
        self.call_count = 0

    def fetch(self, *, league_key: str, game_key: str) -> tuple[YahooLeague, list[YahooTeam]]:
        self.call_count += 1
        return self._league, self._teams


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
        week: int | None = None,
        as_of: datetime.date,
    ) -> Roster:
        return self._roster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LEAGUE_KEY = "449.l.100"
_PRIOR_LEAGUE_KEY = "422.l.200"


def _seed_prior_league(conn: sqlite3.Connection) -> None:
    league_repo = SqliteYahooLeagueRepo(conn)
    league_repo.upsert(
        YahooLeague(
            league_key=_PRIOR_LEAGUE_KEY,
            name="Test League",
            season=2025,
            num_teams=10,
            draft_type="live_standard_draft",
            is_keeper=True,
            game_key="422",
        )
    )
    conn.commit()


def _seed_team(conn: sqlite3.Connection, *, is_user: bool = True) -> None:
    _seed_prior_league(conn)
    team_repo = SqliteYahooTeamRepo(conn)
    team_repo.upsert(
        YahooTeam(
            team_key="422.l.200.t.1",
            league_key=_PRIOR_LEAGUE_KEY,
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
            team_key="422.l.200.t.1",
            yahoo_player_key="449.p.1000",
            player_id=player_id,
            player_name="Mike Trout",
            position="OF",
            cost=30,
        )

        roster = Roster(
            team_key="422.l.200.t.1",
            league_key=_PRIOR_LEAGUE_KEY,
            season=2025,
            week=1,
            as_of=datetime.date(2025, 10, 1),
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
            team_key="422.l.200.t.1",
            yahoo_player_key="449.p.1000",
            player_id=player_id,
            player_name="Mike Trout",
            position="OF",
            cost=30,
        )

        roster = Roster(
            team_key="422.l.200.t.1",
            league_key=_PRIOR_LEAGUE_KEY,
            season=2025,
            week=1,
            as_of=datetime.date(2025, 10, 1),
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
                league_key=_PRIOR_LEAGUE_KEY,
                season=2025,
                week=1,
                as_of=datetime.date(2025, 10, 1),
                entries=(),
            )
        )
        team_repo = SqliteYahooTeamRepo(conn)
        keeper_repo = SqliteKeeperCostRepo(conn)

        with pytest.raises(ValueError, match="No user team found for"):
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


class TestDeriveBestNKeeperCosts:
    def test_assigns_zero_cost_to_all_roster_players(self, conn: sqlite3.Connection) -> None:
        _seed_team(conn)
        player_repo = SqlitePlayerRepo(conn)
        pid1 = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        pid2 = player_repo.upsert(Player(name_first="Aaron", name_last="Judge", mlbam_id=592450))
        conn.commit()

        roster = Roster(
            team_key="422.l.200.t.1",
            league_key=_PRIOR_LEAGUE_KEY,
            season=2025,
            week=1,
            as_of=datetime.date(2025, 10, 1),
            entries=(
                RosterEntry(
                    player_id=pid1,
                    yahoo_player_key="449.p.1000",
                    player_name="Mike Trout",
                    position="CF",
                    roster_status="active",
                    acquisition_type="draft",
                ),
                RosterEntry(
                    player_id=pid2,
                    yahoo_player_key="449.p.2000",
                    player_name="Aaron Judge",
                    position="RF",
                    roster_status="active",
                    acquisition_type="add",
                ),
            ),
        )

        roster_source = FakeRosterSource(roster)
        team_repo = SqliteYahooTeamRepo(conn)
        keeper_repo = SqliteKeeperCostRepo(conn)

        derive_best_n_keeper_costs(
            roster_source=roster_source,
            team_repo=team_repo,
            keeper_repo=keeper_repo,
            prior_league_key=_PRIOR_LEAGUE_KEY,
            prior_season=2025,
            season=2026,
            league_name="test_league",
        )

        costs = keeper_repo.find_by_season_league(2026, "test_league")
        assert len(costs) == 2
        for c in costs:
            assert c.cost == 0.0
            assert c.source == "best_n"

    def test_skips_unresolved_players(self, conn: sqlite3.Connection) -> None:
        _seed_team(conn)
        player_repo = SqlitePlayerRepo(conn)
        pid1 = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        conn.commit()

        roster = Roster(
            team_key="422.l.200.t.1",
            league_key=_PRIOR_LEAGUE_KEY,
            season=2025,
            week=1,
            as_of=datetime.date(2025, 10, 1),
            entries=(
                RosterEntry(
                    player_id=pid1,
                    yahoo_player_key="449.p.1000",
                    player_name="Mike Trout",
                    position="CF",
                    roster_status="active",
                    acquisition_type="draft",
                ),
                RosterEntry(
                    player_id=None,
                    yahoo_player_key="449.p.9999",
                    player_name="Unknown",
                    position="DH",
                    roster_status="active",
                    acquisition_type="add",
                ),
            ),
        )

        roster_source = FakeRosterSource(roster)
        team_repo = SqliteYahooTeamRepo(conn)
        keeper_repo = SqliteKeeperCostRepo(conn)

        derive_best_n_keeper_costs(
            roster_source=roster_source,
            team_repo=team_repo,
            keeper_repo=keeper_repo,
            prior_league_key=_PRIOR_LEAGUE_KEY,
            prior_season=2025,
            season=2026,
            league_name="test_league",
        )

        costs = keeper_repo.find_by_season_league(2026, "test_league")
        assert len(costs) == 1
        assert costs[0].player_id == pid1

    def test_respects_manual_overrides(self, conn: sqlite3.Connection) -> None:
        _seed_team(conn)
        player_repo = SqlitePlayerRepo(conn)
        pid1 = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        conn.commit()

        keeper_repo = SqliteKeeperCostRepo(conn)
        keeper_repo.upsert_batch(
            [
                KeeperCost(
                    player_id=pid1,
                    season=2026,
                    league="test_league",
                    cost=10.0,
                    source="manual",
                    years_remaining=1,
                )
            ]
        )
        conn.commit()

        roster = Roster(
            team_key="422.l.200.t.1",
            league_key=_PRIOR_LEAGUE_KEY,
            season=2025,
            week=1,
            as_of=datetime.date(2025, 10, 1),
            entries=(
                RosterEntry(
                    player_id=pid1,
                    yahoo_player_key="449.p.1000",
                    player_name="Mike Trout",
                    position="CF",
                    roster_status="active",
                    acquisition_type="draft",
                ),
            ),
        )

        roster_source = FakeRosterSource(roster)
        team_repo = SqliteYahooTeamRepo(conn)

        derive_best_n_keeper_costs(
            roster_source=roster_source,
            team_repo=team_repo,
            keeper_repo=keeper_repo,
            prior_league_key=_PRIOR_LEAGUE_KEY,
            prior_season=2025,
            season=2026,
            league_name="test_league",
        )

        costs = keeper_repo.find_by_season_league(2026, "test_league")
        player_costs = [c for c in costs if c.player_id == pid1]
        assert len(player_costs) == 1
        assert player_costs[0].source == "manual"
        assert player_costs[0].cost == 10.0


# ---------------------------------------------------------------------------
# ensure_prior_season_teams tests
# ---------------------------------------------------------------------------

_PRIOR_GAME_KEY = "422"

_PRIOR_LEAGUE = YahooLeague(
    league_key=_PRIOR_LEAGUE_KEY,
    name="Test League",
    season=2025,
    num_teams=10,
    draft_type="live_standard_draft",
    is_keeper=True,
    game_key=_PRIOR_GAME_KEY,
)

_PRIOR_USER_TEAM = YahooTeam(
    team_key="422.l.200.t.1",
    league_key=_PRIOR_LEAGUE_KEY,
    team_id=1,
    name="My Team",
    manager_name="Me",
    is_owned_by_user=True,
)

_PRIOR_OTHER_TEAM = YahooTeam(
    team_key="422.l.200.t.2",
    league_key=_PRIOR_LEAGUE_KEY,
    team_id=2,
    name="Other Team",
    manager_name="Other",
    is_owned_by_user=False,
)


class TestEnsurePriorSeasonTeams:
    def test_no_sync_when_teams_already_exist(self, conn: sqlite3.Connection) -> None:
        """When user team exists in DB, no sync call is made."""
        team_repo = SqliteYahooTeamRepo(conn)
        league_repo = SqliteYahooLeagueRepo(conn)

        # Pre-seed prior league and user team
        league_repo.upsert(_PRIOR_LEAGUE)
        team_repo.upsert(_PRIOR_USER_TEAM)
        conn.commit()

        league_source = FakeLeagueSource(_PRIOR_LEAGUE, [_PRIOR_USER_TEAM])

        ensure_prior_season_teams(
            team_repo=team_repo,
            league_source=league_source,
            league_repo=league_repo,
            prior_league_key=_PRIOR_LEAGUE_KEY,
            prior_game_key=_PRIOR_GAME_KEY,
        )

        assert league_source.call_count == 0

    def test_syncs_when_teams_missing(self, conn: sqlite3.Connection) -> None:
        """When user team is absent, sync is called and teams are populated."""
        team_repo = SqliteYahooTeamRepo(conn)
        league_repo = SqliteYahooLeagueRepo(conn)

        league_source = FakeLeagueSource(_PRIOR_LEAGUE, [_PRIOR_USER_TEAM, _PRIOR_OTHER_TEAM])

        ensure_prior_season_teams(
            team_repo=team_repo,
            league_source=league_source,
            league_repo=league_repo,
            prior_league_key=_PRIOR_LEAGUE_KEY,
            prior_game_key=_PRIOR_GAME_KEY,
        )

        assert league_source.call_count == 1
        assert team_repo.get_user_team(_PRIOR_LEAGUE_KEY) is not None

    def test_raises_when_sync_finds_no_user_team(self, conn: sqlite3.Connection) -> None:
        """When sync completes but no user team exists, ValueError is raised."""
        team_repo = SqliteYahooTeamRepo(conn)
        league_repo = SqliteYahooLeagueRepo(conn)

        # Sync returns teams but none is_owned_by_user
        league_source = FakeLeagueSource(_PRIOR_LEAGUE, [_PRIOR_OTHER_TEAM])

        with pytest.raises(ValueError, match="No user team found"):
            ensure_prior_season_teams(
                team_repo=team_repo,
                league_source=league_source,
                league_repo=league_repo,
                prior_league_key=_PRIOR_LEAGUE_KEY,
                prior_game_key=_PRIOR_GAME_KEY,
            )
