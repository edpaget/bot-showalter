"""Tests for MlbApiPlayerTeamProvider."""

from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain import Player, RosterStint, Team
from fantasy_baseball_manager.repos import (
    SqlitePlayerRepo,
    SqliteRosterStintRepo,
    SqliteTeamRepo,
)
from fantasy_baseball_manager.services.player_team_provider import (
    MlbApiPlayerTeamProvider,
)

if TYPE_CHECKING:
    import sqlite3


def _setup(conn: sqlite3.Connection) -> tuple[SqlitePlayerRepo, SqliteTeamRepo, SqliteRosterStintRepo]:
    player_repo = SqlitePlayerRepo(conn)
    team_repo = SqliteTeamRepo(conn)
    roster_repo = SqliteRosterStintRepo(conn)
    return player_repo, team_repo, roster_repo


def _seed(
    player_repo: SqlitePlayerRepo,
    team_repo: SqliteTeamRepo,
    roster_repo: SqliteRosterStintRepo,
    conn: sqlite3.Connection,
) -> tuple[int, int, int, int]:
    """Seed two teams and two players with 2024 roster stints. Returns (nyy_id, bos_id, judge_id, devers_id)."""
    # Teams use Lahman abbreviations in the DB
    nyy_id = team_repo.upsert(Team(abbreviation="NYA", name="Yankees", league="AL", division="East"))
    bos_id = team_repo.upsert(Team(abbreviation="BOS", name="Red Sox", league="AL", division="East"))
    conn.commit()

    judge_id = player_repo.upsert(Player(name_first="Aaron", name_last="Judge", mlbam_id=592450))
    devers_id = player_repo.upsert(Player(name_first="Rafael", name_last="Devers", mlbam_id=646240))
    conn.commit()

    roster_repo.upsert(RosterStint(player_id=judge_id, team_id=nyy_id, season=2024, start_date="2024-03-28"))
    roster_repo.upsert(RosterStint(player_id=devers_id, team_id=bos_id, season=2024, start_date="2024-03-28"))
    conn.commit()

    return nyy_id, bos_id, judge_id, devers_id


class TestGetPlayerTeams:
    def test_lahman_stints_converted_to_modern(self) -> None:
        conn = create_connection(":memory:")
        player_repo, team_repo, roster_repo = _setup(conn)
        _nyy_id, _bos_id, judge_id, devers_id = _seed(player_repo, team_repo, roster_repo, conn)

        provider = MlbApiPlayerTeamProvider(player_repo, team_repo, roster_repo)
        result = provider.get_player_teams(2024)

        # NYA should be converted to NYY; BOS stays BOS
        assert result[judge_id] == "NYY"
        assert result[devers_id] == "BOS"
        conn.close()

    def test_no_stints_returns_empty(self) -> None:
        conn = create_connection(":memory:")
        player_repo, team_repo, roster_repo = _setup(conn)
        _seed(player_repo, team_repo, roster_repo, conn)

        provider = MlbApiPlayerTeamProvider(player_repo, team_repo, roster_repo)
        result = provider.get_player_teams(2025)

        assert result == {}
        conn.close()

    def test_mlb_api_overlays_updates(self) -> None:
        conn = create_connection(":memory:")
        player_repo, team_repo, roster_repo = _setup(conn)
        _nyy_id, _bos_id, judge_id, _devers_id = _seed(player_repo, team_repo, roster_repo, conn)

        # MLB API says Judge moved to LAD
        def fake_fetcher(_season: int) -> dict[int, str]:
            return {592450: "LAD"}

        provider = MlbApiPlayerTeamProvider(player_repo, team_repo, roster_repo, fetcher=fake_fetcher)
        result = provider.get_player_teams(2024)

        assert result[judge_id] == "LAD"
        conn.close()

    def test_mlb_api_failure_raises(self) -> None:
        conn = create_connection(":memory:")
        player_repo, team_repo, roster_repo = _setup(conn)
        _seed(player_repo, team_repo, roster_repo, conn)

        def failing_fetcher(_season: int) -> dict[int, str]:
            msg = "network error"
            raise RuntimeError(msg)

        provider = MlbApiPlayerTeamProvider(player_repo, team_repo, roster_repo, fetcher=failing_fetcher)

        with pytest.raises(RuntimeError, match="network error"):
            provider.get_player_teams(2024)
        conn.close()

    def test_caches_per_season(self) -> None:
        conn = create_connection(":memory:")
        player_repo, team_repo, roster_repo = _setup(conn)
        _seed(player_repo, team_repo, roster_repo, conn)

        call_count = 0

        def counting_fetcher(_season: int) -> dict[int, str]:
            nonlocal call_count
            call_count += 1
            return {}

        provider = MlbApiPlayerTeamProvider(player_repo, team_repo, roster_repo, fetcher=counting_fetcher)
        provider.get_player_teams(2024)
        provider.get_player_teams(2024)

        assert call_count == 1
        conn.close()

    def test_no_fetcher_uses_stints_only(self) -> None:
        conn = create_connection(":memory:")
        player_repo, team_repo, roster_repo = _setup(conn)
        _nyy_id, _bos_id, judge_id, devers_id = _seed(player_repo, team_repo, roster_repo, conn)

        provider = MlbApiPlayerTeamProvider(player_repo, team_repo, roster_repo, fetcher=None)
        result = provider.get_player_teams(2024)

        assert result[judge_id] == "NYY"
        assert result[devers_id] == "BOS"
        conn.close()
