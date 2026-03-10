from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.cli.factory import SgpContext
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.repos import SqliteYahooLeagueRepo, SqliteYahooTeamStatsRepo

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator

runner = CliRunner()

_STATS_INSERT = (
    "INSERT INTO yahoo_team_season_stats"
    " (team_key, league_key, season, team_name, final_rank, stat_values_json)"
    " VALUES (?, ?, ?, ?, ?, ?)"
)


def _seed_db(conn: sqlite3.Connection) -> None:
    """Insert league and standings data for testing."""
    conn.execute(
        "INSERT INTO yahoo_league"
        " (league_key, name, season, num_teams, draft_type, is_keeper, game_key, renew)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("412.l.91300", "Test League", 2024, 3, "live", 0, "412", "403_91300"),
    )
    conn.execute(
        "INSERT INTO yahoo_league"
        " (league_key, name, season, num_teams, draft_type, is_keeper, game_key, renew)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("403.l.91300", "Test League", 2023, 3, "live", 0, "403", None),
    )

    for i, (hr, era) in enumerate([(10.0, 3.0), (20.0, 3.5), (30.0, 4.0)], start=1):
        conn.execute(
            _STATS_INSERT,
            (f"412.l.91300.t.{i}", "412.l.91300", 2024, f"Team {i}", i, json.dumps({"hr": hr, "era": era})),
        )

    for i, (hr, era) in enumerate([(15.0, 3.2), (25.0, 3.7), (35.0, 4.2)], start=1):
        conn.execute(
            _STATS_INSERT,
            (f"403.l.91300.t.{i}", "403.l.91300", 2023, f"Team {i}", i, json.dumps({"hr": hr, "era": era})),
        )

    conn.commit()


def _build_mock_context(conn: sqlite3.Connection) -> SgpContext:
    return SgpContext(
        conn=conn,
        yahoo_league_repo=SqliteYahooLeagueRepo(SingleConnectionProvider(conn)),
        yahoo_team_stats_repo=SqliteYahooTeamStatsRepo(SingleConnectionProvider(conn)),
    )


@contextmanager
def _mock_sgp_context(ctx: SgpContext) -> Iterator[None]:
    @contextmanager
    def factory(data_dir: str) -> Iterator[SgpContext]:
        yield ctx

    with patch("fantasy_baseball_manager.cli.commands.valuations.build_sgp_context", factory):
        yield


def test_sgp_denominators_command(conn: sqlite3.Connection) -> None:
    _seed_db(conn)
    ctx = _build_mock_context(conn)

    with _mock_sgp_context(ctx):
        result = runner.invoke(
            app,
            ["valuations", "sgp-denominators", "--league", "h2h", "--yahoo-league", "412.l.91300"],
        )

    assert result.exit_code == 0, result.output
    assert "hr" in result.output
    assert "era" in result.output


def test_sgp_denominators_seasons_filter(conn: sqlite3.Connection) -> None:
    _seed_db(conn)
    ctx = _build_mock_context(conn)

    with _mock_sgp_context(ctx):
        result = runner.invoke(
            app,
            ["valuations", "sgp-denominators", "--league", "h2h", "--yahoo-league", "412.l.91300", "--seasons", "1"],
        )

    assert result.exit_code == 0, result.output
    assert "2024" in result.output
    assert "2023" not in result.output
