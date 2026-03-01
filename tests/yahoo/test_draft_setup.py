from typing import TYPE_CHECKING, Any

import pytest

from fantasy_baseball_manager.cli.commands.yahoo import _build_yahoo_draft_setup
from fantasy_baseball_manager.cli.factory import YahooContext
from fantasy_baseball_manager.domain import (
    Valuation,
    YahooLeague,
    YahooTeam,
)
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.repos import (
    SqliteADPRepo,
    SqlitePlayerRepo,
    SqliteProjectionRepo,
    SqliteValuationRepo,
    SqliteYahooDraftRepo,
    SqliteYahooLeagueRepo,
    SqliteYahooPlayerMapRepo,
    SqliteYahooRosterRepo,
    SqliteYahooTeamRepo,
)

if TYPE_CHECKING:
    import sqlite3


class FakeClient:
    def __init__(self) -> None:
        self._game_key = "449"

    def get_game_key(self, season: int) -> str:
        return self._game_key

    def get_draft_results(self, league_key: str) -> dict[str, Any]:
        return {"fantasy_content": {"league": [{}, {"draft_results": {"count": 0}}]}}

    def get_players(self, league_key: str, player_keys: list[str]) -> dict[str, Any]:
        return {"fantasy_content": {"league": [{}]}}


_LEAGUE = LeagueSettings(
    name="test",
    format=LeagueFormat.H2H_CATEGORIES,
    teams=2,
    budget=260,
    roster_batters=3,
    roster_pitchers=2,
    batting_categories=(
        CategoryConfig(key="HR", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
    ),
    pitching_categories=(
        CategoryConfig(key="K", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
    ),
    positions={"OF": 2, "SP": 1},
    roster_util=1,
)


def _make_ctx(conn: sqlite3.Connection) -> YahooContext:
    return YahooContext(
        conn=conn,
        yahoo_league_repo=SqliteYahooLeagueRepo(conn),
        yahoo_team_repo=SqliteYahooTeamRepo(conn),
        yahoo_player_map_repo=SqliteYahooPlayerMapRepo(conn),
        yahoo_roster_repo=SqliteYahooRosterRepo(conn),
        yahoo_draft_repo=SqliteYahooDraftRepo(conn),
        player_repo=SqlitePlayerRepo(conn),
        projection_repo=SqliteProjectionRepo(conn),
        valuation_repo=SqliteValuationRepo(conn),
        adp_repo=SqliteADPRepo(conn),
        client=FakeClient(),  # type: ignore[arg-type]
    )


def _seed_data(conn: sqlite3.Connection) -> None:
    ctx = _make_ctx(conn)

    ctx.yahoo_league_repo.upsert(
        YahooLeague(
            league_key="449.l.100",
            name="Test League",
            season=2026,
            num_teams=2,
            draft_type="live_standard_draft",
            is_keeper=False,
            game_key="449",
        )
    )
    ctx.yahoo_team_repo.upsert(
        YahooTeam(
            team_key="449.l.100.t.1",
            league_key="449.l.100",
            team_id=1,
            name="Team A",
            manager_name="Alice",
            is_owned_by_user=True,
        )
    )
    ctx.yahoo_team_repo.upsert(
        YahooTeam(
            team_key="449.l.100.t.2",
            league_key="449.l.100",
            team_id=2,
            name="Team B",
            manager_name="Bob",
            is_owned_by_user=False,
        )
    )

    player_id = ctx.player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
    ctx.valuation_repo.upsert(
        Valuation(
            player_id=player_id,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="test",
            projection_version="1.0",
            player_type="batter",
            position="OF",
            value=30.0,
            rank=1,
            category_scores={"HR": 2.5},
        )
    )
    conn.commit()


class TestBuildYahooDraftSetup:
    def test_creates_engine_with_correct_config(self, conn: sqlite3.Connection) -> None:
        _seed_data(conn)
        ctx = _make_ctx(conn)

        setup = _build_yahoo_draft_setup(ctx, "449.l.100", 2026, _LEAGUE, "zar", "1.0", "fantasypros")

        assert setup.engine.state.config.teams == 2
        assert setup.engine.state.config.format.value == "snake"
        assert setup.engine.state.config.user_team == 1

    def test_team_map_is_correct(self, conn: sqlite3.Connection) -> None:
        _seed_data(conn)
        ctx = _make_ctx(conn)

        setup = _build_yahoo_draft_setup(ctx, "449.l.100", 2026, _LEAGUE, "zar", "1.0", "fantasypros")

        assert setup.team_map == {"449.l.100.t.1": 1, "449.l.100.t.2": 2}

    def test_board_has_players(self, conn: sqlite3.Connection) -> None:
        _seed_data(conn)
        ctx = _make_ctx(conn)

        setup = _build_yahoo_draft_setup(ctx, "449.l.100", 2026, _LEAGUE, "zar", "1.0", "fantasypros")

        assert len(setup.board.rows) == 1
        assert setup.board.rows[0].player_name == "Mike Trout"

    def test_no_existing_picks_replayed(self, conn: sqlite3.Connection) -> None:
        _seed_data(conn)
        ctx = _make_ctx(conn)

        setup = _build_yahoo_draft_setup(ctx, "449.l.100", 2026, _LEAGUE, "zar", "1.0", "fantasypros")

        assert setup.replayed_count == 0

    def test_auction_format_detected(self, conn: sqlite3.Connection) -> None:
        _seed_data(conn)
        ctx = _make_ctx(conn)

        # Update league to auction type
        ctx.yahoo_league_repo.upsert(
            YahooLeague(
                league_key="449.l.100",
                name="Test League",
                season=2026,
                num_teams=2,
                draft_type="live_auction",
                is_keeper=False,
                game_key="449",
            )
        )
        conn.commit()

        setup = _build_yahoo_draft_setup(ctx, "449.l.100", 2026, _LEAGUE, "zar", "1.0", "fantasypros")

        assert setup.engine.state.config.format.value == "auction"
        assert setup.engine.state.config.budget == 260

    def test_no_teams_raises(self, conn: sqlite3.Connection) -> None:
        ctx = _make_ctx(conn)

        with pytest.raises(ValueError, match="No teams found"):
            _build_yahoo_draft_setup(ctx, "449.l.100", 2026, _LEAGUE, "zar", "1.0", "fantasypros")

    def test_no_user_team_raises(self, conn: sqlite3.Connection) -> None:
        _seed_data(conn)
        ctx = _make_ctx(conn)

        # Reset user team flag
        ctx.yahoo_team_repo.upsert(
            YahooTeam(
                team_key="449.l.100.t.1",
                league_key="449.l.100",
                team_id=1,
                name="Team A",
                manager_name="Alice",
                is_owned_by_user=False,
            )
        )
        conn.commit()

        with pytest.raises(ValueError, match="No user team found"):
            _build_yahoo_draft_setup(ctx, "449.l.100", 2026, _LEAGUE, "zar", "1.0", "fantasypros")
