from typing import TYPE_CHECKING, Any

import pytest

from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain import (
    Valuation,
    YahooLeague,
    YahooTeam,
)
from fantasy_baseball_manager.domain.identity import PlayerType
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
    SqliteValuationRepo,
    SqliteYahooDraftRepo,
    SqliteYahooLeagueRepo,
    SqliteYahooPlayerMapRepo,
    SqliteYahooTeamRepo,
)
from fantasy_baseball_manager.services.yahoo_draft_setup import build_yahoo_draft_setup
from fantasy_baseball_manager.yahoo.draft_source import YahooDraftSource
from fantasy_baseball_manager.yahoo.player_map import YahooPlayerMapper

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


def _make_repos(conn: sqlite3.Connection) -> dict[str, Any]:
    return {
        "league_repo": SqliteYahooLeagueRepo(SingleConnectionProvider(conn)),
        "team_repo": SqliteYahooTeamRepo(SingleConnectionProvider(conn)),
        "player_repo": SqlitePlayerRepo(SingleConnectionProvider(conn)),
        "valuation_repo": SqliteValuationRepo(SingleConnectionProvider(conn)),
        "adp_repo": SqliteADPRepo(SingleConnectionProvider(conn)),
        "draft_repo": SqliteYahooDraftRepo(SingleConnectionProvider(conn)),
    }


def _make_draft_source(conn: sqlite3.Connection) -> YahooDraftSource:
    mapper = YahooPlayerMapper(
        SqliteYahooPlayerMapRepo(SingleConnectionProvider(conn)), SqlitePlayerRepo(SingleConnectionProvider(conn))
    )
    return YahooDraftSource(FakeClient(), mapper)  # type: ignore[arg-type]


def _seed_data(conn: sqlite3.Connection) -> None:
    repos = _make_repos(conn)

    repos["league_repo"].upsert(
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
    repos["team_repo"].upsert(
        YahooTeam(
            team_key="449.l.100.t.1",
            league_key="449.l.100",
            team_id=1,
            name="Team A",
            manager_name="Alice",
            is_owned_by_user=True,
        )
    )
    repos["team_repo"].upsert(
        YahooTeam(
            team_key="449.l.100.t.2",
            league_key="449.l.100",
            team_id=2,
            name="Team B",
            manager_name="Bob",
            is_owned_by_user=False,
        )
    )

    player_id = repos["player_repo"].upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
    repos["valuation_repo"].upsert(
        Valuation(
            player_id=player_id,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="test",
            projection_version="1.0",
            player_type=PlayerType.BATTER,
            position="OF",
            value=30.0,
            rank=1,
            category_scores={"HR": 2.5},
        )
    )
    conn.commit()


def _call_build(conn: sqlite3.Connection) -> Any:
    repos = _make_repos(conn)
    draft_source = _make_draft_source(conn)
    return build_yahoo_draft_setup(
        team_repo=repos["team_repo"],
        league_repo=repos["league_repo"],
        valuation_repo=repos["valuation_repo"],
        player_repo=repos["player_repo"],
        adp_repo=repos["adp_repo"],
        draft_repo=repos["draft_repo"],
        draft_source=draft_source,
        league_key="449.l.100",
        season=2026,
        fbm_league=_LEAGUE,
        system="zar",
        version="1.0",
        provider="fantasypros",
    )


class TestBuildYahooDraftSetup:
    def test_creates_engine_with_correct_config(self, conn: sqlite3.Connection) -> None:
        _seed_data(conn)
        setup = _call_build(conn)

        assert setup.engine.state.config.teams == 2
        assert setup.engine.state.config.format.value == "live"
        assert setup.engine.state.config.user_team == 1

    def test_team_map_is_correct(self, conn: sqlite3.Connection) -> None:
        _seed_data(conn)
        setup = _call_build(conn)

        assert setup.team_map == {"449.l.100.t.1": 1, "449.l.100.t.2": 2}

    def test_board_has_players(self, conn: sqlite3.Connection) -> None:
        _seed_data(conn)
        setup = _call_build(conn)

        assert len(setup.board.rows) == 1
        assert setup.board.rows[0].player_name == "Mike Trout"

    def test_no_existing_picks_replayed(self, conn: sqlite3.Connection) -> None:
        _seed_data(conn)
        setup = _call_build(conn)

        assert setup.replayed_count == 0

    def test_auction_format_detected(self, conn: sqlite3.Connection) -> None:
        _seed_data(conn)
        repos = _make_repos(conn)

        # Update league to auction type
        repos["league_repo"].upsert(
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

        setup = _call_build(conn)

        assert setup.engine.state.config.format.value == "auction"
        assert setup.engine.state.config.budget == 260

    def test_no_teams_raises(self, conn: sqlite3.Connection) -> None:
        repos = _make_repos(conn)
        draft_source = _make_draft_source(conn)

        with pytest.raises(ValueError, match="No teams found"):
            build_yahoo_draft_setup(
                team_repo=repos["team_repo"],
                league_repo=repos["league_repo"],
                valuation_repo=repos["valuation_repo"],
                player_repo=repos["player_repo"],
                adp_repo=repos["adp_repo"],
                draft_repo=repos["draft_repo"],
                draft_source=draft_source,
                league_key="449.l.100",
                season=2026,
                fbm_league=_LEAGUE,
                system="zar",
                version="1.0",
                provider="fantasypros",
            )

    def test_no_user_team_raises(self, conn: sqlite3.Connection) -> None:
        _seed_data(conn)
        repos = _make_repos(conn)

        # Reset user team flag
        repos["team_repo"].upsert(
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

        draft_source = _make_draft_source(conn)
        with pytest.raises(ValueError, match="No user team found"):
            build_yahoo_draft_setup(
                team_repo=repos["team_repo"],
                league_repo=repos["league_repo"],
                valuation_repo=repos["valuation_repo"],
                player_repo=repos["player_repo"],
                adp_repo=repos["adp_repo"],
                draft_repo=repos["draft_repo"],
                draft_source=draft_source,
                league_key="449.l.100",
                season=2026,
                fbm_league=_LEAGUE,
                system="zar",
                version="1.0",
                provider="fantasypros",
            )
