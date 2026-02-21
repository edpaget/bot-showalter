import sqlite3

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.domain.player import Team
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.domain.roster_stint import RosterStint
from fantasy_baseball_manager.repos.player_repo import SqliteTeamRepo
from fantasy_baseball_manager.repos.position_appearance_repo import SqlitePositionAppearanceRepo
from fantasy_baseball_manager.repos.roster_stint_repo import SqliteRosterStintRepo
from fantasy_baseball_manager.tools.player_tools import (
    create_find_players_tool,
    create_get_player_bio_tool,
    create_search_players_tool,
)
from tests.helpers import seed_player


def _seed_team(conn: sqlite3.Connection, abbreviation: str = "NYY", name: str = "Yankees") -> int:
    repo = SqliteTeamRepo(conn)
    return repo.upsert(Team(abbreviation=abbreviation, name=name, league="AL", division="East"))


def _seed_roster_stint(conn: sqlite3.Connection, player_id: int, team_id: int, season: int = 2025) -> None:
    repo = SqliteRosterStintRepo(conn)
    repo.upsert(RosterStint(player_id=player_id, team_id=team_id, season=season, start_date=f"{season}-04-01"))


def _seed_position(conn: sqlite3.Connection, player_id: int, season: int, position: str, games: int = 100) -> None:
    repo = SqlitePositionAppearanceRepo(conn)
    repo.upsert(PositionAppearance(player_id=player_id, season=season, position=position, games=games))


class TestSearchPlayers:
    def test_tool_has_valid_attributes(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_search_players_tool(container)
        assert tool.name == "search_players"
        assert tool.description
        assert tool.args_schema is not None

    def test_matching_results(self, conn: sqlite3.Connection) -> None:
        team_id = _seed_team(conn)
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742, birth_date="1998-10-25")
        _seed_roster_stint(conn, pid, team_id, season=2025)
        _seed_position(conn, pid, 2025, "OF", games=150)
        container = AnalysisContainer(conn)
        tool = create_search_players_tool(container)

        result = tool.run({"name": "Soto", "season": 2025})
        assert "Juan Soto" in result
        assert "NYY" in result

    def test_no_results(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_search_players_tool(container)

        result = tool.run({"name": "Nobody", "season": 2025})
        assert "No players found" in result


class TestGetPlayerBio:
    def test_tool_has_valid_attributes(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_get_player_bio_tool(container)
        assert tool.name == "get_player_bio"
        assert tool.description
        assert tool.args_schema is not None

    def test_single_match_returns_bio(self, conn: sqlite3.Connection) -> None:
        team_id = _seed_team(conn)
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742, birth_date="1998-10-25", bats="L")
        _seed_roster_stint(conn, pid, team_id, season=2025)
        _seed_position(conn, pid, 2025, "OF", games=150)
        container = AnalysisContainer(conn)
        tool = create_get_player_bio_tool(container)

        result = tool.run({"player_name": "Soto", "season": 2025})
        assert "Juan Soto" in result
        assert "Team: NYY" in result
        assert "Position: OF" in result
        assert "Bats/Throws: L/" in result

    def test_multiple_matches_lists_them(self, conn: sqlite3.Connection) -> None:
        team_id = _seed_team(conn)
        pid1 = seed_player(conn, name_first="Joe", name_last="Smith", mlbam_id=100001)
        pid2 = seed_player(conn, name_first="John", name_last="Smith", mlbam_id=100002)
        for pid in (pid1, pid2):
            _seed_roster_stint(conn, pid, team_id, season=2025)
            _seed_position(conn, pid, 2025, "1B")
        container = AnalysisContainer(conn)
        tool = create_get_player_bio_tool(container)

        result = tool.run({"player_name": "Smith", "season": 2025})
        assert "Multiple players found" in result
        assert "Joe Smith" in result
        assert "John Smith" in result

    def test_no_match_returns_message(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_get_player_bio_tool(container)

        result = tool.run({"player_name": "Nobody", "season": 2025})
        assert "No players found" in result


class TestFindPlayers:
    def test_tool_has_valid_attributes(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_find_players_tool(container)
        assert tool.name == "find_players"
        assert tool.description
        assert tool.args_schema is not None

    def test_filter_by_team(self, conn: sqlite3.Connection) -> None:
        nyy_id = _seed_team(conn, "NYY", "Yankees")
        bos_id = _seed_team(conn, "BOS", "Red Sox")
        pid1 = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        pid2 = seed_player(conn, name_first="Rafael", name_last="Devers", mlbam_id=646240)
        _seed_roster_stint(conn, pid1, nyy_id, season=2025)
        _seed_roster_stint(conn, pid2, bos_id, season=2025)
        _seed_position(conn, pid1, 2025, "OF")
        _seed_position(conn, pid2, 2025, "3B")
        container = AnalysisContainer(conn)
        tool = create_find_players_tool(container)

        result = tool.run({"season": 2025, "team": "NYY"})
        assert "Juan Soto" in result
        assert "Rafael Devers" not in result

    def test_no_results(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_find_players_tool(container)

        result = tool.run({"season": 2025})
        assert "No players found" in result
