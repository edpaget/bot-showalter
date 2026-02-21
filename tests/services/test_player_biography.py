import sqlite3

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.player import Team
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.domain.roster_stint import RosterStint
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo, SqliteTeamRepo
from fantasy_baseball_manager.repos.position_appearance_repo import SqlitePositionAppearanceRepo
from fantasy_baseball_manager.repos.roster_stint_repo import SqliteRosterStintRepo
from fantasy_baseball_manager.services.player_biography import PlayerBiographyService
from tests.helpers import seed_player


def _make_service(conn: sqlite3.Connection) -> PlayerBiographyService:
    return PlayerBiographyService(
        player_repo=SqlitePlayerRepo(conn),
        team_repo=SqliteTeamRepo(conn),
        roster_stint_repo=SqliteRosterStintRepo(conn),
        batting_stats_repo=SqliteBattingStatsRepo(conn),
        pitching_stats_repo=SqlitePitchingStatsRepo(conn),
        position_appearance_repo=SqlitePositionAppearanceRepo(conn),
    )


def _seed_team(conn: sqlite3.Connection, abbreviation: str = "NYY", name: str = "Yankees") -> int:
    repo = SqliteTeamRepo(conn)
    return repo.upsert(Team(abbreviation=abbreviation, name=name, league="AL", division="East"))


def _seed_roster_stint(conn: sqlite3.Connection, player_id: int, team_id: int, season: int = 2025) -> None:
    repo = SqliteRosterStintRepo(conn)
    repo.upsert(RosterStint(player_id=player_id, team_id=team_id, season=season, start_date=f"{season}-04-01"))


def _seed_position(conn: sqlite3.Connection, player_id: int, season: int, position: str, games: int = 100) -> None:
    repo = SqlitePositionAppearanceRepo(conn)
    repo.upsert(PositionAppearance(player_id=player_id, season=season, position=position, games=games))


def _seed_batting(conn: sqlite3.Connection, player_id: int, season: int, source: str = "fangraphs") -> None:
    repo = SqliteBattingStatsRepo(conn)
    repo.upsert(BattingStats(player_id=player_id, season=season, source=source, pa=500))


def _seed_pitching(conn: sqlite3.Connection, player_id: int, season: int, source: str = "fangraphs") -> None:
    repo = SqlitePitchingStatsRepo(conn)
    repo.upsert(PitchingStats(player_id=player_id, season=season, source=source, ip=180.0))


class TestSearch:
    def test_match_by_name(self, conn: sqlite3.Connection) -> None:
        team_id = _seed_team(conn)
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742, birth_date="1998-10-25")
        _seed_roster_stint(conn, pid, team_id, season=2025)
        _seed_position(conn, pid, 2025, "OF", games=150)
        _seed_batting(conn, pid, 2024)
        _seed_batting(conn, pid, 2025)
        svc = _make_service(conn)

        results = svc.search("Soto", 2025)
        assert len(results) == 1
        assert results[0].name == "Juan Soto"
        assert results[0].team == "NYY"
        assert results[0].age == 26  # born 1998-10-25, age as of July 1 2025 = 26
        assert results[0].primary_position == "OF"
        assert results[0].experience == 2

    def test_no_match(self, conn: sqlite3.Connection) -> None:
        svc = _make_service(conn)
        results = svc.search("Nobody", 2025)
        assert results == []

    def test_multiple_matches(self, conn: sqlite3.Connection) -> None:
        team_id = _seed_team(conn)
        pid1 = seed_player(conn, name_first="Joe", name_last="Smith", mlbam_id=100001)
        pid2 = seed_player(conn, name_first="John", name_last="Smith", mlbam_id=100002)
        for pid in (pid1, pid2):
            _seed_roster_stint(conn, pid, team_id, season=2025)
            _seed_position(conn, pid, 2025, "1B")
        svc = _make_service(conn)

        results = svc.search("Smith", 2025)
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"Joe Smith", "John Smith"}


class TestFind:
    def test_filter_by_team(self, conn: sqlite3.Connection) -> None:
        nyy_id = _seed_team(conn, "NYY", "Yankees")
        bos_id = _seed_team(conn, "BOS", "Red Sox")
        pid1 = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        pid2 = seed_player(conn, name_first="Rafael", name_last="Devers", mlbam_id=646240)
        _seed_roster_stint(conn, pid1, nyy_id, season=2025)
        _seed_roster_stint(conn, pid2, bos_id, season=2025)
        _seed_position(conn, pid1, 2025, "OF")
        _seed_position(conn, pid2, 2025, "3B")
        svc = _make_service(conn)

        results = svc.find(season=2025, team="NYY")
        assert len(results) == 1
        assert results[0].name == "Juan Soto"
        assert results[0].team == "NYY"

    def test_filter_by_max_age(self, conn: sqlite3.Connection) -> None:
        team_id = _seed_team(conn)
        # Young player: born 2003-01-15 -> age 22 as of July 1 2025
        pid_young = seed_player(conn, name_first="Young", name_last="Player", mlbam_id=100001, birth_date="2003-01-15")
        # Old player: born 1990-06-01 -> age 35 as of July 1 2025
        pid_old = seed_player(conn, name_first="Old", name_last="Player", mlbam_id=100002, birth_date="1990-06-01")
        for pid in (pid_young, pid_old):
            _seed_roster_stint(conn, pid, team_id, season=2025)
            _seed_position(conn, pid, 2025, "OF")
        svc = _make_service(conn)

        results = svc.find(season=2025, max_age=23)
        assert len(results) == 1
        assert results[0].name == "Young Player"

    def test_filter_by_min_experience(self, conn: sqlite3.Connection) -> None:
        team_id = _seed_team(conn)
        # Veteran: 6 seasons of batting stats
        pid_vet = seed_player(conn, name_first="Vet", name_last="Player", mlbam_id=100001)
        for yr in range(2020, 2026):
            _seed_batting(conn, pid_vet, yr)
        # Rookie: 2 seasons
        pid_rook = seed_player(conn, name_first="Rook", name_last="Player", mlbam_id=100002)
        _seed_batting(conn, pid_rook, 2024)
        _seed_batting(conn, pid_rook, 2025)
        for pid in (pid_vet, pid_rook):
            _seed_roster_stint(conn, pid, team_id, season=2025)
            _seed_position(conn, pid, 2025, "OF")
        svc = _make_service(conn)

        results = svc.find(season=2025, min_experience=5)
        assert len(results) == 1
        assert results[0].name == "Vet Player"
        assert results[0].experience == 6

    def test_filter_by_position(self, conn: sqlite3.Connection) -> None:
        team_id = _seed_team(conn)
        pid_ss = seed_player(conn, name_first="Short", name_last="Stop", mlbam_id=100001)
        pid_of = seed_player(conn, name_first="Out", name_last="Fielder", mlbam_id=100002)
        _seed_roster_stint(conn, pid_ss, team_id, season=2025)
        _seed_roster_stint(conn, pid_of, team_id, season=2025)
        _seed_position(conn, pid_ss, 2025, "SS", games=140)
        _seed_position(conn, pid_of, 2025, "OF", games=150)
        svc = _make_service(conn)

        results = svc.find(season=2025, position="SS")
        assert len(results) == 1
        assert results[0].name == "Short Stop"

    def test_combined_filters(self, conn: sqlite3.Connection) -> None:
        nyy_id = _seed_team(conn, "NYY", "Yankees")
        bos_id = _seed_team(conn, "BOS", "Red Sox")
        # Young NYY player -> should match
        pid1 = seed_player(conn, name_first="Young", name_last="Yankee", mlbam_id=100001, birth_date="2003-01-15")
        # Old NYY player -> should NOT match (too old)
        pid2 = seed_player(conn, name_first="Old", name_last="Yankee", mlbam_id=100002, birth_date="1990-06-01")
        # Young BOS player -> should NOT match (wrong team)
        pid3 = seed_player(conn, name_first="Young", name_last="RedSox", mlbam_id=100003, birth_date="2003-06-01")
        _seed_roster_stint(conn, pid1, nyy_id, season=2025)
        _seed_roster_stint(conn, pid2, nyy_id, season=2025)
        _seed_roster_stint(conn, pid3, bos_id, season=2025)
        for pid in (pid1, pid2, pid3):
            _seed_position(conn, pid, 2025, "OF")
        svc = _make_service(conn)

        results = svc.find(season=2025, team="NYY", max_age=23)
        assert len(results) == 1
        assert results[0].name == "Young Yankee"

    def test_no_results(self, conn: sqlite3.Connection) -> None:
        svc = _make_service(conn)
        results = svc.find(season=2025)
        assert results == []


class TestGetBio:
    def test_existing_player(self, conn: sqlite3.Connection) -> None:
        team_id = _seed_team(conn)
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742, birth_date="1998-10-25", bats="L")
        _seed_roster_stint(conn, pid, team_id, season=2025)
        _seed_position(conn, pid, 2025, "OF", games=150)
        _seed_batting(conn, pid, 2022)
        _seed_batting(conn, pid, 2023)
        _seed_batting(conn, pid, 2024)
        svc = _make_service(conn)

        result = svc.get_bio(pid, 2025)
        assert result is not None
        assert result.player_id == pid
        assert result.name == "Juan Soto"
        assert result.team == "NYY"
        assert result.age == 26
        assert result.primary_position == "OF"
        assert result.bats == "L"
        assert result.experience == 3

    def test_nonexistent_player(self, conn: sqlite3.Connection) -> None:
        svc = _make_service(conn)
        result = svc.get_bio(99999, 2025)
        assert result is None
