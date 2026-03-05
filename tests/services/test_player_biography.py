from typing import TYPE_CHECKING

import pytest

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
from fantasy_baseball_manager.services.player_team_provider import (
    MlbApiPlayerTeamProvider,
)
from fantasy_baseball_manager.team_resolver import TeamResolver
from tests.helpers import seed_player

if TYPE_CHECKING:
    import sqlite3

    from fantasy_baseball_manager.repos import TeamResolverProto


def _make_service(
    conn: sqlite3.Connection,
    player_team_provider: MlbApiPlayerTeamProvider | None = None,
    team_resolver: TeamResolverProto | None = None,
) -> PlayerBiographyService:
    return PlayerBiographyService(
        player_repo=SqlitePlayerRepo(conn),
        team_repo=SqliteTeamRepo(conn),
        roster_stint_repo=SqliteRosterStintRepo(conn),
        batting_stats_repo=SqliteBattingStatsRepo(conn),
        pitching_stats_repo=SqlitePitchingStatsRepo(conn),
        position_appearance_repo=SqlitePositionAppearanceRepo(conn),
        player_team_provider=player_team_provider,
        team_resolver=team_resolver,
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

    def test_no_appearances_returns_dh(self, conn: sqlite3.Connection) -> None:
        team_id = _seed_team(conn)
        pid = seed_player(conn, name_first="DH", name_last="Only", mlbam_id=100099)
        _seed_roster_stint(conn, pid, team_id, season=2025)
        # No position appearances seeded
        svc = _make_service(conn)

        result = svc.get_bio(pid, 2025)
        assert result is not None
        assert result.primary_position == "DH"


class TestFindEdgeCases:
    def test_nonexistent_team_returns_empty(self, conn: sqlite3.Connection) -> None:
        svc = _make_service(conn)
        results = svc.find(season=2025, team="ZZZ")
        assert results == []

    def test_skips_player_with_no_birth_date(self, conn: sqlite3.Connection) -> None:
        team_id = _seed_team(conn)
        # Insert player with NULL birth_date via raw SQL
        conn.execute(
            "INSERT INTO player (id, name_first, name_last) VALUES (?, ?, ?)",
            (9001, "NoBD", "Player"),
        )
        conn.commit()
        pid_no_bd = 9001
        pid_normal = seed_player(
            conn, name_first="Normal", name_last="Player", mlbam_id=100011, birth_date="1995-06-15"
        )
        for pid in (pid_no_bd, pid_normal):
            _seed_roster_stint(conn, pid, team_id, season=2025)
            _seed_position(conn, pid, 2025, "OF")
        svc = _make_service(conn)

        # min_age triggers the age check; player with NULL birth_date → age is None → skipped
        results = svc.find(season=2025, min_age=20)
        assert len(results) == 1
        assert results[0].name == "Normal Player"

    def test_filter_by_max_experience(self, conn: sqlite3.Connection) -> None:
        team_id = _seed_team(conn)
        pid_vet = seed_player(conn, name_first="Vet", name_last="Player", mlbam_id=100020)
        for yr in range(2018, 2026):
            _seed_batting(conn, pid_vet, yr)
        pid_rook = seed_player(conn, name_first="Rook", name_last="Player", mlbam_id=100021)
        _seed_batting(conn, pid_rook, 2024)
        _seed_batting(conn, pid_rook, 2025)
        for pid in (pid_vet, pid_rook):
            _seed_roster_stint(conn, pid, team_id, season=2025)
            _seed_position(conn, pid, 2025, "OF")
        svc = _make_service(conn)

        results = svc.find(season=2025, max_experience=3)
        assert len(results) == 1
        assert results[0].name == "Rook Player"

    def test_pitching_seasons_counted(self, conn: sqlite3.Connection) -> None:
        team_id = _seed_team(conn)
        pid = seed_player(conn, name_first="Pitch", name_last="Only", mlbam_id=100030)
        # Only pitching stats, no batting
        _seed_pitching(conn, pid, 2022)
        _seed_pitching(conn, pid, 2023)
        _seed_pitching(conn, pid, 2024)
        _seed_roster_stint(conn, pid, team_id, season=2025)
        _seed_position(conn, pid, 2025, "SP")
        svc = _make_service(conn)

        result = svc.get_bio(pid, 2025)
        assert result is not None
        assert result.experience == 3


class TestProviderFallback:
    """Tests for MLB API provider fallback in find() and _build_summary()."""

    def _make_provider(self, conn: sqlite3.Connection, mapping: dict[int, str]) -> MlbApiPlayerTeamProvider:
        """Create a provider with a fake fetcher returning *mapping* keyed by mlbam_id."""
        player_repo = SqlitePlayerRepo(conn)
        team_repo = SqliteTeamRepo(conn)
        roster_repo = SqliteRosterStintRepo(conn)

        def fake_fetcher(_season: int) -> dict[int, str]:
            return mapping

        return MlbApiPlayerTeamProvider(player_repo, team_repo, roster_repo, fetcher=fake_fetcher)

    def test_find_by_team_with_no_stints_uses_provider(self, conn: sqlite3.Connection) -> None:
        """When roster stints are empty for the season, provider data is used."""
        # Seed player but NO roster stints for 2026
        pid = seed_player(conn, name_first="Aaron", name_last="Judge", mlbam_id=592450)
        _seed_position(conn, pid, 2025, "OF")

        provider = self._make_provider(conn, {592450: "NYY"})
        svc = _make_service(conn, player_team_provider=provider)

        results = svc.find(season=2026, team="NYY")
        assert len(results) == 1
        assert results[0].name == "Aaron Judge"
        assert results[0].team == "NYY"

    def test_find_all_augments_with_provider(self, conn: sqlite3.Connection) -> None:
        """find() without team filter includes players from both stints and provider."""
        team_id = _seed_team(conn)
        pid1 = seed_player(conn, name_first="Aaron", name_last="Judge", mlbam_id=592450)
        _seed_roster_stint(conn, pid1, team_id, season=2025)
        _seed_position(conn, pid1, 2025, "OF")

        # Second player has no stints but is in provider
        pid2 = seed_player(conn, name_first="Rafael", name_last="Devers", mlbam_id=646240)
        _seed_position(conn, pid2, 2025, "3B")

        provider = self._make_provider(conn, {646240: "BOS"})
        svc = _make_service(conn, player_team_provider=provider)

        results = svc.find(season=2025)
        names = {r.name for r in results}
        assert "Aaron Judge" in names
        assert "Rafael Devers" in names

    def test_find_by_modern_abbrev_with_lahman_team(self, conn: sqlite3.Connection) -> None:
        """find(team='NYY') works when DB only has Lahman abbreviation 'NYA'."""
        team_id = _seed_team(conn, "NYA", "Yankees")
        pid = seed_player(conn, name_first="Aaron", name_last="Judge", mlbam_id=592450)
        _seed_roster_stint(conn, pid, team_id, season=2025)
        _seed_position(conn, pid, 2025, "OF")
        svc = _make_service(conn)

        results = svc.find(season=2025, team="NYY")
        assert len(results) == 1
        assert results[0].name == "Aaron Judge"

    def test_build_summary_uses_provider_when_no_stints(self, conn: sqlite3.Connection) -> None:
        """get_bio() returns provider team when no roster stints exist."""
        pid = seed_player(conn, name_first="Aaron", name_last="Judge", mlbam_id=592450)

        provider = self._make_provider(conn, {592450: "NYY"})
        svc = _make_service(conn, player_team_provider=provider)

        result = svc.get_bio(pid, 2026)
        assert result is not None
        assert result.team == "NYY"

    def test_build_summary_defaults_to_fa_without_provider(self, conn: sqlite3.Connection) -> None:
        """get_bio() returns FA when no stints and no provider."""
        pid = seed_player(conn, name_first="Aaron", name_last="Judge", mlbam_id=592450)
        svc = _make_service(conn)

        result = svc.get_bio(pid, 2026)
        assert result is not None
        assert result.team == "FA"

    def test_build_summary_converts_lahman_to_modern(self, conn: sqlite3.Connection) -> None:
        """_build_summary converts Lahman abbreviation (NYA) to modern (NYY)."""
        team_id = _seed_team(conn, "NYA", "Yankees")
        pid = seed_player(conn, name_first="Aaron", name_last="Judge", mlbam_id=592450)
        _seed_roster_stint(conn, pid, team_id, season=2025)
        svc = _make_service(conn)

        result = svc.get_bio(pid, 2025)
        assert result is not None
        assert result.team == "NYY"


class TestFuzzyTeamResolution:
    """Tests for fuzzy team name resolution in find()."""

    @staticmethod
    def _setup_two_teams(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
        """Seed NYY and NYM with one player each. Returns (nyy_id, nym_id, pid_nyy, pid_nym)."""
        nyy_id = _seed_team(conn, "NYY", "New York Yankees")
        nym_id = _seed_team(conn, "NYM", "New York Mets")
        pid_nyy = seed_player(conn, name_first="Aaron", name_last="Judge", mlbam_id=592450)
        pid_nym = seed_player(conn, name_first="Pete", name_last="Alonso", mlbam_id=624413)
        _seed_roster_stint(conn, pid_nyy, nyy_id, season=2025)
        _seed_roster_stint(conn, pid_nym, nym_id, season=2025)
        _seed_position(conn, pid_nyy, 2025, "OF")
        _seed_position(conn, pid_nym, 2025, "1B")
        return nyy_id, nym_id, pid_nyy, pid_nym

    def test_find_by_nickname(self, conn: sqlite3.Connection) -> None:
        """find(team='Yankees') returns same results as find(team='NYY')."""
        self._setup_two_teams(conn)
        resolver = TeamResolver(SqliteTeamRepo(conn))
        svc = _make_service(conn, team_resolver=resolver)

        by_abbrev = svc.find(season=2025, team="NYY")
        by_nickname = svc.find(season=2025, team="Yankees")

        assert len(by_abbrev) == 1
        assert len(by_nickname) == 1
        assert by_abbrev[0].name == by_nickname[0].name == "Aaron Judge"

    def test_find_ambiguous_city(self, conn: sqlite3.Connection) -> None:
        """find(team='New York') returns players from both NYY and NYM."""
        self._setup_two_teams(conn)
        resolver = TeamResolver(SqliteTeamRepo(conn))
        svc = _make_service(conn, team_resolver=resolver)

        results = svc.find(season=2025, team="New York")
        names = {r.name for r in results}
        assert "Aaron Judge" in names
        assert "Pete Alonso" in names

    def test_find_exact_abbrev_still_works(self, conn: sqlite3.Connection) -> None:
        """find(team='NYY') still works with resolver present (backward compatible)."""
        self._setup_two_teams(conn)
        resolver = TeamResolver(SqliteTeamRepo(conn))
        svc = _make_service(conn, team_resolver=resolver)

        results = svc.find(season=2025, team="NYY")
        assert len(results) == 1
        assert results[0].name == "Aaron Judge"

    def test_find_no_match_raises_valueerror(self, conn: sqlite3.Connection) -> None:
        """find(team='xyzabc') raises ValueError when resolver finds no match."""
        self._setup_two_teams(conn)
        resolver = TeamResolver(SqliteTeamRepo(conn))
        svc = _make_service(conn, team_resolver=resolver)

        with pytest.raises(ValueError, match="No team found matching 'xyzabc'"):
            svc.find(season=2025, team="xyzabc")
