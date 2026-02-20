import sqlite3

from fantasy_baseball_manager.ingest.adp_mapper import (
    ADPIngestResult,
    _discover_provider_columns,
    _normalize_name,
    ingest_fantasypros_adp,
)
from fantasy_baseball_manager.repos.adp_repo import SqliteADPRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from tests.helpers import seed_player


def _seed_players(conn: sqlite3.Connection) -> dict[str, int]:
    ids: dict[str, int] = {}
    ids["trout"] = seed_player(conn, name_first="Mike", name_last="Trout", mlbam_id=545361)
    ids["judge"] = seed_player(conn, name_first="Aaron", name_last="Judge", mlbam_id=592450)
    ids["ohtani"] = seed_player(conn, name_first="Shohei", name_last="Ohtani", mlbam_id=660271)
    ids["witt"] = seed_player(conn, name_first="Bobby", name_last="Witt", mlbam_id=677951)
    ids["acuna"] = seed_player(conn, name_first="Ronald", name_last="Acuna", mlbam_id=660670)
    return ids


def _make_row(
    rank: str,
    player: str,
    team: str,
    positions: str,
    avg: str,
    **providers: str,
) -> dict[str, str]:
    row: dict[str, str] = {
        "Rank": rank,
        "Player": player,
        "Team": team,
        "Positions": positions,
    }
    row.update(providers)
    row["AVG"] = avg
    return row


class TestNormalizeName:
    def test_basic(self) -> None:
        assert _normalize_name("Mike Trout") == "mike trout"

    def test_strips_jr_suffix(self) -> None:
        assert _normalize_name("Bobby Witt Jr.") == "bobby witt"

    def test_strips_ii_suffix(self) -> None:
        assert _normalize_name("Ken Griffey II") == "ken griffey"

    def test_strips_batter_parenthetical(self) -> None:
        assert _normalize_name("Shohei Ohtani (Batter)") == "shohei ohtani"

    def test_strips_pitcher_parenthetical(self) -> None:
        assert _normalize_name("Shohei Ohtani (Pitcher)") == "shohei ohtani"

    def test_strips_accents(self) -> None:
        assert _normalize_name("Ronald Acuña") == "ronald acuna"


class TestDiscoverProviderColumns:
    def test_2026_headers(self) -> None:
        header = ["Rank", "Player", "Team", "Positions", "Yahoo", "CBS", "RTS", "NFBC", "FT", "ESPN", "AVG"]
        result = _discover_provider_columns(header)
        slugs = [slug for _, slug in result]
        assert set(slugs) == {"yahoo", "cbs", "rts", "nfbc", "ft", "espn"}

    def test_2016_headers_no_ft(self) -> None:
        header = ["Rank", "Player", "Team", "Positions", "NFBC", "RTS", "Yahoo", "ESPN", "CBS", "AVG"]
        result = _discover_provider_columns(header)
        slugs = [slug for _, slug in result]
        assert set(slugs) == {"nfbc", "rts", "yahoo", "espn", "cbs"}
        assert "ft" not in slugs

    def test_unknown_columns_ignored(self) -> None:
        header = ["Rank", "Player", "Team", "Positions", "UnknownSite", "ESPN", "AVG"]
        result = _discover_provider_columns(header)
        assert len(result) == 1
        assert result[0] == ("ESPN", "espn")


class TestIngestFantasyprosADP:
    def test_basic_ingest(self, conn: sqlite3.Connection) -> None:
        ids = _seed_players(conn)
        repo = SqliteADPRepo(conn)
        rows = [
            _make_row("1", "Mike Trout", "LAA", "CF,RF,DH", "1.0", ESPN="1", Yahoo="1"),
        ]
        result = ingest_fantasypros_adp(rows, repo, SqlitePlayerRepo(conn).all(), season=2026)
        assert result.loaded == 3  # AVG + ESPN + Yahoo
        assert result.skipped == 0
        assert result.unmatched == []

        adps = repo.get_by_player_season(ids["trout"], 2026)
        assert len(adps) == 3
        providers = {a.provider for a in adps}
        assert providers == {"fantasypros", "espn", "yahoo"}

    def test_multi_record_expansion(self, conn: sqlite3.Connection) -> None:
        _seed_players(conn)
        repo = SqliteADPRepo(conn)
        rows = [
            _make_row("1", "Aaron Judge", "NYY", "LF,CF,RF,DH", "1.8", ESPN="2", CBS="1", NFBC="2"),
        ]
        result = ingest_fantasypros_adp(rows, repo, SqlitePlayerRepo(conn).all(), season=2026)
        assert result.loaded == 4  # AVG + ESPN + CBS + NFBC

    def test_two_way_player_batter_and_pitcher_slots(self, conn: sqlite3.Connection) -> None:
        ids = _seed_players(conn)
        repo = SqliteADPRepo(conn)
        # Real CSVs have a consistent column set; empty strings for missing values
        rows = [
            _make_row("1", "Shohei Ohtani", "LAD", "SP,DH", "1.0", Yahoo="", CBS="2", ESPN="1"),
            _make_row("3", "Shohei Ohtani (Batter)", "LAD", "DH", "2.0", Yahoo="2", CBS="", ESPN=""),
            _make_row("95", "Shohei Ohtani (Pitcher)", "LAD", "SP", "92.0", Yahoo="92", CBS="", ESPN=""),
        ]
        result = ingest_fantasypros_adp(rows, repo, SqlitePlayerRepo(conn).all(), season=2026)
        assert result.unmatched == []

        adps = repo.get_by_player_season(ids["ohtani"], 2026)
        by_pos_provider = {(a.positions, a.provider): a.overall_pick for a in adps}
        assert by_pos_provider[("SP,DH", "fantasypros")] == 1.0
        assert by_pos_provider[("SP,DH", "cbs")] == 2.0
        assert by_pos_provider[("SP,DH", "espn")] == 1.0
        assert by_pos_provider[("DH", "fantasypros")] == 2.0
        assert by_pos_provider[("DH", "yahoo")] == 2.0
        assert by_pos_provider[("SP", "fantasypros")] == 92.0
        assert by_pos_provider[("SP", "yahoo")] == 92.0
        assert len(adps) == 7

    def test_empty_provider_values_skipped(self, conn: sqlite3.Connection) -> None:
        _seed_players(conn)
        repo = SqliteADPRepo(conn)
        rows = [
            _make_row("1", "Mike Trout", "LAA", "CF,RF,DH", "1.0", ESPN="", Yahoo="2"),
        ]
        result = ingest_fantasypros_adp(rows, repo, SqlitePlayerRepo(conn).all(), season=2026)
        assert result.loaded == 2  # AVG + Yahoo (ESPN skipped)

    def test_unmatched_player(self, conn: sqlite3.Connection) -> None:
        _seed_players(conn)
        repo = SqliteADPRepo(conn)
        rows = [
            _make_row("1", "Nonexistent Player", "XXX", "OF", "50.0"),
        ]
        result = ingest_fantasypros_adp(rows, repo, SqlitePlayerRepo(conn).all(), season=2026)
        assert result.loaded == 0
        assert result.unmatched == ["Nonexistent Player"]

    def test_ambiguous_name(self, conn: sqlite3.Connection) -> None:
        seed_player(conn, name_first="John", name_last="Smith", mlbam_id=100001)
        seed_player(conn, name_first="John", name_last="Smith", mlbam_id=100002)
        repo_p = SqlitePlayerRepo(conn)
        repo = SqliteADPRepo(conn)
        rows = [
            _make_row("1", "John Smith", "NYY", "OF", "50.0"),
        ]
        result = ingest_fantasypros_adp(rows, repo, repo_p.all(), season=2026)
        assert result.loaded == 0
        assert result.unmatched == ["John Smith"]

    def test_jr_suffix_matches(self, conn: sqlite3.Connection) -> None:
        ids = _seed_players(conn)
        repo = SqliteADPRepo(conn)
        rows = [
            _make_row("4", "Bobby Witt Jr.", "KC", "SS", "3.4"),
        ]
        result = ingest_fantasypros_adp(rows, repo, SqlitePlayerRepo(conn).all(), season=2026)
        assert result.loaded == 1
        adps = repo.get_by_player_season(ids["witt"], 2026)
        assert len(adps) == 1

    def test_accent_matching(self, conn: sqlite3.Connection) -> None:
        ids = _seed_players(conn)
        repo = SqliteADPRepo(conn)
        rows = [
            _make_row("5", "Ronald Acuña Jr.", "ATL", "CF,RF,DH", "5.0"),
        ]
        result = ingest_fantasypros_adp(rows, repo, SqlitePlayerRepo(conn).all(), season=2026)
        assert result.loaded == 1
        adps = repo.get_by_player_season(ids["acuna"], 2026)
        assert len(adps) == 1

    def test_as_of_passed_through(self, conn: sqlite3.Connection) -> None:
        ids = _seed_players(conn)
        repo = SqliteADPRepo(conn)
        rows = [
            _make_row("1", "Mike Trout", "LAA", "CF", "1.0"),
        ]
        ingest_fantasypros_adp(rows, repo, SqlitePlayerRepo(conn).all(), season=2026, as_of="2026-03-01")
        adps = repo.get_by_player_season(ids["trout"], 2026)
        assert adps[0].as_of == "2026-03-01"

    def test_empty_rows(self, conn: sqlite3.Connection) -> None:
        _seed_players(conn)
        repo = SqliteADPRepo(conn)
        result = ingest_fantasypros_adp([], repo, SqlitePlayerRepo(conn).all(), season=2026)
        assert result == ADPIngestResult(loaded=0, skipped=0, unmatched=[])

    def test_avg_always_produced(self, conn: sqlite3.Connection) -> None:
        ids = _seed_players(conn)
        repo = SqliteADPRepo(conn)
        rows = [
            _make_row("1", "Mike Trout", "LAA", "CF", "1.0"),
        ]
        ingest_fantasypros_adp(rows, repo, SqlitePlayerRepo(conn).all(), season=2026)
        adps = repo.get_by_player_season(ids["trout"], 2026)
        assert len(adps) == 1
        assert adps[0].provider == "fantasypros"
        assert adps[0].overall_pick == 1.0
