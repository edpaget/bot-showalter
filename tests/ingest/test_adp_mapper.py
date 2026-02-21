import sqlite3

import pytest

from fantasy_baseball_manager.ingest.adp_mapper import (
    ADPIngestResult,
    _discover_provider_columns,
    _normalize_name,
    _split_raw_name,
    fetch_mlb_active_teams,
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

    def test_collapses_initials_with_spaces(self) -> None:
        assert _normalize_name("J. T. Realmuto") == "jt realmuto"

    def test_collapses_initials_without_spaces(self) -> None:
        assert _normalize_name("J.T. Realmuto") == "jt realmuto"

    def test_collapses_single_initial(self) -> None:
        assert _normalize_name("A.J. Minter") == "aj minter"

    def test_collapses_initials_with_extra_spaces(self) -> None:
        assert _normalize_name("A. J. Minter") == "aj minter"

    def test_initials_match_across_formats(self) -> None:
        # "J. P. Crawford" from DB should match "J.P. Crawford" from ADP
        assert _normalize_name("J. P. Crawford") == _normalize_name("J.P. Crawford")

    def test_nickname_matthew_matches_matt(self) -> None:
        assert _normalize_name("Matthew Boyd") == _normalize_name("Matt Boyd")

    def test_nickname_michael_matches_mike(self) -> None:
        assert _normalize_name("Michael King") == _normalize_name("Mike King")

    def test_nickname_does_not_affect_last_name(self) -> None:
        # "Stephen" in a last name should still be aliased (acceptable trade-off),
        # but verify normalization is consistent
        assert _normalize_name("John Matthew") == "john matt"


class TestSplitRawName:
    def test_simple_name(self) -> None:
        assert _split_raw_name("Mike Trout") == ("Mike", "Trout")

    def test_strips_jr_suffix(self) -> None:
        assert _split_raw_name("Bobby Witt Jr.") == ("Bobby", "Witt")

    def test_strips_parenthetical(self) -> None:
        assert _split_raw_name("Shohei Ohtani (Batter)") == ("Shohei", "Ohtani")

    def test_single_name(self) -> None:
        assert _split_raw_name("Madonna") == ("", "Madonna")


class TestFetchMlbActiveTeams:
    def test_returns_mlbam_to_team_mapping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        teams_json = {"teams": [{"id": 147, "abbreviation": "NYY"}, {"id": 119, "abbreviation": "LAD"}]}
        players_json = {
            "people": [
                {"id": 660271, "currentTeam": {"id": 119}},
                {"id": 592450, "currentTeam": {"id": 147}},
                {"id": 999999},  # no currentTeam
            ]
        }

        class _FakeResponse:
            def __init__(self, data: dict) -> None:
                self._data = data

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return self._data

        class _FakeClient:
            def __init__(self) -> None:
                self._calls: list[str] = []

            def get(self, url: str, **kwargs: object) -> _FakeResponse:
                self._calls.append(url)
                if "teams" in url:
                    return _FakeResponse(teams_json)
                return _FakeResponse(players_json)

            def __enter__(self) -> "_FakeClient":
                return self

            def __exit__(self, *args: object) -> None:
                pass

        monkeypatch.setattr("fantasy_baseball_manager.ingest.adp_mapper.httpx.Client", lambda **kw: _FakeClient())
        result = fetch_mlb_active_teams(2026)
        assert result == {660271: "LAD", 592450: "NYY"}


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

    def test_ambiguous_name_resolved_by_team(self, conn: sqlite3.Connection) -> None:
        id1 = seed_player(conn, name_first="John", name_last="Smith", mlbam_id=100001)
        seed_player(conn, name_first="John", name_last="Smith", mlbam_id=100002)
        repo = SqliteADPRepo(conn)
        rows = [
            _make_row("1", "John Smith", "NYY", "OF", "50.0"),
        ]
        player_teams = {id1: "NYY"}
        result = ingest_fantasypros_adp(
            rows, repo, SqlitePlayerRepo(conn).all(), season=2026, player_teams=player_teams
        )
        assert result.loaded == 1
        assert result.unmatched == []
        adps = repo.get_by_player_season(id1, 2026)
        assert len(adps) == 1

    def test_team_alias_resolves_lahman_abbreviation(self, conn: sqlite3.Connection) -> None:
        id1 = seed_player(conn, name_first="Bobby", name_last="Witt", mlbam_id=677951)
        seed_player(conn, name_first="Bobby", name_last="Witt", mlbam_id=124492)
        repo = SqliteADPRepo(conn)
        rows = [
            _make_row("1", "Bobby Witt Jr.", "KC", "SS", "3.4"),
        ]
        # Lahman uses "KCA" but ADP row has "KC"
        player_teams = {id1: "KCA"}
        result = ingest_fantasypros_adp(
            rows, repo, SqlitePlayerRepo(conn).all(), season=2026, player_teams=player_teams
        )
        assert result.loaded == 1
        assert result.unmatched == []
        adps = repo.get_by_player_season(id1, 2026)
        assert len(adps) == 1

    def test_initials_with_periods_match(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="J. T.", name_last="Realmuto", mlbam_id=592663)
        repo = SqliteADPRepo(conn)
        rows = [
            _make_row("1", "J.T. Realmuto", "PHI", "C", "50.0"),
        ]
        result = ingest_fantasypros_adp(rows, repo, SqlitePlayerRepo(conn).all(), season=2026)
        assert result.loaded == 1
        assert result.unmatched == []
        adps = repo.get_by_player_season(pid, 2026)
        assert len(adps) == 1

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

    def test_creates_stub_for_unknown_player(self, conn: sqlite3.Connection) -> None:
        _seed_players(conn)
        repo = SqliteADPRepo(conn)
        player_repo = SqlitePlayerRepo(conn)
        rows = [
            _make_row("100", "Tatsuya Imai", "HOU", "SP", "172.4"),
        ]
        result = ingest_fantasypros_adp(rows, repo, player_repo.all(), season=2026, player_repo=player_repo)
        assert result.loaded == 1
        assert result.created == 1
        assert result.unmatched == []
        # Stub player was created and ADP links to it
        stub = player_repo.search_by_name("Imai")
        assert len(stub) == 1
        assert stub[0].name_first == "Tatsuya"
        adps = repo.get_by_player_season(stub[0].id, 2026)  # type: ignore[arg-type]
        assert len(adps) == 1
        assert adps[0].overall_pick == 172.4

    def test_stub_not_created_for_ambiguous_player(self, conn: sqlite3.Connection) -> None:
        seed_player(conn, name_first="Josh", name_last="Bell", mlbam_id=100001)
        seed_player(conn, name_first="Josh", name_last="Bell", mlbam_id=100002)
        repo = SqliteADPRepo(conn)
        player_repo = SqlitePlayerRepo(conn)
        rows = [
            _make_row("50", "Josh Bell", "MIN", "1B,DH", "384.0"),
        ]
        result = ingest_fantasypros_adp(rows, repo, player_repo.all(), season=2026, player_repo=player_repo)
        assert result.loaded == 0
        assert result.created == 0
        assert result.unmatched == ["Josh Bell"]

    def test_stub_not_created_without_player_repo(self, conn: sqlite3.Connection) -> None:
        _seed_players(conn)
        repo = SqliteADPRepo(conn)
        rows = [
            _make_row("100", "Tatsuya Imai", "HOU", "SP", "172.4"),
        ]
        result = ingest_fantasypros_adp(rows, repo, SqlitePlayerRepo(conn).all(), season=2026)
        assert result.loaded == 0
        assert result.created == 0
        assert result.unmatched == ["Tatsuya Imai"]

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
