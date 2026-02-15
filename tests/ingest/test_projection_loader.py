import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.projection_accuracy import (
    compare_to_batting_actuals,
    compare_to_pitching_actuals,
)
from fantasy_baseball_manager.ingest.column_maps import (
    make_fg_projection_batting_mapper,
    make_fg_projection_pitching_mapper,
)
from fantasy_baseball_manager.ingest.csv_source import CsvSource
from fantasy_baseball_manager.ingest.loader import StatsLoader
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from tests.ingest.conftest import FakeDataSource


def _seed_player(conn: sqlite3.Connection) -> int:
    repo = SqlitePlayerRepo(conn)
    return repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361, fangraphs_id=10155))


def _batting_projection_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PlayerId": 10155,
                "MLBAMID": 545361,
                "PA": 600,
                "AB": 530,
                "H": 160,
                "2B": 30,
                "3B": 5,
                "HR": 35,
                "RBI": 90,
                "R": 100,
                "SB": 15,
                "CS": 3,
                "BB": 60,
                "SO": 120,
                "AVG": 0.302,
                "OBP": 0.395,
                "SLG": 0.575,
                "OPS": 0.970,
                "wOBA": 0.410,
                "wRC+": 170.0,
                "WAR": 8.5,
            }
        ]
    )


class TestProjectionLoaderIntegration:
    def test_loads_batting_projections_via_stats_loader(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        player_repo = SqlitePlayerRepo(conn)
        players = player_repo.all()
        mapper = make_fg_projection_batting_mapper(players, season=2025, system="steamer", version="2025.1")
        source = FakeDataSource(_batting_projection_df())
        proj_repo = SqliteProjectionRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, proj_repo, log_repo, mapper, "projection", conn=conn)

        log = loader.load()

        assert log.status == "success"
        assert log.rows_loaded == 1
        assert log.target_table == "projection"

        results = proj_repo.get_by_player_season(player_id, 2025, system="steamer")
        assert len(results) == 1
        assert results[0].stat_json["hr"] == 35
        assert results[0].stat_json["avg"] == 0.302
        assert results[0].player_type == "batter"

    def test_loads_pitching_projections(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        player_repo = SqlitePlayerRepo(conn)
        players = player_repo.all()
        mapper = make_fg_projection_pitching_mapper(players, season=2025, system="zips", version="2025.1")
        df = pd.DataFrame(
            [
                {
                    "PlayerId": 10155,
                    "MLBAMID": 545361,
                    "W": 12,
                    "L": 6,
                    "G": 30,
                    "GS": 30,
                    "SV": 0,
                    "H": 130,
                    "ER": 50,
                    "HR": 15,
                    "BB": 40,
                    "SO": 200,
                    "ERA": 2.65,
                    "IP": 185.0,
                    "WHIP": 0.92,
                    "K/9": 9.7,
                    "BB/9": 1.9,
                    "FIP": 2.80,
                    "WAR": 6.5,
                }
            ]
        )
        source = FakeDataSource(df)
        proj_repo = SqliteProjectionRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, proj_repo, log_repo, mapper, "projection", conn=conn)

        log = loader.load()

        assert log.status == "success"
        assert log.rows_loaded == 1

        results = proj_repo.get_by_player_season(player_id, 2025, system="zips")
        assert len(results) == 1
        assert results[0].stat_json["era"] == 2.65
        assert results[0].stat_json["so"] == 200
        assert results[0].player_type == "pitcher"

    def test_csv_source_to_projection(self, tmp_path: Path, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        csv_file = tmp_path / "steamer_batting.csv"
        csv_file.write_text("PlayerId,MLBAMID,PA,HR,AVG,WAR\n" "10155,545361,600,35,0.302,8.5\n")

        player_repo = SqlitePlayerRepo(conn)
        players = player_repo.all()
        mapper = make_fg_projection_batting_mapper(players, season=2025, system="steamer", version="2025.1")
        source = CsvSource(csv_file)
        proj_repo = SqliteProjectionRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, proj_repo, log_repo, mapper, "projection", conn=conn)

        log = loader.load()

        assert log.status == "success"
        assert log.rows_loaded == 1
        assert log.source_type == "csv"

        results = proj_repo.get_by_player_season(player_id, 2025)
        assert len(results) == 1
        assert results[0].stat_json["hr"] == 35

    def test_unknown_players_skipped(self, conn: sqlite3.Connection) -> None:
        _seed_player(conn)
        player_repo = SqlitePlayerRepo(conn)
        players = player_repo.all()
        mapper = make_fg_projection_batting_mapper(players, season=2025, system="steamer", version="2025.1")
        df = pd.DataFrame(
            [
                {"PlayerId": 10155, "MLBAMID": 545361, "HR": 35, "AVG": 0.302},
                {"PlayerId": 99999, "MLBAMID": 999999, "HR": 20, "AVG": 0.250},
            ]
        )
        source = FakeDataSource(df)
        proj_repo = SqliteProjectionRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, proj_repo, log_repo, mapper, "projection", conn=conn)

        log = loader.load()

        assert log.rows_loaded == 1

    def test_upsert_deduplicates_on_reload(self, conn: sqlite3.Connection) -> None:
        _seed_player(conn)
        player_repo = SqlitePlayerRepo(conn)
        players = player_repo.all()
        mapper = make_fg_projection_batting_mapper(players, season=2025, system="steamer", version="2025.1")
        source = FakeDataSource(_batting_projection_df())
        proj_repo = SqliteProjectionRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, proj_repo, log_repo, mapper, "projection", conn=conn)

        loader.load()
        loader.load()

        results = proj_repo.get_by_season(2025, system="steamer")
        assert len(results) == 1

    def test_third_party_import_sets_source_type(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        player_repo = SqlitePlayerRepo(conn)
        players = player_repo.all()
        mapper = make_fg_projection_batting_mapper(
            players,
            season=2025,
            system="steamer",
            version="2025.1",
            source_type="third_party",
        )
        source = FakeDataSource(_batting_projection_df())
        proj_repo = SqliteProjectionRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, proj_repo, log_repo, mapper, "projection", conn=conn)

        log = loader.load()

        assert log.status == "success"
        assert log.rows_loaded == 1

        results = proj_repo.get_by_player_season(player_id, 2025, system="steamer")
        assert len(results) == 1
        assert results[0].source_type == "third_party"


class TestProjectionVsActuals:
    def test_compare_batting_projection_to_actuals(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)

        # Load projection
        player_repo = SqlitePlayerRepo(conn)
        players = player_repo.all()
        mapper = make_fg_projection_batting_mapper(players, season=2025, system="steamer", version="2025.1")
        source = FakeDataSource(_batting_projection_df())
        proj_repo = SqliteProjectionRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, proj_repo, log_repo, mapper, "projection", conn=conn)
        loader.load()

        # Load actual stats
        batting_repo = SqliteBattingStatsRepo(conn)
        batting_repo.upsert(
            BattingStats(
                player_id=player_id,
                season=2025,
                source="fangraphs",
                pa=580,
                hr=28,
                avg=0.275,
                sb=12,
                war=6.5,
            )
        )

        # Compare
        projections = proj_repo.get_by_player_season(player_id, 2025, system="steamer")
        actuals = batting_repo.get_by_player_season(player_id, 2025, source="fangraphs")
        assert len(projections) == 1
        assert len(actuals) == 1

        comparisons = compare_to_batting_actuals(projections[0], actuals[0])
        by_stat = {c.stat_name: c for c in comparisons}

        assert by_stat["hr"].projected == 35.0
        assert by_stat["hr"].actual == 28.0
        assert by_stat["hr"].error == 7.0

        assert by_stat["pa"].projected == 600.0
        assert by_stat["pa"].actual == 580.0
        assert by_stat["pa"].error == 20.0

        assert by_stat["war"].projected == 8.5
        assert by_stat["war"].actual == 6.5
        assert by_stat["war"].error == 2.0

    def test_compare_pitching_projection_to_actuals(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        player_repo = SqlitePlayerRepo(conn)
        players = player_repo.all()
        mapper = make_fg_projection_pitching_mapper(players, season=2025, system="steamer", version="2025.1")
        df = pd.DataFrame(
            [
                {
                    "PlayerId": 10155,
                    "MLBAMID": 545361,
                    "W": 12,
                    "L": 6,
                    "ERA": 3.00,
                    "SO": 200,
                    "IP": 180.0,
                    "WAR": 5.0,
                    "G": 30,
                    "GS": 30,
                    "SV": 0,
                    "H": 130,
                    "ER": 50,
                    "HR": 15,
                    "BB": 40,
                    "WHIP": 0.95,
                    "K/9": 10.0,
                    "BB/9": 2.0,
                    "FIP": 2.90,
                }
            ]
        )
        source = FakeDataSource(df)
        proj_repo = SqliteProjectionRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, proj_repo, log_repo, mapper, "projection", conn=conn)
        loader.load()

        pitching_repo = SqlitePitchingStatsRepo(conn)
        pitching_repo.upsert(
            PitchingStats(
                player_id=player_id,
                season=2025,
                source="fangraphs",
                era=3.50,
                so=180,
                w=10,
                ip=170.0,
                war=4.0,
            )
        )

        projections = proj_repo.get_by_player_season(player_id, 2025, system="steamer")
        actuals = pitching_repo.get_by_player_season(player_id, 2025, source="fangraphs")

        comparisons = compare_to_pitching_actuals(projections[0], actuals[0])
        by_stat = {c.stat_name: c for c in comparisons}

        assert by_stat["era"].projected == 3.00
        assert by_stat["era"].actual == 3.50
        assert by_stat["era"].error == pytest.approx(-0.50)

        assert by_stat["so"].projected == 200.0
        assert by_stat["so"].actual == 180.0
        assert by_stat["so"].error == 20.0
