from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.ingest.column_maps import (
    make_bref_pitching_mapper,
    make_fg_batting_mapper,
)
from fantasy_baseball_manager.ingest.loader import StatsLoader
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from tests.ingest.conftest import ErrorDataSource, FakeDataSource


def _seed_player(
    conn,
    *,
    name_first: str = "Mike",
    name_last: str = "Trout",
    mlbam_id: int = 545361,
    fangraphs_id: int = 10155,
) -> int:
    repo = SqlitePlayerRepo(conn)
    return repo.upsert(
        Player(
            name_first=name_first,
            name_last=name_last,
            mlbam_id=mlbam_id,
            fangraphs_id=fangraphs_id,
        )
    )


def _batting_mapper(row: pd.Series) -> BattingStats | None:
    player_id = row.get("player_id")
    if player_id is None or (isinstance(player_id, float) and pd.isna(player_id)):
        return None
    return BattingStats(
        player_id=int(player_id),
        season=int(row["season"]),
        source="test",
    )


def _pitching_mapper(row: pd.Series) -> PitchingStats | None:
    player_id = row.get("player_id")
    if player_id is None or (isinstance(player_id, float) and pd.isna(player_id)):
        return None
    return PitchingStats(
        player_id=int(player_id),
        season=int(row["season"]),
        source="test",
    )


def _batting_df(*overrides: dict[str, Any]) -> pd.DataFrame:
    defaults: dict[str, Any] = {"player_id": 1, "season": 2024}
    return pd.DataFrame([{**defaults, **o} for o in overrides])


class TestStatsLoader:
    def test_loads_batting_stats_and_writes_success_log(self, conn) -> None:
        player_id = _seed_player(conn)
        source = FakeDataSource(_batting_df({"player_id": player_id}))
        repo = SqliteBattingStatsRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, _batting_mapper, "batting_stats", conn=conn)

        log = loader.load()

        assert log.status == "success"
        assert log.rows_loaded == 1
        assert log.target_table == "batting_stats"
        assert log.source_type == "test"
        assert log.source_detail == "fake"
        assert log.error_message is None

        stats = repo.get_by_player_season(player_id, 2024)
        assert len(stats) == 1

    def test_skips_rows_where_mapper_returns_none(self, conn) -> None:
        player_id = _seed_player(conn)
        df = pd.DataFrame(
            [
                {"player_id": player_id, "season": 2024},
                {"season": 2024},  # missing player_id → mapper returns None
            ]
        )
        source = FakeDataSource(df)
        repo = SqliteBattingStatsRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, _batting_mapper, "batting_stats", conn=conn)

        log = loader.load()

        assert log.rows_loaded == 1

    def test_error_during_fetch_writes_error_log_and_reraises(self, conn) -> None:
        source = ErrorDataSource()
        repo = SqliteBattingStatsRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, _batting_mapper, "batting_stats", conn=conn)

        with pytest.raises(RuntimeError, match="fetch failed"):
            loader.load()

        logs = log_repo.get_by_target_table("batting_stats")
        assert len(logs) == 1
        assert logs[0].status == "error"
        assert logs[0].rows_loaded == 0
        assert "fetch failed" in (logs[0].error_message or "")

    def test_empty_dataframe_loads_zero_rows(self, conn) -> None:
        source = FakeDataSource(pd.DataFrame())
        repo = SqliteBattingStatsRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, _batting_mapper, "batting_stats", conn=conn)

        log = loader.load()

        assert log.status == "success"
        assert log.rows_loaded == 0

    def test_timestamps_are_valid_iso_strings(self, conn) -> None:
        player_id = _seed_player(conn)
        source = FakeDataSource(_batting_df({"player_id": player_id}))
        repo = SqliteBattingStatsRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, _batting_mapper, "batting_stats", conn=conn)

        log = loader.load()

        started = datetime.fromisoformat(log.started_at)
        finished = datetime.fromisoformat(log.finished_at)
        assert started <= finished

    def test_works_with_pitching_repo(self, conn) -> None:
        player_id = _seed_player(conn)
        df = pd.DataFrame([{"player_id": player_id, "season": 2024}])
        source = FakeDataSource(df)
        repo = SqlitePitchingStatsRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, _pitching_mapper, "pitching_stats", conn=conn)

        log = loader.load()

        assert log.status == "success"
        assert log.rows_loaded == 1
        assert log.target_table == "pitching_stats"

        stats = repo.get_by_player_season(player_id, 2024)
        assert len(stats) == 1

    def test_upsert_error_rolls_back_and_writes_error_log(self, conn) -> None:
        player_id = _seed_player(conn)
        conn.commit()
        # First row valid, second row has nonexistent player_id causing FK violation
        df = pd.DataFrame(
            [
                {"player_id": player_id, "season": 2024},
                {"player_id": 99999, "season": 2024},
            ]
        )
        source = FakeDataSource(df)
        repo = SqliteBattingStatsRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, _batting_mapper, "batting_stats", conn=conn)

        with pytest.raises(Exception):
            loader.load()

        # First row's upsert should be rolled back
        stats = repo.get_by_player_season(player_id, 2024)
        assert len(stats) == 0

        # Error log should be committed
        logs = log_repo.get_by_target_table("batting_stats")
        assert len(logs) == 1
        assert logs[0].status == "error"


class TestStatsLoaderIntegration:
    def test_fg_batting_end_to_end(self, conn) -> None:
        player_id = _seed_player(conn)
        player_repo = SqlitePlayerRepo(conn)
        players = player_repo.all()
        mapper = make_fg_batting_mapper(players)

        df = pd.DataFrame(
            [
                {
                    "IDfg": 10155,
                    "Season": 2024,
                    "PA": 500,
                    "AB": 450,
                    "H": 140,
                    "2B": 25,
                    "3B": 3,
                    "HR": 30,
                    "RBI": 85,
                    "R": 90,
                    "SB": 10,
                    "CS": 2,
                    "BB": 45,
                    "SO": 100,
                    "HBP": 4,
                    "SF": 3,
                    "SH": 0,
                    "GDP": 8,
                    "IBB": 5,
                    "AVG": 0.311,
                    "OBP": 0.390,
                    "SLG": 0.560,
                    "OPS": 0.950,
                    "wOBA": 0.400,
                    "wRC+": 160.0,
                    "WAR": 7.0,
                }
            ]
        )
        source = FakeDataSource(df)
        repo = SqliteBattingStatsRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, mapper, "batting_stats", conn=conn)

        log = loader.load()

        assert log.status == "success"
        assert log.rows_loaded == 1

        stats = repo.get_by_player_season(player_id, 2024, source="fangraphs")
        assert len(stats) == 1
        assert stats[0].hr == 30
        assert stats[0].avg == 0.311
        assert stats[0].war == 7.0

    def test_bref_pitching_end_to_end(self, conn) -> None:
        player_id = _seed_player(conn)
        player_repo = SqlitePlayerRepo(conn)
        players = player_repo.all()
        mapper = make_bref_pitching_mapper(players, season=2024)

        df = pd.DataFrame(
            [
                {
                    "mlbID": 545361,
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
                    "SO9": 9.7,
                }
            ]
        )
        source = FakeDataSource(df)
        repo = SqlitePitchingStatsRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, mapper, "pitching_stats", conn=conn)

        log = loader.load()

        assert log.status == "success"
        assert log.rows_loaded == 1

        stats = repo.get_by_player_season(player_id, 2024, source="bbref")
        assert len(stats) == 1
        assert stats[0].w == 12
        assert stats[0].era == 2.65
        assert stats[0].k_per_9 == 9.7
        assert stats[0].hld is None
        assert stats[0].war is None

    def test_unknown_players_skipped(self, conn) -> None:
        player_id = _seed_player(conn)
        player_repo = SqlitePlayerRepo(conn)
        players = player_repo.all()
        mapper = make_fg_batting_mapper(players)

        df = pd.DataFrame(
            [
                {
                    "IDfg": 10155,
                    "Season": 2024,
                    "PA": 500,
                    "AB": 450,
                    "H": 140,
                    "2B": 25,
                    "3B": 3,
                    "HR": 30,
                    "RBI": 85,
                    "R": 90,
                    "SB": 10,
                    "CS": 2,
                    "BB": 45,
                    "SO": 100,
                    "HBP": 4,
                    "SF": 3,
                    "SH": 0,
                    "GDP": 8,
                    "IBB": 5,
                    "AVG": 0.311,
                    "OBP": 0.390,
                    "SLG": 0.560,
                    "OPS": 0.950,
                    "wOBA": 0.400,
                    "wRC+": 160.0,
                    "WAR": 7.0,
                },
                {
                    "IDfg": 99999,
                    "Season": 2024,
                    "PA": 400,
                    "AB": 360,
                    "H": 100,
                    "2B": 20,
                    "3B": 1,
                    "HR": 15,
                    "RBI": 50,
                    "R": 60,
                    "SB": 5,
                    "CS": 1,
                    "BB": 30,
                    "SO": 80,
                    "HBP": 2,
                    "SF": 2,
                    "SH": 0,
                    "GDP": 6,
                    "IBB": 3,
                    "AVG": 0.278,
                    "OBP": 0.340,
                    "SLG": 0.450,
                    "OPS": 0.790,
                    "wOBA": 0.330,
                    "wRC+": 110.0,
                    "WAR": 3.0,
                },
            ]
        )
        source = FakeDataSource(df)
        repo = SqliteBattingStatsRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, mapper, "batting_stats", conn=conn)

        log = loader.load()

        assert log.rows_loaded == 1
        stats = repo.get_by_player_season(player_id, 2024, source="fangraphs")
        assert len(stats) == 1
