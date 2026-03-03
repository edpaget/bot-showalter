from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.position_appearance_repo import SqlitePositionAppearanceRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.services.residual_analyzer import ResidualAnalyzer
from tests.helpers import seed_player

if TYPE_CHECKING:
    import sqlite3

_ServiceTuple = tuple[
    ResidualAnalyzer,
    SqliteProjectionRepo,
    SqlitePlayerRepo,
    SqliteBattingStatsRepo,
    SqlitePitchingStatsRepo,
    SqlitePositionAppearanceRepo,
]


def _make_service(conn: sqlite3.Connection) -> _ServiceTuple:
    proj_repo = SqliteProjectionRepo(conn)
    player_repo = SqlitePlayerRepo(conn)
    batting_repo = SqliteBattingStatsRepo(conn)
    pitching_repo = SqlitePitchingStatsRepo(conn)
    position_repo = SqlitePositionAppearanceRepo(conn)
    analyzer = ResidualAnalyzer(proj_repo, batting_repo, pitching_repo, player_repo, position_repo)
    return analyzer, proj_repo, player_repo, batting_repo, pitching_repo, position_repo


def _seed_batter_data(
    conn: sqlite3.Connection,
    proj_repo: SqliteProjectionRepo,
    batting_repo: SqliteBattingStatsRepo,
    position_repo: SqlitePositionAppearanceRepo,
    *,
    player_id: int,
    name_first: str = "Test",
    name_last: str = "Player",
    predicted_slg: float = 0.400,
    actual_slg: float = 0.350,
    actual_avg: float = 0.270,
    actual_obp: float = 0.340,
    pa: int = 500,
    position: str = "SS",
    birth_date: str = "1995-07-01",
    bats: str = "R",
) -> None:
    seed_player(
        conn,
        player_id=player_id,
        name_first=name_first,
        name_last=name_last,
        birth_date=birth_date,
        bats=bats,
    )
    proj_repo.upsert(
        Projection(
            player_id=player_id,
            season=2024,
            system="test-model",
            version="v1",
            player_type="batter",
            stat_json={"slg": predicted_slg},
        )
    )
    batting_repo.upsert(
        BattingStats(
            player_id=player_id,
            season=2024,
            source="fangraphs",
            pa=pa,
            slg=actual_slg,
            avg=actual_avg,
            obp=actual_obp,
        )
    )
    position_repo.upsert(
        PositionAppearance(
            player_id=player_id,
            season=2024,
            position=position,
            games=100,
        )
    )


class TestAnalyzeBatterBasic:
    def test_returns_correct_top_misses(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)

        # Player 1: residual = 0.450 - 0.300 = +0.150 (big over-prediction)
        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=1,
            name_first="Big",
            name_last="Miss",
            predicted_slg=0.450,
            actual_slg=0.300,
            position="1B",
        )
        # Player 2: residual = 0.400 - 0.380 = +0.020 (small miss)
        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=2,
            name_first="Small",
            name_last="Miss",
            predicted_slg=0.400,
            actual_slg=0.380,
            position="CF",
        )
        # Player 3: residual = 0.350 - 0.450 = -0.100 (under-prediction)
        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=3,
            name_first="Under",
            name_last="Pred",
            predicted_slg=0.350,
            actual_slg=0.450,
            position="RF",
        )
        conn.commit()

        report = analyzer.analyze("test-model", "v1", 2024, "slg", "batter", top_n=2)
        assert report.target == "slg"
        assert report.player_type == "batter"
        assert report.season == 2024
        assert len(report.top_misses) == 2
        # Top misses by abs(residual): player 1 (0.150), player 3 (0.100)
        assert report.top_misses[0].player_id == 1
        assert report.top_misses[0].residual == pytest.approx(0.150)
        assert report.top_misses[1].player_id == 3
        assert report.top_misses[1].residual == pytest.approx(-0.100)

    def test_splits_over_and_under_predictions(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)

        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=1,
            predicted_slg=0.450,
            actual_slg=0.300,
        )
        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=2,
            predicted_slg=0.350,
            actual_slg=0.450,
        )
        conn.commit()

        report = analyzer.analyze("test-model", "v1", 2024, "slg", "batter")
        assert len(report.over_predictions) == 1
        assert report.over_predictions[0].player_id == 1
        assert len(report.under_predictions) == 1
        assert report.under_predictions[0].player_id == 2

    def test_feature_values_populated(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)

        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=1,
            predicted_slg=0.450,
            actual_slg=0.300,
            pa=600,
            birth_date="1994-07-01",
            bats="L",
        )
        conn.commit()

        report = analyzer.analyze("test-model", "v1", 2024, "slg", "batter")
        assert len(report.top_misses) == 1
        fv = report.top_misses[0].feature_values
        assert "age" in fv
        assert fv["age"] == 30.0  # born 1994-07-01, season 2024, age as of July 1
        assert fv["pa"] == 600.0
        assert "bats" in fv

    def test_summary_populated(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)

        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=1,
            predicted_slg=0.450,
            actual_slg=0.300,
            position="1B",
            birth_date="1992-07-01",
        )
        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=2,
            predicted_slg=0.400,
            actual_slg=0.380,
            position="CF",
            birth_date="1998-07-01",
        )
        conn.commit()

        report = analyzer.analyze("test-model", "v1", 2024, "slg", "batter", top_n=1)
        # Top 1 miss is player 1 (residual 0.150)
        assert report.summary.mean_age is not None
        assert report.summary.mean_age == pytest.approx(32.0)  # born 1992, season 2024
        assert "1B" in report.summary.position_distribution


class TestAnalyzeDirectionFilter:
    def test_over_direction_filters(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)

        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=1,
            predicted_slg=0.450,
            actual_slg=0.300,
        )
        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=2,
            predicted_slg=0.350,
            actual_slg=0.450,
        )
        conn.commit()

        report = analyzer.analyze("test-model", "v1", 2024, "slg", "batter", direction="over")
        assert len(report.top_misses) == 1
        assert report.top_misses[0].player_id == 1

    def test_under_direction_filters(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)

        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=1,
            predicted_slg=0.450,
            actual_slg=0.300,
        )
        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=2,
            predicted_slg=0.350,
            actual_slg=0.450,
        )
        conn.commit()

        report = analyzer.analyze("test-model", "v1", 2024, "slg", "batter", direction="under")
        assert len(report.top_misses) == 1
        assert report.top_misses[0].player_id == 2


class TestAnalyzeNoMatchingData:
    def test_no_projections_returns_empty_report(self, conn: sqlite3.Connection) -> None:
        analyzer, _, _, _, _, _ = _make_service(conn)
        report = analyzer.analyze("nonexistent", "v1", 2024, "slg", "batter")
        assert report.top_misses == []
        assert report.over_predictions == []
        assert report.under_predictions == []

    def test_no_matching_actuals_returns_empty_report(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, _, _, _ = _make_service(conn)
        seed_player(conn, player_id=1)
        proj_repo.upsert(
            Projection(
                player_id=1,
                season=2024,
                system="test-model",
                version="v1",
                player_type="batter",
                stat_json={"slg": 0.400},
            )
        )
        conn.commit()
        report = analyzer.analyze("test-model", "v1", 2024, "slg", "batter")
        assert report.top_misses == []


class TestAnalyzePitcher:
    def test_pitcher_residuals(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, _, pitching_repo, position_repo = _make_service(conn)

        seed_player(conn, player_id=10, name_first="Ace", name_last="Pitcher", birth_date="1996-07-01")
        proj_repo.upsert(
            Projection(
                player_id=10,
                season=2024,
                system="test-model",
                version="v1",
                player_type="pitcher",
                stat_json={"era": 3.50},
            )
        )
        pitching_repo.upsert(PitchingStats(player_id=10, season=2024, source="fangraphs", era=4.00, ip=180.0))
        position_repo.upsert(PositionAppearance(player_id=10, season=2024, position="SP", games=30))
        conn.commit()

        report = analyzer.analyze("test-model", "v1", 2024, "era", "pitcher")
        assert len(report.top_misses) == 1
        assert report.top_misses[0].residual == pytest.approx(-0.50)
        assert report.top_misses[0].feature_values["ip"] == 180.0
