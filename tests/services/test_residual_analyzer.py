from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.db.pool import SingleConnectionProvider
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
    proj_repo = SqliteProjectionRepo(SingleConnectionProvider(conn))
    player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
    batting_repo = SqliteBattingStatsRepo(SingleConnectionProvider(conn))
    pitching_repo = SqlitePitchingStatsRepo(SingleConnectionProvider(conn))
    position_repo = SqlitePositionAppearanceRepo(SingleConnectionProvider(conn))
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


def _seed_many_batters(
    conn: sqlite3.Connection,
    proj_repo: SqliteProjectionRepo,
    batting_repo: SqliteBattingStatsRepo,
    position_repo: SqlitePositionAppearanceRepo,
    *,
    count: int = 20,
    poor_count: int = 4,
) -> None:
    """Seed batters: most well-predicted, a few poorly-predicted.

    Well-predicted players have actual ≈ predicted (small residual).
    Poorly-predicted players have large residuals.
    """
    for i in range(1, count + 1):
        is_poor = i <= poor_count
        predicted = 0.400
        actual_slg = 0.250 if is_poor else 0.400 - (i - poor_count) * 0.001

        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=i,
            name_first=f"Player{i}",
            name_last="Test",
            predicted_slg=predicted,
            actual_slg=actual_slg,
            actual_avg=0.270 if not is_poor else 0.220,
            pa=500 if not is_poor else 200,
            birth_date="1995-07-01" if not is_poor else "1988-07-01",
        )
    conn.commit()


class TestDetectFeatureGaps:
    def test_feature_with_perfect_separation_has_high_ks(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)
        _seed_many_batters(conn, proj_repo, batting_repo, position_repo)

        # Add extra features: poorly-predicted (ids 1-4) all have high values,
        # well-predicted all have low values
        extra_features: dict[int, dict[str, float]] = {}
        for i in range(1, 21):
            is_poor = i <= 4
            extra_features[i] = {
                "secret_signal": 100.0 if is_poor else 0.0,
                "noise": float(i),
            }

        report = analyzer.detect_feature_gaps(
            system="test-model",
            version="v1",
            season=2024,
            target="slg",
            player_type="batter",
            model_feature_names=frozenset(["age", "pa", "avg", "obp", "slg"]),
            extra_features=extra_features,
        )
        assert report.target == "slg"
        assert report.player_type == "batter"
        assert len(report.gaps) > 0

        # secret_signal should have one of the highest KS statistics
        secret_gap = next((g for g in report.gaps if g.feature_name == "secret_signal"), None)
        assert secret_gap is not None
        assert secret_gap.ks_statistic > 0.5
        assert secret_gap.in_model is False

    def test_feature_with_no_gap_has_low_ks(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)
        _seed_many_batters(conn, proj_repo, batting_repo, position_repo)

        # Give all players the same value for "uniform_feature"
        extra_features: dict[int, dict[str, float]] = {}
        for i in range(1, 21):
            extra_features[i] = {"uniform_feature": 42.0}

        report = analyzer.detect_feature_gaps(
            system="test-model",
            version="v1",
            season=2024,
            target="slg",
            player_type="batter",
            model_feature_names=frozenset(["age"]),
            extra_features=extra_features,
        )
        uniform_gap = next((g for g in report.gaps if g.feature_name == "uniform_feature"), None)
        # Uniform feature should not appear (no variation) or have KS ≈ 0
        if uniform_gap is not None:
            assert uniform_gap.ks_statistic == pytest.approx(0.0, abs=0.01)

    def test_in_model_flag(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)
        _seed_many_batters(conn, proj_repo, batting_repo, position_repo)

        model_features = frozenset(["age", "pa"])
        report = analyzer.detect_feature_gaps(
            system="test-model",
            version="v1",
            season=2024,
            target="slg",
            player_type="batter",
            model_feature_names=model_features,
        )
        for gap in report.gaps:
            if gap.feature_name in model_features:
                assert gap.in_model is True
            else:
                assert gap.in_model is False

    def test_extra_features_merged(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)
        _seed_many_batters(conn, proj_repo, batting_repo, position_repo)

        extra_features: dict[int, dict[str, float]] = {i: {"new_metric": float(i * 10)} for i in range(1, 21)}
        report = analyzer.detect_feature_gaps(
            system="test-model",
            version="v1",
            season=2024,
            target="slg",
            player_type="batter",
            model_feature_names=frozenset(),
            extra_features=extra_features,
        )
        feature_names = {g.feature_name for g in report.gaps}
        assert "new_metric" in feature_names

    def test_no_residuals_returns_empty_report(self, conn: sqlite3.Connection) -> None:
        analyzer, _, _, _, _, _ = _make_service(conn)
        report = analyzer.detect_feature_gaps(
            system="nonexistent",
            version="v1",
            season=2024,
            target="slg",
            player_type="batter",
            model_feature_names=frozenset(),
        )
        assert report.gaps == []

    def test_gaps_sorted_by_ks_descending(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)
        _seed_many_batters(conn, proj_repo, batting_repo, position_repo)

        report = analyzer.detect_feature_gaps(
            system="test-model",
            version="v1",
            season=2024,
            target="slg",
            player_type="batter",
            model_feature_names=frozenset(),
        )
        ks_values = [g.ks_statistic for g in report.gaps]
        assert ks_values == sorted(ks_values, reverse=True)


def _seed_cohort_batters(
    conn: sqlite3.Connection,
    proj_repo: SqliteProjectionRepo,
    batting_repo: SqliteBattingStatsRepo,
    position_repo: SqlitePositionAppearanceRepo,
) -> None:
    """Seed batters across multiple age/position/hand cohorts.

    Young batters (22-25) have systematic over-prediction (positive residual).
    Older batters (30-33) are well-calibrated (residual ≈ 0).
    """
    # Young cohort: ages 22-25, all over-predicted by ~0.100
    for i, age_birth in enumerate(
        [("2002-07-01", "L", "SS"), ("2001-07-01", "L", "2B"), ("2000-07-01", "L", "SS"), ("1999-07-01", "L", "SS")],
        start=1,
    ):
        birth_date, bats, position = age_birth
        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=i,
            name_first=f"Young{i}",
            name_last="Batter",
            predicted_slg=0.450,
            actual_slg=0.350,
            position=position,
            birth_date=birth_date,
            bats=bats,
        )

    # Older cohort: ages 30-33, well-calibrated
    for i, age_birth in enumerate(
        [
            ("1994-07-01", "R", "1B"),
            ("1993-07-01", "R", "DH"),
            ("1992-07-01", "R", "1B"),
            ("1991-07-01", "S", "DH"),
        ],
        start=5,
    ):
        birth_date, bats, position = age_birth
        _seed_batter_data(
            conn,
            proj_repo,
            batting_repo,
            position_repo,
            player_id=i,
            name_first=f"Vet{i}",
            name_last="Batter",
            predicted_slg=0.400,
            actual_slg=0.400 + (i - 5) * 0.002,  # tiny residuals
            position=position,
            birth_date=birth_date,
            bats=bats,
        )
    conn.commit()


class TestBiasByCohortAge:
    def test_age_dimension_returns_correct_buckets(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)
        _seed_cohort_batters(conn, proj_repo, batting_repo, position_repo)

        report = analyzer.bias_by_cohort("test-model", "v1", 2024, "slg", "batter", "age")
        assert report.dimension == "age"
        assert report.target == "slg"

        labels = {c.cohort_label for c in report.cohorts}
        assert "22-25" in labels
        assert "30-33" in labels

    def test_young_cohort_has_positive_bias(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)
        _seed_cohort_batters(conn, proj_repo, batting_repo, position_repo)

        report = analyzer.bias_by_cohort("test-model", "v1", 2024, "slg", "batter", "age")
        young = next(c for c in report.cohorts if c.cohort_label == "22-25")
        assert young.mean_residual > 0  # over-prediction
        assert young.n == 4

    def test_well_calibrated_cohort_has_near_zero_bias(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)
        _seed_cohort_batters(conn, proj_repo, batting_repo, position_repo)

        report = analyzer.bias_by_cohort("test-model", "v1", 2024, "slg", "batter", "age")
        vet = next(c for c in report.cohorts if c.cohort_label == "30-33")
        assert abs(vet.mean_residual) < 0.01

    def test_significance_for_biased_cohort(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)
        _seed_cohort_batters(conn, proj_repo, batting_repo, position_repo)

        report = analyzer.bias_by_cohort("test-model", "v1", 2024, "slg", "batter", "age")
        young = next(c for c in report.cohorts if c.cohort_label == "22-25")
        assert young.significant is True


class TestBiasByCohortPosition:
    def test_position_dimension_groups_correctly(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)
        _seed_cohort_batters(conn, proj_repo, batting_repo, position_repo)

        report = analyzer.bias_by_cohort("test-model", "v1", 2024, "slg", "batter", "position")
        assert report.dimension == "position"
        labels = {c.cohort_label for c in report.cohorts}
        assert "SS" in labels
        assert "1B" in labels


class TestBiasByCohortHandedness:
    def test_handedness_dimension(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)
        _seed_cohort_batters(conn, proj_repo, batting_repo, position_repo)

        report = analyzer.bias_by_cohort("test-model", "v1", 2024, "slg", "batter", "handedness")
        assert report.dimension == "handedness"
        labels = {c.cohort_label for c in report.cohorts}
        assert "L" in labels
        assert "R" in labels


class TestBiasByCohortExperience:
    def test_experience_dimension(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)
        _seed_cohort_batters(conn, proj_repo, batting_repo, position_repo)

        # Seed batting stats for prior seasons to establish experience
        # Players 1-4 (young): 2 seasons of data (2023, 2024)
        for pid in range(1, 5):
            batting_repo.upsert(BattingStats(player_id=pid, season=2023, source="fangraphs", pa=400, slg=0.350))
        # Players 5-8 (veteran): 8 seasons of data (2017-2024)
        for pid in range(5, 9):
            for year in range(2017, 2024):
                batting_repo.upsert(BattingStats(player_id=pid, season=year, source="fangraphs", pa=500, slg=0.400))
        conn.commit()

        report = analyzer.bias_by_cohort("test-model", "v1", 2024, "slg", "batter", "experience")
        assert report.dimension == "experience"
        labels = {c.cohort_label for c in report.cohorts}
        # Young players: 2 seasons → "1-2" bucket
        assert "1-2" in labels
        # Veteran players: 8 seasons → "6-10" bucket
        assert "6-10" in labels


class TestBiasByCohortAllDimensions:
    def test_returns_four_reports(self, conn: sqlite3.Connection) -> None:
        analyzer, proj_repo, _, batting_repo, _, position_repo = _make_service(conn)
        _seed_cohort_batters(conn, proj_repo, batting_repo, position_repo)

        # Seed experience data
        for pid in range(1, 9):
            batting_repo.upsert(BattingStats(player_id=pid, season=2023, source="fangraphs", pa=400, slg=0.350))
        conn.commit()

        reports = analyzer.bias_by_cohort_all_dimensions("test-model", "v1", 2024, "slg", "batter")
        assert len(reports) == 4
        dims = {r.dimension for r in reports}
        assert dims == {"age", "position", "handedness", "experience"}


class TestBiasByCohortNoData:
    def test_no_residuals_returns_empty_cohorts(self, conn: sqlite3.Connection) -> None:
        analyzer, _, _, _, _, _ = _make_service(conn)
        report = analyzer.bias_by_cohort("nonexistent", "v1", 2024, "slg", "batter", "age")
        assert report.cohorts == []
