import pytest

from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.pipeline.stages.finalizers import StandardFinalizer
from fantasy_baseball_manager.pipeline.types import PlayerRates


class TestStandardFinalizerBatting:
    def test_produces_batting_projection(self) -> None:
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=29,
            rates={
                "singles": 0.15,
                "doubles": 0.05,
                "triples": 0.005,
                "hr": 0.04,
                "bb": 0.08,
                "so": 0.20,
                "hbp": 0.008,
                "sf": 0.005,
                "sh": 0.003,
                "sb": 0.02,
                "cs": 0.005,
                "r": 0.13,
                "rbi": 0.15,
            },
            opportunities=555.0,
        )
        finalizer = StandardFinalizer()
        result = finalizer.finalize_batting([p])
        assert len(result) == 1
        assert isinstance(result[0], BattingProjection)
        assert result[0].player_id == "p1"
        assert result[0].pa == pytest.approx(555.0)

    def test_counting_stats_are_rate_times_pa(self) -> None:
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=29,
            rates={
                "hr": 0.04,
                "bb": 0.08,
                "so": 0.20,
                "singles": 0.15,
                "doubles": 0.05,
                "triples": 0.005,
                "hbp": 0.008,
                "sf": 0.005,
                "sh": 0.003,
                "sb": 0.02,
                "cs": 0.005,
                "r": 0.13,
                "rbi": 0.15,
            },
            opportunities=500.0,
        )
        finalizer = StandardFinalizer()
        result = finalizer.finalize_batting([p])
        proj = result[0]
        assert proj.hr == pytest.approx(0.04 * 500)
        assert proj.bb == pytest.approx(0.08 * 500)
        assert proj.so == pytest.approx(0.20 * 500)

    def test_h_derived_from_components(self) -> None:
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=29,
            rates={
                "singles": 0.15,
                "doubles": 0.05,
                "triples": 0.005,
                "hr": 0.04,
                "bb": 0.08,
                "so": 0.20,
                "hbp": 0.008,
                "sf": 0.005,
                "sh": 0.003,
                "sb": 0.02,
                "cs": 0.005,
                "r": 0.13,
                "rbi": 0.15,
            },
            opportunities=500.0,
        )
        finalizer = StandardFinalizer()
        proj = finalizer.finalize_batting([p])[0]
        expected_h = 500 * (0.15 + 0.05 + 0.005 + 0.04)
        assert proj.h == pytest.approx(expected_h)

    def test_ab_derived_correctly(self) -> None:
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=29,
            rates={
                "singles": 0.15,
                "doubles": 0.05,
                "triples": 0.005,
                "hr": 0.04,
                "bb": 0.08,
                "so": 0.20,
                "hbp": 0.008,
                "sf": 0.005,
                "sh": 0.003,
                "sb": 0.02,
                "cs": 0.005,
                "r": 0.13,
                "rbi": 0.15,
            },
            opportunities=500.0,
        )
        finalizer = StandardFinalizer()
        proj = finalizer.finalize_batting([p])[0]
        expected_ab = 500 - 500 * (0.08 + 0.008 + 0.005 + 0.003)
        assert proj.ab == pytest.approx(expected_ab)


class TestStandardFinalizerPitching:
    def test_produces_pitching_projection(self) -> None:
        p = PlayerRates(
            player_id="sp1",
            name="Test SP",
            year=2025,
            age=29,
            rates={
                "h": 0.30,
                "bb": 0.10,
                "so": 0.25,
                "hr": 0.03,
                "hbp": 0.01,
                "er": 0.10,
                "w": 0.02,
                "sv": 0.0,
                "hld": 0.0,
                "bs": 0.0,
            },
            opportunities=501.0,  # outs
            metadata={"ip_per_year": [180.0, 170.0], "is_starter": True},
        )
        finalizer = StandardFinalizer()
        result = finalizer.finalize_pitching([p])
        assert len(result) == 1
        assert isinstance(result[0], PitchingProjection)
        assert result[0].ip == pytest.approx(167.0)

    def test_era_computed(self) -> None:
        p = PlayerRates(
            player_id="sp1",
            name="Test SP",
            year=2025,
            age=29,
            rates={
                "h": 0.30,
                "bb": 0.10,
                "so": 0.25,
                "hr": 0.03,
                "hbp": 0.01,
                "er": 0.10,
                "w": 0.02,
                "sv": 0.0,
                "hld": 0.0,
                "bs": 0.0,
            },
            opportunities=501.0,
            metadata={"ip_per_year": [180.0, 170.0], "is_starter": True},
        )
        finalizer = StandardFinalizer()
        proj = finalizer.finalize_pitching([p])[0]
        expected_er = 0.10 * 501
        expected_era = (expected_er / 167.0) * 9
        assert proj.era == pytest.approx(expected_era)

    def test_reliever_gs_is_zero(self) -> None:
        p = PlayerRates(
            player_id="rp1",
            name="Test RP",
            year=2025,
            age=29,
            rates={
                "h": 0.30,
                "bb": 0.10,
                "so": 0.30,
                "hr": 0.02,
                "hbp": 0.01,
                "er": 0.08,
                "w": 0.01,
                "sv": 0.05,
                "hld": 0.03,
                "bs": 0.01,
            },
            opportunities=199.5,
            metadata={"ip_per_year": [70.0, 65.0], "is_starter": False},
        )
        finalizer = StandardFinalizer()
        proj = finalizer.finalize_pitching([p])[0]
        assert proj.gs == pytest.approx(0.0)
        assert proj.nsvh == pytest.approx(199.5 * (0.05 + 0.03 - 0.01))
