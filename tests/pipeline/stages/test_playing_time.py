import pytest

from fantasy_baseball_manager.pipeline.stages.playing_time import MarcelPlayingTime
from fantasy_baseball_manager.pipeline.types import PlayerRates


class TestMarcelPlayingTime:
    def test_batting_playing_time(self) -> None:
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=29,
            rates={"hr": 0.04},
            metadata={"pa_per_year": [600, 550, 500]},
        )
        projector = MarcelPlayingTime()
        result = projector.project([p])
        # 0.5*600 + 0.1*550 + 200 = 555
        assert result[0].opportunities == pytest.approx(555.0)

    def test_batting_single_year(self) -> None:
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=29,
            rates={"hr": 0.04},
            metadata={"pa_per_year": [600]},
        )
        projector = MarcelPlayingTime()
        result = projector.project([p])
        # 0.5*600 + 0.1*0 + 200 = 500
        assert result[0].opportunities == pytest.approx(500.0)

    def test_pitching_starter_playing_time(self) -> None:
        p = PlayerRates(
            player_id="sp1",
            name="Test SP",
            year=2025,
            age=29,
            rates={"so": 0.2},
            metadata={"ip_per_year": [180.0, 170.0], "is_starter": True},
        )
        projector = MarcelPlayingTime()
        result = projector.project([p])
        # IP = 0.5*180 + 0.1*170 + 60 = 167
        # Outs = 167 * 3 = 501
        assert result[0].opportunities == pytest.approx(501.0)

    def test_pitching_reliever_playing_time(self) -> None:
        p = PlayerRates(
            player_id="rp1",
            name="Test RP",
            year=2025,
            age=29,
            rates={"so": 0.3},
            metadata={"ip_per_year": [70.0, 65.0], "is_starter": False},
        )
        projector = MarcelPlayingTime()
        result = projector.project([p])
        # IP = 0.5*70 + 0.1*65 + 25 = 66.5
        # Outs = 66.5 * 3 = 199.5
        assert result[0].opportunities == pytest.approx(199.5)
