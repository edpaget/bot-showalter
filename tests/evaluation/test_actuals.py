import pytest

from fantasy_baseball_manager.evaluation.actuals import (
    actuals_as_projections,
    batting_stats_to_projection,
    pitching_stats_to_projection,
)
from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)


def _make_batting_stats(
    player_id: str = "b1",
    name: str = "Hitter",
    year: int = 2024,
    pa: int = 600,
    ab: int = 540,
    h: int = 160,
    singles: int = 100,
    doubles: int = 30,
    triples: int = 5,
    hr: int = 25,
    bb: int = 50,
    so: int = 120,
    hbp: int = 5,
    sf: int = 3,
    sh: int = 2,
    sb: int = 10,
    cs: int = 3,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=28,
        pa=pa,
        ab=ab,
        h=h,
        singles=singles,
        doubles=doubles,
        triples=triples,
        hr=hr,
        bb=bb,
        so=so,
        hbp=hbp,
        sf=sf,
        sh=sh,
        sb=sb,
        cs=cs,
        r=80,
        rbi=90,
    )


def _make_pitching_stats(
    player_id: str = "p1",
    name: str = "Pitcher",
    year: int = 2024,
    ip: float = 180.0,
    g: int = 32,
    gs: int = 32,
    er: int = 60,
    h: int = 150,
    bb: int = 50,
    so: int = 180,
    hr: int = 20,
    hbp: int = 5,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=28,
        ip=ip,
        g=g,
        gs=gs,
        er=er,
        h=h,
        bb=bb,
        so=so,
        hr=hr,
        hbp=hbp,
    )


class FakeDataSource:
    def __init__(
        self,
        batting: dict[int, list[BattingSeasonStats]],
        pitching: dict[int, list[PitchingSeasonStats]],
    ) -> None:
        self._batting = batting
        self._pitching = pitching

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return self._batting.get(year, [])

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        return self._pitching.get(year, [])

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        return []

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        return []


class TestBattingStatsToProjection:
    def test_field_conversion(self) -> None:
        stats = _make_batting_stats(pa=600, hr=25, sb=10)
        proj = batting_stats_to_projection(stats)
        assert proj.player_id == "b1"
        assert proj.name == "Hitter"
        assert proj.year == 2024
        assert proj.pa == 600.0
        assert proj.hr == 25.0
        assert proj.sb == 10.0
        assert isinstance(proj.pa, float)
        assert isinstance(proj.hr, float)


class TestPitchingStatsToProjection:
    def test_era_whip_computed(self) -> None:
        stats = _make_pitching_stats(ip=180.0, er=60, h=150, bb=50)
        proj = pitching_stats_to_projection(stats)
        assert proj.era == pytest.approx(60 / 180.0 * 9)
        assert proj.whip == pytest.approx((150 + 50) / 180.0)

    def test_zero_ip(self) -> None:
        stats = _make_pitching_stats(ip=0.0, er=0, h=0, bb=0)
        proj = pitching_stats_to_projection(stats)
        assert proj.era == 0.0
        assert proj.whip == 0.0


class TestActualsAsProjections:
    def test_returns_batting_and_pitching(self) -> None:
        ds = FakeDataSource(
            batting={2024: [_make_batting_stats()]},
            pitching={2024: [_make_pitching_stats()]},
        )
        batting, pitching = actuals_as_projections(ds, 2024)
        assert len(batting) == 1
        assert len(pitching) == 1

    def test_min_pa_filter(self) -> None:
        ds = FakeDataSource(
            batting={
                2024: [
                    _make_batting_stats(player_id="b1", pa=100),
                    _make_batting_stats(player_id="b2", pa=500),
                ]
            },
            pitching={2024: []},
        )
        batting, _ = actuals_as_projections(ds, 2024, min_pa=200)
        assert len(batting) == 1
        assert batting[0].player_id == "b2"

    def test_min_ip_filter(self) -> None:
        ds = FakeDataSource(
            batting={2024: []},
            pitching={
                2024: [
                    _make_pitching_stats(player_id="p1", ip=30.0),
                    _make_pitching_stats(player_id="p2", ip=100.0),
                ]
            },
        )
        _, pitching = actuals_as_projections(ds, 2024, min_ip=50.0)
        assert len(pitching) == 1
        assert pitching[0].player_id == "p2"
